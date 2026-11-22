#!/usr/bin/env python3
"""Real-time bridge: camera machine webcam -> FaceFusion (on the GPU machine) -> stream back to the camera machine (OBS, ffplay).

Input: the camera machine sends its C920 as H.264 in NUT over TCP (send-cam.sh), with the capture time in the pts;
this script listens on --in-host, else on the Tailscale IP when there is one, else on every interface
(open 5101/tcp to the camera machine only, or put the tailnet interface in a trusted firewall zone).
Output: H.264 (NVENC) in MPEG-TS over UDP to --out-host:--out-port, pts = capture time (latency can be measured at the far end).
FaceFusion is not modified: we reuse its frame-by-frame processing (process_stream_frame).

Low latency and smoothness:
- decoding and encoding happen in-process (PyAV): no raw-frame pipe to ffmpeg (+60 ms measured);
- only the latest received frame is kept, with at most --max-inflight frames being computed (beyond that, the frame is skipped);
- frames come out in camera order (FaceFusion alone returns them in completion order);
- MPEG-TS muxing with no delay (0.7 s by default otherwise).
"""
import argparse
import os
import queue
import socket
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction

FF_DIR = os.environ.get('FACEFUSION_DIR') or os.path.expanduser('~/src/facefusion')
LIVE_DIR = os.environ.get('LIVE_DIR') or os.path.expanduser('~/scratch/deepfake-direct')
START_DIR = os.getcwd()  # relative paths given on the command line are resolved from here, not from FaceFusion
sys.path.insert(0, FF_DIR)
os.chdir(FF_DIR)

import av  # noqa: E402
import cv2  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument('--source', required=True, help='photo of the head to apply (generated face or consenting person)')
parser.add_argument('--mode', choices=['swap', 'animate'], default='swap',
                    help='swap: your face is replaced; animate: the whole photo is animated by your camera (LivePortrait)')
parser.add_argument('--model', default='hyperswap_1b_256')
parser.add_argument('--weight', type=float, default=0.5, help='identity strength (0.5 neutral, 1.0 stronger identity)')
parser.add_argument('--expression', type=int, default=0, help='expression restoration (mouth, eyes) 0-100, 0 = off')
parser.add_argument('--pixel-boost', default='256x256', help='256x256, 512x512 (sharper face, slower)')
parser.add_argument('--enhancer', default='', help='e.g. gfpgan_1.4, gpen_bfr_512 (sharper, slower)')
parser.add_argument('--enhancer-blend', type=int, default=80)
parser.add_argument('--mask', nargs='+', default=['box'], help='box, occlusion (hands in front of the face), area, region')
parser.add_argument('--mouth', choices=['swap', 'real'], default='swap',
                    help='swap: the mouth comes from the swap; real: keep your real mouth (exact lips, beard around it stays visible), slightly slower')
parser.add_argument('--in-format', default='nut', choices=['nut', 'h264', 'mjpeg'], help='must match send-cam.sh')
parser.add_argument('--encoder', default='h264_nvenc', choices=['h264_nvenc', 'libx264'])
parser.add_argument('--analyse', choices=['light', 'full'], default='light',
                    help='light: scrfd 320 detection, no 68 landmarks or age/gender (not needed here), much faster; full: full analysis')
parser.add_argument('--provider', default='cuda', choices=['cuda', 'tensorrt'], help='tensorrt: faster, long first launch (engines are built, then cached)')
parser.add_argument('--fps', type=int, default=30)
parser.add_argument('--in-port', type=int, default=5101)
parser.add_argument('--in-host', default='', help='listen address (default: this machine\'s Tailscale IP if any, else every interface: open 5101/tcp to the camera machine only)')
parser.add_argument('--out-host', default=os.environ.get('CAM_HOST', 'surface'), help='camera machine that receives the swapped stream (default: $CAM_HOST or surface)')
parser.add_argument('--out-port', type=int, default=6000)
parser.add_argument('--threads', type=int, default=4)
parser.add_argument('--max-inflight', type=int, default=4, help='maximum number of frames being computed (bounds latency)')
parser.add_argument('--bitrate', type=int, default=6_000_000)
parser.add_argument('--snapshot', default=os.path.join(LIVE_DIR, 'live-snap.jpg'), help='last output frame, rewritten now and then (default: $LIVE_DIR/live-snap.jpg)')
parser.add_argument('--seconds', type=float, default=0, help='automatic stop (0 = never)')
opts = parser.parse_args()


def tailscale_ip() -> str:
    """This machine's Tailscale IPv4, or '' without Tailscale (then --in-host, or every interface)."""
    try:
        out = subprocess.run(['tailscale', 'ip', '-4'], capture_output=True, text=True, timeout=5).stdout.split()
        return out[0] if out else ''
    except (OSError, subprocess.SubprocessError):
        return ''


# --- FaceFusion: same initialization as its CLI, without launching the interface ---
from facefusion import logger, state_manager  # noqa: E402
from facefusion.args import apply_args  # noqa: E402
from facefusion.core import common_pre_check, processors_pre_check  # noqa: E402
from facefusion.program import create_program  # noqa: E402

processors = ['face_swapper'] + (['expression_restorer'] if opts.expression else []) + (['face_enhancer'] if opts.enhancer else [])
if opts.mode == 'animate':
    processors = ['face_editor']  # loads the LivePortrait models
os.makedirs(os.path.dirname(os.path.abspath(opts.snapshot)), exist_ok=True)
ff_argv = ['run', '--source-paths', os.path.join(START_DIR, os.path.expanduser(opts.source)), '--processors', *processors,
           '--face-swapper-model', opts.model, '--face-swapper-weight', str(opts.weight),
           '--face-swapper-pixel-boost', opts.pixel_boost, '--face-mask-types', *opts.mask,
           '--execution-providers', opts.provider, '--execution-thread-count', str(opts.threads),
           '--face-selector-mode', 'one', '--log-level', 'warn']
if opts.analyse == 'light':
    ff_argv += ['--face-detector-model', 'scrfd', '--face-detector-size', '320x320', '--face-landmarker-score', '0']
if opts.mouth == 'real':  # region mask without the mouth or lips: the original mouth stays
    ff_argv[ff_argv.index('--face-mask-types') + 1:ff_argv.index('--face-mask-types') + 1 + len(opts.mask)] = [*opts.mask, 'region']
    ff_argv += ['--face-mask-regions', 'skin', 'left-eyebrow', 'right-eyebrow', 'left-eye', 'right-eye', 'nose']
if opts.expression:
    ff_argv += ['--expression-restorer-factor', str(opts.expression)]
if opts.enhancer:
    ff_argv += ['--face-enhancer-model', opts.enhancer, '--face-enhancer-blend', str(opts.enhancer_blend)]
sys.argv = ['facefusion.py', *ff_argv]  # some FaceFusion modules re-read sys.argv: give them their options
apply_args(vars(create_program().parse_args(ff_argv)), state_manager.init_item)
logger.init(state_manager.get_item('log_level'))
if not common_pre_check() or not processors_pre_check():
    sys.exit('FaceFusion: missing models or pre-check failed')

if opts.analyse == 'light':  # shortcuts measured with the profiler (see ff_speedups.py)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import ff_speedups
    ff_speedups.apply()

from facefusion.content_analyser import analyse_stream  # noqa: E402
from facefusion.streamer import process_stream_frame  # noqa: E402
from facefusion.vision import read_static_images  # noqa: E402


class LatestFrame:
    """Receives the camera (TCP socket -> PyAV) and keeps only the latest decoded frame, with its capture time."""

    def __init__(self) -> None:
        self.cond = threading.Condition()
        self.frame = None
        self.capture_time = 0.0
        self.received = 0
        self.seen = 0
        self.rx_lat = []
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        host = opts.in_host or tailscale_ip() or '0.0.0.0'
        try:
            server.bind((host, opts.in_port))  # bind directly on tailscale0: no tailscale serve relay (it vanishes on its own)
        except OSError as error:
            print(f'[bridge] cannot listen on {host}:{opts.in_port} ({error}): is another bridge running? exiting', flush=True)
            os._exit(1)
        print(f'[bridge] listening on {host}:{opts.in_port}', flush=True)
        server.listen(1)
        while True:
            conn, _ = server.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print('[bridge] camera connected', flush=True)
            try:
                container = av.open(conn.makefile('rb', buffering=0), mode='r', format=opts.in_format, buffer_size=4096,
                                    options={'fflags': 'nobuffer', 'probesize': '32768', 'analyzeduration': '0'})
                stream = container.streams.video[0]
                stream.codec_context.flags |= av.codec.context.Flags.low_delay
                for packet in container.demux(stream):
                    for frame in packet.decode():
                        image = frame.to_ndarray(format='bgr24')
                        capture = float(frame.pts * frame.time_base) if frame.pts is not None else time.time()
                        now = time.time()
                        with self.cond:
                            self.frame, self.capture_time = image, capture
                            self.received += 1
                            self.rx_lat.append((now - capture) * 1000)
                            self.cond.notify_all()
            except Exception as error:  # stream cut or corrupted: go back to listening
                print(f'[bridge] input interrupted: {error}', flush=True)
            conn.close()
            print('[bridge] camera disconnected', flush=True)

    def next(self):
        """Waits for a frame newer than the last one returned; returns (image, capture time)."""
        with self.cond:
            while self.received == self.seen:
                self.cond.wait()
            self.seen = self.received
            return self.frame.copy(), self.capture_time


class Output:
    """Encodes to H.264 (NVENC, minimal latency) and sends over MPEG-TS/UDP, pts = capture time."""

    def __init__(self, width: int, height: int) -> None:
        self.container = av.open(f'udp://{opts.out_host}:{opts.out_port}?pkt_size=1316', mode='w', format='mpegts',
                                 options={'muxdelay': '0', 'muxpreload': '0', 'flush_packets': '1'})
        self.stream = self.container.add_stream(opts.encoder, rate=opts.fps)
        self.stream.width, self.stream.height, self.stream.pix_fmt = width, height, 'yuv420p'
        self.stream.bit_rate = opts.bitrate
        self.stream.time_base = Fraction(1, 1000)
        self.stream.codec_context.gop_size = opts.fps
        self.stream.codec_context.max_b_frames = 0
        if opts.encoder == 'h264_nvenc':
            self.stream.codec_context.options = {'preset': 'p1', 'tune': 'ull', 'zerolatency': '1', 'delay': '0', 'rc': 'cbr'}
        else:
            self.stream.codec_context.options = {'preset': 'ultrafast', 'tune': 'zerolatency'}
        self.last_pts = -1

    def send(self, image, capture_time: float) -> None:
        frame = av.VideoFrame.from_ndarray(image, format='bgr24').reformat(format='yuv420p')
        pts = max(int(capture_time * 1000), self.last_pts + 1)
        frame.pts, frame.time_base, self.last_pts = pts, Fraction(1, 1000), pts
        for packet in self.stream.encode(frame):
            self.container.mux(packet)


camera = LatestFrame()
source_frames = read_static_images(state_manager.get_item('source_paths'))
if opts.mode == 'animate':
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from animate import PhotoAnimator
    animator = PhotoAnimator(source_frames[0])

    def process(frame):
        return animator(frame)
else:
    def process(frame):
        return process_stream_frame(source_frames, frame)
executor = ThreadPoolExecutor(max_workers=opts.threads)
slots = threading.Semaphore(opts.max_inflight)
ordered = queue.Queue()
stats = {'out': 0, 'dropped': 0, 'calc': [], 'total': []}
stats_lock = threading.Lock()


def emitter() -> None:
    """Outputs frames in camera order, as soon as each one is computed."""
    output = None
    last_snap = 0.0
    while True:
        capture, picked, future = ordered.get()
        try:
            frame = future.result()
            if output is None:
                output = Output(frame.shape[1], frame.shape[0])
            output.send(frame, capture)
        finally:
            slots.release()
        now = time.time()
        with stats_lock:
            stats['out'] += 1
            stats['calc'].append((now - picked) * 1000)
            stats['total'].append((now - capture) * 1000)
        if now - last_snap > 5:
            cv2.imwrite(opts.snapshot, frame)
            last_snap = now


def mean_p95(values):
    values = sorted(values)
    return f'{statistics.mean(values):4.0f} ms (p95 {values[max(int(len(values) * 0.95) - 1, 0)]:4.0f})'


def reporter() -> None:
    last_out, last_rx, last_t = 0, 0, time.time()
    while True:
        time.sleep(5)
        now = time.time()
        with stats_lock:
            calc, total, stats['calc'], stats['total'] = stats['calc'], stats['total'], [], []
            out, dropped = stats['out'], stats['dropped']
        with camera.cond:
            rx, camera.rx_lat = camera.rx_lat, []
        span = now - last_t
        msg = f'[bridge] out {(out - last_out) / span:4.1f} fps | in {(camera.received - last_rx) / span:4.1f} fps'
        if rx and calc:
            msg += f' | capture->bridge {mean_p95(rx)} | compute {mean_p95(calc)} | capture->send {mean_p95(total)}'
        print(msg + f' | skipped {dropped}', flush=True)
        last_out, last_rx, last_t = out, camera.received, now


threading.Thread(target=emitter, daemon=True).start()
threading.Thread(target=reporter, daemon=True).start()
print(f'Waiting for the camera ({opts.in_format}) on {opts.in_host or tailscale_ip() or "0.0.0.0"}:{opts.in_port}, output {opts.encoder} '
      f'udp://{opts.out_host}:{opts.out_port} | processors {processors}', flush=True)
start = time.time()
while not opts.seconds or time.time() - start < opts.seconds:
    frame, capture = camera.next()
    if analyse_stream(frame, opts.fps):  # FaceFusion content check, as in its webcam mode
        print('[bridge] content rejected by FaceFusion: exiting', flush=True)
        break
    if not slots.acquire(blocking=False):
        with stats_lock:
            stats['dropped'] += 1
        continue
    ordered.put((capture, time.time(), executor.submit(process, frame)))
os._exit(0)  # end (--seconds): exit without waiting for the compute threads
