# deepfake-live

Real-time face swap for a webcam over the network: the camera runs on one machine (a laptop), the GPU on another
(a desktop with an RTX card), and the swapped stream comes back to the laptop (OBS or ffplay). No sudo, no
v4l2loopback. Engine: [FaceFusion](https://github.com/facefusion/facefusion), unmodified.

```
camera machine: C920 --H.264 (NUT) over TCP--> GPU machine:5101 (bridge.py + FaceFusion) --H.264 over UDP--> camera machine:6000
```

> **Consent.** Use a generated face (for example from thispersondoesnotexist.com) or the face of a person who agreed
> to it. Do not use this to impersonate someone, to deceive, or to harass. You are responsible for how you use it.

## Requirements

- GPU machine: NVIDIA GPU, a FaceFusion checkout with its `.venv` (`FACEFUSION_DIR`, default
  `~/repos/tools/facefusion`), Python packages `av` (PyAV) in that venv. Optional: TensorRT 10.12 in the venv
  (the version FaceFusion recommends) for the speeds below.
- Camera machine: `ffmpeg`, `v4l2-ctl` (v4l-utils), a webcam (tested with a Logitech C920), OBS or `ffplay`.
- Both machines on the same private network (tested over Tailscale). The bridge listens on `--in-host` if given,
  else on the machine's Tailscale IP, else on every interface: then open 5101/tcp on the GPU machine to the camera
  machine only (the stream is not authenticated). The camera machine receives the result on 6000/udp.

## Usage

- `run-live.sh --source face.jpg [--model hyperswap_1c_256] [--weight 1.0] [--mouth real] [--enhancer gfpgan_1.4]
  [--expression 80] [--pixel-boost 512x512] [--provider tensorrt] [--out-host laptop] [--in-port 5101]
  [--out-port 6000]`: on the GPU machine (the bridge). Metrics printed every 5 s. `./run-live.sh --help` lists every
  option.
- `send-cam.sh [bridge_host] [port]`: on the camera machine, or from the GPU machine with
  `ssh <camera-host> timeout 7200 bash -s -- <bridge-host> < send-cam.sh` (the form `stop-live.sh` stops). Finds the
  camera by name (`CAM_NAME`, `C920` by default), sets a manual exposure (`CAM_EXPOSURE`, `CAM_GAIN`).
  `CAM_CODEC=mjpeg` sends raw MJPEG (~100 Mbit/s): start the bridge with `--in-format mjpeg`.
- `view-on-surface.sh`: opens an ffplay window on the camera machine (`CAM_HOST`; GNOME on Wayland, it uses the
  Xwayland auth file). In OBS: Sources > + > Media Source, untick Local File, Input `udp://0.0.0.0:6000`.
- `obs_ws.py`: drives OBS through its websocket (source settings, a screenshot to check); needs the `websockets`
  package and reads OBS's websocket config from `~/.config/obs-studio`.
- `mode.sh swap|animate [photo]`: switches the live stream between face swap and animated photo (LivePortrait), for
  at most 2 h. Default photo `$LIVE_DIR/face-source-1.png`; `PROVIDER=cuda` without TensorRT.
- `recalibrate.sh`: animate mode, takes the current pose as neutral (face the camera, mouth closed).
- `animate_test.py photo driving.mp4 out.mp4`: offline test of the animate mode.
- `stop-live.sh`: stops the bridge and the sender (anchored patterns, never a broad `pkill`).

Environment: `FACEFUSION_DIR`, `CAM_HOST` (camera machine: ssh target and default `--out-host`, default `surface`),
`BRIDGE_HOST` (GPU machine, default `minisforum`), `LIVE_DIR` (log, snapshot and recalibration flag, default
`~/scratch/deepfake-direct`), `CURL_LIMIT` (optional rate cap for FaceFusion's model downloads, e.g. `3M`).

Optional GPU arbiter: if `EGPU` points to one, or an `egpu` command is on the PATH, `run-live.sh` runs the bridge as
`egpu avec gpu --raison deepfake-live --calme 20s -- <bridge>`, so a machine that shares its GPU with a local LLM
server waits for running requests, unloads the model, and puts it back when the bridge stops. `EGPU=` (empty)
disables it; without it, the bridge simply starts.

## Measurements (RTX 3080 20 GB, C920 1280x720, hyperswap_1c_256, weight 1.0)

- 30 fps held, FaceFusion compute 44 ms per frame, GPU ~70 %.
- Latency (capture time carried in the pts, clocks NTP-synced to ~2 ms): capture -> bridge ~30 ms, capture -> bridge
  output ~75 ms, capture -> frame decoded on the laptop ~160 ms (p95 ~180). OBS and the screen come on top.
- Before tuning: 350-400 ms and 17 fps. Fixed: the C920's auto exposure (125 ms in a dark room), OBS's buffer (1 MB ≈
  1.3 s), out-of-order frames, MJPEG at ~100 Mbit/s over Wi-Fi, raw-frame pipes to ffmpeg (+60 ms each, replaced by
  PyAV), a userspace relay (the bridge now listens directly on the private interface).
- Offline speed (720p file): swap alone 47 fps; + expression restorer 14; + gfpgan 21; pixel boost 512 25;
  occlusion mask 35; real mouth (region mask) 27 offline, 18 live.

## TensorRT and shortcuts

- TensorRT 10.12.0.36 in the venv; engines cached (`.caches/`). The first run of a new model takes several minutes,
  and the bridge is frozen while the engine builds.
- `ff_speedups.py`: 87 -> 47 ms per frame (cache fingerprint, identity embedding, blurred mask, age/gender).
- Live: swap 1c + 512 px + restored mouth at 30.8 fps (58 ms compute); animated photo at 29.8 fps (73 ms).
  Before: 12 and 15 fps.

## Animated photo mode (LivePortrait, FaceFusion's models)

The whole photo (head, hair, shoulders) follows the camera's pose and expressions. Neutral pose = average of the
first 15 frames with a face; size and position locked to the photo (otherwise the face grows). Without TensorRT:
15-19 fps live (heavy LivePortrait generator).

## Pitfalls

- The C920 opens only once: if OBS holds it, the sender fails ("Device or resource busy").
- If a `tailscale serve` rule already relays port 5101, binding 0.0.0.0 fails ("Address already in use"): the
  bridge listens on the Tailscale IP or the address given with `--in-host`.

## License

MIT (these scripts). FaceFusion and its models keep their own licenses.
