#!/usr/bin/env python3
"""Offline test of the animated photo mode: a driving video -> animated video, with the measured speed.
usage (FaceFusion venv): animate_test.py photo.png driving.mp4 out.mp4
ANIM_PROVIDER: FaceFusion execution provider (tensorrt by default, or cuda).
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

FF_DIR = os.environ.get('FACEFUSION_DIR') or os.path.expanduser('~/src/facefusion')
sys.path.insert(0, FF_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
photo_path, driving_path, out_path = (os.path.abspath(p) for p in sys.argv[1:4])
os.chdir(FF_DIR)

import cv2  # noqa: E402

from facefusion import logger, state_manager  # noqa: E402
from facefusion.args import apply_args  # noqa: E402
from facefusion.core import common_pre_check, processors_pre_check  # noqa: E402
from facefusion.program import create_program  # noqa: E402

apply_args(vars(create_program().parse_args(['run', '--source-paths', photo_path, '--processors', 'face_editor',
                                             '--execution-providers', os.environ.get('ANIM_PROVIDER', 'tensorrt'), '--execution-thread-count', '4',
                                             '--log-level', 'warn'])), state_manager.init_item)
logger.init('warn')
if not common_pre_check() or not processors_pre_check():
    sys.exit('missing models')

from animate import PhotoAnimator  # noqa: E402

animator = PhotoAnimator(cv2.imread(photo_path))
capture = cv2.VideoCapture(driving_path)
frames = []
while True:
    ok, frame = capture.read()
    if not ok:
        break
    frames.append(frame)
writer = cv2.VideoWriter(out_path + '.tmp.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, animator.canvas)
start = time.time()
with ThreadPoolExecutor(max_workers=4) as pool:
    for frame in frames[:15]:  # calibration: neutral pose = average of the first frames
        animator(frame)
    for result in pool.map(animator, frames):
        writer.write(result)
elapsed = time.time() - start
writer.release()
os.system(f'ffmpeg -hide_banner -loglevel error -y -i "{out_path}.tmp.mp4" -c:v libx264 -crf 20 -pix_fmt yuv420p "{out_path}" && rm -f "{out_path}.tmp.mp4"')
print(f'{len(frames)} frames in {elapsed:.1f} s: {len(frames) / elapsed:.1f} fps -> {out_path}')
