#!/usr/bin/env bash
# Stops the bridge (GPU machine) and the camera sender (camera machine), nothing else: anchored patterns, never a broad pkill.
FF="${FACEFUSION_DIR:-$HOME/src/facefusion}"; CAM_HOST="${CAM_HOST:-surface}"
pgrep -f "^$FF/.venv/bin/python .*bridge.py" | xargs -r kill
pgrep -f "^ssh $CAM_HOST timeout [0-9]+ bash -s" | xargs -r kill
timeout 15 ssh "$CAM_HOST" "pgrep -f '^timeout [0-9]+ bash -s' | xargs -r kill; pgrep -f '^ffmpeg -nostdin .* -f v4l2 ' | xargs -r kill"
