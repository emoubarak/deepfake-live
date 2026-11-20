#!/usr/bin/env bash
# From the GPU machine: opens an ffplay window with the swapped stream on the camera machine's screen (no OBS needed).
CAM_HOST="${CAM_HOST:-surface}"
ssh "$CAM_HOST" 'DISPLAY=:0 XAUTHORITY=$(ls /run/user/$(id -u)/.mutter-Xwaylandauth.* | head -1) \
  setsid nohup ffplay -hide_banner -loglevel error -fflags nobuffer -flags low_delay -framedrop \
  -window_title "FaceFusion live" udp://0.0.0.0:6000 >/dev/null 2>&1 &'
