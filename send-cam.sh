#!/usr/bin/env bash
# On the camera machine, or from the GPU machine (the form stop-live.sh knows how to stop):
#   ssh <camera-host> timeout 7200 bash -s -- <bridge-host> < send-cam.sh
# Sends the Logitech C920 to the bridge.
# The camera is found by name (CAM_NAME, "C920" by default): /dev/video0 is often a laptop's built-in camera.
# CAM_CODEC=h264 (default): encoded on the camera machine, ~8 Mbit/s, light on Wi-Fi, in NUT (bridge: --in-format nut);
#   mjpeg: raw copy (~100 Mbit/s, bridge: --in-format mjpeg).
# TS_OVERLAY=1: burns the capture time in the top-left corner (latency measurement only).
# usage: ./send-cam.sh [bridge_host=$BRIDGE_HOST] [port=5101]   (the bridge host is required: argument or BRIDGE_HOST)
HOST="${1:-${BRIDGE_HOST:-}}"
[ -n "$HOST" ] || { echo "bridge host missing: ./send-cam.sh <bridge_host> [port], or set BRIDGE_HOST" >&2; exit 2; }
PORT="${2:-5101}"; NAME="${CAM_NAME:-C920}"; CODEC="${CAM_CODEC:-h264}"
DEV=""
for d in /sys/class/video4linux/video*; do
  if grep -q "$NAME" "$d/name" 2>/dev/null && [ "$(cat "$d/index")" = 0 ]; then DEV="/dev/$(basename "$d")"; break; fi
done
[ -z "$DEV" ] && { echo "camera '$NAME' not found" >&2; exit 1; }
echo "camera: $DEV ($NAME), sending $CODEC to $HOST:$PORT" >&2
# Manual exposure: on auto, the C920 goes up to 125 ms in a dark room -> 15-17 fps, blur and lag.
# CAM_EXPOSURE in 1/10 ms (156 = 15.6 ms: 30 fps; 312 drops it to 24 fps; accepted values 156, 312, 625...), CAM_GAIN 0-255; CAM_EXPOSURE=auto to leave it to the camera.
EXPO="${CAM_EXPOSURE:-156}"; GAIN="${CAM_GAIN:-230}"
apply_exposure() {  # applied once the stream is open: the C920 resets its exposure when it starts
  if [ "$EXPO" = auto ]; then
    v4l2-ctl -d "$DEV" --set-ctrl=auto_exposure=3 2>/dev/null
  else
    v4l2-ctl -d "$DEV" --set-ctrl=auto_exposure=1 2>/dev/null
    v4l2-ctl -d "$DEV" --set-ctrl=exposure_time_absolute="$EXPO" 2>/dev/null
    v4l2-ctl -d "$DEV" --set-ctrl=gain="$GAIN" 2>/dev/null
  fi
}

# -ts mono2abs + -copyts: the capture time travels in the pts all the way (the bridge measures latency)
IN=(-fflags nobuffer -f v4l2 -ts mono2abs -input_format mjpeg -video_size 1280x720 -framerate 30)
if [ "$CODEC" = mjpeg ]; then
  OUT=(-c:v copy -f mjpeg)
else
  VF="format=yuv420p"
  [ -n "$TS_OVERLAY" ] && VF="drawtext=text='%{pts\:hms}':x=20:y=20:fontsize=40:fontcolor=yellow:box=1:boxcolor=black@0.7,$VF"
  OUT=(-copyts -vf "$VF" -c:v libx264 -preset ultrafast -tune zerolatency -bf 0 -g 30 -b:v 8M -maxrate 8M -bufsize 1M
       -flush_packets 1 -f nut)  # NUT: each frame's size is known, the bridge decodes without waiting for the next one
fi
# loop: reconnects if the bridge restarts or the camera is briefly busy
while true; do
  ffmpeg -nostdin -hide_banner -loglevel error "${IN[@]}" -i "$DEV" "${OUT[@]}" "tcp://$HOST:$PORT?tcp_nodelay=1" &
  FF=$!
  sleep 2; apply_exposure
  wait "$FF"
  sleep 1
done
