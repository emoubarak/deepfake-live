#!/usr/bin/env bash
# Switches the live mode: ./mode.sh swap | animate [photo]; the bridge restarts, the camera sender reconnects by itself.
# LIVE_DIR: log, snapshot and recalibration flag (default ~/scratch/deepfake-direct); the default photo is
# $LIVE_DIR/face-source-1.png. PROVIDER: tensorrt (default, needs TensorRT in the venv) or cuda.
# The bridge runs for at most 2 h (--seconds 7200).
FF="${FACEFUSION_DIR:-$HOME/repos/tools/facefusion}"; LIVE="${LIVE_DIR:-$HOME/scratch/deepfake-direct}"
MODE="${1:-swap}"; SRC="${2:-$LIVE/face-source-1.png}"; D="$(dirname "$(readlink -f "$0")")"; P="${PROVIDER:-tensorrt}"
case "$MODE" in
  swap) ARGS=(--mode swap --model hyperswap_1c_256 --weight 1.0 --pixel-boost 512x512 --expression 80 --provider "$P") ;;
  animate) ARGS=(--mode animate --provider "$P") ;;
  *) echo "unknown mode: $MODE (swap | animate)"; exit 1 ;;
esac
[ -f "$SRC" ] || { echo "photo not found: $SRC" >&2; exit 1; }
pgrep -f "^$FF/.venv/bin/python .*bridge.py" | xargs -r kill; sleep 2
mkdir -p "$LIVE"; : > "$LIVE/live.log"
nohup "$D/run-live.sh" --source "$SRC" "${ARGS[@]}" --seconds 7200 > "$LIVE/live.log" 2>&1 &
echo "bridge restarted in $MODE mode; metrics: tail -f $LIVE/live.log"
