#!/usr/bin/env bash
# On the GPU machine: starts the bridge (receives the camera stream, sends the swapped stream back).
# usage: ./run-live.sh --source ~/path/face.jpg [--model hyperswap_1b_256] [--enhancer gfpgan_1.4] [--seconds 60]
# FACEFUSION_DIR: your FaceFusion checkout (default ~/src/facefusion), with its .venv.
# CURL_LIMIT=3M: optional, caps FaceFusion's model downloads (curl) at that rate, resumable.
# EGPU: optional GPU arbiter command (see README); EGPU= (empty) disables the automatic lookup.
FF="${FACEFUSION_DIR:-$HOME/src/facefusion}"
[ -x "$FF/.venv/bin/python" ] || { echo "FaceFusion venv not found in $FF (set FACEFUSION_DIR)" >&2; exit 1; }
export FACEFUSION_DIR="$FF"
HERE="$(dirname "$(readlink -f "$0")")"
NV=$(ls -d "$FF"/.venv/lib/python3.*/site-packages/nvidia/*/lib "$FF"/.venv/lib/python3.*/site-packages/tensorrt_libs 2>/dev/null | tr '\n' ':')
[ -n "$NV" ] && export LD_LIBRARY_PATH="${NV%:}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [ -n "${CURL_LIMIT:-}" ]; then
  CH="${XDG_CACHE_HOME:-$HOME/.cache}/deepfake-live/curl"; mkdir -p "$CH"
  printf 'limit-rate = %s\ncontinue-at = -\n' "$CURL_LIMIT" > "$CH/.curlrc"
  export CURL_HOME="$CH"
fi
CMD=("$FF/.venv/bin/python" "$HERE/bridge.py" "$@")
case " $* " in *" -h "*|*" --help "*) exec "${CMD[@]}" ;; esac   # help: no GPU reservation
# Optional GPU arbiter, for a machine that shares its GPU with a local LLM server: the command must accept
# `<egpu> avec gpu --raison TEXT --calme DURATION -- command…` (reserve the GPU for the command's lifetime).
EGPU="${EGPU-$(command -v egpu 2>/dev/null)}"
if [ -n "$EGPU" ] && [ -x "$EGPU" ]; then
  exec "$EGPU" avec gpu --raison "deepfake-live" --calme 20s -- "${CMD[@]}"
fi
exec "${CMD[@]}"
