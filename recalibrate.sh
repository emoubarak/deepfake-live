#!/usr/bin/env bash
# Animate mode: takes your current pose and expression as neutral (run it facing the camera, mouth closed).
LIVE="${LIVE_DIR:-$HOME/scratch/deepfake-direct}"
mkdir -p "$LIVE" && touch "$LIVE/recalibrate" && echo "recalibration requested (next frame)"
