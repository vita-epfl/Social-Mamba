#!/usr/bin/env bash
set -euo pipefail
CKPT=${1:-checkpoints/nba/social_mamba_nba.pth.tar}
python -m social_mamba.evaluate --ckpt "$CKPT" --metric ade_fde
