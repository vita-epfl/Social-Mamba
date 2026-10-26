#!/usr/bin/env bash
set -euo pipefail
CKPT=${1:-checkpoints/nba_full/social_mamba_nba_full.pth.tar}
python -m social_mamba.evaluate --ckpt "$CKPT" --metric ade_fde
