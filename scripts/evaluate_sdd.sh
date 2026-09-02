#!/usr/bin/env bash
set -euo pipefail
CKPT=${1:-checkpoints/sdd/social_mamba_sdd.pth.tar}
python -m social_mamba.evaluate_sdd --ckpt "$CKPT" --metric ade_fde
