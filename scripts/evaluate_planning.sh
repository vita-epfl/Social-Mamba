#!/usr/bin/env bash
set -euo pipefail
CKPT=${1:-checkpoints/jrdb/social_mamba_jrdb.pth.tar}
python -m social_mamba.evaluate_planning --ckpt "$CKPT" --methods zero,cv,model --neighbor-policy cv
