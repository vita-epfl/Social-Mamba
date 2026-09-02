#!/usr/bin/env bash
set -euo pipefail
CKPT=${1:-checkpoints/jrdb/social_mamba_jrdb.pth.tar}
python -m social_mamba.evaluate_efficiency --ckpt "$CKPT" --batch-size 32 --warmup-batches 5
