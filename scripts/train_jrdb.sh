#!/usr/bin/env bash
set -euo pipefail
python -m social_mamba.train \
  --cfg social_mamba/configs/UniHuMotion_jrdb.yaml \
  --exp_name social_mamba_jrdb
