#!/usr/bin/env bash
set -euo pipefail

OBS_LEN=${OBS_LEN:-10}
PRED_LEN=${PRED_LEN:-20}
MAX_AGENTS=${MAX_AGENTS:-8}
CHUNK_SIZE=${CHUNK_SIZE:-4}

for split in train val test; do
  python extract_data/NBA/extract_NBA.py --split "$split"
done

mkdir -p UniHuMotion_trajnetpp/data/UniHuMotion_NBA
cp extract_data/NBA/output_csv/nba_train.csv UniHuMotion_trajnetpp/data/UniHuMotion_NBA/
cp extract_data/NBA/output_csv/nba_val.csv UniHuMotion_trajnetpp/data/UniHuMotion_NBA/
cp extract_data/NBA/output_csv/nba_test.csv UniHuMotion_trajnetpp/data/UniHuMotion_NBA/
mkdir -p data/UniHuMotion/UHM_NBA/train data/UniHuMotion/UHM_NBA/val data/UniHuMotion/UHM_NBA/test

(
  cd UniHuMotion_trajnetpp

  rm -rf output_pre
  python -m trajnetdataset.convert \
    --acceptance 1.0 1.0 1.0 1.0 \
    --train_fraction 1.0 \
    --val_fraction 0.0 \
    --fps 5 \
    --obs_len "$OBS_LEN" \
    --pred_len "$PRED_LEN" \
    --chunk_stride 1
  cp output_pre/train/*.ndjson ../data/UniHuMotion/UHM_NBA/train/

  rm -rf output_pre
  python -m trajnetdataset.convert \
    --acceptance 1.0 1.0 1.0 1.0 \
    --train_fraction 0.0 \
    --val_fraction 1.0 \
    --fps 5 \
    --obs_len "$OBS_LEN" \
    --pred_len "$PRED_LEN" \
    --chunk_stride 1
  cp output_pre/val/*.ndjson ../data/UniHuMotion/UHM_NBA/val/

  rm -rf output_pre
  python -m trajnetdataset.convert \
    --acceptance 1.0 1.0 1.0 1.0 \
    --train_fraction 0.0 \
    --val_fraction 0.0 \
    --fps 5 \
    --obs_len "$OBS_LEN" \
    --pred_len "$PRED_LEN" \
    --chunk_stride 1
  cp output_pre/test/*.ndjson ../data/UniHuMotion/UHM_NBA/test/
)

python UniHuMotion_cache/cache_generator.py \
  --UniHuMotion_dataset UHM_NBA \
  --split all \
  --obs-len "$OBS_LEN" \
  --max-agents "$MAX_AGENTS" \
  --chunk-size "$CHUNK_SIZE"
