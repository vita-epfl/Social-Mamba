#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

DATASET_NAME=${DATASET_NAME:-NBA}
PREPROCESS_SPLIT=${PREPROCESS_SPLIT:-all}
RAW_DIR=${RAW_DIR:-$REPO_ROOT/extract_data/NBA/raw}
OBS_LEN=${OBS_LEN:-10}
PRED_LEN=${PRED_LEN:-20}
MAX_AGENTS=${MAX_AGENTS:-8}
CHUNK_SIZE=${CHUNK_SIZE:-4}

case "$DATASET_NAME" in
  NBA|NBA_full) ;;
  *) echo "DATASET_NAME must be NBA or NBA_full" >&2; exit 1 ;;
esac
case "$PREPROCESS_SPLIT" in
  all) splits=(train val test) ;;
  train|val|test) splits=("$PREPROCESS_SPLIT") ;;
  *) echo "PREPROCESS_SPLIT must be train, val, test, or all" >&2; exit 1 ;;
esac
if [[ "$DATASET_NAME" == NBA_full && "$MAX_AGENTS" != 11 ]]; then
  echo "NBA-Full requires MAX_AGENTS=11; eight-agent caches are incompatible." >&2
  exit 1
fi

uhm_dir="$REPO_ROOT/data/UniHuMotion/UHM_$DATASET_NAME"
cache_root=${SOCIAL_MAMBA_CACHE_DIR:-$REPO_ROOT/data/cache}
shopt -s nullglob
for split in "${splits[@]}"; do
  for existing in "$uhm_dir/$split/"*.ndjson "$cache_root/$DATASET_NAME/$split/"*.pkl; do
    echo "Existing preprocessing output: $existing. Move it aside before rebuilding." >&2
    exit 1
  done
done

mkdir -p "$REPO_ROOT/extract_data/NBA/output_csv"
csv_workdir=$(mktemp -d "$REPO_ROOT/extract_data/NBA/output_csv/preprocess.XXXXXX")
conversion_workdir=$(mktemp -d "$REPO_ROOT/UniHuMotion_trajnetpp/output_nba.XXXXXX")
trap 'rm -rf "$csv_workdir" "$conversion_workdir"' EXIT

for split in "${splits[@]}"; do
  python extract_data/NBA/extract_NBA.py --split "$split" --raw-dir "$RAW_DIR" --output-dir "$csv_workdir"
  case "$split" in
    train) train_fraction=1.0; val_fraction=0.0 ;;
    val) train_fraction=0.0; val_fraction=1.0 ;;
    test) train_fraction=0.0; val_fraction=0.0 ;;
  esac
  (
    cd UniHuMotion_trajnetpp
    python -m trajnetdataset.convert \
      --input_file "$csv_workdir/nba_$split.csv" \
      --output_dir "$conversion_workdir" \
      --acceptance 1.0 1.0 1.0 1.0 \
      --train_fraction "$train_fraction" \
      --val_fraction "$val_fraction" \
      --fps 5 --obs_len "$OBS_LEN" --pred_len "$PRED_LEN" --chunk_stride 1
  )
  mkdir -p "$uhm_dir/$split"
  cp "$conversion_workdir/$split/"*.ndjson "$uhm_dir/$split/"
done

python UniHuMotion_cache/cache_generator.py \
  --UniHuMotion_dataset "UHM_$DATASET_NAME" \
  --split "$PREPROCESS_SPLIT" --obs-len "$OBS_LEN" \
  --max-agents "$MAX_AGENTS" --chunk-size "$CHUNK_SIZE"
