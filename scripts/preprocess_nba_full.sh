#!/usr/bin/env bash
set -euo pipefail
export DATASET_NAME=NBA_full
export MAX_AGENTS=${MAX_AGENTS:-11}
exec bash "$(dirname "${BASH_SOURCE[0]}")/preprocess_nba.sh" "$@"
