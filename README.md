# Social-Mamba

<p align="center">
  <img src="assets/social-mamba-logo.png" alt="Social-Mamba logo" width="180">
</p>

Official code release for **Social-Mamba**, an ECCV 2026 paper on socially aware trajectory forecasting built on the Multi-Transmotion unified human motion framework.

[[Paper]](https://arxiv.org/abs/2605.15424)

<p align="center">
  <em>"Job's not finished."</em> - Kobe Bryant<br>
  <strong>Mamba is back on the NBA court.</strong>
</p>

<p align="center">
  <img src="assets/kobe.png" alt="Mamba-inspired basketball court illustration" width="720">
</p>

## Repository Layout

```text
social_mamba/              # model, training, and evaluation code
social_mamba/configs/      # public configs for NBA, NBA event splits, JRDB, and SDD
social_mamba/utils/        # metrics and shared utilities
UniHuMotion_trajnetpp/     # Multi-Transmotion CSV-to-UniHuMotion converter
UniHuMotion_cache/         # UniHuMotion-to-cache generator used by training/eval
extract_data/NBA/          # NBA raw npy-to-csv extraction example
scripts/                   # small runnable examples
checkpoints/               # local checkpoint placement, not tracked by git
data/                      # local data/caches, not tracked by git
```

## Installation

Create an environment with Python 3.10 or newer. Install PyTorch for your CUDA version first, following the official PyTorch instructions, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

`mamba-ssm` may require a CUDA-enabled PyTorch build. If installation fails, install CUDA/PyTorch-compatible `mamba-ssm` and `causal-conv1d` wheels for your machine.

## Data Format

Social-Mamba uses the same two-stage preprocessing pipeline as Multi-Transmotion:

```text
raw dataset -> UniHuMotion .ndjson -> training cache .pkl
```

The preprocessing and cache format are from Multi-Transmotion. The raw/benchmark data sources used by our experiments follow the established sources from prior trajectory-forecasting work:

- NBA follows the LED data setup: https://github.com/MediaBrain-SJTU/LED
- SDD follows the SocialVAE data setup: https://github.com/xupei0610/SocialVAE
- JRDB follows the NMRF data setup: https://github.com/AdaCompNUS/NMRF_TrajectoryPrediction

Please download the raw datasets according to the upstream repositories and their licenses, then use the bundled Multi-Transmotion-style conversion scripts to build the Social-Mamba cache files.

The training and evaluation code reads cache files from:

```text
data/cache/<dataset>/<split>/*.pkl
```

Examples:

```text
data/cache/NBA/train/*.pkl
data/cache/NBA/val/*.pkl
data/cache/NBA/test/*.pkl
data/cache/nba_score/test/*.pkl
data/cache/nba_rebound/test/*.pkl
data/cache/jrdb/test/*.pkl
data/cache/SDD/test/*.pkl
```

To store caches elsewhere, set:

```bash
export SOCIAL_MAMBA_CACHE_DIR=/path/to/data/cache
```

### UniHuMotion Schema

The bundled converter writes Multi-Transmotion-style `.ndjson` files. Each row follows the unified human motion representation with frame id, agent id, trajectory, boxes, and pose attributes. Missing values are represented as `null`.

```text
[0]       frame id
[1]       agent id
[2:4]     trajectory x, y
[4:8]     3D bounding box h, w, l, rot_z
[8:12]    2D bounding box left, top, width, height
[12:129]  3D pose keypoints, x/y/z for 39 joints
[129:207] 2D pose keypoints, x/y for 39 joints
```

For trajectory-only datasets such as NBA/SDD-style inputs, unavailable box and pose values can remain `null`; the cache generator preserves the full 81-token layout expected by the model code.

## Preprocessing

### Preprocess NBA From Raw Files

Prepare the NBA trajectories following the LED/Multi-Transmotion data convention.

Place the raw NBA files as:

```text
extract_data/NBA/raw/nba_train.npy
extract_data/NBA/raw/nba_test.npy
```

Then run the bundled end-to-end script:

```bash
bash scripts/preprocess_nba.sh
```

This script performs three steps:

```text
extract_data/NBA/raw/*.npy
  -> extract_data/NBA/output_csv/nba_{train,val,test}.csv
  -> data/UniHuMotion/UHM_NBA/{train,val,test}/*.ndjson
  -> data/cache/NBA/{train,val,test}/*.pkl
```

The NBA defaults match the public config: `obs_len=10`, `pred_len=20`, `max_agents=8`, and `chunk_size=4`. They can be overridden:

```bash
OBS_LEN=10 PRED_LEN=20 MAX_AGENTS=8 CHUNK_SIZE=4 bash scripts/preprocess_nba.sh
```

For the NBA event-split tasks, use the same LED/Multi-Transmotion data convention and cache names `nba_score` and `nba_rebound`. These configs use `obs_len=8` and `pred_len=12`.

### Build Caches From UniHuMotion `.ndjson`

For JRDB, SDD, or any dataset already converted to UniHuMotion `.ndjson`, first prepare the raw trajectories using the corresponding upstream convention, then arrange converted files as:

```text
data/UniHuMotion/UHM_<DATASET>/train/*.ndjson
data/UniHuMotion/UHM_<DATASET>/val/*.ndjson
data/UniHuMotion/UHM_<DATASET>/test/*.ndjson
```

Then generate caches:

```bash
python UniHuMotion_cache/cache_generator.py \
  --UniHuMotion_dataset UHM_jrdb \
  --split all \
  --obs-len 9 \
  --max-agents 8 \
  --chunk-size 4
```

For SDD:

```bash
python UniHuMotion_cache/cache_generator.py \
  --UniHuMotion_dataset UHM_SDD \
  --split all \
  --obs-len 8 \
  --max-agents 8 \
  --chunk-size 4
```

For NBA, the equivalent cache-only command is:

```bash
python UniHuMotion_cache/cache_generator.py \
  --UniHuMotion_dataset UHM_NBA \
  --split all \
  --obs-len 10 \
  --max-agents 8 \
  --chunk-size 4
```

## Checkpoints

Pretrained weights are distributed through this repository's GitHub Releases page rather than tracked in git. Download the release assets and place them in this layout:

```text
checkpoints/nba/social_mamba_nba.pth.tar
checkpoints/nba_score/social_mamba_nba_score.pth.tar
checkpoints/nba_rebound/social_mamba_nba_rebound.pth.tar
checkpoints/jrdb/social_mamba_jrdb.pth.tar
checkpoints/sdd/social_mamba_sdd.pth.tar
```

The planned release assets are:

```text
social_mamba_nba.pth.tar
social_mamba_nba_score.pth.tar
social_mamba_nba_rebound.pth.tar
social_mamba_jrdb.pth.tar
social_mamba_sdd.pth.tar
```

If you publish from a fork or mirror, attach the same files to the corresponding GitHub Release and keep the local filenames unchanged.

## Evaluation

NBA:

```bash
python -m social_mamba.evaluate \
  --ckpt checkpoints/nba/social_mamba_nba.pth.tar \
  --metric ade_fde
```

NBA-score:

```bash
python -m social_mamba.evaluate \
  --ckpt checkpoints/nba_score/social_mamba_nba_score.pth.tar \
  --metric ade_fde
```

NBA-rebound:

```bash
python -m social_mamba.evaluate \
  --ckpt checkpoints/nba_rebound/social_mamba_nba_rebound.pth.tar \
  --metric ade_fde
```

JRDB:

```bash
python -m social_mamba.evaluate_jrdb \
  --ckpt checkpoints/jrdb/social_mamba_jrdb.pth.tar \
  --metric ade_fde
```

SDD:

```bash
python -m social_mamba.evaluate_sdd \
  --ckpt checkpoints/sdd/social_mamba_sdd.pth.tar \
  --metric ade_fde
```

The helper scripts in `scripts/` run the same commands with default checkpoint paths.

## Training

A minimal JRDB training command is:

```bash
python -m social_mamba.train \
  --cfg social_mamba/configs/UniHuMotion_jrdb.yaml \
  --exp_name social_mamba_jrdb
```

Weights & Biases logging is disabled by default. Enable it explicitly after logging in locally with `wandb login`:

```bash
python -m social_mamba.train \
  --cfg social_mamba/configs/UniHuMotion_jrdb.yaml \
  --exp_name social_mamba_jrdb \
  --use_wandb
```

For the NBA event-split tasks, use `social_mamba/configs/UniHuMotion_nba-score.yaml` or `social_mamba/configs/UniHuMotion_nba-rebound.yaml`.

## Notes for Reproducibility

- `padding_mask` follows the PyTorch convention: `True` means padded/invalid.
- The default social-grid sorting mode is `none`; pass `--social_grid_type distance`, `risk`, `intent`, `kinematic`, or `random` to override it.
- In JRDB/SDD experiments the effective padded agent budget is fixed at `N=8`. Runtime therefore reflects a fixed agent-token budget in the reported setting. If the configured agent budget is increased, Social-Mamba scales approximately linearly with the padded number of agents because its Mamba blocks process serialized agent-time tokens.
- SDD pixel-scale reporting uses `data/SDD_meter2pixel_scale.npy` when available; otherwise normalized coordinates are reported.

## Acknowledgements

This implementation builds on the Multi-Transmotion codebase and unified cache format.

```bibtex
@inproceedings{gao2024multi,
  title={Multi-Transmotion: Pre-trained Model for Human Motion Prediction},
  author={Gao, Yang and Luan, Po-Chien and Alahi, Alexandre},
  booktitle={8th Annual Conference on Robot Learning},
  year={2024}
}
```

A Social-Mamba citation will be added after publication.
