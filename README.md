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
social_mamba/configs/      # NBA-Full, NBA/LED, NBA event splits, JRDB, and SDD
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
data/cache/NBA_full/test/*.pkl
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

**NBA-Full (Table 1):** keep all 11 agents (10 players and the ball). Run:

```bash
bash scripts/preprocess_nba_full.sh
```

This creates `data/UniHuMotion/UHM_NBA_full/{train,val,test}/*.ndjson` and `data/cache/NBA_full/{train,val,test}/*.pkl`. To build only the evaluation split:

```bash
PREPROCESS_SPLIT=test bash scripts/preprocess_nba_full.sh
```

The full test set contains all **47,940 clips**, with each of the 11 agents evaluated as primary, for **527,340 primary trajectories**. Use 10 observed frames, 20 predicted frames (4 seconds), and 20 prediction modes. No test subset or additional checkpoint selection is required. The original 81-token cache format is large; ensure sufficient storage before preprocessing.

**NBA/LED (legacy eight-agent setup):** the older `NBA` checkpoint and cache are separate from the NBA-Full Table 1 experiment. To prepare this setup instead:

```bash
bash scripts/preprocess_nba.sh
```

This script performs three steps:

```text
extract_data/NBA/raw/*.npy
  -> temporary nba_{train,val,test}.csv
  -> data/UniHuMotion/UHM_NBA/{train,val,test}/*.ndjson
  -> data/cache/NBA/{train,val,test}/*.pkl
```

Both setups use `obs_len=10`, `pred_len=20`, and `chunk_size=4`. NBA/LED uses `max_agents=8`; NBA-Full requires `max_agents=11`. Do not rename or reuse an eight-agent cache as `NBA_full`. Existing preprocessing outputs are not overwritten; move them aside before rebuilding. Raw files and the cache root can be configured:

```bash
RAW_DIR=/path/to/nba/raw SOCIAL_MAMBA_CACHE_DIR=/path/to/cache \
  PREPROCESS_SPLIT=test bash scripts/preprocess_nba_full.sh
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

For NBA-Full, the equivalent cache-only command is:

```bash
python UniHuMotion_cache/cache_generator.py \
  --UniHuMotion_dataset UHM_NBA_full \
  --split all \
  --obs-len 10 \
  --max-agents 11 \
  --chunk-size 4
```

## Checkpoints

Pretrained weights are distributed through [GitHub Releases](https://github.com/vita-epfl/Social-Mamba/releases). Download the **NBA-Full Table 1 checkpoint** `social_mamba_nba_full.pth.tar` from [v1.1](https://github.com/vita-epfl/Social-Mamba/releases/tag/v1.1). The [v1.0 archive](https://github.com/vita-epfl/Social-Mamba/releases/tag/v1.0) contains the other five checkpoints, including the legacy NBA/LED checkpoint, not the NBA-Full Table 1 model.

Place the files in this layout:

```text
checkpoints/nba_full/social_mamba_nba_full.pth.tar
checkpoints/nba/social_mamba_nba.pth.tar
checkpoints/nba_score/social_mamba_nba_score.pth.tar
checkpoints/nba_rebound/social_mamba_nba_rebound.pth.tar
checkpoints/jrdb/social_mamba_jrdb.pth.tar
checkpoints/sdd/social_mamba_sdd.pth.tar
```

The NBA-Full checkpoint has SHA-256:

```text
ee661f5ae77c56d694f97aa8b1856201cd5f4503e30925f5d294c19df7dfc64b
```

It is the epoch-81 checkpoint from `UMamba_v3_dset_NBA_full_lr_0.0005_decay_1.0_l_local_4_l_global_6_l_ego_2_l_goal_2_dropout_0.1_batch_128`.

## Evaluation

NBA-Full (Table 1):

```bash
python -m social_mamba.evaluate \
  --ckpt checkpoints/nba_full/social_mamba_nba_full.pth.tar \
  --metric ade_fde --json-out nba_full_results.json
```

Verified on a V100 using the compiled Mamba kernel:

| Setup | Input agents | Parameters | ADE (4 s) | FDE (4 s) |
| --- | ---: | ---: | ---: | ---: |
| NBA-Full, Table 1 | 11 | 1,914,030 | 0.7198 | 0.9187 |

The reported errors use LED's court-length convention: errors in exact meters are multiplied by `(1 / 0.3048) * (28 / 94)`. Evaluation reads the configuration embedded in the checkpoint, not a separate YAML file. `--batch-size` sets cache chunks per batch, normally four primary trajectories per chunk.

NBA/LED (legacy checkpoint, not Table 1):

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

For the NBA-Full architecture and training hyperparameters used in Table 1:

```bash
python -m social_mamba.train \
  --cfg social_mamba/configs/UniHuMotion_NBA_full.yaml \
  --exp_name social_mamba_nba_full
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
- NBA-Full uses local/global/ego/goal depths `4/6/2/2`, hidden dimension 128, learning rate 0.0005, and batch size 128. Its temporal and global encoders reuse their block weights across depth, matching the original implementation and the 1.9M parameter count.
- Earlier release code deep-copied those encoder blocks, inflating the parameter count; this is corrected. The original checkpoints store shared weights, so restoring sharing does not change their loaded predictions. The older released NBA checkpoint belongs to the separate NBA/LED run and does not reproduce Table 1.
- Training now saves `config.yaml` after command-line overrides. For older runs, the checkpoint's embedded config is authoritative because the on-disk YAML was saved before overrides.
- The default social-grid sorting mode is `none`; pass `--social_grid_type distance`, `risk`, `intent`, `kinematic`, or `random` to override it.
- In JRDB/SDD experiments the effective padded agent budget is fixed at `N=8`. Runtime therefore reflects a fixed agent-token budget in the reported setting. If the configured agent budget is increased, Social-Mamba scales approximately linearly with the padded number of agents because its Mamba blocks process serialized agent-time tokens.
- SDD pixel-scale reporting uses `data/SDD_meter2pixel_scale.npy` when available; otherwise normalized coordinates are reported.

Run the focused reproducibility regression tests in the installed environment:

```bash
python -m unittest discover -s tests -v
```

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
