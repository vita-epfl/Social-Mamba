# Checkpoints

Download `social_mamba_nba_full.pth.tar` from [v1.1](https://github.com/vita-epfl/Social-Mamba/releases/tag/v1.1) for the NBA-Full Table 1 model. The [v1.0 archive](https://github.com/vita-epfl/Social-Mamba/releases/tag/v1.0) contains the other five checkpoints. Its `social_mamba_nba.pth.tar` is the legacy NBA/LED checkpoint, not the NBA-Full model.

```text
checkpoints/
  nba_full/social_mamba_nba_full.pth.tar
  nba/social_mamba_nba.pth.tar
  nba_score/social_mamba_nba_score.pth.tar
  nba_rebound/social_mamba_nba_rebound.pth.tar
  jrdb/social_mamba_jrdb.pth.tar
  sdd/social_mamba_sdd.pth.tar
```

NBA-Full checkpoint SHA-256:

```text
ee661f5ae77c56d694f97aa8b1856201cd5f4503e30925f5d294c19df7dfc64b
```

Use a separate `NBA_full` cache with **11 agents**, 10 observed frames, and 20 predicted frames. See the root README for preprocessing and evaluation commands. The checkpoint's embedded configuration is authoritative.
