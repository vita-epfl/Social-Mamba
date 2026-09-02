# Checkpoints

Large model weights are not tracked in git. Download the pretrained checkpoint assets from this repository's GitHub Releases page and place them in this layout:

```text
checkpoints/
  nba/social_mamba_nba.pth.tar
  nba_score/social_mamba_nba_score.pth.tar
  nba_rebound/social_mamba_nba_rebound.pth.tar
  jrdb/social_mamba_jrdb.pth.tar
  sdd/social_mamba_sdd.pth.tar
```

The release should include one asset for each supported benchmark:

```text
social_mamba_nba.pth.tar
social_mamba_nba_score.pth.tar
social_mamba_nba_rebound.pth.tar
social_mamba_jrdb.pth.tar
social_mamba_sdd.pth.tar
```
