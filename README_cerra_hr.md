# Aurora Europe 5.5 km (Path B)

- Native 5.5 km fine-tune over Europe using CERRA (2021–2023).
- No core edits; project layer only.

## Quickstart
1) Edit `configs/region_europe.yaml` with exact H,W and levels.
2) Run `scripts/smoke_forward.py` — should print `[1,H,W]`.
3) Put NetCDF in `data/inter/nc/...` and compute `data/stats/cerra_norm_stats.json`.
4) Run `train/train_pathb.py` to fine-tune (bf16 + checkpointing).
