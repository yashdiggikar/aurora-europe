# Aurora Europe 5.5 km 

Fine-tuning Microsoft **Aurora** for **Europe** at **native 5.5 km** resolution using **CERRA (2021–2023)**.  
This project follows a **project-layer only** approach — **no core edits** to `aurora/`. We configure variables/region, build the pretrained model with our settings, and fine-tune in stages (decoder → level_agg → backbone) using **BF16** + **activation checkpointing**.

---

## Quickstart

1. **Configure region & grid**  
   Edit `configs/region_europe.yaml` with exact `H`, `W`, `levels_hpa`, `bbox`, `dt_hours`, and ensure `patch_size` divides both `H` and `W`.

2. **Sanity check forward**  
   Run `scripts/smoke_forward.py` — it should print shapes like `OK 2t: torch.Size([1, H, W])`.

3. **Prepare data & stats**  
   Place CERRA NetCDF under `data/inter/nc/...`, then run `dataio/compute_stats.py` to create `data/stats/cerra_norm_stats.json`.

4. **Fine-tune (Path B)**  
   Run `train/train_pathb.py` (uses BF16 + activation checkpointing). Checkpoints are saved to `runs/ckpts/` by default.

5. **Evaluate / Rollout**  
   Use `eval/` tools (e.g., `rollout.py`, `rmse_acc.py`) to generate forecasts and compute metrics (RMSE/ACC).

---

## How this project works (overview)

- **Instantiate** the official `AuroraPretrained` with our **vars** and **Europe grid**.
- **Load** the official **pretrained checkpoint** with `strict=False` (since we use `rh` instead of `q`).
- **Zero-init** the new `rh` patch/token embedding; enable **BF16** and **activation checkpointing**.
- **Fine-tune in stages** as configured in `configs/train_pathB.yaml`.
- **Save** CERRA fine-tuned weights under `runs/ckpts/`.

---

## File & Directory Guide 

### `configs/`
- **`region_europe.yaml`** — Defines **domain** (lat/lon bbox), **grid** (`H`,`W`), **pressure levels**, `dt_hours`, and `patch_size`. Must match your actual CERRA data.
- **`vars.yaml`** — Lists **input/target variables**. Uses `rh` (relative humidity) instead of `q` in `atmos_vars`, plus surface/static sets.
- **`train_pathB.yaml`** — **Training hyperparameters** (batch size, LR, epochs, grad clipping), **staged unfreezing** (`decoder` → `level_agg` → `backbone`), **loss weights** per variable, save/validation cadence, and `save_dir`.

### `dataio/`
- **`compute_stats.py`** — Computes per-variable (and per-level) **mean/std** from your CERRA training split. Writes `data/stats/cerra_norm_stats.json`.
- **`loader_pathb.py`** — PyTorch **Dataset/DataLoader** for Path B: reads CERRA NetCDF, crops to region/levels, assembles **history**, **normalizes** using stats, builds `Batch` + `Metadata`, returns `(batch, targets)`.
- **`convert_grib_to_nc.py`** *(optional)* — Helper to **convert GRIB → NetCDF** with consistent names/coords.

### `models/`
- **`build_model_eu.py`** — Builder that **instantiates Aurora** with your vars/grid, enables **BF16** + **activation checkpointing**, **loads pretrained** with `strict=False`, **zero-inits `rh`** embedding, moves to GPU, and returns a **ready-to-train** model. Supports loading your fine-tuned ckpt via `ckpt_path`.

### `train/`
- **`train_pathb.py`** — **Main fine-tune script**. Loads configs, builds model, creates loaders, and runs **staged training**:  
  Stage 1 = `decoder` → Stage 2 = `decoder + level_agg` → Stage 3 = `decoder + level_agg + backbone`.  
  Uses **gradient clipping**, periodic **validation**, and **saves checkpoints** to `runs/ckpts/`.

### `eval/`
- **`rollout.py`** — **Autoregressive forecasting** utilities (multi-step rollout): feed model outputs back as inputs to produce forecast sequences.
- **`rmse_acc.py`** — Metric helpers for **RMSE** and **Anomaly Correlation (ACC)**, with optional **latitude weighting**.

### `losses/`
- **`lat_weight.py`** — **Cosine-latitude weighting** helpers for loss/metrics (area-aware to avoid high-lat dominance).

### `utils/`
- **`determinism.py`** — `set_seed(seed)` for **reproducibility** (Python/NumPy/PyTorch seeds + safe backend flags).
- **`grids.py`** — `latlon_mesh(...)` builds `[H, W]` **lat/lon 2D arrays** used in `Metadata` (positional context and area weighting).

### `scripts/`
- **`smoke_forward.py`** — **Sanity check**: builds a tiny dummy batch (or zeros), calls `build_model_eu`, runs one forward pass, prints output shapes like `[1, H, W]`. Use before training to catch config/shape issues.

---

## Training Details 

- **Precision & memory** — BF16 for speed/stability on modern GPUs; **activation checkpointing** to save memory during backprop.
- **Staged unfreezing**  
  - **Stage 1 – Decoder**: Stabilize heads on new vars/domain.  
  - **Stage 2 – +Level Aggregation**: Adapt **vertical mixing** across pressure levels.  
  - **Stage 3 – +Backbone**: Carefully adapt the large core with a **smaller LR**.
- **Variables** — If you add/replace vars (e.g., `rh` for `q`), load with `strict=False` and **zero-init** the new token embedding.
- **Normalization** — Always use **CERRA-specific** stats (`cerra_norm_stats.json`) for both inputs and targets if loss is computed in normalized space.

---

## Checkpoints

- **Default save dir** — `runs/ckpts/` (configurable via `configs/train_pathB.yaml → save_dir`).
- **Files** — `epochXX.pt`, `best.pt`, `final.pt`.
- **Load logic**  
  - **Start training** from official pretrained: builder calls `load_checkpoint(strict=False)` **without** a path.  
  - **Resume/Inference** with your weights: builder calls `load_checkpoint(path="runs/ckpts/best.pt", strict=False)`.

---

## Tips 

- Ensure `patch_size` **divides** both `H` and `W`.
- `levels_hpa` in `region_europe.yaml` must match your dataset’s pressure levels exactly.
- Compute stats **after** your data is in place and **for the exact var set** you train.
- If GPU memory is tight: lower batch size, keep BF16 + activation checkpointing, and use gradient clipping (already enabled).
