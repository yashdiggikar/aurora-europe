# aurora-europe/projects/cerra_hr/train/train_pathb.py
"""
Path B fine-tuning loop for Aurora on CERRA Europe (5.5 km).
- Uses bf16 + activation checkpointing (per official docs)
- Gradient clipping
- Supports 'rh' (relative humidity) instead of 'q' => strict=False load
- Optional higher LR for new 'rh' token embeddings
- Optional zero-init of encoder token for 'rh' is handled in build_model_eu(...)
"""

import os
import torch
import yaml
from torch.utils.data import DataLoader, random_split
from torch.nn.utils import clip_grad_norm_

from projects.cerra_hr.models.build_model_eu import build_model_eu
from projects.cerra_hr.losses.lat_weight import weighted_mse
from projects.cerra_hr.dataio.loader_pathb import CerraHRDataset
from projects.cerra_hr.utils.determinism import set_seed


# ------------------------------
# Freezing / unfreezing helpers
# ------------------------------
def freeze_except(model, parts):
    """
    Enable gradients only for parameter names containing any tag in `parts`.
    Example tags: "decoder", "level_agg", "backbone"
    """
    tags = tuple(parts)
    for n, p in model.named_parameters():
        p.requires_grad = any(tag in n for tag in tags)


# ------------------------------
# Dataloaders
# ------------------------------
def make_dataloaders(cfg_vars, cfg_region, cfg_train):
    """
    Build train/val dataloaders.
    Assumes data is already preprocessed into a single dataset with:
      - surface vars: [time, lat, lon]
      - atmos vars (incl. rh): [time, level, lat, lon]
    """
    ds = CerraHRDataset(
        paths=["data/inter/nc/2023/cerra_202301.nc"],   # <-- replace/expand with your files/globs
        stats_json="data/stats/cerra_norm_stats.json",
        lead_hours=cfg_region["dt_hours"],
        vars_surf=cfg_vars["surf_vars"],
        vars_atmos=cfg_vars["atmos_vars"],   # includes 'rh'
        vars_static=cfg_vars["static_vars"],
        bbox=cfg_region["bbox"],
        target_surf=tuple(cfg_train["target_vars"]["surf"]),
        target_atmos=tuple(cfg_train["target_vars"]["atmos"]),
    )
    n = len(ds)
    n_val = max(1, int(0.05 * n))
    n_train = max(1, n - n_val)
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg_train["batch_size"],
        shuffle=True,
        num_workers=cfg_train["num_workers"],
        pin_memory=True
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg_train["batch_size"],
        shuffle=False,
        num_workers=cfg_train["num_workers"],
        pin_memory=True
    )
    return train_dl, val_dl


# ------------------------------
# Loss
# ------------------------------
def compute_loss(pred, target, lat2d, loss_w):
    """
    pred:
      {"surf": {var: [B,H,W], ...},
       "atmos": {var: [B,L,H,W], ...}}
    target:
      {"surf": {var: [H,W], ...},
       "atmos": {var: [L,H,W], ...}}
    lat2d: [H,W]
    """
    loss = 0.0

    # surface vars
    for v, w in loss_w.items():
        if not v.startswith("surf:"):
            continue
        name = v.split(":", 1)[1]
        if name in pred["surf"] and name in target["surf"]:
            loss = loss + w * weighted_mse(pred["surf"][name], target["surf"][name], lat2d)

    # atmospheric vars (average over levels)
    for v, w in loss_w.items():
        if not v.startswith("atmos:"):
            continue
        name = v.split(":", 1)[1]
        if name in pred["atmos"] and name in target["atmos"]:
            p = pred["atmos"][name]  # [B,L,H,W]
            t = target["atmos"][name]  # [L,H,W]
            lsum = 0.0
            for li in range(p.shape[1]):
                lsum = lsum + weighted_mse(p[:, li], t[li], lat2d)
            loss = loss + w * (lsum / p.shape[1])

    return loss


def model_forward(model, batch, cfg_vars):
    """
    Run model and return dicts aligned to compute_loss expectations.
    """
    out = model(batch)
    pred = {
        "surf":  {k: out.surf_vars[k]  for k in cfg_vars["surf_vars"]  if k in out.surf_vars},
        "atmos": {k: out.atmos_vars[k] for k in cfg_vars["atmos_vars"] if k in out.atmos_vars},
    }
    return pred


# ------------------------------
# Checkpoint helpers
# ------------------------------
def maybe_save(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if hasattr(model, "save_checkpoint"):
        model.save_checkpoint(path)
    else:
        torch.save({"model": model.state_dict()}, path)


# ------------------------------
# Evaluation
# ------------------------------
def evaluate(model, val_dl, cfg_vars, cfg_train):
    model.eval()
    tot, cnt = 0.0, 0
    with torch.no_grad(), torch.cuda.amp.autocast():
        for batch, target in val_dl:
            batch = batch.cuda(non_blocking=True)
            lat2d_b = batch.metadata.lat.cuda(non_blocking=True)
            target_gpu = {
                "surf":  {k: v.cuda(non_blocking=True) for k, v in target["surf"].items()},
                "atmos": {k: v.cuda(non_blocking=True) for k, v in target["atmos"].items()},
            }
            pred = model_forward(model, batch, cfg_vars)
            loss = compute_loss(pred, target_gpu, lat2d_b, cfg_train["loss_weights"])
            tot += float(loss.item()); cnt += 1
    return tot / max(cnt, 1)


# ------------------------------
# Optimizer with higher LR for new 'rh' token embeddings
# ------------------------------
def make_optimizer(model, base_lr=3e-4, new_token_lr=1e-3):
    """
    Two-LR optimizer: higher LR for new 'rh' token embeddings if found.
    Falls back to a single group if those params aren't discoverable.
    """
    new_params, base_params = [], []
    for n, p in model.named_parameters():
        nm = n.lower()
        is_token = ("token_embed" in nm) or ("token_embeds" in nm)
        if is_token and ("rh" in nm):
            new_params.append(p)
        else:
            base_params.append(p)

    if not new_params:
        # Fallback: single param group still trains fine.
        return torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)

    return torch.optim.AdamW(
        [{"params": base_params, "lr": base_lr},
         {"params": new_params, "lr": new_token_lr}],
        weight_decay=1e-4
    )


# ------------------------------
# Train stage
# ------------------------------
def run_stage(model, train_dl, val_dl, lat2d, cfg_vars, cfg_train, train_parts, lr, epochs, clip=1.0):
    """
    One training stage (freeze/unfreeze as specified).
    """
    freeze_except(model, train_parts)
    # Build optimizer with higher LR for 'rh' token embeds when present:
    opt = make_optimizer(model, base_lr=lr, new_token_lr=max(lr * 3, 1e-3))
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    step, best_val = 0, float("inf")
    for ep in range(1, epochs + 1):
        model.train()
        for batch, target in train_dl:
            batch = batch.cuda(non_blocking=True)
            lat2d_b = batch.metadata.lat.cuda(non_blocking=True)
            target_gpu = {
                "surf":  {k: v.cuda(non_blocking=True) for k, v in target["surf"].items()},
                "atmos": {k: v.cuda(non_blocking=True) for k, v in target["atmos"].items()},
            }

            with torch.cuda.amp.autocast():
                pred = model_forward(model, batch, cfg_vars)
                loss = compute_loss(pred, target_gpu, lat2d_b, cfg_train["loss_weights"])

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            # Gradient clipping (per docs recommendation to mitigate explosions)
            # Also returns total norm (can log if desired)
            total_norm = clip_grad_norm_( [p for p in model.parameters() if p.requires_grad],
                                          max_norm=clip )
            scaler.step(opt)
            scaler.update()

            step += 1
            if step % cfg_train["val_interval_steps"] == 0:
                val_loss = evaluate(model, val_dl, cfg_vars, cfg_train)
                if val_loss < best_val:
                    best_val = val_loss
                    maybe_save(model, os.path.join(cfg_train["save_dir"], "best.pt"))

        # Save epoch checkpoint
        maybe_save(model, os.path.join(cfg_train["save_dir"], f"epoch{ep:02d}.pt"))


# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    set_seed(42)

    # Load configs
    cfg_vars   = yaml.safe_load(open("projects/cerra_hr/configs/vars.yaml"))
    cfg_region = yaml.safe_load(open("projects/cerra_hr/configs/region_europe.yaml"))
    cfg_train  = yaml.safe_load(open("projects/cerra_hr/configs/train_pathB.yaml"))

    # Build model (bf16 + activation checkpointing inside builder)
    # We extended variables (q -> rh), so strict=False and optionally zero-init RH token in builder.
    model = build_model_eu(
        cfg_vars["surf_vars"],
        cfg_vars["static_vars"],
        cfg_vars["atmos_vars"],           # includes 'rh'
        patch_size=cfg_region["patch_size"],
        bf16=True,
        stabilise=False,                   # if gradients explode, set True AND re-run
        strict=False,                      # q->rh extension => strict=False per docs
        zero_init_rh_encoder=True,         # gently introduce new var
    ).cuda()

    # Data
    train_dl, val_dl = make_dataloaders(cfg_vars, cfg_region, cfg_train)
    # Cache a lat2d example for loss weighting
    lat2d_example = next(iter(train_dl))[0].metadata.lat.cuda()

    # Run staged training (decoder -> level_agg -> backbone)
    for stage in ("stage1", "stage2", "stage3"):
        hp = cfg_train[stage]
        run_stage(
            model,
            train_dl,
            val_dl,
            lat2d=lat2d_example,
            cfg_vars=cfg_vars,
            cfg_train=cfg_train,
            train_parts=hp["train_parts"],
            lr=hp["lr"],
            epochs=hp["epochs"],
            clip=cfg_train["clip_grad_norm"],
        )

    # Final save (best already saved during training)
    maybe_save(model, os.path.join(cfg_train["save_dir"], "final.pt"))
