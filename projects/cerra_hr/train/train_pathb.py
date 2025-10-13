import os, torch, yaml
from torch.utils.data import DataLoader, random_split
from torch.nn.utils import clip_grad_norm_
from projects.cerra_hr.models.build_model_eu import build_model_eu
from projects.cerra_hr.losses.lat_weight import weighted_mse
from projects.cerra_hr.dataio.loader_pathb import CerraHRDataset
from projects.cerra_hr.utils.determinism import set_seed

def freeze_except(model, parts):
    tags = tuple(parts)
    for n, p in model.named_parameters():
        p.requires_grad = any(tag in n for tag in tags)

def make_dataloaders(cfg_vars, cfg_region, cfg_train):
    ds = CerraHRDataset(
        paths=["data/inter/nc/2023/cerra_202301.nc"],   # <-- put real files later
        stats_json="data/stats/cerra_norm_stats.json",
        lead_hours=cfg_region["dt_hours"],
        vars_surf=cfg_vars["surf_vars"],
        vars_atmos=cfg_vars["atmos_vars"],
        vars_static=cfg_vars["static_vars"],
        bbox=cfg_region["bbox"],
        target_surf=tuple(cfg_train["target_vars"]["surf"]),
        target_atmos=tuple(cfg_train["target_vars"]["atmos"]),
    )
    n = len(ds)
    n_val = max(1, int(0.05 * n))
    n_train = n - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    train_dl = DataLoader(train_ds, batch_size=cfg_train["batch_size"], shuffle=True,
                          num_workers=cfg_train["num_workers"], pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=cfg_train["batch_size"], shuffle=False,
                          num_workers=cfg_train["num_workers"], pin_memory=True)
    return train_dl, val_dl

def compute_loss(pred, target, lat2d, loss_w):
    loss = 0.0
    for v, w in loss_w.items():
        if v.startswith("surf:"):
            name = v.split(":",1)[1]
            if name in pred["surf"] and name in target["surf"]:
                loss = loss + w * weighted_mse(pred["surf"][name], target["surf"][name], lat2d)
    for v, w in loss_w.items():
        if v.startswith("atmos:"):
            name = v.split(":",1)[1]
            if name in pred["atmos"] and name in target["atmos"]:
                p = pred["atmos"][name]  # [B,L,H,W]
                t = target["atmos"][name]
                lsum = 0.0
                for li in range(p.shape[1]):
                    lsum = lsum + weighted_mse(p[:,li], t[li], lat2d)
                loss = loss + w * (lsum / p.shape[1])
    return loss

def model_forward(model, batch, cfg_vars):
    out = model(batch)
    pred = {
        "surf":  {k: out.surf_vars[k]  for k in cfg_vars["surf_vars"]  if k in out.surf_vars},
        "atmos": {k: out.atmos_vars[k] for k in cfg_vars["atmos_vars"] if k in out.atmos_vars},
    }
    return pred

def maybe_save(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if hasattr(model, "save_checkpoint"):
        model.save_checkpoint(path)
    else:
        torch.save({"model": model.state_dict()}, path)

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
    return tot / max(cnt,1)

def run_stage(model, train_dl, val_dl, lat2d, cfg_vars, cfg_train, train_parts, lr, epochs, clip=1.0):
    freeze_except(model, train_parts)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    step, best_val = 0, float("inf")
    for ep in range(1, epochs+1):
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
            clip_grad_norm_(params, clip)
            scaler.step(opt); scaler.update()

            step += 1
            if step % cfg_train["val_interval_steps"] == 0:
                val_loss = evaluate(model, val_dl, cfg_vars, cfg_train)
                if val_loss < best_val:
                    best_val = val_loss
                    maybe_save(model, os.path.join(cfg_train["save_dir"], "best.pt"))
        maybe_save(model, os.path.join(cfg_train["save_dir"], f"epoch{ep:02d}.pt"))

if __name__ == "__main__":
    set_seed(42)
    cfg_vars   = yaml.safe_load(open("projects/cerra_hr/configs/vars.yaml"))
    cfg_region = yaml.safe_load(open("projects/cerra_hr/configs/region_europe.yaml"))
    cfg_train  = yaml.safe_load(open("projects/cerra_hr/configs/train_pathB.yaml"))

    model = build_model_eu(cfg_vars["surf_vars"], cfg_vars["static_vars"], cfg_vars["atmos_vars"],
                           patch_size=cfg_region["patch_size"], bf16=True, stabilise=False, strict=True).cuda()
    train_dl, val_dl = make_dataloaders(cfg_vars, cfg_region, cfg_train)

    lat2d_example = next(iter(train_dl))[0].metadata.lat.cuda()
    for stage in ("stage1","stage2","stage3"):
        hp = cfg_train[stage]
        run_stage(model, train_dl, val_dl, lat2d=lat2d_example,
                  cfg_vars=cfg_vars, cfg_train=cfg_train,
                  train_parts=hp["train_parts"], lr=hp["lr"], epochs=hp["epochs"],
                  clip=cfg_train["clip_grad_norm"])
