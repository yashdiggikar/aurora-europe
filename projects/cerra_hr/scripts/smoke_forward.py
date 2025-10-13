# aurora-europe/projects/cerra_hr/scripts/smoke_forward.py
import torch, yaml
from datetime import datetime
from aurora.batch import Batch, Metadata
from projects.cerra_hr.models.build_model_eu import build_model_eu
from projects.cerra_hr.utils.grids import latlon_mesh

# --- Configs ---
cfg_vars   = yaml.safe_load(open("projects/cerra_hr/configs/vars.yaml"))
cfg_region = yaml.safe_load(open("projects/cerra_hr/configs/region_europe.yaml"))
H, W = cfg_region["grid"]["H"], cfg_region["grid"]["W"]
levels = tuple(cfg_region["levels_hpa"])
L = len(levels)

# --- Zero inputs (shape smoke test only) ---
B, T = 1, 1
surf  = {k: torch.zeros(B, T, H, W) for k in cfg_vars["surf_vars"]}
atmos = {k: torch.zeros(B, T, L, H, W) for k in cfg_vars["atmos_vars"]}  # includes 'rh'
static = {"z": torch.zeros(H, W), "lsm": torch.zeros(H, W)}

lat, lon = latlon_mesh(cfg_region["bbox"]["lat_min"], cfg_region["bbox"]["lat_max"],
                       cfg_region["bbox"]["lon_min"], cfg_region["bbox"]["lon_max"], H, W)
meta = Metadata(
    lat=lat, lon=lon,
    time=(datetime(2023, 1, 1, 0, 0),),
    atmos_levels=levels,
    rollout_step=0
)
batch = Batch(surf_vars=surf, atmos_vars=atmos, static_vars=static, metadata=meta)

# --- Model (bf16 + activation checkpointing; strict=False because q->rh) ---
model = build_model_eu(
    cfg_vars["surf_vars"],
    cfg_vars["static_vars"],
    cfg_vars["atmos_vars"],
    patch_size=cfg_region["patch_size"],
    bf16=True,
    stabilise=False,
    strict=False,                 # IMPORTANT: q -> rh extension
    zero_init_rh_encoder=True
).cuda().eval()
model.configure_activation_checkpointing()

with torch.no_grad():
    out = model(batch.cuda())
    print("OK 2t:", out.surf_vars["2t"].shape)  # Expect: torch.Size([1, H, W])
