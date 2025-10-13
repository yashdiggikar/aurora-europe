import torch, yaml
from datetime import datetime
from aurora.batch import Batch, Metadata
from projects.cerra_hr.models.build_model_eu import build_model_eu
from projects.cerra_hr.utils.grids import latlon_mesh

cfg_vars   = yaml.safe_load(open("projects/cerra_hr/configs/vars.yaml"))
cfg_region = yaml.safe_load(open("projects/cerra_hr/configs/region_europe.yaml"))
H, W = cfg_region["grid"]["H"], cfg_region["grid"]["W"]

B, T, L = 1, 1, 13
surf = {k: torch.zeros(B,T,H,W) for k in cfg_vars["surf_vars"]}
atmos= {k: torch.zeros(B,T,L,H,W) for k in cfg_vars["atmos_vars"]}
static={"z": torch.zeros(H,W), "lsm": torch.zeros(H,W)}
lat, lon = latlon_mesh(cfg_region["bbox"]["lat_min"], cfg_region["bbox"]["lat_max"],
                       cfg_region["bbox"]["lon_min"], cfg_region["bbox"]["lon_max"], H, W)
levels = tuple(cfg_region["levels_hpa"])

meta  = Metadata(lat=lat, lon=lon, time=(datetime(2023,1,1,0,0),), atmos_levels=levels, rollout_step=0)
batch = Batch(surf_vars=surf, atmos_vars=atmos, static_vars=static, metadata=meta)

model = build_model_eu(cfg_vars["surf_vars"], cfg_vars["static_vars"], cfg_vars["atmos_vars"],
                       patch_size=cfg_region["patch_size"], bf16=True, stabilise=False, strict=True).cuda().eval()
model.configure_activation_checkpointing()

with torch.no_grad():
    out = model(batch.cuda())
    print("OK 2t:", out.surf_vars["2t"].shape)  # Expect [1,H,W]
