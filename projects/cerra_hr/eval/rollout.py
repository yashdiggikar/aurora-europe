import torch
from aurora.batch import Batch, Metadata

@torch.no_grad()
def autoregressive_rollout(model, init_batch: Batch, steps: int, dt_hours: int):
    model.eval()
    batch = init_batch.cuda()
    outs = []
    for k in range(steps):
        out = model(batch)  # predicts t+Δ
        outs.append(out)
        next_surf  = {k: v.unsqueeze(1) for k, v in out.surf_vars.items()}   # [B,1,H,W]
        next_atmos = {k: v.unsqueeze(1) for k, v in out.atmos_vars.items()}  # [B,1,L,H,W]
        meta = Metadata(
            lat=batch.metadata.lat, lon=batch.metadata.lon,
            time=(f"{k+1}",), atmos_levels=batch.metadata.atmos_levels,
            rollout_step=(batch.metadata.rollout_step + 1) if batch.metadata.rollout_step is not None else (k+1),
        )
        batch = Batch(surf_vars=next_surf, atmos_vars=next_atmos, static_vars=batch.static_vars, metadata=meta).cuda()
    return outs
