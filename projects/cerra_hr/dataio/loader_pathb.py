import xarray as xr, numpy as np, torch, json
from torch.utils.data import Dataset
from aurora.batch import Batch, Metadata

def _to_torch(x): return torch.from_numpy(x.astype("float32"))

class CerraHRDataset(Dataset):
    def __init__(self, paths, stats_json, lead_hours, vars_surf, vars_atmos, vars_static, bbox,
                 target_surf=("2t",), target_atmos=("t","u","v")):
        self.ds = xr.open_mfdataset(paths, combine="by_coords")
        self.stats = json.load(open(stats_json))
        self.lead = np.timedelta64(lead_hours, "h")
        self.ds = self.ds.sel(latitude=slice(bbox["lat_max"], bbox["lat_min"]),
                              longitude=slice(bbox["lon_min"], bbox["lon_max"]))
        times = self.ds.time.values
        ts = set(times)
        self.pairs = [(t, t+self.lead) for t in times if (t+self.lead) in ts]
        self.VS, self.VA, self.VC = tuple(vars_surf), tuple(vars_atmos), tuple(vars_static)
        self.TS, self.TA = tuple(target_surf), tuple(target_atmos)

        self.lat = _to_torch(self.ds.latitude.values).unsqueeze(1).repeat(1, self.ds.longitude.size)
        self.lon = _to_torch(self.ds.longitude.values).unsqueeze(0).repeat(self.ds.latitude.size, 1)
        self.levels = tuple(self.ds.level.values.tolist()) if "level" in self.ds[self.VA[0]].dims else ()

    def norm(self, x, name):
        st = self.stats.get(name, None)
        if not st: return (x - float(x.mean())) / (float(x.std()) or 1.0)
        m, s = st["mean"], st["std"]
        return (x - m) / (s or 1.0)

    def __len__(self): return len(self.pairs)

    def __getitem__(self, i):
        t0, t1 = self.pairs[i]
        H, W = self.ds.latitude.size, self.ds.longitude.size

        surf = {v: _to_torch(self.norm(self.ds[v].sel(time=t0).values, v)).unsqueeze(0).unsqueeze(0)
                for v in self.VS}  # [B=1,T=1,H,W]
        atmos= {v: _to_torch(self.norm(self.ds[v].sel(time=t0).values, v)).unsqueeze(0).unsqueeze(0)
                for v in self.VA}  # [1,1,L,H,W]
        static= {v: _to_torch(self.ds[v].values) for v in self.VC}

        meta  = Metadata(lat=self.lat, lon=self.lon, time=(str(t0),), atmos_levels=self.levels, rollout_step=0)
        batch = Batch(surf_vars=surf, atmos_vars=atmos, static_vars=static, metadata=meta)

        tgt_surf = {v: _to_torch(self.ds[v].sel(time=t1).values) for v in self.TS}   # [H,W]
        tgt_atm  = {v: _to_torch(self.ds[v].sel(time=t1).values) for v in self.TA}   # [L,H,W]

        target = {"surf": tgt_surf, "atmos": tgt_atm, "coords": {"time1": str(t1)}}
        return batch, target
