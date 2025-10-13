# aurora-europe/projects/cerra_hr/dataio/loader_pathb.py
"""
CERRA Path-B dataset → Aurora Batch + targets.

Key features:
- Works with atmos vars including 'rh' (relative humidity) instead of 'q'.
- Auto-detects common coordinate names: latitude/lat, longitude/lon, level/isobaricInhPa.
- Crops to bbox and pairs (t, t+Δ) where Δ = lead_hours from region config.
- Normalises inputs using stats JSON; falls back to per-file mean/std if missing.
- Returns (batch, target) ready for training:
    batch  -> Batch(surf_vars=[B, T=1, H, W], atmos_vars=[B, T=1, L, H, W], static=[H, W])
    target -> {"surf": {var:[H,W]}, "atmos": {var:[L,H,W]}, "coords":{"time1": str(t1)}}
"""

from __future__ import annotations
import json
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset
from aurora.batch import Batch, Metadata


def _to_torch_f32(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(x, dtype=np.float32))


def _detect_coords(ds: xr.Dataset) -> Tuple[str, str, str]:
    """Return names for (lat, lon, level) coords if present; level may be '' if absent."""
    lat_name = "latitude" if "latitude" in ds.coords else ("lat" if "lat" in ds.coords else None)
    lon_name = "longitude" if "longitude" in ds.coords else ("lon" if "lon" in ds.coords else None)
    # pressure levels in hPa often called 'level' or 'isobaricInhPa'
    lev_name = "level" if "level" in ds.coords else ("isobaricInhPa" if "isobaricInhPa" in ds.coords else None)

    if lat_name is None or lon_name is None:
        raise ValueError(f"Could not find latitude/longitude coords in dataset. Found coords: {list(ds.coords)}")

    return lat_name, lon_name, lev_name or ""


class CerraHRDataset(Dataset):
    def __init__(
        self,
        paths: Sequence[str],
        stats_json: str,
        lead_hours: int,
        vars_surf: Sequence[str],
        vars_atmos: Sequence[str],
        vars_static: Sequence[str],
        bbox: Mapping[str, float],
        target_surf: Sequence[str] = ("2t",),
        target_atmos: Sequence[str] = ("t", "u", "v", "rh"),
        ensure_no_nans: bool = True,
    ):
        """
        Args:
            paths: list/glob of NetCDF files for a continuous period.
            stats_json: path to normalisation stats JSON (means/stds per var).
            lead_hours: Δ (hours) between input and target times (e.g., 6).
            vars_*: variable names present in NetCDF (must match dataset).
            bbox: dict with lat_min, lat_max, lon_min, lon_max for cropping.
            target_*: which vars to supervise in the loss.
            ensure_no_nans: if True, raise if NaNs are found in inputs/targets.
        """
        # Open and unify
        self.ds = xr.open_mfdataset(paths, combine="by_coords")
        self.stats = json.load(open(stats_json))
        self.lead = np.timedelta64(lead_hours, "h")

        # Detect coordinate names
        self.lat_name, self.lon_name, self.lev_name = _detect_coords(self.ds)

        # Crop to bbox
        lat_min, lat_max = bbox["lat_min"], bbox["lat_max"]
        lon_min, lon_max = bbox["lon_min"], bbox["lon_max"]
        if lon_min >= lon_max:
            raise ValueError("Expected lon_min < lon_max (no wrap-around). Recenter your data if needed.")
        self.ds = self.ds.sel(
            **{
                self.lat_name: slice(lat_max, lat_min),  # north -> south
                self.lon_name: slice(lon_min, lon_max),  # west -> east
            }
        )

        # Build time pairs (t, t+Δ)
        times = self.ds.time.values
        ts = set(times.tolist())  # ensure Python hashables
        self.pairs: List[Tuple[np.datetime64, np.datetime64]] = [
            (t0, t0 + self.lead) for t0 in times if (t0 + self.lead) in ts
        ]
        if not self.pairs:
            raise ValueError("No (t, t+Δ) time pairs found. Check dt_hours/lead_hours and data cadence.")

        # Save var lists
        self.VS, self.VA, self.VC = tuple(vars_surf), tuple(vars_atmos), tuple(vars_static)
        self.TS, self.TA = tuple(target_surf), tuple(target_atmos)

        # Cache 2D lat/lon meshes from 1D coords (exact alignment with the file)
        H = self.ds[self.lat_name].size
        W = self.ds[self.lon_name].size
        lat_1d = self.ds[self.lat_name].values
        lon_1d = self.ds[self.lon_name].values
        self.lat2d = _to_torch_f32(np.repeat(lat_1d.reshape(H, 1), W, axis=1))
        self.lon2d = _to_torch_f32(np.repeat(lon_1d.reshape(1, W), H, axis=0))

        # Levels tuple (may be empty if no level coord on atmos vars)
        if self.VA:
            first_var = self.VA[0]
            has_level = self.lev_name and (self.lev_name in self.ds[first_var].dims)
            self.levels = tuple(self.ds[self.lev_name].values.tolist()) if has_level else ()
        else:
            self.levels = ()

        self.ensure_no_nans = ensure_no_nans

    # ---------- helpers ----------
    def _norm(self, x: np.ndarray, name: str) -> np.ndarray:
        st = self.stats.get(name)
        if st is None:
            # Fallback: per-file mean/std (warn-ish behavior)
            mu = float(np.nanmean(x))
            sd = float(np.nanstd(x)) or 1.0
            return (x - mu) / sd
        mu, sd = st["mean"], st["std"] or 1.0
        return (x - mu) / sd

    def _assert_no_nans(self, arr: np.ndarray | torch.Tensor, label: str):
        if not self.ensure_no_nans:
            return
        if isinstance(arr, torch.Tensor):
            has_nan = torch.isnan(arr).any().item()
        else:
            has_nan = np.isnan(arr).any()
        if has_nan:
            raise ValueError(f"Found NaNs in {label}. Check preprocessing/stats/data integrity.")

    # ---------- dataset API ----------
    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int):
        t0, t1 = self.pairs[i]

        # ----- Inputs @ t0 -----
        # surface: [H, W] -> [B=1, T=1, H, W]
        surf: Dict[str, torch.Tensor] = {}
        for v in self.VS:
            a = self.ds[v].sel(time=t0).values
            a_n = self._norm(a, v)
            tensor = _to_torch_f32(a_n).unsqueeze(0).unsqueeze(0)
            self._assert_no_nans(tensor, f"input surf {v} at {t0}")
            surf[v] = tensor

        # atmos: [L, H, W] -> [B=1, T=1, L, H, W]
        atmos: Dict[str, torch.Tensor] = {}
        for v in self.VA:
            a = self.ds[v].sel(time=t0).values
            a_n = self._norm(a, v)
            tensor = _to_torch_f32(a_n).unsqueeze(0).unsqueeze(0)
            self._assert_no_nans(tensor, f"input atmos {v} at {t0}")
            atmos[v] = tensor

        # static: [H, W]
        static: Dict[str, torch.Tensor] = {}
        for v in self.VC:
            a = self.ds[v].values
            tensor = _to_torch_f32(a)
            self._assert_no_nans(tensor, f"static {v}")
            static[v] = tensor

        # Metadata
        levels = tuple(self.levels)
        meta = Metadata(
            lat=self.lat2d,
            lon=self.lon2d,
            time=(str(t0),),
            atmos_levels=levels,
            rollout_step=0,
        )
        batch = Batch(surf_vars=surf, atmos_vars=atmos, static_vars=static, metadata=meta)

        # ----- Targets @ t1 (denormalized truth) -----
        tgt_surf: Dict[str, torch.Tensor] = {}
        for v in self.TS:
            a = self.ds[v].sel(time=t1).values  # [H, W]
            tensor = _to_torch_f32(a)
            self._assert_no_nans(tensor, f"target surf {v} at {t1}")
            tgt_surf[v] = tensor

        tgt_atm: Dict[str, torch.Tensor] = {}
        for v in self.TA:
            a = self.ds[v].sel(time=t1).values  # [L, H, W]
            tensor = _to_torch_f32(a)
            self._assert_no_nans(tensor, f"target atmos {v} at {t1}")
            tgt_atm[v] = tensor

        target = {"surf": tgt_surf, "atmos": tgt_atm, "coords": {"time1": str(t1)}}
        return batch, target
