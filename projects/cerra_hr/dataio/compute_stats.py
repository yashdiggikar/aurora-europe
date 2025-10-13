# aurora-europe/projects/cerra_hr/dataio/compute_stats.py
"""
Compute normalisation statistics (mean/std) for CERRA variables.

- Supports surf/static vars (2D) and atmos vars (3D with 'level' dim).
- Default: global mean/std per variable across all times/levels/space.
- Optional: per-level stats for atmos vars via --per_level (writes dicts keyed by level).
- NaN-safe (skipna) and works with multi-file datasets.

Output: JSON mapping
{
  "2t": {"mean": ..., "std": ...},
  "t":  {"mean": ..., "std": ...},             # global
  # or when --per_level:
  "t":  {"mean": { "50": ..., "100": ... }, "std": { "50": ..., "100": ... } }
}
"""

import argparse
import json
import xarray as xr

# EDIT THESE ONLY IF YOUR VAR NAMES DIFFER FROM vars.yaml
V_SURF   = ["2t", "10u", "10v", "msl", "sp"]
V_ATMOS  = ["t", "u", "v", "rh", "z"]   # we use rh, not q
V_STATIC = ["z", "lsm"]                 # static maps (orography, land-sea mask)


def global_stats(da: xr.DataArray) -> dict:
    """Compute global mean/std across all dims (time/level/lat/lon)."""
    # NaN-safe; use float32 to avoid dtype issues
    mu  = float(da.astype("float32").mean(skipna=True).values)
    std = float(da.astype("float32").std(skipna=True).values) or 1.0
    return {"mean": mu, "std": std}


def per_level_stats(da: xr.DataArray, level_name: str) -> dict:
    """
    Compute mean/std per level; returns dicts keyed by level value (as string).
    Expects `level_name` to be present in da.dims.
    """
    lvl_vals = da[level_name].values
    means, stds = {}, {}
    # Iterate over levels to keep memory moderate if chunks are large
    for lv in lvl_vals:
        sub = da.sel({level_name: lv}).astype("float32")
        m = float(sub.mean(skipna=True).values)
        s = float(sub.std(skipna=True).values) or 1.0
        means[str(float(lv)) if hasattr(lv, "item") else str(lv)] = m
        stds[str(float(lv)) if hasattr(lv, "item") else str(lv)]  = s
    return {"mean": means, "std": stds}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", nargs="+", required=True,
                    help="One or more NetCDF files (glob expanded by shell or pass multiple).")
    ap.add_argument("--out", required=True, help="Output JSON path for stats.")
    ap.add_argument("--per_level", action="store_true",
                    help="Compute per-level stats for atmos vars (keys per pressure).")
    ap.add_argument("--level_name", default="", help="Override level coord name if not 'level'/'isobaricInhPa'.")
    args = ap.parse_args()

    # Open multi-file dataset (lazy if dask is available)
    ds = xr.open_mfdataset(args.paths, combine="by_coords")

    # Detect coordinate names
    lat_name  = "latitude" if "latitude" in ds.coords else ("lat" if "lat" in ds.coords else None)
    lon_name  = "longitude" if "longitude" in ds.coords else ("lon" if "lon" in ds.coords else None)
    # Level coordinate can vary; allow override
    if args.level_name:
        level_name = args.level_name
    else:
        level_name = "level" if "level" in ds.coords else ("isobaricInhPa" if "isobaricInhPa" in ds.coords else None)

    out = {}
    missing = []

    # Surf vars
    for v in V_SURF:
        if v in ds:
            out[v] = global_stats(ds[v])
        else:
            missing.append(v)

    # Atmos vars
    for v in V_ATMOS:
        if v in ds:
            if args.per_level:
                if level_name and (level_name in ds[v].dims):
                    out[v] = per_level_stats(ds[v], level_name)
                else:
                    # Fallback to global if no level dim present
                    out[v] = global_stats(ds[v])
            else:
                out[v] = global_stats(ds[v])
        else:
            missing.append(v)

    # Static vars
    for v in V_STATIC:
        if v in ds:
            out[v] = global_stats(ds[v])
        else:
            missing.append(v)

    # Write JSON
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    # Friendly report on missing vars (not fatal)
    if missing:
        print("Warning: variables not found in dataset and skipped:", ", ".join(missing))


if __name__ == "__main__":
    main()
