# aurora-europe/projects/cerra_hr/dataio/convert_grib_to_nc.py
"""
Optional helper to convert GRIB -> NetCDF with light preprocessing.

Use when CDS only gives GRIB, or you need to:
- crop to a bbox,
- rename variables/coords to match your vars.yaml,
- set chunking/compression for faster IO,
- wrap longitudes from 0..360 to -180..180.

If you already have NetCDF from CDS that matches your naming/layout, you can skip this.
"""

import os
import argparse
import json
import xarray as xr
import numpy as np

def wrap_longitudes(ds, lon_name="longitude"):
    """Convert 0..360 longitudes to -180..180 (no-op if already in desired range)."""
    if lon_name not in ds.coords:
        return ds
    lon = ds[lon_name]
    if np.nanmin(lon.values) >= 0 and np.nanmax(lon.values) > 180:
        lon_wrapped = ((lon + 180) % 360) - 180
        ds = ds.assign_coords({lon_name: lon_wrapped}).sortby(lon_name)
    return ds

def maybe_rename(ds, rename_json):
    """Rename variables/coords using a JSON mapping file, e.g. {"t2m":"2t","u10":"10u"}."""
    if not rename_json:
        return ds
    with open(rename_json) as f:
        mapping = json.load(f)
    # allow both var and coord renames
    to_rename = {k:v for k,v in mapping.items() if (k in ds.data_vars or k in ds.coords)}
    return ds.rename(to_rename)

def convert(
    grib_path,
    nc_path,
    bbox=None,                 # dict: {lat_min, lat_max, lon_min, lon_max}
    rename_json=None,          # path to JSON mapping for var/coord rename
    chunk="auto",              # "auto" or dict like {"time":1,"level":13,"latitude":720,"longitude":1200}
    compress_level=4,          # NetCDF zlib compression level
):
    ds = xr.open_dataset(grib_path, engine="cfgrib")

    # Optional: wrap longitudes to -180..180 for consistency with bbox & Aurora configs
    ds = wrap_longitudes(ds, lon_name="longitude" if "longitude" in ds.coords else "lon")

    # Optional: rename variables/coords to match your vars.yaml (e.g., t2m->2t, u10->10u, v10->10v)
    ds = maybe_rename(ds, rename_json)

    # Optional: crop to bbox
    if bbox is not None:
        lat_name = "latitude" if "latitude" in ds.coords else "lat"
        lon_name = "longitude" if "longitude" in ds.coords else "lon"
        ds = ds.sel(
            **{
                lat_name: slice(bbox["lat_max"], bbox["lat_min"]),  # north -> south
                lon_name: slice(bbox["lon_min"], bbox["lon_max"]),  # west -> east
            }
        )

    # Optional: set chunking (improves IO when training)
    if chunk == "auto":
        ds = ds.chunk("auto")
    elif isinstance(chunk, dict):
        ds = ds.chunk(chunk)

    # Compression settings
    encoding = {var: {"zlib": True, "complevel": compress_level} for var in ds.data_vars}

    os.makedirs(os.path.dirname(nc_path) or ".", exist_ok=True)
    ds.to_netcdf(nc_path, encoding=encoding)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grib", required=True, help="Input GRIB file")
    ap.add_argument("--out",  required=True, help="Output NetCDF file")
    ap.add_argument("--bbox", type=str, default="", help='JSON string: {"lat_min":35,"lat_max":72,"lon_min":-25,"lon_max":45}')
    ap.add_argument("--rename_json", type=str, default="", help="Path to JSON mapping for renames (vars/coords)")
    ap.add_argument("--chunk", type=str, default="auto", help='Either "auto" or JSON string of dict, e.g. {"time":1}')
    ap.add_argument("--compress_level", type=int, default=4)
    args = ap.parse_args()

    bbox = json.loads(args.bbox) if args.bbox else None
    chunk = json.loads(args.chunk) if (args.chunk and args.chunk != "auto") else "auto"

    convert(args.grib, args.out, bbox=bbox, rename_json=args.rename_json or None,
            chunk=chunk, compress_level=args.compress_level)
