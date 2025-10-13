import xarray as xr, os, argparse

def convert(grib_path, nc_path):
    ds = xr.open_dataset(grib_path, engine="cfgrib")
    os.makedirs(os.path.dirname(nc_path), exist_ok=True)
    ds.to_netcdf(nc_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grib"); ap.add_argument("--out")
    a = ap.parse_args()
    convert(a.grib, a.out)
