import xarray as xr, json, argparse

V_SURF = ["2t","10u","10v","msl","sp"]
V_ATM  = ["t","u","v","q","z"]
V_STATIC=["z","lsm"]

def get_stats(ds, var):
    x = ds[var].astype("float32")
    mu = float(x.mean().values); sd = float(x.std().values) or 1.0
    return {"mean": mu, "std": sd}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", nargs="+")
    ap.add_argument("--out")
    a = ap.parse_args()
    ds = xr.open_mfdataset(a.paths, combine="by_coords")
    out = {}
    for v in V_SURF + V_ATM + V_STATIC:
        if v in ds:
            out[v] = get_stats(ds, v)
    with open(a.out, "w") as f: json.dump(out, f, indent=2)
