"""
Interpolate and upsample area3 variables to 1/12 deg grid (49 x 13 = 637).

Target domain:
- latitude: 29.0 to 33.0
- longitude: 122.0 to 123.0
- resolution: 1/12 deg

Default variables:
- sal, uo, vo, ssh

Input directory:
- /root/autodl-tmp/data/raw

Output directory:
- /root/autodl-tmp/data/upsampled/1-12/area3

No land_mask is used or generated.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.interpolate import griddata


INPUT_DIR = "/root/autodl-tmp/data/raw"
OUTPUT_DIR = "/root/autodl-tmp/data/upsampled/1-12/area3"
DEFAULT_VARS = ["sal", "uo", "vo", "ssh"]

LAT_MIN, LAT_MAX = 29.0, 33.0
LON_MIN, LON_MAX = 122.0, 123.0
RES = 1.0 / 12.0


def parse_colname(col):
    col = str(col).strip()
    if col.lower() == "date":
        return None, None
    if "_" in col:
        parts = col.split("_")
        if len(parts) == 2:
            try:
                return float(parts[0]), float(parts[1])
            except ValueError:
                return None, None
    return None, None


def build_target_grid():
    n_lat = int(round((LAT_MAX - LAT_MIN) / RES)) + 1
    n_lon = int(round((LON_MAX - LON_MIN) / RES)) + 1
    target_lat = np.linspace(LAT_MIN, LAT_MAX, n_lat, dtype=np.float64)
    target_lon = np.linspace(LON_MIN, LON_MAX, n_lon, dtype=np.float64)
    lon_grid, lat_grid = np.meshgrid(target_lon, target_lat)
    target_points = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
    return target_lat, target_lon, lat_grid, lon_grid, target_points


def target_column_names(lat_grid, lon_grid):
    la = lat_grid.ravel()
    lo = lon_grid.ravel()
    return [f"{float(la[i]):.2f}_{float(lo[i]):.2f}" for i in range(la.size)]


def locate_input_path(var_name):
    csv_path = os.path.join(INPUT_DIR, f"{var_name}.csv")
    pkl_path = os.path.join(INPUT_DIR, f"{var_name}.pkl")
    if os.path.exists(csv_path):
        return csv_path
    if os.path.exists(pkl_path):
        return pkl_path
    raise FileNotFoundError(f"Missing source file for {var_name}: {csv_path} or {pkl_path}")


def read_table(path):
    if path.endswith(".pkl"):
        return pd.read_pickle(path)
    return pd.read_csv(path)


def interpolate_one(var_name, lat_grid, lon_grid, target_points):
    input_path = locate_input_path(var_name)
    print(f"\n[{var_name}] reading: {input_path}", flush=True)
    df = read_table(input_path)

    if "date" not in df.columns:
        raise ValueError(f"{var_name}: expected a 'date' column.")

    cols = [c for c in df.columns if c != "date"]
    src_lats, src_lons, src_cols = [], [], []
    for c in cols:
        lat, lon = parse_colname(c)
        if lat is None:
            continue
        if LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX:
            src_lats.append(lat)
            src_lons.append(lon)
            src_cols.append(c)

    if len(src_cols) < 3:
        raise ValueError(f"{var_name}: not enough domain columns ({len(src_cols)}).")

    values = df[src_cols].to_numpy(dtype=np.float64)
    dates = df["date"].values
    src_points = np.column_stack([np.asarray(src_lats), np.asarray(src_lons)])

    print(
        f"[{var_name}] source shape={values.shape}, source domain points={len(src_cols)}, target points={target_points.shape[0]}",
        flush=True,
    )
    print(f"[{var_name}] raw NaN count={int(np.isnan(values).sum())}", flush=True)

    n_times = values.shape[0]
    n_target = target_points.shape[0]
    out = np.full((n_times, n_target), np.nan, dtype=np.float32)

    for t in range(n_times):
        row = values[t]
        valid = np.isfinite(row)
        if valid.sum() < 3:
            continue

        # Linear interpolation in space, nearest as fallback for uncovered points.
        linear_interp = griddata(src_points[valid], row[valid], target_points, method="linear")
        if np.any(~np.isfinite(linear_interp)):
            nearest_interp = griddata(src_points[valid], row[valid], target_points, method="nearest")
            linear_interp[~np.isfinite(linear_interp)] = nearest_interp[~np.isfinite(linear_interp)]
        out[t] = linear_interp.astype(np.float32)

        if (t + 1) % 500 == 0 or t == n_times - 1:
            print(f"[{var_name}] interpolated {t + 1}/{n_times}", flush=True)

    if np.any(~np.isfinite(out)):
        # Temporal median fallback per grid if an entire timestamp is too sparse.
        col_med = np.nanmedian(out, axis=0)
        col_med = np.where(np.isfinite(col_med), col_med, 0.0)
        bad = ~np.isfinite(out)
        out[bad] = np.take(col_med, np.where(bad)[1]).astype(np.float32)

    out_df = pd.DataFrame(out, columns=target_column_names(lat_grid, lon_grid))
    out_df.insert(0, "date", dates)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{var_name}.pkl")
    out_df.to_pickle(out_path)
    print(f"[{var_name}] saved: {out_path} shape={out_df.shape}", flush=True)
    print(f"[{var_name}] output NaN count={int(np.isnan(out).sum())}", flush=True)


def main():
    vars_to_process = DEFAULT_VARS
    if len(sys.argv) > 1:
        vars_to_process = [v.strip() for v in sys.argv[1].split(",") if v.strip()]
        if not vars_to_process:
            raise ValueError("No variable names provided.")

    _, _, lat_grid, lon_grid, target_points = build_target_grid()
    print(
        f"Target grid fixed: lat[{LAT_MIN},{LAT_MAX}], lon[{LON_MIN},{LON_MAX}], "
        f"res={RES:.8f}, points={target_points.shape[0]}",
        flush=True,
    )

    for var_name in vars_to_process:
        interpolate_one(var_name, lat_grid, lon_grid, target_points)

    print("\nAll done.", flush=True)


if __name__ == "__main__":
    main()
