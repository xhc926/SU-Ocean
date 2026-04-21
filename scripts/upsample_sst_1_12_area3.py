"""
Upsample SST from 0.25 deg to 1/12 deg for area3.

Target domain:
- latitude: 29.0 to 33.0 (inclusive)
- longitude: 122.0 to 123.0 (inclusive)
- resolution: 1/12 deg

Input:
- /root/autodl-tmp/data/raw/sst.csv

Output:
- /root/autodl-tmp/data/upsampled/1-12/area3/sst.pkl

Notes:
- Interpolate with linear method using valid source points at each timestamp.
- Fill remaining NaN with nearest interpolation fallback.
- No land mask is generated.
"""

import os
import numpy as np
import pandas as pd
from scipy.interpolate import griddata


INPUT_PATH = "/root/autodl-tmp/data/raw/sst.csv"
OUTPUT_DIR = "/root/autodl-tmp/data/upsampled/1-12/area3"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "sst.pkl")

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


def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Input not found: {INPUT_PATH}")

    print(f"Reading input: {INPUT_PATH}", flush=True)
    df = pd.read_csv(INPUT_PATH)
    if "date" not in df.columns:
        raise ValueError("Input must contain 'date' column.")

    data_cols = [c for c in df.columns if c != "date"]
    src_lats, src_lons, valid_cols = [], [], []
    for c in data_cols:
        lat, lon = parse_colname(c)
        if lat is None:
            continue
        # Keep only source points inside requested domain.
        if LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX:
            src_lats.append(lat)
            src_lons.append(lon)
            valid_cols.append(c)

    if len(valid_cols) < 3:
        raise ValueError(f"Not enough valid source points in domain, got {len(valid_cols)}.")

    src_points = np.column_stack([np.asarray(src_lats), np.asarray(src_lons)])
    values = df[valid_cols].to_numpy(dtype=np.float64)
    dates = df["date"].values

    target_lat, target_lon, lat_grid, lon_grid, target_points = build_target_grid()
    print(
        f"Target grid: {len(target_lat)} x {len(target_lon)} = {target_points.shape[0]} points",
        flush=True,
    )

    n_times = values.shape[0]
    n_target = target_points.shape[0]
    out = np.full((n_times, n_target), np.nan, dtype=np.float32)

    for t in range(n_times):
        row = values[t]
        valid = np.isfinite(row)
        if valid.sum() < 3:
            continue

        linear_interp = griddata(src_points[valid], row[valid], target_points, method="linear")
        if np.any(~np.isfinite(linear_interp)):
            nearest_interp = griddata(src_points[valid], row[valid], target_points, method="nearest")
            linear_interp[~np.isfinite(linear_interp)] = nearest_interp[~np.isfinite(linear_interp)]

        out[t] = linear_interp.astype(np.float32)

        if (t + 1) % 500 == 0 or t == n_times - 1:
            print(f"Interpolated {t + 1}/{n_times}", flush=True)

    # If any timestamp has too many NaN due to sparse data, use column-wise temporal median fallback.
    if np.any(~np.isfinite(out)):
        col_med = np.nanmedian(out, axis=0)
        col_med = np.where(np.isfinite(col_med), col_med, 0.0)
        bad = ~np.isfinite(out)
        out[bad] = np.take(col_med, np.where(bad)[1]).astype(np.float32)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_df = pd.DataFrame(out, columns=target_column_names(lat_grid, lon_grid))
    out_df.insert(0, "date", dates)
    out_df.to_pickle(OUTPUT_PATH)
    print(f"Saved: {OUTPUT_PATH} shape={out_df.shape}", flush=True)


if __name__ == "__main__":
    main()
