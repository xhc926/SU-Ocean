"""
Upsample gridded CSV/PKL time series onto a regular lat/lon grid (default 0.25 deg),
rebuild ocean/land mask, and save PKL under upsampled/1-4.

Input columns may be "(lat, lon)" or "lat_lon" (e.g. 26.50_114.00).
Output PKL columns use the same lat_lon string format as the original raw tables.
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

LAND_FILL = 0.0
RES_DEG = 0.25

DEFAULT_INPUT_DIR = "/root/autodl-tmp/data/raw"
DEFAULT_OUTPUT_DIR = "/root/autodl-tmp/data/upsampled/1-4"


def parse_colname(col):
    col = str(col).strip()
    if col.lower() == "date":
        return None, None
    # "(lat, lon)" style
    parts = col.replace("(", "").replace(")", "").split(",")
    if len(parts) == 2:
        try:
            return float(parts[0].strip()), float(parts[1].strip())
        except ValueError:
            pass
    # "lat_lon" style (e.g. 26.50_114.00)
    if "_" in col:
        segs = col.split("_")
        if len(segs) == 2:
            try:
                return float(segs[0].strip()), float(segs[1].strip())
            except ValueError:
                pass
    return None, None


def format_colname_lat_lon(lat, lon):
    """Column header style matching raw data: lat_lon e.g. 26.50_114.00"""
    return f"{float(lat):.2f}_{float(lon):.2f}"


def column_names_from_grids(lat_grid, lon_grid):
    la, lo = lat_grid.ravel(), lon_grid.ravel()
    return [format_colname_lat_lon(la[i], lo[i]) for i in range(la.size)]


def columns_to_points(df):
    data_cols = [c for c in df.columns if c != "date"]
    src_lats, src_lons, valid_cols = [], [], []
    for c in data_cols:
        lat, lon = parse_colname(c)
        if lat is not None:
            src_lats.append(lat)
            src_lons.append(lon)
            valid_cols.append(c)
    return np.asarray(src_lats, dtype=np.float64), np.asarray(src_lons, dtype=np.float64), valid_cols


def build_target_grid(lat_min, lat_max, lon_min, lon_max, res=RES_DEG):
    lat0 = np.floor(lat_min / res) * res
    lat1 = np.ceil(lat_max / res) * res
    lon0 = np.floor(lon_min / res) * res
    lon1 = np.ceil(lon_max / res) * res
    n_lat = int(round((lat1 - lat0) / res)) + 1
    n_lon = int(round((lon1 - lon0) / res)) + 1
    target_lat = np.linspace(lat0, lat1, n_lat)
    target_lon = np.linspace(lon0, lon1, n_lon)
    lon_grid, lat_grid = np.meshgrid(target_lon, target_lat)
    target_points = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
    return target_lat, target_lon, target_points, lat_grid, lon_grid


def collect_bounds_from_dir(input_dir):
    lat_min, lat_max = np.inf, -np.inf
    lon_min, lon_max = np.inf, -np.inf
    for fn in sorted(os.listdir(input_dir)):
        if not (fn.endswith(".csv") or fn.endswith(".pkl")):
            continue
        if fn.startswith("land_mask"):
            continue
        path = os.path.join(input_dir, fn)
        if fn.endswith(".pkl"):
            df = pd.read_pickle(path)
        else:
            df = pd.read_csv(path, nrows=0)
        lats, lons, _ = columns_to_points(df)
        if len(lats) == 0:
            continue
        lat_min = min(lat_min, float(lats.min()))
        lat_max = max(lat_max, float(lats.max()))
        lon_min = min(lon_min, float(lons.min()))
        lon_max = max(lon_max, float(lons.max()))
    if not np.isfinite(lat_min):
        raise ValueError(f"No valid (lat,lon) columns found under {input_dir}")
    return lat_min, lat_max, lon_min, lon_max


def upsample_file(input_path, output_dir, target_points, lat_grid, lon_grid, mask_accumulator, save_csv=False):
    print(f"Processing: {input_path}", flush=True)
    if input_path.endswith(".pkl"):
        df = pd.read_pickle(input_path)
    else:
        df = pd.read_csv(input_path)

    base = os.path.basename(input_path)
    var_name = os.path.splitext(base)[0].replace("_cleaned", "")

    has_date = "date" in df.columns
    if has_date:
        dates = df["date"].values
        data_cols = [c for c in df.columns if c != "date"]
    else:
        dates = None
        data_cols = list(df.columns)

    src_lats, src_lons, valid_cols = [], [], []
    for c in data_cols:
        lat, lon = parse_colname(c)
        if lat is not None:
            src_lats.append(lat)
            src_lons.append(lon)
            valid_cols.append(c)

    if not valid_cols:
        print(f"  WARNING: no valid (lat,lon) columns found, skipping {input_path}", flush=True)
        return

    src_points = np.column_stack([src_lats, src_lons])
    data_matrix = df[valid_cols].values.astype(np.float64)
    n_times = data_matrix.shape[0]
    n_target = target_points.shape[0]

    result = np.full((n_times, n_target), np.nan, dtype=np.float32)
    ocean_mask = np.zeros(n_target, dtype=np.float32)

    for t in range(n_times):
        row = data_matrix[t]
        valid = ~np.isnan(row)
        if valid.sum() < 3:
            continue
        interp = griddata(src_points[valid], row[valid], target_points, method="linear")
        result[t] = interp.astype(np.float32)
        ocean_mask[np.isfinite(interp)] = 1.0

        if (t + 1) % 500 == 0 or t == n_times - 1:
            print(f"  interpolated {t+1}/{n_times} time steps", flush=True)

    result = np.where(np.isnan(result) | np.isinf(result), LAND_FILL, result)

    target_colnames = column_names_from_grids(lat_grid, lon_grid)
    out_df = pd.DataFrame(result, columns=target_colnames)
    if has_date:
        out_df.insert(0, "date", dates)

    os.makedirs(output_dir, exist_ok=True)
    out_pkl = os.path.join(output_dir, f"{var_name}.pkl")
    out_df.to_pickle(out_pkl)
    print(f"  Saved: {out_pkl}  shape={out_df.shape}", flush=True)
    if save_csv:
        out_df.to_csv(os.path.join(output_dir, f"{var_name}.csv"), index=False)

    mask_accumulator[:] = np.maximum(mask_accumulator, ocean_mask)


def write_land_mask(output_dir, mask_flat, lat_grid, lon_grid):
    n_target = mask_flat.size
    target_colnames = column_names_from_grids(lat_grid, lon_grid)
    mask_df = pd.DataFrame(mask_flat.reshape(1, -1), columns=target_colnames)
    mask_pkl = os.path.join(output_dir, "land_mask.pkl")
    mask_df.to_pickle(mask_pkl)
    print(f"Saved land_mask: {mask_pkl}", flush=True)


def main():
    input_dir = DEFAULT_INPUT_DIR
    output_dir = DEFAULT_OUTPUT_DIR
    save_csv = False

    args = sys.argv[1:]
    if "-h" in args or "--help" in args:
        print(
            "Usage: python upsample_to_1_12deg.py [input_dir] [output_dir] [--csv]\n"
            f"  Default input:  {DEFAULT_INPUT_DIR}\n"
            f"  Default output: {DEFAULT_OUTPUT_DIR}\n"
            f"  Target resolution: {RES_DEG} deg (mask regenerated from all variables)."
        )
        sys.exit(0)
    if len(args) >= 1:
        input_dir = args[0]
    if len(args) >= 2:
        output_dir = args[1]
    if "--csv" in args:
        save_csv = True

    if not os.path.isdir(input_dir):
        print(f"ERROR: input dir not found: {input_dir}", flush=True)
        sys.exit(1)

    lat_min, lat_max, lon_min, lon_max = collect_bounds_from_dir(input_dir)
    print(
        f"Domain bounds from raw columns: lat [{lat_min:.4f}, {lat_max:.4f}], "
        f"lon [{lon_min:.4f}, {lon_max:.4f}], res={RES_DEG} deg",
        flush=True,
    )
    target_lat, target_lon, target_points, lat_grid, lon_grid = build_target_grid(
        lat_min, lat_max, lon_min, lon_max, res=RES_DEG
    )
    print(
        f"Target grid: {len(target_lat)} x {len(target_lon)} = {target_points.shape[0]} points",
        flush=True,
    )

    mask_accumulator = np.zeros(target_points.shape[0], dtype=np.float32)

    for fn in sorted(os.listdir(input_dir)):
        if not (fn.endswith(".csv") or fn.endswith(".pkl")):
            continue
        if fn.startswith("land_mask"):
            continue
        upsample_file(
            os.path.join(input_dir, fn),
            output_dir,
            target_points,
            lat_grid,
            lon_grid,
            mask_accumulator,
            save_csv=save_csv,
        )

    write_land_mask(output_dir, mask_accumulator, lat_grid, lon_grid)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
