"""
Upsample gridded CSV/PKL time series onto a regular lat/lon grid (default 0.25 deg),
and save PKL under the chosen output directory.

Input columns may be "(lat, lon)" or "lat_lon" (e.g. 26.50_114.00).
Output PKL columns use the same lat_lon string format as the original raw tables.

Land mask (independent of interpolation): uses raw SWH on a 0.5° coarse grid.
Fine grid is RES_DEG (default 0.25°). For each fine point:
  • On a coarse vertex (lat and lon are multiples of 0.5°): ocean iff that coarse
    SWH column has data (any finite timestep).
  • Exactly one of lat/lon on a coarse line: ocean iff the two adjacent coarse
    SWH vertices along the non-aligned axis both have data.
  • Neither on a coarse line: ocean iff at least three of the four surrounding
    coarse vertices have SWH data.
Missing coarse columns count as no data (land).
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

LAND_FILL = 0.0
RES_DEG = 0.25
# SWH coarse grid step (degrees); fine mask rules are defined relative to this.
SWH_COARSE_STEP = 0.5

DEFAULT_INPUT_DIR = "/root/autodl-tmp/data/raw/area1"
DEFAULT_OUTPUT_DIR = "/root/autodl-tmp/data/upsampled/1-4/area1"


def parse_colname(col):
    col = str(col).strip()
    if col.lower() == "date":
        return None, None
    parts = col.replace("(", "").replace(")", "").split(",")
    if len(parts) == 2:
        try:
            return float(parts[0].strip()), float(parts[1].strip())
        except ValueError:
            pass
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


def on_coarse_multiple(val, step=SWH_COARSE_STEP, eps=1e-4):
    """True if val lies on coarse grid (multiple of step)."""
    k = val / step
    return abs(k - np.round(k)) * step < eps


def find_swh_input_path(input_dir):
    """Pick one SWH table (stem swh after stripping _cleaned)."""
    candidates = []
    for fn in sorted(os.listdir(input_dir)):
        if not (fn.endswith(".csv") or fn.endswith(".pkl")):
            continue
        if fn.startswith("land_mask"):
            continue
        stem = os.path.splitext(fn)[0].replace("_cleaned", "")
        if stem == "swh":
            candidates.append(os.path.join(input_dir, fn))
    if not candidates:
        raise FileNotFoundError(
            f"No swh.csv / swh.pkl (or swh_cleaned.pkl) under {input_dir} for coarse mask."
        )
    return candidates[-1]


def build_coarse_swh_has_data(swh_df):
    """
    Map column key "lat_lon" -> True if that coarse SWH column has any finite value
    in any timestep (has data = ocean candidate).
    """
    has_data = {}
    data_cols = [c for c in swh_df.columns if c != "date"]
    for c in data_cols:
        lat, lon = parse_colname(c)
        if lat is None:
            continue
        key = format_colname_lat_lon(lat, lon)
        col = swh_df[c].values.astype(np.float64)
        has_data[key] = bool(np.any(np.isfinite(col)))
    return has_data


def coarse_has_data(coarse_has_data_map, lat, lon):
    return coarse_has_data_map.get(format_colname_lat_lon(lat, lon), False)


def fine_mask_ocean_at(lat, lon, coarse_map, step=SWH_COARSE_STEP):
    """
    Ocean (True) / land (False) for fine grid point (lat, lon) from coarse SWH presence.
    """
    lat_on = on_coarse_multiple(lat, step)
    lon_on = on_coarse_multiple(lon, step)

    if lat_on and lon_on:
        return coarse_has_data(coarse_map, lat, lon)

    if lat_on ^ lon_on:
        if lon_on:
            lat_lo = np.floor(lat * 2) / 2
            lat_hi = lat_lo + step
            lon_snap = np.round(lon * 2) / 2
            a = coarse_has_data(coarse_map, lat_lo, lon_snap)
            b = coarse_has_data(coarse_map, lat_hi, lon_snap)
            return a and b
        lat_snap = np.round(lat * 2) / 2
        lon_lo = np.floor(lon * 2) / 2
        lon_hi = lon_lo + step
        a = coarse_has_data(coarse_map, lat_snap, lon_lo)
        b = coarse_has_data(coarse_map, lat_snap, lon_hi)
        return a and b

    lat_lo = np.floor(lat * 2) / 2
    lat_hi = lat_lo + step
    lon_lo = np.floor(lon * 2) / 2
    lon_hi = lon_lo + step
    corners = [
        coarse_has_data(coarse_map, lat_lo, lon_lo),
        coarse_has_data(coarse_map, lat_lo, lon_hi),
        coarse_has_data(coarse_map, lat_hi, lon_lo),
        coarse_has_data(coarse_map, lat_hi, lon_hi),
    ]
    return sum(1 for x in corners if x) >= 3


def build_fine_land_mask_from_coarse_swh(target_points, coarse_map):
    """target_points: (N, 2) with columns [lat, lon]. Returns float32 vector 1=ocean, 0=land."""
    out = np.zeros(target_points.shape[0], dtype=np.float32)
    for i in range(target_points.shape[0]):
        la, lo = float(target_points[i, 0]), float(target_points[i, 1])
        if fine_mask_ocean_at(la, lo, coarse_map):
            out[i] = 1.0
    return out


def upsample_file(
    input_path,
    output_dir,
    target_points,
    lat_grid,
    lon_grid,
    save_csv=False,
):
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

    for t in range(n_times):
        row = data_matrix[t]
        valid = ~np.isnan(row)
        if valid.sum() < 3:
            continue
        interp = griddata(src_points[valid], row[valid], target_points, method="linear")
        result[t] = interp.astype(np.float32)

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


def write_land_mask(output_dir, mask_flat, lat_grid, lon_grid):
    target_colnames = column_names_from_grids(lat_grid, lon_grid)
    mask_df = pd.DataFrame(mask_flat.reshape(1, -1), columns=target_colnames)
    mask_pkl = os.path.join(output_dir, "land_mask.pkl")
    mask_df.to_pickle(mask_pkl)
    print(f"Saved land_mask: {mask_pkl}", flush=True)


def main():
    input_dir = DEFAULT_INPUT_DIR
    output_dir = DEFAULT_OUTPUT_DIR
    save_csv = False

    args = list(sys.argv[1:])

    if "-h" in args or "--help" in args:
        print(
            "Usage: python upsample.py [input_dir] [output_dir] [--csv]\n"
            f"  Default input:  {DEFAULT_INPUT_DIR}\n"
            f"  Default output: {DEFAULT_OUTPUT_DIR}\n"
            f"  Target resolution: {RES_DEG} deg\n"
            f"  land_mask: from raw SWH ({SWH_COARSE_STEP}° coarse) rules on fine grid; "
            "independent of interpolation."
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

    swh_path = find_swh_input_path(input_dir)
    if swh_path.endswith(".pkl"):
        swh_df = pd.read_pickle(swh_path)
    else:
        swh_df = pd.read_csv(swh_path)
    coarse_map = build_coarse_swh_has_data(swh_df)
    n_coarse = sum(1 for v in coarse_map.values() if v)
    print(
        f"Coarse SWH mask map from {swh_path}: {len(coarse_map)} columns, "
        f"{n_coarse} with any finite data",
        flush=True,
    )

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

    mask_flat = build_fine_land_mask_from_coarse_swh(target_points, coarse_map)
    n_ocean = int(mask_flat.sum())
    print(f"Fine land_mask: {n_ocean}/{mask_flat.size} ocean (1)", flush=True)

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
            save_csv=save_csv,
        )

    write_land_mask(output_dir, mask_flat, lat_grid, lon_grid)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
