#!/usr/bin/env python3
"""
OLinear：谱分解时间 Q 矩阵（UniOcean area3 四要素）

与仓库内原 Generate_corrmat.ipynb 同一套公式：对训练段序列按通道构造滞后块协方差，
归一化对角后按通道平均得 Σ，再 eigh 取特征向量并 flip 得到 Q_mat。

输入：AREA_ROOT 下 sal.pkl, uo.pkl, vo.pkl, ssh.pkl（与 run.py 中 sal_1_12 等数据根目录一致）。
输出：OUT_DIR（默认 AREA_ROOT/olinear_q/）下的 {变量名}_{time_lag}_ratio{r}.npy，形状 (time_lag, time_lag)。

训练示例：
  --q_mat_file olinear_q/sal_32_ratio0.7.npy 对应 --seq_len 32
  --q_out_mat_file olinear_q/sal_16_ratio0.7.npy 对应 --pred_len 16
  若 seq_len=16 且 pred_len=8，还需 sal_8_ratio0.7.npy（--lags 必须含 8）。

用法：
  python generate_corrmat.py
  python generate_corrmat.py --area-root /path/to/area3 --lags 8,12,16,24,32,48
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from numpy.linalg import eigh


def temporal_q_mat(A: np.ndarray, time_lag: int) -> np.ndarray:
    """A: (time, n_features), float32. Returns (time_lag, time_lag)."""
    sigma_list = []
    n_feat = A.shape[1]
    for feature_idx in range(n_feat):
        lagged_matrix = np.array(
            [A[i : A.shape[0] - time_lag + i + 1, feature_idx] for i in range(time_lag)]
        )
        if np.isnan(lagged_matrix).any():
            lagged_matrix = np.nan_to_num(lagged_matrix)
        cov_matrix = np.cov(lagged_matrix)
        diag_vec = np.diag(cov_matrix)
        if (diag_vec < 1e-4).any():
            continue
        cov_matrix = cov_matrix / diag_vec
        sigma_list.append(np.asarray(cov_matrix, dtype=np.float32))
    if not sigma_list:
        raise RuntimeError(f"No valid features for time_lag={time_lag}; check data")
    sigma = np.mean(sigma_list, axis=0)
    _, eigenvectors = eigh(sigma)
    q_mat = np.flip(eigenvectors.T, axis=0)
    return q_mat.astype(np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate OLinear temporal Q matrices (.npy) for area3 factors.")
    p.add_argument(
        "--area-root",
        type=str,
        default="/root/autodl-tmp/data/upsampled/1-4/area2",
        help="Directory containing swh.pkl, u10.pkl, v10.pkl",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Output directory (default: AREA_ROOT/olinear_q)",
    )
    p.add_argument(
        "--lags",
        type=str,
        default="8,12,16,24,32,48",
        help="Comma-separated lags; must cover each training seq_len AND pred_len (e.g. seq_len/2 when label_len=pred_len=seq/2)",
    )
    p.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Train fraction for statistics (matches Dataset_Custom ~0.7 split)",
    )
    p.add_argument(
        "--base-ratio",
        type=float,
        default=1.0,
        help="Same meaning as original notebook base_ratio",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    area_root = os.path.abspath(args.area_root)
    out_dir = args.out_dir.strip() or os.path.join(area_root, "olinear_q")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    time_lags = sorted({int(x.strip()) for x in args.lags.split(",") if x.strip()})
    if not time_lags:
        print("No lags in --lags", file=sys.stderr)
        return 1

    train_ratio = args.train_ratio
    ratio = train_ratio * args.base_ratio

    datasets = [
        ("swh", "swh.pkl"),
        ("u10", "u10.pkl"),
        ("v10", "v10.pkl")
    ]

    print(f"AREA_ROOT={area_root}")
    print(f"OUT_DIR={out_dir}")
    print(f"TIME_LAGS={time_lags}, ratio={ratio:.2f}")

    for name, pkl_name in datasets:
        fp = os.path.join(area_root, pkl_name)
        if not os.path.isfile(fp):
            print(f"SKIP (missing): {fp}")
            continue
        df = pd.read_pickle(fp)
        cols = [c for c in df.columns if c != "date"]
        data = df[cols].values.astype(np.float32)
        train_length = int(data.shape[0] * train_ratio)
        a = data[train_length - int(data.shape[0] * ratio) : train_length]
        print(f"{name}: A.shape={a.shape}")
        for time_lag in time_lags:
            if a.shape[0] < time_lag + 2:
                print(f"  skip lag={time_lag} (need more rows)")
                continue
            q_mat = temporal_q_mat(a, time_lag)
            out_name = f"{name}_{time_lag}_ratio{ratio:.1f}.npy"
            out_path = os.path.join(out_dir, out_name)
            np.save(out_path, q_mat)
            print(f"  -> {out_path}  shape={q_mat.shape}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
