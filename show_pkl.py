#!/usr/bin/env python3
"""
查看 uniocean/data 目录下的 pkl/npy/npz/joblib 文件的简单脚本。
用法示例：
  列出数据文件:
    python uniocean\show_pkl.py --list
  查看文件摘要:
    python uniocean\show_pkl.py --file some.pkl
  指定数据目录:
    python uniocean\show_pkl.py -d uniocean/data --list
"""
import argparse
import os
import pickle
import sys
import joblib
import numpy as np
import pandas as pd


def list_pkls(path):
    out = []
    for root, _, files in os.walk(path):
        for f in files:
            if f.lower().endswith(('.pkl', '.pickle', '.npy')):
                out.append(os.path.join(root, f))
    return sorted(out)


def load_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        return np.load(path, allow_pickle=True)
    if ext == '.npz':
        return np.load(path, allow_pickle=True)
    if ext == '.joblib':
        if joblib is None:
            raise RuntimeError('joblib not available in this environment')
        return joblib.load(path)
    with open(path, 'rb') as f:
        return pickle.load(f)


def summarize(obj, name='root', max_items=8):
    t = type(obj)
    print(f"{name}: {t}")
    if isinstance(obj, np.ndarray):
        print(f"  numpy.ndarray shape={getattr(obj, 'shape', None)} dtype={getattr(obj, 'dtype', None)}")
    elif isinstance(obj, (pd.DataFrame, pd.Series)):
        print(f"  pandas {type(obj).__name__} shape={obj.shape}")
        with pd.option_context('display.max_rows', 10, 'display.max_columns', 10):
            print(obj.head( min(10, len(obj)) ))
    elif isinstance(obj, dict):
        print(f"  dict with {len(obj)} keys")
        for i, (k, v) in enumerate(obj.items()):
            if i >= max_items:
                print("  ...")
                break
            print(f"   - {k}: {type(v)}", end='')
            if isinstance(v, np.ndarray):
                print(f" shape={getattr(v,'shape',None)} dtype={getattr(v,'dtype',None)}")
            elif isinstance(v, (pd.DataFrame, pd.Series)):
                print(f" shape={getattr(v,'shape',None)}")
            else:
                print("")
    elif isinstance(obj, (list, tuple, set)):
        print(f"  {t.__name__} len={len(obj)}")
        for i, v in enumerate(list(obj)[:max_items]):
            print(f"   - [{i}] type={type(v)}")
    else:
        try:
            s = repr(obj)
            if len(s) > 300:
                s = s[:300] + '...'
            print("  repr:", s)
        except Exception:
            print("  (no repr available)")


def print_mask_column_stats(df, n_examples=5, row_index=0):
    """
    For a mask DataFrame (e.g. land_mask.pkl: one row, many lat_lon columns),
    count columns with value 0 vs 1, list n_examples of each, and report others.
    Uses df.iloc[row_index] only.
    """
    if not isinstance(df, pd.DataFrame) or df.shape[1] == 0:
        print("不是非空的 DataFrame，跳过掩码统计。")
        return
    if row_index < 0 or row_index >= len(df):
        print(f"row_index {row_index} 越界，行数={len(df)}")
        return
    row = df.iloc[row_index]
    zero_cols, one_cols, other = [], [], []
    for col in df.columns:
        v = row[col]
        if pd.isna(v):
            other.append((col, "nan"))
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            other.append((col, repr(v)))
            continue
        if np.isclose(fv, 0.0):
            zero_cols.append(col)
        elif np.isclose(fv, 1.0):
            one_cols.append(col)
        else:
            other.append((col, fv))

    total = len(df.columns)
    print(f"行索引: {row_index}（共 {len(df)} 行）")
    print(f"列数总计: {total}")
    print(f"值为 0 的列数: {len(zero_cols)}")
    print(f"值为 1 的列数: {len(one_cols)}")
    if other:
        print(f"非 0/1 或 NaN 的列数: {len(other)}")
        for col, val in other[:10]:
            print(f"  示例: {col!r} -> {val}")
        if len(other) > 10:
            print(f"  ... 另有 {len(other) - 10} 列")
    print(f"\n前 {min(n_examples, len(zero_cols))} 个「值为 0」的列名:")
    for c in zero_cols[:n_examples]:
        print(f"  {c}")
    print(f"\n前 {min(n_examples, len(one_cols))} 个「值为 1」的列名:")
    for c in one_cols[:n_examples]:
        print(f"  {c}")


def main():
    parser = argparse.ArgumentParser(description='查看 uniocean/data 中的 pkl/npy/npz/joblib 文件')
    parser.add_argument('--data-dir', '-d', default=os.path.join(os.path.dirname(__file__), 'data'),
                        help='数据目录（默认：uniocean/data）')
    parser.add_argument('--list', '-l', action='store_true', help='列出数据文件')
    parser.add_argument('--file', '-f', help='要查看的文件名或路径（相对 data-dir 或绝对路径）')
    parser.add_argument('--keys', '-k', action='store_true', help='当文件内容是 dict 时，打印每个 key 的详细摘要')
    parser.add_argument('--head', '-n', type=int, default=10, help='DataFrame 显示前 n 行（默认 10）')
    parser.add_argument('--mask', '-m', action='store_true',
                        help='DataFrame 为单行/掩码表时：按列统计 0/1 个数并各列出 5 个列名（不截断列）')
    parser.add_argument('--mask-examples', type=int, default=5, metavar='N',
                        help='与 --mask 联用：各列举 N 个 0 列与 1 列（默认 5）')
    parser.add_argument('--mask-row', type=int, default=0, metavar='I',
                        help='与 --mask 联用：使用第 I 行（默认 0）')
    args = parser.parse_args()

    data_dir = args.data_dir
    if not os.path.exists(data_dir):
        print('数据目录不存在：', data_dir)
        sys.exit(1)

    if args.list:
        files = list_pkls(data_dir)
        if not files:
            print('未在目录中找到匹配的文件：', data_dir)
            return
        for p in files:
            print(p)
        return

    if not args.file:
        print('请指定 --file 或使用 --list 列出可用文件')
        return

    p = args.file
    if not os.path.isabs(p):
        p = os.path.join(data_dir, p)
    if not os.path.exists(p):
        print('文件不存在：', p)
        return

    try:
        obj = load_file(p)
    except Exception as e:
        print('加载失败：', e)
        return

    print('文件：', p)
    print('对象类型：', type(obj))

    if isinstance(obj, dict):
        print('keys:', list(obj.keys()))
        if args.keys:
            for k, v in obj.items():
                print('\n--- key:', k, 'type=', type(v))
                summarize(v, name=str(k))
        else:
            summarize(obj)
    elif isinstance(obj, (pd.DataFrame, pd.Series)):
        if isinstance(obj, pd.DataFrame) and args.mask:
            print_mask_column_stats(obj, n_examples=args.mask_examples, row_index=args.mask_row)
            print("\n--- DataFrame 摘要（宽表不打印全部单元格）---")
            print(f"shape: {obj.shape}")
            try:
                obj.info(verbose=False, memory_usage="deep")
            except Exception:
                obj.info()
        else:
            with pd.option_context("display.max_columns", 20, "display.width", 200):
                print(obj.head(args.head))
            try:
                print("\ninfo:")
                obj.info()
            except Exception:
                pass
    else:
        summarize(obj)


if __name__ == '__main__':
    main()
