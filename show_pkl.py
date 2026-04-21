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


def main():
    parser = argparse.ArgumentParser(description='查看 uniocean/data 中的 pkl/npy/npz/joblib 文件')
    parser.add_argument('--data-dir', '-d', default=os.path.join(os.path.dirname(__file__), 'data'),
                        help='数据目录（默认：uniocean/data）')
    parser.add_argument('--list', '-l', action='store_true', help='列出数据文件')
    parser.add_argument('--file', '-f', help='要查看的文件名或路径（相对 data-dir 或绝对路径）')
    parser.add_argument('--keys', '-k', action='store_true', help='当文件内容是 dict 时，打印每个 key 的详细摘要')
    parser.add_argument('--head', '-n', type=int, default=10, help='DataFrame 显示前 n 行（默认 10）')
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
        print(obj.head(args.head))
        try:
            print('\ninfo:')
            obj.info()
        except Exception:
            pass
    else:
        summarize(obj)


if __name__ == '__main__':
    main()
