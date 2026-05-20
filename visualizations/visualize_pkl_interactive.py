#!/usr/bin/env python3
"""
像 visualize_interactive.py 一样，对 pkl 数据做交互式可视化（仅第一列）。
pkl 格式与 data_loader 一致：DataFrame 含 'date' 列 + 若干数据列。

用法:
  python visualize_pkl_interactive.py
  python visualize_pkl_interactive.py --pkl /path/to/other.pkl --out my_vis.html
"""

import argparse
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def main():
    parser = argparse.ArgumentParser(description='pkl 第一列交互式可视化')
    parser.add_argument('--pkl', type=str, default='/root/autodl-tmp/SAL8.pkl', help='pkl 文件路径')
    parser.add_argument('--out', type=str, default='SAL8_col0_interactive.html', help='输出 html 文件名')
    parser.add_argument('--title', type=str, default='SAL8 第1列', help='图表标题')
    parser.add_argument('--window_days', type=int, default=336, help='滑动窗口天数')
    parser.add_argument('--step_days', type=int, default=14, help='窗口步长天数')
    args = parser.parse_args()

    df = pd.read_pickle(args.pkl)
    if 'date' not in df.columns:
        raise ValueError("pkl 需包含 'date' 列，当前列: {}".format(list(df.columns)))
    df['date'] = pd.to_datetime(df['date'])

    # 第一列数据：去掉 date 后的第一个列
    data_cols = [c for c in df.columns if c != 'date']
    col = data_cols[0] if data_cols else None
    if col is None:
        raise ValueError("pkl 中除 date 外无数据列")

    fig = make_subplots(rows=1, cols=1, subplot_titles=(f'{col} (第1列)',), shared_xaxes=True)
    fig.add_trace(
        go.Scatter(x=df['date'], y=df[col], name=col, mode='lines', line=dict(width=1, color='#1f77b4')),
        row=1, col=1
    )
    fig.update_yaxes(title_text=f'{col}', row=1, col=1)
    xaxis_title_row = 1

    fig.update_layout(
        title=args.title,
        hovermode='x unified',
        height=500,
        template='plotly_white',
        showlegend=True
    )
    fig.update_xaxes(title_text='时间', row=xaxis_title_row, col=1)

    t0 = df['date'].min()
    t1 = df['date'].max()
    window_size_days = args.window_days
    step_days = args.step_days

    windows = []
    start = t0
    while start + pd.Timedelta(days=window_size_days) <= t1:
        end = start + pd.Timedelta(days=window_size_days)
        windows.append((start, end))
        start = start + pd.Timedelta(days=step_days)
    if not windows:
        windows = [(t0, t1)]

    steps = []
    for i, (s, e) in enumerate(windows):
        label = s.strftime('%Y-%m-%d')
        args_relayout = [{
            'xaxis.range': [s.strftime('%Y-%m-%d %H:%M:%S'), e.strftime('%Y-%m-%d %H:%M:%S')]
        }]
        step = {'label': label, 'method': 'relayout', 'args': args_relayout}
        steps.append(step)

    slider = {'active': len(steps) - 1, 'currentvalue': {'prefix': '窗口开始: '}, 'pad': {'t': 50}, 'steps': steps}
    fig.update_layout(sliders=[slider])

    fig.write_html(args.out)
    print("交互式图表已生成: {}".format(args.out))
    print("支持滑动窗口。可调参数: --window_days --step_days")
    fig.show()

if __name__ == '__main__':
    main()
