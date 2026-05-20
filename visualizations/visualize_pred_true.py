#!/usr/bin/env python3
"""
使用 pred.npy 和 true.npy 做预测值与真实值的简单对比可视化。

数据含义（以单要素为例）:
  pred.npy / true.npy 形状: (N, pred_len, D)
  - N: 测试样本数（每段历史对应一个“样本”）
  - pred_len: 预测的时间步数（如 24 表示未来 24 步）
  - D: 空间格点数（如 6411、64800）

生成的 4 张图分别画什么:
  1) vis_timeseries_one_point.png
     只看“第 1 个样本、第 1 个格点”：横轴是未来 24 步，纵轴是该格点的数值；
     蓝线=真实值，红线=预测值。用来看单点随时间好不好跟。
  3) vis_scatter.png
     把所有样本、所有时间步、所有格点的数值摊平，每个点 = (真实值, 预测值)；
     点越贴近红色虚线 y=x，说明预测越准。
  4) vis_multi_samples.png
     选同一个格点，画 3 个不同样本的“未来 24 步”曲线（每个样本一子图）；
     每个子图里蓝=真实、红=预测，看不同时段、同一点上的表现。

用法:
  python visualize_pred_true.py "/root/autodl-tmp/results/你的实验setting名"
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_data(result_dir):
    pred_path = os.path.join(result_dir, 'pred.npy')
    true_path = os.path.join(result_dir, 'true.npy')
    if not os.path.exists(pred_path):
        raise FileNotFoundError('未找到 pred.npy，请确认路径: {}'.format(pred_path))
    if not os.path.exists(true_path):
        raise FileNotFoundError('未找到 true.npy，请确认路径: {}'.format(true_path))
    preds = np.load(pred_path)
    trues = np.load(true_path)
    return preds, trues


def plot_timeseries_one_point(preds, trues, sample_idx=0, spatial_idx=0, save_path=None):
    """单一样本、单一空间点的预测 vs 真实 时间序列"""
    # preds/trues: (N, pred_len, D) 或 (N, pred_len, D, F)
    if preds.ndim == 4:
        pred = preds[sample_idx, :, spatial_idx, 0]   # 多要素取第一个
        true = trues[sample_idx, :, spatial_idx, 0]
    else:
        pred = preds[sample_idx, :, spatial_idx]
        true = trues[sample_idx, :, spatial_idx]
    steps = np.arange(len(pred))
    plt.figure(figsize=(8, 4))
    plt.plot(steps, true, 'b-', label='True', alpha=0.8)
    plt.plot(steps, pred, 'r--', label='Pred', alpha=0.8)
    plt.xlabel('Prediction step')
    plt.ylabel('Value')
    plt.legend()
    plt.title('Sample {}, Spatial point {}'.format(sample_idx, spatial_idx))
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_scatter(preds, trues, max_points=50000, save_path=None):
    """散点图：预测值 vs 真实值（抽样避免点过多）"""
    pred_flat = preds.ravel()
    true_flat = trues.ravel()
    if len(pred_flat) > max_points:
        idx = np.random.choice(len(pred_flat), max_points, replace=False)
        pred_flat = pred_flat[idx]
        true_flat = true_flat[idx]
    plt.figure(figsize=(6, 6))
    plt.scatter(true_flat, pred_flat, alpha=0.3, s=5)
    vmin = min(true_flat.min(), pred_flat.min())
    vmax = max(true_flat.max(), pred_flat.max())
    plt.plot([vmin, vmax], [vmin, vmax], 'r--', lw=2, label='y=x')
    plt.xlabel('True')
    plt.ylabel('Pred')
    plt.legend()
    plt.title('Pred vs True (scatter)')
    plt.axis('equal')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_multiple_samples(preds, trues, n_samples=3, spatial_idx=0, save_path=None):
    """多个样本在同一空间点上的预测 vs 真实（子图）。
    测试集是滑窗得到的，连续样本(0,1,2...)的真实值重叠大、曲线几乎一样；
    这里改为在时间上均匀取样本（如 0, N/3, 2N/3），使每条曲线对应不同时段。"""
    N = preds.shape[0]
    if preds.ndim == 4:
        preds_2d = preds[:, :, spatial_idx, 0]  # (N, pred_len)
        trues_2d = trues[:, :, spatial_idx, 0]
    else:
        preds_2d = preds[:, :, spatial_idx]
        trues_2d = trues[:, :, spatial_idx]
    n_plot = min(n_samples, N)
    # 均匀取样本索引，避免连续样本导致真实曲线重叠、几乎重合
    if n_plot >= N:
        indices = np.arange(N)
    else:
        indices = np.linspace(0, N - 1, n_plot, dtype=int)
    fig, axes = plt.subplots(n_plot, 1, figsize=(8, 2.5 * n_plot), sharex=True)
    if n_plot == 1:
        axes = [axes]
    steps = np.arange(preds_2d.shape[1])
    for k, i in enumerate(indices):
        axes[k].plot(steps, trues_2d[i], 'b-', label='True', alpha=0.8)
        axes[k].plot(steps, preds_2d[i], 'r--', label='Pred', alpha=0.8)
        axes[k].set_ylabel('Value')
        axes[k].legend(loc='upper right', fontsize=8)
        axes[k].set_title('Sample {} (time window index {})'.format(k, i))
    axes[-1].set_xlabel('Prediction step')
    plt.suptitle('Spatial point {} (samples spread over test set)'.format(spatial_idx))
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Pred vs True 简单可视化')
    parser.add_argument('path', nargs='?', default=None,
                        help='结果目录完整路径（含空格时请用引号包住，推荐此方式）')
    parser.add_argument('--result_dir', type=str, default='/root/autodl-tmp/results/',
                        help='结果目录；与 --setting 合用。若已用 path 指定完整路径则忽略')
    parser.add_argument('--setting', type=str, default=None,
                        help='实验名（与 result_dir 拼接）；路径含空格时请改用位置参数 path')
    parser.add_argument('--sample_idx', type=int, default=0, help='时间序列图使用的样本下标')
    parser.add_argument('--spatial_idx', type=int, default=0, help='使用的空间点下标')
    parser.add_argument('--n_samples', type=int, default=3, help='多样本子图数量')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='图片保存目录，默认与 result_dir 相同')
    args = parser.parse_args()

    if args.path:
        result_dir = args.path.rstrip('/')
    elif args.setting:
        result_dir = os.path.join(args.result_dir, args.setting)
    else:
        result_dir = args.result_dir.rstrip('/')
    out_dir = args.out_dir or result_dir
    os.makedirs(out_dir, exist_ok=True)

    preds, trues = load_data(result_dir)
    print('pred shape: {}, true shape: {}'.format(preds.shape, trues.shape))

    # 1) 单样本单点 时间序列
    plot_timeseries_one_point(
        preds, trues,
        sample_idx=args.sample_idx,
        spatial_idx=min(args.spatial_idx, preds.shape[-1] - 1) if preds.ndim == 3 else 0,
        save_path=os.path.join(out_dir, 'vis_timeseries_one_point.png')
    )
    print('Saved: vis_timeseries_one_point.png')

    # 3) 散点图
    plot_scatter(preds, trues, save_path=os.path.join(out_dir, 'vis_scatter.png'))
    print('Saved: vis_scatter.png')

    # 4) 多样本同一空间点
    plot_multiple_samples(
        preds, trues,
        n_samples=args.n_samples,
        spatial_idx=min(args.spatial_idx, preds.shape[2] - 1) if preds.ndim == 3 else 0,
        save_path=os.path.join(out_dir, 'vis_multi_samples.png')
    )
    print('Saved: vis_multi_samples.png')

    print('Done. Figures saved to:', out_dir)


if __name__ == '__main__':
    main()

# 用法示例（路径含空格时用引号包住，推荐用位置参数）:
#   python visualize_pred_true.py "/root/autodl-tmp/results/itransformer_uo_ftM_sl96_ll48_pl24_..._ms[8, 4, 2, 1]_peFalse_itr0"
#   python visualize_pred_true.py --result_dir /root/autodl-tmp/results/ --setting "itransformer_uo_ftM_..._ms[8, 4, 2, 1]_peFalse_itr0"