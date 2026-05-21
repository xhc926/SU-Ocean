# UHSM：一种统一层次化空间多尺度多要素海况预测框架

本项目是 UHSM（A Unified Hierarchical Spatial Multi-scale framework for multi-factor oceanic forecasting）的 PyTorch 实现。

## 方法简介

UHSM 由两条核心路径组成：

- 多尺度特征编码：先对多要素海况数据进行多尺度空间下采样，提取不同粒度特征；再通过层次化 Transformer 编码器-解码器结构在各尺度进行时空建模与跨尺度融合，形成由粗到细的表征学习路径，最终输出最细粒度结果。
- 多要素协同预测：通过全要素特征级联与跨要素混叠实现联合训练，并在输出端为各要素配置 MLP 头，通过多任务学习完成一体化预报。

## 当前支持模型

- `olinear`
- `simpletm`
- `itransformer` / `itransformerUHSM` / `itransformerUHSM4` / `itransformerUHSMAbl`
- `emaformer` / `emaformerUHSM` / `emaformerUHSM4`
- `dualformer`

## 环境要求

- Python >= 3.8（推荐 3.8）
- PyTorch（测试版本 2.0.0）
- CUDA（测试版本 11.8，可选）
- numpy
- pandas == 2.0.0
- scikit-learn
- reformer-pytorch == 1.4.4
- PyWavelets == 1.4.1

## 数据准备

- 数据入口由 `run.py` 中的 `--data` 与 `data_parser` 管理。
- 常用数据名包括 `ALL1/ALL2/ALL3/ALL4`，会自动映射到对应 `data_path` 与 `root_path`。
- 可选传入 `--land_mask_path` 用于海陆掩码加权评估。

## 运行方式

### 使用脚本批量运行

- 多尺度实验：`bash scripts/run_multiscale.sh`
- 其他示例：`scripts/run_baseline.sh`、`scripts/run_simpletm.sh`、`scripts/run_olinear.sh`、`scripts/run_dualformer.sh`、`scripts/run_ablation.sh`

## 输出说明

- 日志默认输出到 `--log_dir`
- 模型权重默认输出到 `--checkpoints`
- 结果默认输出到 `--results_dir`（含 `pred.npy`、`true.npy`、`metrics.npy` 等）
