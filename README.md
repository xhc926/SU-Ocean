# UHSM: A Unified Hierarchical Spatial Multi-scale framework for multi-factor oceanic forecasting

This repository provides a PyTorch implementation of UHSM

## Method Overview

UHSM has two key ideas:

- Multi-scale feature encoding: multi-factor ocean fields are spatially downsampled at multiple scales, then modeled by a hierarchical Transformer encoder-decoder with cross-scale fusion, forming a coarse-to-fine representation path and producing predictions at the finest scale.
- Multi-factor collaborative forecasting: all factors are concatenated and mixed for joint training; factor-specific MLP heads are used at the output side for integrated multi-task forecasting.

## Supported Models

- `olinear`
- `simpletm`
- `itransformer` / `itransformerUHSM` / `itransformerUHSM4` / `itransformerUHSMAbl`
- `emaformer` / `emaformerUHSM` / `emaformerUHSM4`
- `dualformer`

## Requirements

- Python >= 3.8 (3.8 recommended)
- PyTorch (tested with 2.0.0)
- CUDA (tested with 11.8, optional)
- numpy
- pandas == 2.0.0
- scikit-learn
- reformer-pytorch == 1.4.4
- PyWavelets == 1.4.1

## Data Setup

- Dataset routing is configured in `run.py` via `--data` and `data_parser`.
- Common dataset keys include `ALL1/ALL2/ALL3/ALL4`, which auto-map to corresponding `data_path` and `root_path`.
- Optional `--land_mask_path` can be used for land/sea mask weighted evaluation.

## Get Started

### Run provided scripts

- Multi-scale experiments: `bash scripts/run_multiscale.sh`
- Other examples: `scripts/run_baseline.sh`, `scripts/run_simpletm.sh`, `scripts/run_olinear.sh`, `scripts/run_dualformer.sh`, `scripts/run_ablation.sh`

## Outputs

- Logs: `--log_dir`
- Checkpoints: `--checkpoints`
- Results: `--results_dir` (e.g., `pred.npy`, `true.npy`, `metrics.npy`)
