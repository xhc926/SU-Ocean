#!/usr/bin/env bash
# 顺序执行：run_baseline.sh → run_multiscale.sh
#
# 用法（在仓库根目录）:
#   bash scripts/run_all.sh
#   bash scripts/run_all.sh swh_1_4    # 仅传给 run_baseline.sh 的第一个参数
#
# multiscale 阶段固定为 run_multiscale.sh 当前配置（ALL2 等）；MS_DATA 已不再使用。

set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "############################################"
echo "### Phase 1/2: run_baseline.sh"
echo "############################################"
bash scripts/run_baseline.sh "$@"

echo ""
echo "############################################"
echo "### Phase 2/2: run_multiscale.sh"
echo "############################################"
bash scripts/run_multiscale.sh

echo ""
echo ">>> all phases completed."
