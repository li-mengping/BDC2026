#!/usr/bin/env bash
set -euo pipefail

# 训练两个排序模型，并按 validation Top5 选择 best iteration。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
python -m code.models.xgboost.train
