#!/usr/bin/env bash
set -euo pipefail

# 加载已训练模型，生成 output/result.csv。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
uv run python -m code.ranker.predict
