#!/usr/bin/env bash
CONFIG="$1"
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

# torchrun 直接执行 train.py 时不会自动加入仓库根目录，显式注入以保证可导入 basicsr。
PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
CUDA_VISIBLE_DEVICES=0 torchrun \
  --nproc_per_node=1 \
  --master_port=4322 \
  "${PROJECT_ROOT}/basicsr/train.py" \
  -opt "$CONFIG" \
  --launcher pytorch






