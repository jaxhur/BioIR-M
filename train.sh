#!/usr/bin/env bash
CONFIG="$1"
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

# torchrun 直接执行 train.py 时不会自动加入仓库根目录，显式注入以保证可导入 basicsr。
# 不在此处固定 CUDA_VISIBLE_DEVICES；4090/5090 服务器由调用者按实际租用卡设置。
PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
torchrun \
  --nproc_per_node=1 \
  --master_port="${MASTER_PORT:-4322}" \
  "${PROJECT_ROOT}/basicsr/train.py" \
  -opt "$CONFIG" \
  --launcher pytorch






