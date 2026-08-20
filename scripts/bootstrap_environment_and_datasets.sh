#!/usr/bin/env bash
# 根据 README 创建 BioIR 环境、安装依赖、下载/解压并校验 LOL-v1 与 LOL-v2。
# 本脚本面向 Linux/AutoDL 等远程服务器；不启动训练，也不会默认删除已有环境或数据。

set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="bioir"
DATA_ROOT="${PROJECT_ROOT}/datasets"
DATA_SOURCE="gdrive"
AUTODL_ROOT="/root/autodl-fs"
RECREATE_ENV=0
REPLACE_DATASETS=0
SKIP_ENV=0
SKIP_DATA=0
KEEP_ARCHIVES=0

LOL_V1_URL="https://drive.google.com/uc?id=1mAN3ll5wWwt1Xz0C7uio31-NJu-50S8Z"
LOL_V2_URL="https://drive.google.com/uc?id=1L0UnJg6gZ4Eb7It2EuNxP0L3lQNmKMaP"

usage() {
  cat <<'EOF'
用法：
  bash scripts/bootstrap_environment_and_datasets.sh [选项]

默认行为：创建或复用 bioir (Python 3.9) 环境，安装 README 指定依赖，
从 Google Drive 下载 LOL-v1/LOL-v2 ZIP，解压为项目配置需要的目录并校验配对。

选项：
  --env-name NAME          Conda 环境名，默认 bioir。
  --data-root PATH         数据根目录，默认 <项目>/datasets。
  --data-source SOURCE     gdrive、autodl 或 skip，默认 gdrive。
  --autodl-root PATH       AutoDL 归档所在目录，默认 /root/autodl-fs。
  --recreate-env           明确删除并重建同名 Conda 环境。
  --replace-datasets       明确删除校验失败的 LOL-v1/LOL-v2 目标目录后重新解压。
  --skip-env               跳过环境创建与依赖安装，要求目标环境已存在。
  --skip-data              跳过下载和解压，只校验现有数据目录。
  --keep-archives          成功后保留 datasets/.downloads 中的 ZIP 文件。
  -h, --help               显示本帮助。
EOF
}

die() {
  echo "[错误] $*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name)
      ENV_NAME="${2:?缺少环境名}"; shift 2 ;;
    --data-root)
      DATA_ROOT="${2:?缺少数据根目录}"; shift 2 ;;
    --data-source)
      DATA_SOURCE="${2:?缺少数据来源}"; shift 2 ;;
    --autodl-root)
      AUTODL_ROOT="${2:?缺少 AutoDL 目录}"; shift 2 ;;
    --recreate-env)
      RECREATE_ENV=1; shift ;;
    --replace-datasets)
      REPLACE_DATASETS=1; shift ;;
    --skip-env)
      SKIP_ENV=1; shift ;;
    --skip-data)
      SKIP_DATA=1; shift ;;
    --keep-archives)
      KEEP_ARCHIVES=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      die "未知参数：$1" ;;
  esac
done

[[ "$DATA_SOURCE" == "gdrive" || "$DATA_SOURCE" == "autodl" || "$DATA_SOURCE" == "skip" ]] \
  || die "--data-source 只能是 gdrive、autodl 或 skip。"

command -v conda >/dev/null 2>&1 || die "未找到 conda。请先加载 Miniconda/Anaconda。"

environment_exists() {
  conda env list | awk 'NR > 2 {gsub(/\*/, "", $1); print $1}' | grep -Fxq "$ENV_NAME"
}

run_in_env() {
  conda run -n "$ENV_NAME" "$@"
}

ensure_environment() {
  if [[ "$SKIP_ENV" -eq 1 ]]; then
    environment_exists || die "--skip-env 需要已有 Conda 环境：$ENV_NAME"
    echo "[环境] 跳过创建，复用已有环境：$ENV_NAME"
    return
  fi

  if environment_exists && [[ "$RECREATE_ENV" -eq 1 ]]; then
    echo "[环境] 删除旧环境：$ENV_NAME"
    conda env remove -n "$ENV_NAME" -y
  fi
  if ! environment_exists; then
    echo "[环境] 创建 Python 3.9 环境：$ENV_NAME"
    conda create -n "$ENV_NAME" python=3.9 -y
  else
    echo "[环境] 复用已有环境：$ENV_NAME"
  fi

  local python_version
  python_version="$(run_in_env python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  [[ "$python_version" == "3.9" ]] || die "环境 $ENV_NAME 的 Python 为 $python_version，README 要求 Python 3.9；请使用 --recreate-env。"

  echo "[环境] 安装 PyTorch 2.4.0 + CUDA 12.4 轮子"
  run_in_env python -m pip install --upgrade pip
  run_in_env python -m pip install --no-cache-dir \
    torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu124
  echo "[环境] 安装项目依赖"
  run_in_env python -m pip install --no-cache-dir \
    opencv-python lmdb tqdm einops scipy scikit-image tensorboard natsort \
    pyiqa joblib lpips ptflops scikit-learn pandas thop gdown
  run_in_env python -m pip install -e "$PROJECT_ROOT"
  run_in_env python -c 'import basicsr, torch; print("[环境] basicsr=", basicsr.__file__); print("[环境] torch=", torch.__version__, "cuda=", torch.version.cuda, "available=", torch.cuda.is_available())'
}

verify_datasets() {
  run_in_env python "$PROJECT_ROOT/scripts/prepare_lol_datasets.py" \
    --data-root "$DATA_ROOT" --verify-only
}

download_with_gdrive() {
  local output_path="$1"
  local url="$2"
  if [[ -f "$output_path" ]]; then
    echo "[数据] 复用已下载归档：$output_path"
    return
  fi
  echo "[数据] 下载：$url"
  run_in_env python -m gdown --fuzzy "$url" --output "$output_path"
}

prepare_datasets() {
  if [[ "$SKIP_DATA" -eq 1 || "$DATA_SOURCE" == "skip" ]]; then
    echo "[数据] 跳过下载与解压，仅校验现有数据。"
    verify_datasets
    return
  fi

  if verify_datasets; then
    echo "[数据] LOL-v1、LOL-v2 已完整配对，跳过下载。"
    return
  fi

  mkdir -p "$DATA_ROOT/.downloads"
  local v1_archive="$DATA_ROOT/.downloads/LOL-v1.zip"
  local v2_archive="$DATA_ROOT/.downloads/LOL-v2-renamed.zip"
  if [[ "$DATA_SOURCE" == "gdrive" ]]; then
    download_with_gdrive "$v1_archive" "$LOL_V1_URL"
    download_with_gdrive "$v2_archive" "$LOL_V2_URL"
  else
    [[ -f "$AUTODL_ROOT/LOL-v1.zip" ]] || die "缺少 AutoDL 归档：$AUTODL_ROOT/LOL-v1.zip"
    [[ -f "$AUTODL_ROOT/LOL-v2-renamed.zip" ]] || die "缺少 AutoDL 归档：$AUTODL_ROOT/LOL-v2-renamed.zip"
    cp -f -- "$AUTODL_ROOT/LOL-v1.zip" "$v1_archive"
    cp -f -- "$AUTODL_ROOT/LOL-v2-renamed.zip" "$v2_archive"
  fi

  local replace_option=()
  if [[ "$REPLACE_DATASETS" -eq 1 ]]; then
    replace_option=(--replace-existing)
  fi
  run_in_env python "$PROJECT_ROOT/scripts/prepare_lol_datasets.py" \
    --data-root "$DATA_ROOT" \
    --lol-v1-archive "$v1_archive" \
    --lol-v2-archive "$v2_archive" \
    "${replace_option[@]}"
  if [[ "$KEEP_ARCHIVES" -eq 0 ]]; then
    rm -f -- "$v1_archive" "$v2_archive"
  fi
}

echo "[开始] 项目目录：$PROJECT_ROOT"
echo "[开始] 环境名称：$ENV_NAME"
echo "[开始] 数据目录：$DATA_ROOT"
ensure_environment
prepare_datasets
echo "[完成] 环境与数据集已准备。"
echo "[下一步] conda activate $ENV_NAME"
