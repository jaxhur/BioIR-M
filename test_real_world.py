"""在无配对 GT 的真实低光数据集上测试 BasicSR 模型。"""

import argparse
import csv
import hashlib
import json
import logging
import shlex
import sys
from importlib import metadata
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from basicsr.models import create_model
from basicsr.utils.options import parse


METHOD_NAME = "CDRNet"
DEFAULT_FACTOR = 64
DATASET_NAMES = ("DICM", "LIME", "MEF", "NPE", "VV")
EXPECTED_COUNTS = {
    "DICM": {64},
    "LIME": {10},
    "MEF": {17},
    "NPE": {84, 85},
    "VV": {24},
}
IMAGE_EXTENSIONS = {
    ".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"
}


def sha256_file(path):
    """分块计算文件 SHA-256，避免一次性读取大权重。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def list_images(folder):
    """递归返回目录中按相对路径排序的非隐藏图像。"""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"数据集目录不存在：{folder}")
    images = [
        path for path in folder.rglob("*")
        if (path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            and not any(
                part.startswith(".") for part in path.relative_to(folder).parts
            ))
    ]
    if not images:
        raise RuntimeError(f"数据集目录中没有支持的图像：{folder}")
    return sorted(images, key=lambda path: path.relative_to(folder).as_posix())


def build_image_entries(dataset_root):
    """建立输入图像与 PNG 输出路径的稳定映射，并拒绝同名冲突。"""
    entries = []
    stem_keys = {}
    for input_path in list_images(dataset_root):
        relative_path = input_path.relative_to(dataset_root)
        stem_key = relative_path.with_suffix("").as_posix().casefold()
        if stem_key in stem_keys:
            raise ValueError(
                "去除扩展名后出现重复相对路径："
                f"{stem_keys[stem_key]} 与 {relative_path.as_posix()}"
            )
        stem_keys[stem_key] = relative_path.as_posix()
        entries.append((input_path, relative_path, relative_path.with_suffix(".png")))
    return entries


def validate_dataset_count(dataset_name, image_count, allow_mismatch):
    """核对公开数据集图像数；NPE 同时接受 84/85 张版本。"""
    expected = EXPECTED_COUNTS[dataset_name]
    if image_count in expected:
        return
    message = (
        f"{dataset_name} 图像数为 {image_count}，预期为 "
        f"{sorted(expected)}。请检查下载版本或使用 --allow-count-mismatch。"
    )
    if allow_mismatch:
        logging.warning(message)
        return
    raise ValueError(message)


def load_rgb(path):
    """读取 RGB 图像并固定为 uint8 三通道数组。"""
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def save_rgb(path, image):
    """以无损 PNG 保存 uint8 RGB 图像。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="RGB").save(path, format="PNG")


def write_csv(path, fieldnames, rows):
    """用 Excel 兼容的 UTF-8 编码写入结构化结果。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def configure_logger(log_path):
    """让终端和测试日志接收同一条消息。"""
    logger = logging.getLogger(f"real_world_{METHOD_NAME}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    for handler in (
        logging.StreamHandler(),
        logging.FileHandler(log_path, mode="w", encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def load_state_dict(weights_path):
    """读取常见生成器 checkpoint 包装并清理 DataParallel 前缀。"""
    weights_path = Path(weights_path)
    if weights_path.suffix.lower() == ".state":
        raise ValueError(f"测试只能加载模型权重，不能加载训练状态：{weights_path}")
    checkpoint = torch.load(weights_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"不支持的 checkpoint 类型：{type(checkpoint)}")
    for key in ("params_ema", "params", "state_dict", "model", "net_g"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            checkpoint = value
            break
    if not checkpoint or not all(torch.is_tensor(value) for value in checkpoint.values()):
        raise ValueError(f"checkpoint 中没有可识别的生成器参数：{weights_path}")
    return {
        key[7:] if key.startswith("module.") else key: value
        for key, value in checkpoint.items()
    }


def load_model(option_path, weights_path, device):
    """按 YAML 创建生成器，并严格加载命令行指定权重。"""
    opt = parse(str(option_path), is_train=False)
    opt["dist"] = False
    opt["num_gpu"] = 0 if device.type == "cpu" else 1
    opt.setdefault("path", {})["pretrain_network_g"] = None
    model = create_model(opt).net_g
    model.load_state_dict(load_state_dict(weights_path), strict=True)
    model.to(device).eval()
    return model, opt


def unwrap_model_output(output):
    """提取多阶段网络的最终增强 Tensor。"""
    if (isinstance(output, tuple) and len(output) == 2
            and isinstance(output[1], dict)):
        output = output[0]
    elif isinstance(output, (list, tuple)):
        output = output[-1]
    if not torch.is_tensor(output) or output.ndim != 4:
        shape = tuple(output.shape) if torch.is_tensor(output) else None
        raise TypeError(f"模型输出必须是 BCHW Tensor，当前为 {type(output)}，shape={shape}")
    return output


def infer_one(model, image_rgb, device, factor):
    """整图推理，仅做必要补边，并在保存前恢复原始尺寸。"""
    tensor = torch.from_numpy(
        image_rgb.astype(np.float32) / 255.0
    ).permute(2, 0, 1).unsqueeze(0).to(device)
    _, _, height, width = tensor.shape
    pad_height = (factor - height % factor) % factor
    pad_width = (factor - width % factor) % factor
    if pad_height or pad_width:
        # 反射补边要求 pad 小于对应维度；极小图像退回复制补边。
        mode = (
            "reflect"
            if pad_height < height and pad_width < width
            else "replicate"
        )
        tensor = F.pad(tensor, (0, pad_width, 0, pad_height), mode=mode)
    with torch.inference_mode():
        restored = unwrap_model_output(model(tensor))
    if restored.shape[0] != 1 or restored.shape[1] != 3:
        raise ValueError(f"模型输出应为 1x3xHxW，当前为 {tuple(restored.shape)}")
    restored = restored[:, :, :height, :width].clamp(0, 1)
    restored = restored[0].permute(1, 2, 0).cpu().numpy()
    return np.round(restored * 255.0).astype(np.uint8)


def create_no_reference_metrics(device):
    """创建统一的 PyIQA NIQE 与 BRISQUE 指标。"""
    try:
        import pyiqa
    except ImportError as exc:
        raise ImportError(
            "缺少 PyIQA，请先执行：python -m pip install pyiqa"
        ) from exc
    niqe = pyiqa.create_metric("niqe", device=str(device)).eval()
    brisque = pyiqa.create_metric("brisque", device=str(device)).eval()
    niqe.requires_grad_(False)
    brisque.requires_grad_(False)
    try:
        version = metadata.version("pyiqa")
    except metadata.PackageNotFoundError:
        version = "unknown"
    return niqe, brisque, version


def calculate_no_reference_metrics(image_rgb, niqe, brisque, device):
    """在保存后的 uint8 像素口径上计算单图 NIQE 与 BRISQUE。"""
    tensor = torch.from_numpy(
        image_rgb.astype(np.float32) / 255.0
    ).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.inference_mode():
        niqe_value = float(niqe(tensor).reshape(-1)[0].item())
        brisque_value = float(brisque(tensor).reshape(-1)[0].item())
    if not np.isfinite(niqe_value) or not np.isfinite(brisque_value):
        raise ValueError(
            f"无参考指标出现非有限值：NIQE={niqe_value}, BRISQUE={brisque_value}"
        )
    return niqe_value, brisque_value


def validate_existing_outputs(enhanced_root, entries, resume):
    """阻止旧结果混入当前运行，并允许受控覆盖中断结果。"""
    enhanced_root = Path(enhanced_root)
    if not enhanced_root.exists():
        return
    existing_paths = [
        path for path in enhanced_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not existing_paths:
        return
    existing = {
        path.relative_to(enhanced_root).with_suffix("").as_posix().casefold()
        for path in existing_paths
    }
    expected = {
        output_relative.with_suffix("").as_posix().casefold()
        for _, _, output_relative in entries
    }
    unexpected = sorted(existing - expected)
    if unexpected:
        raise ValueError(
            f"输出目录含有 {len(unexpected)} 个当前数据集之外的旧结果：{unexpected[:5]}"
        )
    if existing and not resume:
        raise FileExistsError(
            f"输出目录已有 {len(existing)} 张图像：{enhanced_root}。"
            "确认同一权重后可加 --resume 重新覆盖。"
        )


def write_manifest(dataset_name, dataset_root, entries, manifest_path):
    """冻结实际输入文件清单、尺寸和哈希，并返回清单哈希。"""
    rows = []
    for input_path, relative_path, _ in entries:
        with Image.open(input_path) as image:
            width, height = image.size
        rows.append({
            "dataset": dataset_name,
            "image": relative_path.as_posix(),
            "width": width,
            "height": height,
            "bytes": input_path.stat().st_size,
            "sha256": sha256_file(input_path),
        })
    write_csv(
        manifest_path,
        ["dataset", "image", "width", "height", "bytes", "sha256"],
        rows,
    )
    return sha256_file(manifest_path)


def resolve_device(device_name):
    """解析自动、CUDA 或 CPU 设备，并对不可用 CUDA 明确报错。"""
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("已请求 CUDA，但当前环境 torch.cuda.is_available() 为 False。")
    return device


def parse_args():
    """解析真实场景无参考评测参数。"""
    parser = argparse.ArgumentParser(
        description=f"{METHOD_NAME} 真实场景无参考测试"
    )
    parser.add_argument("--opt", required=True, type=Path, help="模型 YAML 配置。")
    parser.add_argument("--weights", required=True, type=Path, help="生成器权重。")
    parser.add_argument(
        "--data-root", required=True, type=Path,
        help="包含 DICM/LIME/MEF/NPE/VV 子目录的数据根目录。",
    )
    parser.add_argument(
        "--output-root", default=Path("real_world_results"), type=Path,
        help="全部方法结果的公共根目录。",
    )
    parser.add_argument(
        "--run-name", required=True,
        help="本次方法与权重标识，例如 CDRNet_LOLv1。",
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASET_NAMES,
        default=list(DATASET_NAMES), help="需要测试的数据集。",
    )
    parser.add_argument(
        "--device", default="auto", help="auto、cpu、cuda 或 cuda:0。"
    )
    parser.add_argument(
        "--factor", type=int, default=None,
        help=f"推理补边倍数；默认 {DEFAULT_FACTOR}。",
    )
    parser.add_argument(
        "--allow-count-mismatch", action="store_true",
        help="允许数据集数量偏离已知公开版本，但仍冻结实际清单。",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="受控覆盖同一输出目录中断产生的已有图像。",
    )
    return parser.parse_args()


def main():
    """依次完成权重加载、增强图保存、逐图指标和数据集均值。"""
    args = parse_args()
    if not args.opt.is_file():
        raise FileNotFoundError(f"配置文件不存在：{args.opt}")
    if not args.weights.is_file():
        raise FileNotFoundError(f"权重文件不存在：{args.weights}")
    if not args.data_root.is_dir():
        raise FileNotFoundError(f"数据根目录不存在：{args.data_root}")
    if args.factor is not None and args.factor <= 0:
        raise ValueError(f"--factor 必须为正整数，当前为 {args.factor}")

    device = resolve_device(args.device)
    model, opt = load_model(args.opt, args.weights, device)
    factor = args.factor
    if factor is None:
        factor = int(opt.get("network_g", {}).get("global_patch", DEFAULT_FACTOR))
    if factor <= 0:
        raise ValueError(f"最终推理补边倍数必须为正整数，当前为 {factor}")
    niqe, brisque, pyiqa_version = create_no_reference_metrics(device)

    run_root = (args.output_root / args.run_name).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    logger = configure_logger(run_root / "test.log")
    checkpoint_sha256 = sha256_file(args.weights)
    option_sha256 = sha256_file(args.opt)
    logger.info("命令：%s", shlex.join(sys.argv))
    logger.info(
        "方法=%s，设备=%s，补边倍数=%d，权重=%s",
        METHOD_NAME, device, factor, args.weights.resolve(),
    )

    summary_rows = []
    for dataset_name in args.datasets:
        dataset_root = (args.data_root / dataset_name).resolve()
        entries = build_image_entries(dataset_root)
        validate_dataset_count(
            dataset_name, len(entries), args.allow_count_mismatch
        )
        dataset_output = run_root / dataset_name
        enhanced_root = dataset_output / "enhanced"
        validate_existing_outputs(enhanced_root, entries, args.resume)
        enhanced_root.mkdir(parents=True, exist_ok=True)

        manifest_path = dataset_output / "input_manifest.csv"
        manifest_sha256 = write_manifest(
            dataset_name, dataset_root, entries, manifest_path
        )
        per_image_rows = []
        niqe_values = []
        brisque_values = []
        progress = tqdm(entries, desc=f"{METHOD_NAME}/{dataset_name}", unit="image")
        for input_path, relative_path, output_relative in progress:
            restored = infer_one(model, load_rgb(input_path), device, factor)
            output_path = enhanced_root / output_relative
            save_rgb(output_path, restored)
            niqe_value, brisque_value = calculate_no_reference_metrics(
                restored, niqe, brisque, device
            )
            niqe_values.append(niqe_value)
            brisque_values.append(brisque_value)
            per_image_rows.append({
                "dataset": dataset_name,
                "image": relative_path.as_posix(),
                "enhanced_image": output_relative.as_posix(),
                "niqe": f"{niqe_value:.6f}",
                "brisque": f"{brisque_value:.6f}",
            })

        # 推理完成后要求输出与冻结输入清单逐一对应，拒绝漏图或陈旧结果。
        output_entries = build_image_entries(enhanced_root)
        output_keys = {
            relative.with_suffix("").as_posix().casefold()
            for _, relative, _ in output_entries
        }
        expected_keys = {
            output_relative.with_suffix("").as_posix().casefold()
            for _, _, output_relative in entries
        }
        if output_keys != expected_keys:
            raise RuntimeError(
                f"{dataset_name} 输出覆盖不完整："
                f"missing={sorted(expected_keys - output_keys)[:5]}，"
                f"extra={sorted(output_keys - expected_keys)[:5]}"
            )

        niqe_mean = float(np.mean(niqe_values))
        brisque_mean = float(np.mean(brisque_values))
        write_csv(
            dataset_output / "per_image_metrics.csv",
            ["dataset", "image", "enhanced_image", "niqe", "brisque"],
            per_image_rows,
        )
        metric_row = {
            "method": METHOD_NAME,
            "run_name": args.run_name,
            "dataset": dataset_name,
            "num_images": len(entries),
            "niqe": f"{niqe_mean:.6f}",
            "brisque": f"{brisque_mean:.6f}",
            "aggregation": "arithmetic mean over images within this dataset",
            "direction": "NIQE lower; BRISQUE lower",
            "metric_library": "pyiqa",
            "metric_library_version": pyiqa_version,
            "metric_input": "saved uint8 RGB PNG mapped to [0,1]",
            "checkpoint": args.weights.resolve().as_posix(),
            "checkpoint_sha256": checkpoint_sha256,
            "option": args.opt.resolve().as_posix(),
            "option_sha256": option_sha256,
            "input_manifest": manifest_path.resolve().as_posix(),
            "input_manifest_sha256": manifest_sha256,
            "enhanced_images": enhanced_root.resolve().as_posix(),
        }
        write_csv(dataset_output / "metric.csv", list(metric_row), [metric_row])
        summary_rows.append(metric_row)
        logger.info(
            "[%s] images=%d, NIQE=%.6f, BRISQUE=%.6f, manifest=%s",
            dataset_name, len(entries), niqe_mean, brisque_mean,
            manifest_sha256,
        )

    write_csv(run_root / "summary.csv", list(summary_rows[0]), summary_rows)
    run_metadata = {
        "method": METHOD_NAME,
        "run_name": args.run_name,
        "datasets": list(args.datasets),
        "cross_dataset_mean": False,
        "command": shlex.join(sys.argv),
        "device": str(device),
        "factor": factor,
        "checkpoint": args.weights.resolve().as_posix(),
        "checkpoint_sha256": checkpoint_sha256,
        "option": args.opt.resolve().as_posix(),
        "option_sha256": option_sha256,
        "pyiqa_version": pyiqa_version,
    }
    with (run_root / "run.json").open("w", encoding="utf-8") as file:
        json.dump(run_metadata, file, ensure_ascii=False, indent=2)
    logger.info("逐数据集汇总：%s", (run_root / "summary.csv").resolve())
    logger.info("本脚本不计算跨数据集 Mean/AVG。")


if __name__ == "__main__":
    main()
