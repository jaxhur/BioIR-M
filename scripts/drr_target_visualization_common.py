"""DRR 先验目标可视化脚本的公共实现。

本模块负责 LOL-v1、LOL-v2-syn、LOL-v2-real 的严格成对读取、现有
``R*`` 公式、路由尺度聚合、固定值域图片保存和逐图/数据集统计。两个入口
脚本只分别定义旧版与新版 ``A*``，从而保证两版比较使用完全相同的数据与口径。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
STATISTIC_NAMES = (
    "mean",
    "spatial_std",
    "q05",
    "q25",
    "q50",
    "q75",
    "q95",
    "q95_minus_q05",
    "minimum",
    "maximum",
)


@dataclass(frozen=True)
class DatasetSpec:
    """描述一套 LOL 数据集的训练与测试配对目录。

    Args:
        name: 命令行和输出目录中使用的数据集名称。
        train_lq: 相对于数据根目录的训练低照度目录。
        train_gt: 相对于数据根目录的训练正常曝光目录。
        test_lq: 相对于数据根目录的测试低照度目录。
        test_gt: 相对于数据根目录的测试正常曝光目录。
    """

    name: str
    train_lq: Path
    train_gt: Path
    test_lq: Path
    test_gt: Path

    def directories(self, split: str) -> tuple[Path, Path]:
        """返回指定 split 的 LQ/GT 相对目录。

        Args:
            split: ``train`` 或 ``test``。

        Returns:
            ``(lq_relative, gt_relative)``。

        Raises:
            ValueError: split 不受支持时抛出。
        """

        if split == "train":
            return self.train_lq, self.train_gt
        if split == "test":
            return self.test_lq, self.test_gt
        raise ValueError(f"不支持的数据划分：{split}")


@dataclass(frozen=True)
class ImagePair:
    """保存按规范化相对路径严格匹配的一对 LQ/GT 图像。

    Args:
        relative_path: 图像相对于 LQ/GT 根目录的共同路径。
        lq_path: 低照度图像路径。
        gt_path: 正常曝光 GT 路径。
    """

    relative_path: Path
    lq_path: Path
    gt_path: Path


@dataclass(frozen=True)
class ReliabilityParameters:
    """保存原方案 ``R*`` 构造公式的固定参数。

    Args:
        structure_tau: GT 结构存在度的软阈值。
        magnitude_tau: LQ/GT 梯度幅值一致性的容忍尺度。
        log_epsilon: 对数亮度的稳定常数。
        reliability_epsilon: 方向余弦分母的稳定常数。
    """

    structure_tau: float = 0.10
    magnitude_tau: float = 0.05
    log_epsilon: float = 0.02
    reliability_epsilon: float = 1e-6


DATASET_SPECS = (
    DatasetSpec(
        name="LOL-v1",
        train_lq=Path("LOL-v1/our485/low"),
        train_gt=Path("LOL-v1/our485/high"),
        test_lq=Path("LOL-v1/eval15/low"),
        test_gt=Path("LOL-v1/eval15/high"),
    ),
    DatasetSpec(
        name="LOL-v2-syn",
        train_lq=Path("LOL-v2/Synthetic/Train/Low"),
        train_gt=Path("LOL-v2/Synthetic/Train/Normal"),
        test_lq=Path("LOL-v2/Synthetic/Test/Low"),
        test_gt=Path("LOL-v2/Synthetic/Test/Normal"),
    ),
    DatasetSpec(
        name="LOL-v2-real",
        train_lq=Path("LOL-v2/Real_captured/Train/Low"),
        train_gt=Path("LOL-v2/Real_captured/Train/Normal"),
        test_lq=Path("LOL-v2/Real_captured/Test/Low"),
        test_gt=Path("LOL-v2/Real_captured/Test/Normal"),
    ),
)
DATASET_BY_NAME = {spec.name: spec for spec in DATASET_SPECS}

# demand_builder 接收已计算好的 LQ/GT 亮度和数据集名称，返回 [0,1] 的 A*。
DemandBuilder = Callable[[torch.Tensor, torch.Tensor, str], torch.Tensor]


def add_common_arguments(
    parser: argparse.ArgumentParser,
    default_output_root: Path,
) -> None:
    """向两版入口脚本加入共享命令行参数。

    Args:
        parser: 待扩展的参数解析器。
        default_output_root: 当前公式版本的默认输出根目录。
    """

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("datasets"),
        help="LOL-v1 与 LOL-v2 所在的数据根目录。",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=default_output_root,
        help="目标图片、逐图 CSV 和汇总 CSV 的输出根目录。",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=[spec.name for spec in DATASET_SPECS] + ["all"],
        default=["all"],
        help="需要处理的数据集；all 表示三套数据集。",
    )
    parser.add_argument(
        "--split",
        choices=("train", "test"),
        default="train",
        help="统计并可视化训练集或完整测试集，默认 train。",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=16,
        help="模拟 ADRI 区域 gate 的平均池化窗口，默认 16。",
    )
    parser.add_argument(
        "--reliability-topk",
        type=int,
        default=16,
        help="每个 pool 窗口聚合 R* 时保留的最大响应数量，默认 16。",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="每套数据集最多参与统计的图片数；0 表示完整 split。",
    )
    parser.add_argument(
        "--max-visualizations",
        type=int,
        default=20,
        help="每套数据集最多保存的可视化数量；0 表示全部，-1 表示不保存。",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="计算设备，例如 cpu、cuda、cuda:0；auto 自动选择。",
    )
    parser.add_argument(
        "--structure-tau",
        type=float,
        default=0.10,
        help="R* 中 GT 结构存在度软阈值。",
    )
    parser.add_argument(
        "--magnitude-tau",
        type=float,
        default=0.05,
        help="R* 中梯度幅值一致性容忍尺度。",
    )
    parser.add_argument(
        "--log-epsilon",
        type=float,
        default=0.02,
        help="R* 对数亮度稳定常数。",
    )
    parser.add_argument(
        "--reliability-epsilon",
        type=float,
        default=1e-6,
        help="R* 方向余弦分母稳定常数。",
    )


def validate_common_arguments(args: argparse.Namespace) -> None:
    """验证两版脚本共享参数。

    Args:
        args: 已解析的命令行参数。

    Raises:
        ValueError: 参数不满足公式或聚合约束时抛出。
    """

    if args.pool_size <= 0:
        raise ValueError("--pool-size 必须为正整数。")
    window_area = args.pool_size * args.pool_size
    if not 1 <= args.reliability_topk <= window_area:
        raise ValueError("--reliability-topk 必须位于 [1, pool_size^2]。")
    if args.max_images < 0:
        raise ValueError("--max-images 不能为负数。")
    if args.max_visualizations < -1:
        raise ValueError("--max-visualizations 只能为 -1、0 或正整数。")
    for name in ("structure_tau", "magnitude_tau", "log_epsilon",
                 "reliability_epsilon"):
        if getattr(args, name) <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} 必须大于 0。")


def selected_specs(dataset_names: Sequence[str]) -> tuple[DatasetSpec, ...]:
    """按预定义顺序返回用户选择的数据集。

    Args:
        dataset_names: 命令行中选择的数据集名称。

    Returns:
        去重且保持固定顺序的数据集定义。
    """

    selected = (set(DATASET_BY_NAME) if "all" in dataset_names else
                set(dataset_names))
    return tuple(spec for spec in DATASET_SPECS if spec.name in selected)


def resolve_device(device_name: str) -> torch.device:
    """解析计算设备并在请求 CUDA 但不可用时明确报错。

    Args:
        device_name: ``auto`` 或 PyTorch 设备字符串。

    Returns:
        可用于张量计算的设备。
    """

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求了 CUDA，但当前 PyTorch 环境检测不到可用 GPU。")
    return device


def discover_images(directory: Path) -> dict[str, tuple[Path, Path]]:
    """递归发现图像并建立大小写无关的相对路径索引。

    Args:
        directory: LQ 或 GT 根目录。

    Returns:
        键为规范化相对路径，值为原相对路径和真实路径。

    Raises:
        FileNotFoundError: 输入目录不存在时抛出。
        ValueError: 没有图像或发现重复相对路径时抛出。
    """

    if not directory.is_dir():
        raise FileNotFoundError(f"图像目录不存在：{directory}")
    index: dict[str, tuple[Path, Path]] = {}
    for image_path in sorted(directory.rglob("*"),
                             key=lambda path: path.as_posix().casefold()):
        if (not image_path.is_file() or
                image_path.suffix.casefold() not in IMAGE_EXTENSIONS):
            continue
        relative_path = image_path.relative_to(directory)
        key = relative_path.as_posix().casefold()
        if key in index:
            raise ValueError(
                f"发现大小写不敏感的重复相对路径：{index[key][1]} 与 {image_path}")
        index[key] = (relative_path, image_path)
    if not index:
        raise ValueError(f"目录中没有支持的图像：{directory}")
    return index


def build_pairs(
    data_root: Path,
    spec: DatasetSpec,
    split: str,
    max_images: int = 0,
) -> list[ImagePair]:
    """按规范化相对路径严格构造指定 split 的 LQ/GT 图像对。

    Args:
        data_root: LOL-v1 与 LOL-v2 的共同数据根目录。
        spec: 当前数据集定义。
        split: ``train`` 或 ``test``。
        max_images: 最多返回多少对；0 表示全部。

    Returns:
        按相对路径排序的图像对。

    Raises:
        ValueError: LQ/GT 缺失配对时抛出。
    """

    lq_relative, gt_relative = spec.directories(split)
    lq_root = data_root / lq_relative
    gt_root = data_root / gt_relative
    lq_index = discover_images(lq_root)
    gt_index = discover_images(gt_root)
    lq_keys = set(lq_index)
    gt_keys = set(gt_index)
    missing_gt = sorted(lq_keys - gt_keys)
    missing_lq = sorted(gt_keys - lq_keys)
    if missing_gt or missing_lq:
        details = []
        if missing_gt:
            details.append(f"缺少 GT：{missing_gt[:5]}")
        if missing_lq:
            details.append(f"缺少 LQ：{missing_lq[:5]}")
        raise ValueError(f"{spec.name}/{split} 配对失败；" + "；".join(details))

    pairs = [
        ImagePair(
            relative_path=lq_index[key][0],
            lq_path=lq_index[key][1],
            gt_path=gt_index[key][1],
        ) for key in sorted(lq_keys)
    ]
    return pairs[:max_images] if max_images > 0 else pairs


def load_rgb_tensor(path: Path, device: torch.device) -> torch.Tensor:
    """读取 RGB 图像并转换为 ``1×3×H×W`` 的 ``[0,1]`` Tensor。

    Args:
        path: 输入图像路径。
        device: 目标计算设备。

    Returns:
        ``float32`` RGB Tensor。
    """

    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 255.0
    return (torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device))


def calculate_luminance(image: torch.Tensor) -> torch.Tensor:
    """按方案中的固定 RGB 权重计算单通道亮度。

    Args:
        image: ``B×3×H×W``、值域 ``[0,1]`` 的 RGB Tensor。

    Returns:
        ``B×1×H×W`` 亮度 Tensor。
    """

    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("RGB 输入必须为 B×3×H×W。")
    weights = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    return (image * weights).sum(dim=1, keepdim=True)


def safe_reflect_pad(
    tensor: torch.Tensor,
    padding: tuple[int, int, int, int],
) -> torch.Tensor:
    """优先反射补边，极小 Tensor 自动退化为复制补边。

    Args:
        tensor: 待补边 Tensor。
        padding: PyTorch 顺序的 ``(left,right,top,bottom)``。

    Returns:
        补边后的 Tensor。
    """

    left, right, top, bottom = padding
    height, width = tensor.shape[-2:]
    if left < width and right < width and top < height and bottom < height:
        return F.pad(tensor, padding, mode="reflect")
    return F.pad(tensor, padding, mode="replicate")


def log_sobel_features(
    luminance: torch.Tensor,
    log_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """严格复现方案中的 3×3 高斯、log 与 Sobel 结构特征。

    Args:
        luminance: ``B×1×H×W`` 亮度图。
        log_epsilon: 对数变换的稳定常数。

    Returns:
        水平梯度、垂直梯度和梯度二范数幅值。
    """

    gaussian = luminance.new_tensor(
        [[0.07511361, 0.1238414, 0.07511361],
         [0.1238414, 0.20417996, 0.1238414],
         [0.07511361, 0.1238414, 0.07511361]],
    ).view(1, 1, 3, 3)
    sobel_x = luminance.new_tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0],
         [-1.0, 0.0, 1.0]],
    ).view(1, 1, 3, 3) / 8.0
    sobel_y = luminance.new_tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0],
         [1.0, 2.0, 1.0]],
    ).view(1, 1, 3, 3) / 8.0

    smoothed = F.conv2d(safe_reflect_pad(luminance, (1, 1, 1, 1)), gaussian)
    log_luminance = torch.log(smoothed.clamp_min(0.0) + log_epsilon)
    padded_log = safe_reflect_pad(log_luminance, (1, 1, 1, 1))
    gradient_x = F.conv2d(padded_log, sobel_x)
    gradient_y = F.conv2d(padded_log, sobel_y)
    magnitude = torch.sqrt(gradient_x.square() + gradient_y.square())
    return gradient_x, gradient_y, magnitude


def build_reliability_target(
    luminance_lq: torch.Tensor,
    luminance_gt: torch.Tensor,
    parameters: ReliabilityParameters,
) -> tuple[torch.Tensor, torch.Tensor]:
    """按原方案构造结构可靠性 ``R*`` 与 GT 结构存在度 ``E_g``。

    Args:
        luminance_lq: LQ 亮度，形状 ``B×1×H×W``。
        luminance_gt: GT 亮度，形状同 LQ。
        parameters: ``R*`` 公式参数。

    Returns:
        ``(reliability, edge_presence)``，二者范围均为 ``[0,1]``。
    """

    grad_lq_x, grad_lq_y, magnitude_lq = log_sobel_features(
        luminance_lq, parameters.log_epsilon)
    grad_gt_x, grad_gt_y, magnitude_gt = log_sobel_features(
        luminance_gt, parameters.log_epsilon)
    edge_presence = 1.0 - torch.exp(
        -magnitude_gt / parameters.structure_tau)
    magnitude_consistency = torch.exp(
        -(magnitude_lq - magnitude_gt).abs() /
        (magnitude_gt + parameters.magnitude_tau))
    direction_dot = grad_lq_x * grad_gt_x + grad_lq_y * grad_gt_y
    direction_consistency = direction_dot / (
        magnitude_lq * magnitude_gt + parameters.reliability_epsilon)
    direction_consistency = direction_consistency.clamp(0.0, 1.0)
    reliability = edge_presence * magnitude_consistency * direction_consistency
    return reliability.clamp(0.0, 1.0), edge_presence.clamp(0.0, 1.0)


def gaussian_kernel_1d(
    reference: torch.Tensor,
    kernel_size: int,
    sigma: float,
) -> torch.Tensor:
    """生成归一化的一维离散 Gaussian 核。

    Args:
        reference: 用于继承 dtype 和 device 的 Tensor。
        kernel_size: 奇数核宽度。
        sigma: Gaussian 标准差，单位为像素。

    Returns:
        长度为 ``kernel_size``、权重和为 1 的一维核。
    """

    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("Gaussian kernel_size 必须为正奇数。")
    if sigma <= 0.0:
        raise ValueError("Gaussian sigma 必须大于 0。")
    radius = kernel_size // 2
    positions = torch.arange(
        -radius, radius + 1, device=reference.device, dtype=reference.dtype)
    kernel = torch.exp(-positions.square() / (2.0 * sigma * sigma))
    return kernel / kernel.sum()


def gaussian_low_pass(
    luminance: torch.Tensor,
    kernel_size: int,
    sigma: float,
) -> torch.Tensor:
    """使用固定、可分离的 Gaussian 低通平滑亮度图。

    Args:
        luminance: ``B×1×H×W`` 亮度图。
        kernel_size: Gaussian 核边长。
        sigma: Gaussian 标准差，单位为像素。

    Returns:
        与输入形状一致的低频亮度图。
    """

    if luminance.ndim != 4 or luminance.shape[1] != 1:
        raise ValueError("低通输入必须为 B×1×H×W。")
    kernel = gaussian_kernel_1d(luminance, kernel_size, sigma)
    radius = kernel_size // 2
    horizontal = kernel.view(1, 1, 1, kernel_size)
    vertical = kernel.view(1, 1, kernel_size, 1)
    smoothed = F.conv2d(
        safe_reflect_pad(luminance, (radius, radius, 0, 0)), horizontal)
    return F.conv2d(
        safe_reflect_pad(smoothed, (0, 0, radius, radius)), vertical)


def average_pool_map(target: torch.Tensor, pool_size: int) -> torch.Tensor:
    """裁去不能形成完整窗口的右/下边缘后执行非重叠平均池化。

    该裁剪口径用于复现此前对 LOL 原始整图的统计。训练使用的 256×256 patch
    可被 16 整除，因此不会发生裁剪。

    Args:
        target: ``B×1×H×W`` 目标图。
        pool_size: 非重叠池化窗口和步长。

    Returns:
        ``B×1×floor(H/pool)×floor(W/pool)`` 区域平均图。
    """

    height, width = target.shape[-2:]
    usable_height = height - height % pool_size
    usable_width = width - width % pool_size
    if usable_height == 0 or usable_width == 0:
        raise ValueError("目标图尺寸必须不小于 pool_size。")
    return F.avg_pool2d(
        target[..., :usable_height, :usable_width],
        kernel_size=pool_size,
        stride=pool_size,
    )


def topk_pool_map(
    target: torch.Tensor,
    pool_size: int,
    topk: int,
) -> torch.Tensor:
    """按原 ADRI 逻辑计算每个区域内最大 ``topk`` 响应的均值。

    Args:
        target: ``B×1×H×W`` 结构可靠性图。
        pool_size: 非重叠区域边长。
        topk: 每个区域保留的最大响应数量。

    Returns:
        与平均池化相同区域网格上的 Top-k 均值图。
    """

    height, width = target.shape[-2:]
    usable_height = height - height % pool_size
    usable_width = width - width % pool_size
    if usable_height == 0 or usable_width == 0:
        raise ValueError("目标图尺寸必须不小于 pool_size。")
    cropped = target[..., :usable_height, :usable_width]
    patches = F.unfold(cropped, kernel_size=pool_size, stride=pool_size)
    pooled = patches.topk(topk, dim=1).values.mean(dim=1, keepdim=True)
    grid_height = usable_height // pool_size
    grid_width = usable_width // pool_size
    return pooled.view(target.shape[0], 1, grid_height, grid_width)


def tensor_to_numpy_map(target: torch.Tensor) -> np.ndarray:
    """将单张单通道目标转换为二维 ``float32`` 数组。

    Args:
        target: ``1×1×H×W`` 或 ``1×H×W`` Tensor。

    Returns:
        二维数组。
    """

    array = target.detach().float().cpu().numpy()
    return np.squeeze(array).astype(np.float32, copy=False)


def map_statistics(values: np.ndarray) -> dict[str, float]:
    """计算单张目标图的空间分布统计。

    Args:
        values: 二维目标图。

    Returns:
        均值、总体标准差、分位数、鲁棒跨度和极值。
    """

    flattened = np.asarray(values, dtype=np.float64).reshape(-1)
    q05, q25, q50, q75, q95 = np.quantile(
        flattened, [0.05, 0.25, 0.50, 0.75, 0.95])
    return {
        "mean": float(flattened.mean()),
        "spatial_std": float(flattened.std(ddof=0)),
        "q05": float(q05),
        "q25": float(q25),
        "q50": float(q50),
        "q75": float(q75),
        "q95": float(q95),
        "q95_minus_q05": float(q95 - q05),
        "minimum": float(flattened.min()),
        "maximum": float(flattened.max()),
    }


def attach_statistics(
    record: dict[str, str | int | float],
    prefix: str,
    values: np.ndarray,
) -> None:
    """把单张目标图统计追加到逐图记录。

    Args:
        record: 待更新的 CSV 行。
        prefix: 字段前缀，例如 ``a_star``。
        values: 二维目标图。
    """

    for name, value in map_statistics(values).items():
        record[f"{prefix}_{name}"] = value


def summarize_records(
    records: Sequence[Mapping[str, str | int | float]],
    dataset: str,
    split: str,
    formula_name: str,
    target_prefixes: Sequence[tuple[str, str]],
) -> list[dict[str, str | int | float]]:
    """先逐图统计，再对各图统计量取中位数和算术平均。

    Args:
        records: 当前数据集全部逐图记录。
        dataset: 数据集名称。
        split: 数据划分。
        formula_name: A* 公式版本名称。
        target_prefixes: ``(显示名, CSV 前缀)`` 列表。

    Returns:
        每种目标图一行的数据集汇总结果。
    """

    summaries: list[dict[str, str | int | float]] = []
    for display_name, prefix in target_prefixes:
        summary: dict[str, str | int | float] = {
            "formula": formula_name,
            "dataset": dataset,
            "split": split,
            "target": display_name,
            "num_images": len(records),
        }
        for statistic_name in STATISTIC_NAMES:
            values = np.asarray(
                [float(record[f"{prefix}_{statistic_name}"])
                 for record in records],
                dtype=np.float64,
            )
            summary[f"median_of_image_{statistic_name}"] = float(
                np.median(values))
            summary[f"mean_of_image_{statistic_name}"] = float(values.mean())
        summaries.append(summary)
    return summaries


def write_csv(
    rows: Sequence[Mapping[str, str | int | float]],
    output_path: Path,
) -> None:
    """以 UTF-8 BOM 编码写出表格，便于 Excel 直接打开中文字段。

    Args:
        rows: 具有相同字段的 CSV 行。
        output_path: 输出文件路径。

    Raises:
        ValueError: 没有可写记录时抛出。
    """

    if not rows:
        raise ValueError(f"没有可写入的 CSV 记录：{output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def rgb_tensor_to_image(image: torch.Tensor) -> Image.Image:
    """把 ``[0,1]`` RGB Tensor 转换为 PIL RGB 图像。

    Args:
        image: ``1×3×H×W`` 或 ``3×H×W`` Tensor。

    Returns:
        PIL RGB 图像。
    """

    array = image.detach().float().cpu().clamp(0.0, 1.0).numpy()
    array = np.squeeze(array)
    array = np.moveaxis(array, 0, -1)
    return Image.fromarray(np.rint(array * 255.0).astype(np.uint8), mode="RGB")


def map_to_image(values: np.ndarray) -> Image.Image:
    """按固定 ``[0,1]`` 值域保存目标图，不做逐图 min-max 拉伸。

    Args:
        values: 二维目标图。

    Returns:
        PIL 灰度图像。
    """

    clipped = np.clip(values, 0.0, 1.0)
    return Image.fromarray(
        np.rint(clipped * 255.0).astype(np.uint8), mode="L")


def expand_region_map(
    region_values: np.ndarray,
    pool_size: int,
    output_height: int,
    output_width: int,
) -> np.ndarray:
    """把区域 gate 用最近邻展开到原图尺寸，便于观察实际路由块。

    Args:
        region_values: ``H_grid×W_grid`` 区域图。
        pool_size: 每个区域对应的输入像素边长。
        output_height: 原图高度。
        output_width: 原图宽度。

    Returns:
        与原图同尺寸的二维图；不足的右/下边缘复制最后一个区域值。
    """

    expanded = np.repeat(
        np.repeat(region_values, pool_size, axis=0), pool_size, axis=1)
    pad_bottom = max(output_height - expanded.shape[0], 0)
    pad_right = max(output_width - expanded.shape[1], 0)
    if pad_bottom or pad_right:
        expanded = np.pad(
            expanded,
            ((0, pad_bottom), (0, pad_right)),
            mode="edge",
        )
    return expanded[:output_height, :output_width]


def load_panel_font(size: int = 16) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    """加载常见字体，失败时回退到 Pillow 默认字体。

    Args:
        size: 字号。

    Returns:
        可用于面板标题的字体。
    """

    for font_path in (
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ):
        if font_path.is_file():
            return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def make_panel_tile(
    image: Image.Image,
    title: str,
    width: int,
    height: int,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
) -> Image.Image:
    """将一幅图缩放居中并添加不影响图像内容的标题栏。

    Args:
        image: RGB 或灰度 PIL 图像。
        title: 面板标题。
        width: 图像区域宽度。
        height: 图像区域高度。
        font: 标题字体。

    Returns:
        RGB 面板单元。
    """

    title_height = 32
    source = image.convert("RGB").copy()
    source.thumbnail((width, height), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (width, height + title_height), color=(20, 22, 26))
    x_offset = (width - source.width) // 2
    y_offset = title_height + (height - source.height) // 2
    tile.paste(source, (x_offset, y_offset))
    ImageDraw.Draw(tile).text((8, 7), title, font=font, fill=(240, 242, 246))
    return tile


def save_visualizations(
    output_directory: Path,
    lq: torch.Tensor,
    gt: torch.Tensor,
    demand: np.ndarray,
    demand_regions: np.ndarray,
    reliability: np.ndarray,
    reliability_regions: np.ndarray,
    formula_name: str,
    pool_size: int,
) -> None:
    """保存单图目标、实际区域 gate 展开图和五联对照面板。

    Args:
        output_directory: 当前图像对的独立输出目录。
        lq: 低照度 RGB Tensor。
        gt: 正常曝光 RGB Tensor。
        demand: 完整分辨率 A*。
        demand_regions: 平均池化后的 A* 区域网格。
        reliability: 完整分辨率 R*。
        reliability_regions: Top-k 聚合后的 R* 区域网格。
        formula_name: 面板中的 A* 版本标识。
        pool_size: 区域边长。
    """

    output_directory.mkdir(parents=True, exist_ok=True)
    lq_image = rgb_tensor_to_image(lq)
    gt_image = rgb_tensor_to_image(gt)
    demand_image = map_to_image(demand)
    reliability_image = map_to_image(reliability)
    height, width = demand.shape
    demand_region_image = map_to_image(
        expand_region_map(demand_regions, pool_size, height, width))
    reliability_region_image = map_to_image(
        expand_region_map(reliability_regions, pool_size, height, width))
    formula_label = ("original" if formula_name.startswith("original") else
                     "revised" if formula_name.startswith("revised") else
                     formula_name[:16])

    lq_image.save(output_directory / "input_lq.png")
    gt_image.save(output_directory / "ground_truth.png")
    demand_image.save(output_directory / "A_star.png")
    demand_region_image.save(output_directory / f"A_star_avgpool{pool_size}.png")
    reliability_image.save(output_directory / "R_star.png")
    reliability_region_image.save(
        output_directory / f"R_star_topk{pool_size}.png")

    font = load_panel_font()
    tile_width, tile_height = 320, 240
    tiles = [
        make_panel_tile(lq_image, "LQ input", tile_width, tile_height, font),
        make_panel_tile(gt_image, "GT", tile_width, tile_height, font),
        make_panel_tile(
            demand_image,
            f"A* {formula_label}  mean={demand.mean():.3f}",
            tile_width,
            tile_height,
            font,
        ),
        make_panel_tile(
            demand_region_image,
            f"A* AvgPool{pool_size}  std={demand_regions.std():.3f}",
            tile_width,
            tile_height,
            font,
        ),
        make_panel_tile(
            reliability_image,
            f"R*  mean={reliability.mean():.3f}",
            tile_width,
            tile_height,
            font,
        ),
        make_panel_tile(
            reliability_region_image,
            f"R* Top-k  std={reliability_regions.std():.3f}",
            tile_width,
            tile_height,
            font,
        ),
    ]
    panel = Image.new(
        "RGB",
        (tile_width * 3, (tile_height + 32) * 2),
        color=(10, 11, 14),
    )
    for index, tile in enumerate(tiles):
        panel.paste(
            tile,
            ((index % 3) * tile_width, (index // 3) * (tile_height + 32)),
        )
    panel.save(output_directory / "comparison_panel.png")


def print_key_summary(summary_rows: Sequence[Mapping[str, str | int | float]]) -> None:
    """在终端打印用户关注的三项中位统计。

    Args:
        summary_rows: 当前数据集的汇总行。
    """

    print("  target                 median(mean)  median(std)  median(q95-q05)")
    for row in summary_rows:
        print(
            f"  {str(row['target']):<22}"
            f"{float(row['median_of_image_mean']):>13.6f}"
            f"{float(row['median_of_image_spatial_std']):>13.6f}"
            f"{float(row['median_of_image_q95_minus_q05']):>18.6f}")


def analyze_dataset(
    data_root: Path,
    output_root: Path,
    spec: DatasetSpec,
    split: str,
    formula_name: str,
    demand_builder: DemandBuilder,
    formula_parameters: Mapping[str, str | int | float],
    reliability_parameters: ReliabilityParameters,
    pool_size: int,
    reliability_topk: int,
    max_images: int,
    max_visualizations: int,
    device: torch.device,
) -> list[dict[str, str | int | float]]:
    """处理一套数据集并保存图片、逐图统计和汇总统计。

    Args:
        data_root: LOL 数据根目录。
        output_root: 当前公式版本的输出根目录。
        spec: 当前数据集定义。
        split: ``train`` 或 ``test``。
        formula_name: A* 公式版本名称。
        demand_builder: 由 LQ/GT 亮度构造 A* 的函数。
        formula_parameters: 写入 metadata 的 A* 参数。
        reliability_parameters: R* 参数。
        pool_size: A 平均池化与 R Top-k 区域边长。
        reliability_topk: 每个 R 区域保留的最大响应数。
        max_images: 最多参与统计的图像数量；0 表示全部。
        max_visualizations: 最多保存的图片数量；0 全部，-1 不保存。
        device: PyTorch 计算设备。

    Returns:
        当前数据集的汇总统计行。
    """

    pairs = build_pairs(data_root, spec, split, max_images=max_images)
    dataset_output = output_root / split / spec.name
    visual_count = (len(pairs) if max_visualizations == 0 else
                    max(max_visualizations, 0))
    records: list[dict[str, str | int | float]] = []
    print(
        f"[{formula_name}][{spec.name}/{split}] 配对验证通过，共 {len(pairs)} 张；"
        f"device={device}。")

    with torch.inference_mode():
        for index, pair in enumerate(pairs, start=1):
            lq = load_rgb_tensor(pair.lq_path, device)
            gt = load_rgb_tensor(pair.gt_path, device)
            if lq.shape != gt.shape:
                raise ValueError(
                    f"LQ/GT 尺寸不一致：{pair.lq_path} {tuple(lq.shape)}；"
                    f"{pair.gt_path} {tuple(gt.shape)}")
            luminance_lq = calculate_luminance(lq)
            luminance_gt = calculate_luminance(gt)
            demand = demand_builder(luminance_lq, luminance_gt, spec.name)
            reliability, edge_presence = build_reliability_target(
                luminance_lq, luminance_gt, reliability_parameters)
            demand_regions = average_pool_map(demand, pool_size)
            reliability_regions = topk_pool_map(
                reliability, pool_size, reliability_topk)

            demand_np = tensor_to_numpy_map(demand)
            reliability_np = tensor_to_numpy_map(reliability)
            edge_presence_np = tensor_to_numpy_map(edge_presence)
            demand_regions_np = tensor_to_numpy_map(demand_regions)
            reliability_regions_np = tensor_to_numpy_map(reliability_regions)
            height, width = demand_np.shape
            record: dict[str, str | int | float] = {
                "formula": formula_name,
                "dataset": spec.name,
                "split": split,
                "relative_path": pair.relative_path.as_posix(),
                "lq_path": pair.lq_path.resolve().as_posix(),
                "gt_path": pair.gt_path.resolve().as_posix(),
                "width": width,
                "height": height,
                "pool_size": pool_size,
                "pooled_width": demand_regions_np.shape[1],
                "pooled_height": demand_regions_np.shape[0],
                "reliability_topk": reliability_topk,
            }
            attach_statistics(record, "a_star", demand_np)
            attach_statistics(record, "a_star_pooled", demand_regions_np)
            attach_statistics(record, "r_star", reliability_np)
            attach_statistics(record, "r_star_topk", reliability_regions_np)
            attach_statistics(record, "edge_presence", edge_presence_np)
            records.append(record)

            if index <= visual_count:
                relative_stem = pair.relative_path.with_suffix("")
                image_output = (dataset_output / "images" /
                                relative_stem.parent / relative_stem.name)
                save_visualizations(
                    output_directory=image_output,
                    lq=lq,
                    gt=gt,
                    demand=demand_np,
                    demand_regions=demand_regions_np,
                    reliability=reliability_np,
                    reliability_regions=reliability_regions_np,
                    formula_name=formula_name,
                    pool_size=pool_size,
                )
            if index == len(pairs) or index % 25 == 0:
                print(f"[{formula_name}][{spec.name}/{split}] {index}/{len(pairs)}")

    summary_rows = summarize_records(
        records=records,
        dataset=spec.name,
        split=split,
        formula_name=formula_name,
        target_prefixes=(
            ("A_star", "a_star"),
            (f"A_star_AvgPool{pool_size}", "a_star_pooled"),
            ("R_star", "r_star"),
            (f"R_star_TopK{reliability_topk}", "r_star_topk"),
        ),
    )
    write_csv(records, dataset_output / "per_image_statistics.csv")
    write_csv(summary_rows, dataset_output / "summary_statistics.csv")
    lq_relative, gt_relative = spec.directories(split)
    metadata = {
        "formula": formula_name,
        "dataset": spec.name,
        "split": split,
        "num_images": len(records),
        "data_root": data_root.resolve().as_posix(),
        "lq_relative": lq_relative.as_posix(),
        "gt_relative": gt_relative.as_posix(),
        "pool_size": pool_size,
        "pool_boundary": "crop incomplete right/bottom windows",
        "reliability_topk": reliability_topk,
        "formula_parameters": dict(formula_parameters),
        "reliability_parameters": asdict(reliability_parameters),
        "fixed_visualization_range": [0.0, 1.0],
    }
    with (dataset_output / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    print_key_summary(summary_rows)
    return summary_rows


def run_selected_datasets(
    args: argparse.Namespace,
    formula_name: str,
    demand_builders: Mapping[str, DemandBuilder],
    formula_parameters: Mapping[str, Mapping[str, str | int | float]],
) -> None:
    """按命令行选择运行多个数据集并写出总汇总表。

    Args:
        args: 两版脚本的完整命令行参数。
        formula_name: A* 公式版本名称。
        demand_builders: 数据集名称到 A* 构造函数的映射。
        formula_parameters: 数据集名称到公式参数的映射。
    """

    validate_common_arguments(args)
    device = resolve_device(args.device)
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    reliability_parameters = ReliabilityParameters(
        structure_tau=args.structure_tau,
        magnitude_tau=args.magnitude_tau,
        log_epsilon=args.log_epsilon,
        reliability_epsilon=args.reliability_epsilon,
    )
    all_summaries: list[dict[str, str | int | float]] = []
    for spec in selected_specs(args.datasets):
        all_summaries.extend(
            analyze_dataset(
                data_root=data_root,
                output_root=output_root,
                spec=spec,
                split=args.split,
                formula_name=formula_name,
                demand_builder=demand_builders[spec.name],
                formula_parameters=formula_parameters[spec.name],
                reliability_parameters=reliability_parameters,
                pool_size=args.pool_size,
                reliability_topk=args.reliability_topk,
                max_images=args.max_images,
                max_visualizations=args.max_visualizations,
                device=device,
            ))
    write_csv(all_summaries,
              output_root / args.split / "all_datasets_summary.csv")
    print(f"完成：结果位于 {output_root.as_posix()}")


def estimate_positive_quantile(
    pairs: Sequence[ImagePair],
    device: torch.device,
    kernel_size: int,
    sigma: float,
    quantile: float,
    histogram_bins: int,
) -> tuple[float, int]:
    """以流式直方图估计训练集正亮度缺口的固定分位数。

    逐图累计直方图避免一次保存数亿像素。估计误差不超过一个 bin 宽度，默认
    65536 bins 时约为 ``1.53e-5``。

    Args:
        pairs: 用于统计的训练 LQ/GT 图像对。
        device: PyTorch 计算设备。
        kernel_size: 新公式 Gaussian 低通核边长。
        sigma: 新公式 Gaussian 标准差。
        quantile: 需要估计的正缺口分位数。
        histogram_bins: ``[0,1]`` 区间上的直方图 bin 数量。

    Returns:
        ``(tau_a, positive_pixel_count)``。

    Raises:
        ValueError: 没有任何正亮度缺口时抛出。
    """

    histogram = np.zeros(histogram_bins, dtype=np.int64)
    positive_pixel_count = 0
    with torch.inference_mode():
        for index, pair in enumerate(pairs, start=1):
            lq = load_rgb_tensor(pair.lq_path, device)
            gt = load_rgb_tensor(pair.gt_path, device)
            luminance_lq = gaussian_low_pass(
                calculate_luminance(lq), kernel_size, sigma)
            luminance_gt = gaussian_low_pass(
                calculate_luminance(gt), kernel_size, sigma)
            deficit = (luminance_gt - luminance_lq).clamp_min(0.0)
            positive = deficit[deficit > 0.0]
            if positive.numel() > 0:
                counts = torch.histc(
                    positive.float(),
                    bins=histogram_bins,
                    min=0.0,
                    max=1.0,
                ).to(dtype=torch.int64, device="cpu").numpy()
                histogram += counts
                positive_pixel_count += int(positive.numel())
            if index == len(pairs) or index % 50 == 0:
                print(f"[tau_A] {index}/{len(pairs)}")

    if positive_pixel_count == 0:
        raise ValueError("训练集中没有正亮度缺口，无法估计 tau_A。")
    target_rank = max(1, math.ceil(quantile * positive_pixel_count))
    bin_index = int(np.searchsorted(np.cumsum(histogram), target_rank, side="left"))
    # 使用命中 bin 的上边界，保证归一化尺度不会系统性偏小。
    tau_a = min((bin_index + 1) / histogram_bins, 1.0)
    return float(tau_a), positive_pixel_count
