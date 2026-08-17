"""可视化 LOL 测试集中的亮度、频率与结构先验。

该脚本只执行确定性的图像变换，不加载模型或权重，也不需要训练。默认处理
LOL-v1、LOL-v2 Synthetic 和 LOL-v2 Real 的全部测试图像，并在处理前严格
检查低照度图像与正常曝光图像能否按相对路径一一配对。
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
GAUSSIAN_KERNEL = np.asarray(
    [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
    dtype=np.float32,
) / 16.0
SOBEL_X = np.asarray(
    [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
    dtype=np.float32,
)
SOBEL_Y = np.asarray(
    [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
    dtype=np.float32,
)


@dataclass(frozen=True)
class DatasetSpec:
    """描述一套 LOL 测试集的低照度目录与配对 GT 目录。

    Args:
        name: 输出目录和清单中使用的数据集名称。
        low_relative: 相对于数据根目录的低照度图像目录。
        normal_relative: 相对于数据根目录的正常曝光图像目录。
    """

    name: str
    low_relative: Path
    normal_relative: Path


@dataclass(frozen=True)
class ImagePair:
    """保存按相对路径严格匹配的一对测试图像。

    Args:
        relative_path: 图像相对于 low/normal 目录的共同路径。
        low_path: 低照度输入图像路径。
        normal_path: 正常曝光 GT 图像路径，仅用于配对验证。
    """

    relative_path: Path
    low_path: Path
    normal_path: Path


DATASET_SPECS = (
    DatasetSpec(
        name="LOL-v1",
        low_relative=Path("LOL-v1/eval15/low"),
        normal_relative=Path("LOL-v1/eval15/high"),
    ),
    DatasetSpec(
        name="LOL-v2-syn",
        low_relative=Path("LOL-v2/Synthetic/Test/Low"),
        normal_relative=Path("LOL-v2/Synthetic/Test/Normal"),
    ),
    DatasetSpec(
        name="LOL-v2-real",
        low_relative=Path("LOL-v2/Real_captured/Test/Low"),
        normal_relative=Path("LOL-v2/Real_captured/Test/Normal"),
    ),
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        包含数据路径、频域参数和可视化尺寸的参数对象。
    """

    parser = argparse.ArgumentParser(
        description="无需训练，批量可视化 LOL 测试集的 Y/色度/频率/结构先验。"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("datasets"),
        help="LOL-v1 与 LOL-v2 所在的数据根目录。",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("analysis_artifacts/prior_visualization"),
        help="面板、总览图和 CSV 清单的输出根目录。",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=[spec.name for spec in DATASET_SPECS] + ["all"],
        default=["all"],
        help="需要处理的数据集，默认处理全部三套测试集。",
    )
    parser.add_argument(
        "--sigma-f",
        type=float,
        default=0.1,
        help="归一化频率坐标中的高斯低通标准差。",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=256,
        help="九宫格中每个可视化分量的最大边长。",
    )
    parser.add_argument(
        "--overview-count",
        type=int,
        default=3,
        help="每套数据集总览图中均匀抽取的面板数量。",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="每套数据集最多处理多少张；0 表示处理完整测试集。",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """在开始批量处理前验证参数。

    Args:
        args: 命令行参数。

    Raises:
        ValueError: 参数超出合理范围时抛出。
    """

    if args.sigma_f <= 0.0:
        raise ValueError("--sigma-f 必须大于 0。")
    if args.tile_size < 64:
        raise ValueError("--tile-size 不能小于 64。")
    if args.overview_count < 0:
        raise ValueError("--overview-count 不能为负数。")
    if args.max_images < 0:
        raise ValueError("--max-images 不能为负数。")


def selected_specs(dataset_names: Sequence[str]) -> tuple[DatasetSpec, ...]:
    """根据命令行选择返回需要处理的数据集定义。

    Args:
        dataset_names: 用户选择的数据集名称，包含 ``all`` 时返回全部。

    Returns:
        去重且保持预定义顺序的数据集定义元组。
    """

    selected = {spec.name for spec in DATASET_SPECS} if "all" in dataset_names else set(dataset_names)
    return tuple(spec for spec in DATASET_SPECS if spec.name in selected)


def discover_images(directory: Path) -> dict[str, tuple[Path, Path]]:
    """递归发现图像并建立大小写无关的相对路径索引。

    Args:
        directory: 需要扫描的 low 或 normal 目录。

    Returns:
        键为规范化相对路径，值为原始相对路径与绝对路径的映射。

    Raises:
        FileNotFoundError: 目录不存在时抛出。
        ValueError: 出现仅大小写不同的重复相对路径或目录内没有图像时抛出。
    """

    if not directory.is_dir():
        raise FileNotFoundError(f"图像目录不存在：{directory}")

    index: dict[str, tuple[Path, Path]] = {}
    for image_path in sorted(directory.rglob("*"), key=lambda path: path.as_posix().casefold()):
        if not image_path.is_file() or image_path.suffix.casefold() not in IMAGE_EXTENSIONS:
            continue
        relative_path = image_path.relative_to(directory)
        key = relative_path.as_posix().casefold()
        if key in index:
            raise ValueError(
                f"发现大小写不敏感的重复相对路径：{index[key][1]} 与 {image_path}"
            )
        index[key] = (relative_path, image_path)

    if not index:
        raise ValueError(f"目录中没有支持的图像：{directory}")
    return index


def build_pairs(data_root: Path, spec: DatasetSpec) -> list[ImagePair]:
    """按规范化相对路径严格构造测试图像对。

    Args:
        data_root: 数据集根目录。
        spec: 当前数据集目录定义。

    Returns:
        按相对路径排序的测试图像对。

    Raises:
        ValueError: low 与 normal 目录存在缺失配对时抛出。
    """

    low_index = discover_images(data_root / spec.low_relative)
    normal_index = discover_images(data_root / spec.normal_relative)
    low_keys = set(low_index)
    normal_keys = set(normal_index)
    missing_normal = sorted(low_keys - normal_keys)
    missing_low = sorted(normal_keys - low_keys)
    if missing_normal or missing_low:
        examples = []
        if missing_normal:
            examples.append(f"缺少 normal：{missing_normal[:5]}")
        if missing_low:
            examples.append(f"缺少 low：{missing_low[:5]}")
        raise ValueError(f"{spec.name} 配对失败；" + "；".join(examples))

    return [
        ImagePair(
            relative_path=low_index[key][0],
            low_path=low_index[key][1],
            normal_path=normal_index[key][1],
        )
        for key in sorted(low_keys)
    ]


def load_rgb(path: Path) -> np.ndarray:
    """读取图像并转换为 ``[0, 1]`` 的 RGB 浮点数组。

    Args:
        path: 输入图像路径。

    Returns:
        形状为 ``H × W × 3`` 的 ``float32`` 数组。
    """

    with Image.open(path) as image:
        rgb = image.convert("RGB")
        return np.asarray(rgb, dtype=np.float32) / 255.0


def rgb_to_ycbcr(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按照 BT.601/JPEG 形式把归一化 RGB 转换为 YCbCr。

    Args:
        rgb: 形状为 ``H × W × 3``、范围为 ``[0, 1]`` 的 RGB 数组。

    Returns:
        ``Y``、``Cb``、``Cr`` 三个二维数组；Cb/Cr 的中性值为 0.5。
    """

    red, green, blue = np.moveaxis(rgb, -1, 0)
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    cb = -0.168736 * red - 0.331264 * green + 0.5 * blue + 0.5
    cr = 0.5 * red - 0.418688 * green - 0.081312 * blue + 0.5
    return luminance.astype(np.float32), cb.astype(np.float32), cr.astype(np.float32)


def fft_gaussian_split(luminance: np.ndarray, sigma_f: float) -> tuple[np.ndarray, np.ndarray]:
    """使用中心化高斯低通掩码分解亮度低频与有符号高频残差。

    Args:
        luminance: 归一化亮度图。
        sigma_f: 归一化频率半径上的高斯标准差。

    Returns:
        ``Y_lf`` 与 ``Y_hf = Y - Y_lf``。
    """

    height, width = luminance.shape
    vertical = (np.arange(height, dtype=np.float32) - height // 2) / float(height)
    horizontal = (np.arange(width, dtype=np.float32) - width // 2) / float(width)
    radius_squared = vertical[:, None] ** 2 + horizontal[None, :] ** 2
    low_pass = np.exp(-radius_squared / (2.0 * sigma_f**2)).astype(np.float32)

    centered_spectrum = np.fft.fftshift(np.fft.fft2(luminance))
    low_frequency = np.fft.ifft2(np.fft.ifftshift(centered_spectrum * low_pass)).real
    low_frequency = low_frequency.astype(np.float32)
    high_frequency = (luminance - low_frequency).astype(np.float32)
    return low_frequency, high_frequency


def convolve_reflect(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """使用反射填充对二维单通道图像执行小核卷积。

    Args:
        image: 二维输入图像。
        kernel: 高、宽均为奇数的二维卷积核。

    Returns:
        与输入尺寸一致的卷积结果。
    """

    kernel_height, kernel_width = kernel.shape
    if kernel_height % 2 == 0 or kernel_width % 2 == 0:
        raise ValueError("卷积核高宽必须为奇数。")

    pad_y = kernel_height // 2
    pad_x = kernel_width // 2
    padded = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)), mode="reflect")
    result = np.zeros_like(image, dtype=np.float32)
    height, width = image.shape
    for row in range(kernel_height):
        for column in range(kernel_width):
            result += kernel[row, column] * padded[row : row + height, column : column + width]
    return result


def sobel_magnitude(luminance: np.ndarray) -> np.ndarray:
    """计算亮度图的二维 Sobel 梯度幅值。

    Args:
        luminance: 原始或高斯平滑后的二维亮度图。

    Returns:
        非负的 Sobel 结构响应图。
    """

    gradient_x = convolve_reflect(luminance, SOBEL_X)
    gradient_y = convolve_reflect(luminance, SOBEL_Y)
    return np.sqrt(gradient_x**2 + gradient_y**2 + 1e-12).astype(np.float32)


def robust_limit(values: Iterable[np.ndarray], percentile: float = 99.5) -> float:
    """计算多幅图共享的鲁棒可视化上限。

    Args:
        values: 需要共享尺度的二维数组。
        percentile: 用于避开极少数异常峰值的百分位数。

    Returns:
        严格为正的共享尺度。
    """

    limits = [float(np.percentile(np.abs(value), percentile)) for value in values]
    return max(max(limits, default=0.0), 1e-6)


def grayscale(values: np.ndarray, lower: float = 0.0, upper: float = 1.0) -> np.ndarray:
    """把标量图按固定范围转换为三通道灰度图。

    Args:
        values: 二维标量图。
        lower: 显示下限。
        upper: 显示上限。

    Returns:
        形状为 ``H × W × 3`` 的 uint8 图像。
    """

    normalized = np.clip((values - lower) / max(upper - lower, 1e-12), 0.0, 1.0)
    gray = np.rint(normalized * 255.0).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=-1)


def divergent_map(values: np.ndarray, limit: float) -> np.ndarray:
    """把有符号响应映射为蓝—白—红发散色图。

    Args:
        values: 有正有负的二维响应图。
        limit: 正负两侧共享的绝对值上限。

    Returns:
        RGB uint8 可视化图像。
    """

    normalized = np.clip(values / max(limit, 1e-12), -1.0, 1.0)
    positive = np.clip(normalized, 0.0, 1.0)[..., None]
    negative = np.clip(-normalized, 0.0, 1.0)[..., None]
    white = np.ones((*normalized.shape, 3), dtype=np.float32)
    red = np.asarray([0.75, 0.04, 0.08], dtype=np.float32)
    blue = np.asarray([0.04, 0.20, 0.72], dtype=np.float32)
    colors = white * (1.0 - positive - negative) + red * positive + blue * negative
    return np.rint(np.clip(colors, 0.0, 1.0) * 255.0).astype(np.uint8)


def heat_map(values: np.ndarray, limit: float) -> np.ndarray:
    """把非负响应映射为深蓝—青—黄热力图。

    Args:
        values: 非负二维响应图。
        limit: 鲁棒显示上限。

    Returns:
        RGB uint8 可视化图像。
    """

    normalized = np.clip(values / max(limit, 1e-12), 0.0, 1.0)
    anchors = np.asarray(
        [
            [0.03, 0.02, 0.18],
            [0.08, 0.22, 0.48],
            [0.00, 0.58, 0.66],
            [0.93, 0.88, 0.18],
        ],
        dtype=np.float32,
    )
    scaled = normalized * (len(anchors) - 1)
    lower = np.floor(scaled).astype(np.int32)
    upper = np.minimum(lower + 1, len(anchors) - 1)
    weight = (scaled - lower)[..., None]
    colors = anchors[lower] * (1.0 - weight) + anchors[upper] * weight
    return np.rint(colors * 255.0).astype(np.uint8)


def load_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    """优先加载 Windows Arial，失败时回退到 Pillow 默认字体。

    Args:
        size: 字号。

    Returns:
        可用于面板标题绘制的字体对象。
    """

    for font_path in (Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/calibri.ttf")):
        if font_path.is_file():
            return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def make_tile(image: np.ndarray, title: str, tile_size: int, font: ImageFont.ImageFont) -> Image.Image:
    """将一个分量缩放、居中并添加标题栏。

    Args:
        image: RGB uint8 分量图。
        title: 英文标题及必要的显示范围说明。
        tile_size: 图像区域的正方形边长。
        font: 标题字体。

    Returns:
        带标题栏的 Pillow 图像。
    """

    title_height = 34
    source = Image.fromarray(image, mode="RGB")
    source.thumbnail((tile_size, tile_size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (tile_size, tile_size + title_height), color=(20, 22, 26))
    x_offset = (tile_size - source.width) // 2
    y_offset = title_height + (tile_size - source.height) // 2
    canvas.paste(source, (x_offset, y_offset))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title, font=font, fill=(238, 240, 244))
    return canvas


def compose_panel(
    rgb: np.ndarray,
    luminance: np.ndarray,
    cb: np.ndarray,
    cr: np.ndarray,
    low_frequency: np.ndarray,
    high_frequency: np.ndarray,
    edge_raw: np.ndarray,
    edge_smooth: np.ndarray,
    tile_size: int,
) -> Image.Image:
    """把输入和八种先验组织成共享尺度的三乘三面板。

    Args:
        rgb: 原始低照度 RGB 图像。
        luminance: Y 亮度图。
        cb: 带 0.5 中心偏置的 Cb 图。
        cr: 带 0.5 中心偏置的 Cr 图。
        low_frequency: FFT 高斯低通得到的 Y_lf。
        high_frequency: 有符号亮度高频残差 Y_hf。
        edge_raw: 直接在 Y 上计算的 Sobel 结构图。
        edge_smooth: 高斯平滑后计算的 Sobel 结构图。
        tile_size: 单个分量的显示尺寸。

    Returns:
        三乘三 RGB 面板。
    """

    high_limit = robust_limit((high_frequency,))
    chroma_limit = robust_limit((cb - 0.5, cr - 0.5))
    edge_limit = robust_limit((edge_raw, edge_smooth))
    rgb_uint8 = np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    font = load_font(max(13, tile_size // 18))
    tiles = (
        make_tile(rgb_uint8, "Low-light RGB", tile_size, font),
        make_tile(grayscale(luminance), "Y  [0, 1]", tile_size, font),
        make_tile(divergent_map(cb - 0.5, chroma_limit), "Cb - 0.5  (signed)", tile_size, font),
        make_tile(divergent_map(cr - 0.5, chroma_limit), "Cr - 0.5  (signed)", tile_size, font),
        make_tile(grayscale(low_frequency), "Y_lf  FFT Gaussian", tile_size, font),
        make_tile(divergent_map(high_frequency, high_limit), "Y_hf  (signed)", tile_size, font),
        make_tile(heat_map(np.abs(high_frequency), high_limit), "|Y_hf|", tile_size, font),
        make_tile(heat_map(edge_raw, edge_limit), "E  Sobel(Y)", tile_size, font),
        make_tile(heat_map(edge_smooth, edge_limit), "E  Sobel(Gauss(Y))", tile_size, font),
    )

    tile_height = tiles[0].height
    panel = Image.new("RGB", (tile_size * 3, tile_height * 3), color=(12, 13, 16))
    for index, tile in enumerate(tiles):
        panel.paste(tile, ((index % 3) * tile_size, (index // 3) * tile_height))
    return panel


def create_overview(panel_paths: Sequence[Path], output_path: Path, count: int) -> None:
    """从完整结果中均匀抽取若干面板并纵向拼接为总览图。

    Args:
        panel_paths: 当前数据集生成的全部面板路径。
        output_path: 总览图输出路径。
        count: 均匀抽取的面板数量；0 表示不生成。
    """

    if count == 0 or not panel_paths:
        return
    sample_count = min(count, len(panel_paths))
    indices = np.linspace(0, len(panel_paths) - 1, num=sample_count, dtype=np.int32)
    panels: list[Image.Image] = []
    for index in indices:
        with Image.open(panel_paths[int(index)]) as panel:
            panels.append(panel.convert("RGB").copy())

    gap = 16
    width = max(panel.width for panel in panels)
    height = sum(panel.height for panel in panels) + gap * (len(panels) - 1)
    overview = Image.new("RGB", (width, height), color=(8, 9, 12))
    y_offset = 0
    for panel in panels:
        overview.paste(panel, ((width - panel.width) // 2, y_offset))
        y_offset += panel.height + gap
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overview.save(output_path, format="PNG", compress_level=6)


def process_dataset(
    data_root: Path,
    output_root: Path,
    spec: DatasetSpec,
    sigma_f: float,
    tile_size: int,
    overview_count: int,
    max_images: int,
) -> list[dict[str, str | int | float]]:
    """处理一套完整测试集并返回可写入 CSV 的统计记录。

    Args:
        data_root: 数据根目录。
        output_root: 结果输出根目录。
        spec: 当前数据集定义。
        sigma_f: FFT 高斯低通标准差。
        tile_size: 面板单元尺寸。
        overview_count: 总览图抽样数量。
        max_images: 最大处理数量，0 表示全部。

    Returns:
        当前数据集每张输入图像的路径、尺寸和先验统计记录。
    """

    pairs = build_pairs(data_root, spec)
    if max_images > 0:
        pairs = pairs[:max_images]
    dataset_output = output_root / spec.name
    panels_output = dataset_output / "panels"
    panel_paths: list[Path] = []
    records: list[dict[str, str | int | float]] = []

    print(f"[{spec.name}] 已验证配对，开始处理 {len(pairs)} 张低照度测试图。")
    for index, pair in enumerate(pairs, start=1):
        rgb = load_rgb(pair.low_path)
        luminance, cb, cr = rgb_to_ycbcr(rgb)
        low_frequency, high_frequency = fft_gaussian_split(luminance, sigma_f)
        edge_raw = sobel_magnitude(luminance)
        smoothed_luminance = convolve_reflect(luminance, GAUSSIAN_KERNEL)
        edge_smooth = sobel_magnitude(smoothed_luminance)
        panel = compose_panel(
            rgb=rgb,
            luminance=luminance,
            cb=cb,
            cr=cr,
            low_frequency=low_frequency,
            high_frequency=high_frequency,
            edge_raw=edge_raw,
            edge_smooth=edge_smooth,
            tile_size=tile_size,
        )

        relative_panel = pair.relative_path.with_suffix("")
        panel_path = panels_output / relative_panel.parent / f"{relative_panel.name}_priors.png"
        panel_path.parent.mkdir(parents=True, exist_ok=True)
        panel.save(panel_path, format="PNG", compress_level=6)
        panel_paths.append(panel_path)
        height, width = luminance.shape
        records.append(
            {
                "dataset": spec.name,
                "relative_path": pair.relative_path.as_posix(),
                "low_path": str(pair.low_path.resolve()),
                "normal_path": str(pair.normal_path.resolve()),
                "panel_path": str(panel_path.resolve()),
                "width": width,
                "height": height,
                "sigma_f": sigma_f,
                "y_mean": float(luminance.mean()),
                "y_lf_mean": float(low_frequency.mean()),
                "y_hf_abs_mean": float(np.abs(high_frequency).mean()),
                "cb_offset_abs_mean": float(np.abs(cb - 0.5).mean()),
                "cr_offset_abs_mean": float(np.abs(cr - 0.5).mean()),
                "edge_raw_mean": float(edge_raw.mean()),
                "edge_smooth_mean": float(edge_smooth.mean()),
            }
        )
        if index == len(pairs) or index % 25 == 0:
            print(f"[{spec.name}] {index}/{len(pairs)}")

    create_overview(panel_paths, dataset_output / "overview.png", overview_count)
    return records


def write_manifest(records: Sequence[dict[str, str | int | float]], output_path: Path) -> None:
    """写出包含路径、尺寸和先验均值的 UTF-8 CSV 清单。

    Args:
        records: 所有已处理图像的统计记录。
        output_path: CSV 输出路径。
    """

    if not records:
        raise ValueError("没有可写入清单的记录。")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    """验证配对、生成全部面板与每套数据集的代表性总览。"""

    args = parse_args()
    validate_args(args)
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    records: list[dict[str, str | int | float]] = []
    for spec in selected_specs(args.datasets):
        records.extend(
            process_dataset(
                data_root=data_root,
                output_root=output_root,
                spec=spec,
                sigma_f=args.sigma_f,
                tile_size=args.tile_size,
                overview_count=args.overview_count,
                max_images=args.max_images,
            )
        )
    write_manifest(records, output_root / "manifest.csv")
    print(f"完成：共生成 {len(records)} 个面板，结果位于 {output_root}")


if __name__ == "__main__":
    main()
