"""使用已训练的 BEAR-BioIR 权重导出测试集路由行为可视化。

示例命令：``python visualize_bear_routing.py --opt options/BEAR-LOLv1.yml
--weights experiments/BEAR-BioIR-v2-LOLv1/models/best_G.pth``。

未指定 ``--image`` 或 ``--input-dir`` 时，脚本读取 YAML 中完整的
``datasets.val.dataroot_lq``。每张图只执行一次前向推理，不修改权重，也不
需要重新训练。所有单通道图均使用固定 ``[0,1]`` 标尺保存，避免逐图归一化
夸大微弱响应。
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from basicsr.utils.options import parse
from test_lol import (list_images, load_model, load_rgb, save_grayscale,
                      save_rgb)


PROJECT_ROOT = Path(__file__).resolve().parent


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """返回可能被 DataParallel 包装的实际生成网络。"""
    return model.module if hasattr(model, 'module') else model


def _validate_visualization_model(model: torch.nn.Module) -> torch.nn.Module:
    """确认当前配置能产生可解释的预测路由量。

    Args:
        model: 已加载训练权重的生成网络。

    Returns:
        去除并行包装后的 BEAR-BioIR 网络。

    Raises:
        ValueError: 当前 YAML 使用常量路由或固定结构先验。
    """
    bare_model = _unwrap_model(model)
    if getattr(bare_model, 'routing_mode', None) != 'bepr':
        raise ValueError(
            'Routing visualization requires network_g.routing_mode=bepr; '
            'constant routing only produces fixed A/R maps.')
    if (getattr(bare_model, 'structure_source', None) != 'predicted'
            or getattr(bare_model, 'structure_predictor', None) is None):
        raise ValueError(
            'S_hat visualization requires '
            'network_g.structure_source=predicted.')
    for name in ('pad_multiple', 'route_patch'):
        if int(getattr(bare_model, name, 0)) <= 0:
            raise ValueError(f'BEAR-BioIR is missing a valid {name}.')
    return bare_model


def _check_unit_map(name: str, value: torch.Tensor,
                    expected_batch: int = 1) -> torch.Tensor:
    """校验单通道概率图并将极小浮点越界裁回 ``[0,1]``。"""
    if (not torch.is_tensor(value) or value.ndim != 4
            or value.shape[:2] != (expected_batch, 1)):
        shape = tuple(value.shape) if torch.is_tensor(value) else None
        raise ValueError(
            f'{name} must have shape {expected_batch}x1xHxW, got {shape}.')
    if not torch.isfinite(value).all():
        raise ValueError(f'{name} contains NaN or Inf values.')
    minimum = float(value.min().item())
    maximum = float(value.max().item())
    if minimum < -1e-6 or maximum > 1.0 + 1e-6:
        raise ValueError(
            f'{name} must stay in [0,1], got [{minimum}, {maximum}].')
    return value.detach().float().clamp(0.0, 1.0)


def _route_grid_to_dense(name: str, grid: torch.Tensor,
                         padded_size: Tuple[int, int],
                         original_size: Tuple[int, int]) -> np.ndarray:
    """将区域路由网格按最近邻展开并裁回低光输入尺寸。

    最近邻展开保留每个 ``route_patch`` 区域的原始常数值，不会生成模型
    实际未预测的平滑过渡。
    """
    grid = _check_unit_map(name, grid)
    dense = F.interpolate(grid, size=padded_size, mode='nearest')
    height, width = original_size
    return dense[0, 0, :height, :width].cpu().numpy().astype(np.float32)


def infer_routing_behavior(model: torch.nn.Module, image_rgb: np.ndarray,
                           device: torch.device) -> Dict[str, object]:
    """在一次前向中提取增强结果、三尺度 ``A/R`` 和 ``S_hat``。

    Args:
        model: 已严格加载目标 checkpoint 且处于 eval 模式的生成网络。
        image_rgb: 值域为 ``[0,255]`` 的 ``H×W×3`` uint8 RGB 低光图。
        device: 模型当前使用的计算设备。

    Returns:
        包含增强图、原始路由网格、输入对齐稠密图、预测结构图和预算的字典。
    """
    if (image_rgb.ndim != 3 or image_rgb.shape[2] != 3
            or image_rgb.dtype != np.uint8):
        raise ValueError(
            'Input image must be an HxWx3 uint8 RGB array, '
            f'got shape={image_rgb.shape}, dtype={image_rgb.dtype}.')
    bare_model = _validate_visualization_model(model)
    height, width = image_rgb.shape[:2]
    pad_multiple = int(bare_model.pad_multiple)
    route_patch = int(bare_model.route_patch)
    padded_height = math.ceil(height / pad_multiple) * pad_multiple
    padded_width = math.ceil(width / pad_multiple) * pad_multiple
    expected_grid_size = (padded_height // route_patch,
                          padded_width // route_patch)

    image = image_rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.inference_mode():
        output = model(tensor, return_aux=True)
    if not (isinstance(output, tuple) and len(output) == 2):
        raise TypeError(
            'BEAR-BioIR must return (restored, aux) when return_aux=True.')
    restored, auxiliary = output
    if not isinstance(auxiliary, dict):
        raise TypeError('BEAR-BioIR auxiliary output must be a dictionary.')
    if isinstance(restored, list):
        restored = restored[-1]
    if (not torch.is_tensor(restored) or restored.ndim != 4
            or restored.shape[:2] != (1, 3)):
        shape = tuple(restored.shape) if torch.is_tensor(restored) else None
        raise ValueError(f'Restored image must have shape 1x3xHxW, got {shape}.')
    restored = restored[:, :, :height, :width].clamp(0.0, 1.0)
    restored_image = torch.round(restored[0].permute(1, 2, 0) * 255.0)
    restored_image = restored_image.to(torch.uint8).cpu().numpy()

    scopes = auxiliary.get('scopes')
    reliabilities = auxiliary.get('reliabilities')
    if (not isinstance(scopes, (list, tuple)) or len(scopes) != 3
            or not isinstance(reliabilities, (list, tuple))
            or len(reliabilities) != 3):
        raise ValueError(
            'Auxiliary output must contain three scopes and three reliabilities.')

    route_grids = {}
    dense_maps = {}
    for prefix, values in (('A', scopes), ('R', reliabilities)):
        for index, value in enumerate(values, start=1):
            name = f'{prefix}_{index}'
            value = _check_unit_map(name, value)
            if tuple(value.shape[-2:]) != expected_grid_size:
                raise ValueError(
                    f'{name} grid must be {expected_grid_size}, '
                    f'got {tuple(value.shape[-2:])}.')
            route_grids[name] = value[0, 0].cpu().numpy().astype(np.float32)
            dense_maps[name] = _route_grid_to_dense(
                name, value, (padded_height, padded_width), (height, width))

    structure = _check_unit_map('S_hat', auxiliary.get('structure'))
    if (structure.shape[-2] < height or structure.shape[-1] < width):
        raise ValueError(
            'S_hat is smaller than the original input: '
            f'{tuple(structure.shape[-2:])} vs {(height, width)}.')
    structure_map = structure[0, 0, :height, :width]
    structure_map = structure_map.cpu().numpy().astype(np.float32)

    budget = auxiliary.get('budget')
    if (not torch.is_tensor(budget) or budget.numel() != 1
            or not torch.isfinite(budget).all()):
        raise ValueError('Auxiliary budget must contain one finite value.')
    budget_value = float(budget.detach().reshape(-1)[0].cpu().item())
    if not 0.0 <= budget_value <= 1.0:
        raise ValueError(f'Budget must stay in [0,1], got {budget_value}.')

    return {
        'restored': restored_image,
        'route_grids': route_grids,
        'dense_maps': dense_maps,
        'structure': structure_map,
        'budget': budget_value,
        'original_size': (height, width),
        'padded_size': (padded_height, padded_width),
        'route_patch': route_patch,
    }


def unit_map_to_grayscale(value: np.ndarray) -> np.ndarray:
    """按固定 ``[0,1]`` 标尺把浮点图转换为 uint8 灰度图。"""
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError('Unit map must be a finite two-dimensional array.')
    if float(value.min()) < -1e-6 or float(value.max()) > 1.0 + 1e-6:
        raise ValueError('Unit map values must stay in [0,1].')
    return np.round(np.clip(value, 0.0, 1.0) * 255.0).astype(np.uint8)


def unit_map_to_heatmap(value: np.ndarray) -> np.ndarray:
    """使用固定色标把 ``[0,1]`` 单通道图转换为 RGB 热力图。"""
    grayscale = unit_map_to_grayscale(value)
    colormap = getattr(cv2, 'COLORMAP_TURBO', cv2.COLORMAP_JET)
    heatmap_bgr = cv2.applyColorMap(grayscale, colormap)
    return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)


def _labeled_tile(image: np.ndarray, label: str,
                  label_height: int = 32) -> np.ndarray:
    """为总览图中的单个 RGB 面板添加黑底标题。"""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Overview tile must be an HxWx3 RGB image.')
    header = np.zeros((label_height, image.shape[1], 3), dtype=np.uint8)
    cv2.putText(header, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([header, image], axis=0)


def build_overview(low: np.ndarray, result: Dict[str, object]) -> np.ndarray:
    """生成“输入/结构/输出 + 三尺度 A + 三尺度 R”的 3×3 总览图。"""
    dense_maps = result['dense_maps']
    structure = result['structure']
    restored = result['restored']
    rows = [
        [
            _labeled_tile(low, 'Low-light input'),
            _labeled_tile(
                unit_map_to_heatmap(structure),
                f'S_hat [0,1] mean={float(structure.mean()):.3f}'),
            _labeled_tile(restored, 'Enhanced result'),
        ],
        [
            _labeled_tile(
                unit_map_to_heatmap(dense_maps[name]),
                f'{name} scope mean={float(dense_maps[name].mean()):.3f}')
            for name in ('A_1', 'A_2', 'A_3')
        ],
        [
            _labeled_tile(
                unit_map_to_heatmap(dense_maps[name]),
                f'{name} reliability mean={float(dense_maps[name].mean()):.3f}')
            for name in ('R_1', 'R_2', 'R_3')
        ],
    ]
    return np.concatenate(
        [np.concatenate(row, axis=1) for row in rows], axis=0)


def _statistics_rows(
        maps: Iterable[Tuple[str, np.ndarray]]) -> List[Dict[str, str]]:
    """计算固定标尺图的基础分布统计，便于发现饱和或塌缩。"""
    rows = []
    for name, value in maps:
        rows.append({
            'map': name,
            'min': f'{float(value.min()):.6f}',
            'max': f'{float(value.max()):.6f}',
            'mean': f'{float(value.mean()):.6f}',
            'std': f'{float(value.std()):.6f}',
        })
    return rows


def save_routing_visualization(output_directory: Path, low: np.ndarray,
                               result: Dict[str, object], image_path: Path,
                               option_path: Path,
                               weights_path: Path) -> None:
    """保存论文观察所需单图、总览图、原始数组和元数据。

    Args:
        output_directory: 当前样本的独立输出目录。
        low: 原始低光 RGB 图像。
        result: :func:`infer_routing_behavior` 返回的推理结果。
        image_path: 输入图路径，用于记录可复现实验元数据。
        option_path: 网络 YAML 路径。
        weights_path: 已训练生成器权重路径。
    """
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    heatmap_directory = output_directory / 'heatmaps'
    heatmap_directory.mkdir(parents=True, exist_ok=True)

    save_rgb(output_directory / 'input_low.png', low)
    save_rgb(output_directory / 'enhanced.png', result['restored'])
    all_maps = dict(result['dense_maps'])
    all_maps['S_hat'] = result['structure']
    for name, value in all_maps.items():
        save_grayscale(
            output_directory / f'{name}.png', unit_map_to_grayscale(value))
        save_rgb(
            heatmap_directory / f'{name}_heatmap.png',
            unit_map_to_heatmap(value))
    save_rgb(
        output_directory / 'routing_overview.png',
        build_overview(low, result))

    arrays = {
        **{name: value for name, value in all_maps.items()},
        **{
            f'{name}_grid': value
            for name, value in result['route_grids'].items()
        },
    }
    np.savez_compressed(output_directory / 'routing_arrays.npz', **arrays)

    statistics = _statistics_rows(all_maps.items())
    with (output_directory / 'routing_statistics.csv').open(
            'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(
            file, fieldnames=['map', 'min', 'max', 'mean', 'std'])
        writer.writeheader()
        writer.writerows(statistics)

    metadata = {
        'image': str(Path(image_path).expanduser().resolve()),
        'option': str(Path(option_path).expanduser().resolve()),
        'weights': str(Path(weights_path).expanduser().resolve()),
        'budget': result['budget'],
        'original_size_hw': list(result['original_size']),
        'padded_size_hw': list(result['padded_size']),
        'route_patch': result['route_patch'],
        'map_range': [0.0, 1.0],
        'route_resize': 'nearest',
        'per_image_normalization': False,
    }
    with (output_directory / 'metadata.json').open(
            'w', encoding='utf-8') as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)


def resolve_input_images(
        opt: Dict[str, object], image_path: str = None,
        input_directory: str = None) -> Tuple[Path, List[Tuple[Path, Path]]]:
    """解析单图覆盖或 YAML 中的完整验证输入目录。

    Args:
        opt: BasicSR 解析后的完整 YAML 配置。
        image_path: 可选单图路径，用于只重跑指定样本。
        input_directory: 可选输入目录，用于覆盖 YAML 验证集路径。

    Returns:
        输入根目录，以及 ``(绝对或工作目录路径, 相对路径)`` 列表。

    Raises:
        ValueError: 同时指定单图与目录，或 YAML 缺少验证集 LQ 路径。
        FileNotFoundError: 显式单图不存在。
        RuntimeError: 测试目录中没有支持的图片。
    """
    if image_path is not None and input_directory is not None:
        raise ValueError('--image and --input-dir cannot be used together.')
    if image_path is not None:
        image = Path(image_path).expanduser()
        if not image.is_file():
            raise FileNotFoundError(f'Input image does not exist: {image}')
        return image.parent, [(image, Path(image.name))]

    if input_directory is None:
        try:
            input_directory = opt['datasets']['val']['dataroot_lq']
        except (KeyError, TypeError) as exc:
            raise ValueError(
                'YAML must define datasets.val.dataroot_lq when neither '
                '--image nor --input-dir is provided.') from exc
    input_root = Path(input_directory).expanduser()
    images = list_images(input_root)
    if not images:
        raise RuntimeError(f'No test images found in: {input_root}')
    return input_root, [
        (image, image.relative_to(input_root)) for image in images
    ]


def resolve_output_root(output_directory: str, experiment_name: str) -> Path:
    """返回显式输出根目录或当前实验的默认路由分析目录。"""
    if output_directory is not None:
        return Path(output_directory).expanduser().resolve()
    return (PROJECT_ROOT / 'analysis_artifacts' / 'routing_behavior'
            / experiment_name)


def _dataset_statistics_row(relative_path: Path,
                            sample_output: Path,
                            result: Dict[str, object]) -> Dict[str, object]:
    """汇总单张图的预算及各路由图均值和标准差。"""
    row = {
        'image': relative_path.as_posix(),
        'budget': f'{float(result["budget"]):.6f}',
        'output_directory': str(sample_output.resolve()),
    }
    maps = dict(result['dense_maps'])
    maps['S_hat'] = result['structure']
    for name, value in maps.items():
        row[f'{name}_mean'] = f'{float(value.mean()):.6f}'
        row[f'{name}_std'] = f'{float(value.std()):.6f}'
    return row


def visualize_input_set(
        model: torch.nn.Module, image_entries: List[Tuple[Path, Path]],
        output_root: Path, device: torch.device, option_path: Path,
        weights_path: Path) -> List[Dict[str, object]]:
    """逐图导出完整测试集路由图并写入数据集级统计表。

    Args:
        model: 已加载目标 checkpoint 的 BEAR-BioIR 生成网络。
        image_entries: ``(输入路径, 相对路径)`` 列表。
        output_root: 当前实验的路由可视化输出根目录。
        device: 模型推理设备。
        option_path: 当前网络 YAML 路径。
        weights_path: 当前生成器权重路径。

    Returns:
        每张图片对应的一行路由分布统计。
    """
    if not image_entries:
        raise ValueError('image_entries must contain at least one image.')
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    total = len(image_entries)
    for index, (image_path, relative_path) in enumerate(
            image_entries, start=1):
        low = load_rgb(image_path)
        result = infer_routing_behavior(model, low, device)
        # 去除图片扩展名后作为目录，保留测试集内的相对层级与同名文件隔离。
        sample_output = output_root / relative_path.with_suffix('')
        save_routing_visualization(
            sample_output, low, result, image_path, option_path, weights_path)
        rows.append(
            _dataset_statistics_row(relative_path, sample_output, result))
        print(
            f'[{index}/{total}] {relative_path.as_posix()} -> '
            f'{sample_output.resolve()}')

    summary_path = output_root / 'routing_dataset_statistics.csv'
    with summary_path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    """解析命令行并对完整测试集或指定单图导出路由可视化。"""
    parser = argparse.ArgumentParser(
        description='Visualize trained BEAR-BioIR routing behavior.')
    parser.add_argument('--opt', required=True, help='BEAR option YAML.')
    parser.add_argument('--weights', required=True, help='Generator *.pth.')
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        '--image', default=None,
        help='只处理指定低光图；省略时默认处理 YAML 完整验证集。')
    input_group.add_argument(
        '--input-dir', default=None,
        help='覆盖 YAML datasets.val.dataroot_lq 的测试图片目录。')
    parser.add_argument(
        '--output-dir', default=None,
        help=('输出根目录；默认使用 analysis_artifacts/routing_behavior/'
              '<实验名>。'))
    parser.add_argument(
        '--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable.')

    option_path = Path(args.opt)
    weights_path = Path(args.weights)
    opt = parse(str(option_path), is_train=False)
    input_root, image_entries = resolve_input_images(
        opt, image_path=args.image, input_directory=args.input_dir)
    model = load_model(opt, weights_path, device)
    output_root = resolve_output_root(args.output_dir, opt['name'])
    rows = visualize_input_set(
        model, image_entries, output_root, device, option_path, weights_path)
    print(f'Input root: {input_root.resolve()}')
    print(f'Processed images: {len(rows)}')
    print(f'Routing visualizations: {output_root.resolve()}')
    print(
        f'Dataset statistics: '
        f'{(output_root / "routing_dataset_statistics.csv").resolve()}')
    print('Fixed map range=[0,1]; route resize=nearest')


if __name__ == '__main__':
    main()
