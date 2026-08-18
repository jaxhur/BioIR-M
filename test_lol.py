"""按统一 LOL 复现口径评测 BioIR。"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from basicsr.metrics import calculate_psnr, calculate_ssim
from basicsr.models import create_model
from basicsr.utils import get_file_logger
from basicsr.utils.options import parse


IMAGE_EXTENSIONS = {'.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff', '.webp'}
DATASET_SPLITS = {
    'LOL-v1': ('our485', 'eval15'),
    'LOL-v2-syn': ('Synthetic/Train', 'Synthetic/Test'),
    'LOL-v2-real': ('Real_captured/Train', 'Real_captured/Test'),
}
PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_output_root(output_root, experiment_name, dataset_name):
    """返回当前实验和数据集对应的独立测试结果目录。"""
    if output_root is not None:
        return Path(output_root).expanduser().resolve()
    return PROJECT_ROOT / 'test_result' / experiment_name / dataset_name


def list_images(folder):
    """返回目录中按相对路径排序的非隐藏图像。"""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f'Image folder does not exist: {folder}')
    return sorted(
        path for path in folder.rglob('*')
        if (path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            and not any(part.startswith('.')
                        for part in path.relative_to(folder).parts)))


def make_pairs(lq_dir, gt_dir):
    """严格按相对路径双向校验并配对低照度图与 GT。"""
    lq_dir = Path(lq_dir)
    gt_dir = Path(gt_dir)
    lq_index = {
        path.relative_to(lq_dir).as_posix(): path
        for path in list_images(lq_dir)
    }
    gt_index = {
        path.relative_to(gt_dir).as_posix(): path
        for path in list_images(gt_dir)
    }
    missing_gt = sorted(set(lq_index) - set(gt_index))
    missing_lq = sorted(set(gt_index) - set(lq_index))
    if missing_gt or missing_lq:
        raise ValueError(
            'Strict relative-path pairing failed. '
            f'Missing GT: {len(missing_gt)} {missing_gt[:5]}; '
            f'missing LQ: {len(missing_lq)} {missing_lq[:5]}.')
    if not lq_index:
        raise RuntimeError(f'No image pairs found in {lq_dir} and {gt_dir}')
    return [(lq_index[key], gt_index[key]) for key in sorted(lq_index)]


def load_rgb(path):
    """读取 uint8 RGB 图像。"""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f'Could not read image: {path}')
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_rgb(path, image):
    """保存 uint8 RGB 图像，并按需创建父目录。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise OSError(f'Could not write image: {path}')


def center_label(image, text, height=28):
    """为可选对比图添加标题栏。"""
    label = np.zeros((height, image.shape[1], 3), dtype=np.uint8)
    cv2.putText(label, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([label, image], axis=0)


def save_comparison(path, low, restored, gt):
    """保存仅用于观察、不参与指标计算的横向对比图。"""
    if low.shape != restored.shape or low.shape != gt.shape:
        raise ValueError(
            f'Comparison images must have identical shapes: '
            f'{low.shape}, {restored.shape}, {gt.shape}')
    canvas = np.concatenate([
        center_label(low, 'Low'),
        center_label(restored, 'BioIR'),
        center_label(gt, 'GT')
    ], axis=1)
    save_rgb(path, canvas)


def load_state_dict(weights_path):
    """从生成器权重文件读取参数，并拒绝误传训练状态文件。"""
    weights_path = Path(weights_path)
    if weights_path.suffix.lower() == '.state':
        raise ValueError(
            f'Expected generator weights (*.pth), got state: {weights_path}')
    checkpoint = torch.load(weights_path, map_location='cpu')
    if not isinstance(checkpoint, dict):
        raise TypeError(f'Unsupported checkpoint object: {type(checkpoint)}')
    for key in ('params_ema', 'params', 'state_dict'):
        if key in checkpoint:
            checkpoint = checkpoint[key]
            break
    if not checkpoint or not all(torch.is_tensor(value)
                                 for value in checkpoint.values()):
        raise ValueError(
            f'Checkpoint does not contain generator parameters: {weights_path}')
    return {
        key[7:] if key.startswith('module.') else key: value
        for key, value in checkpoint.items()
    }


def load_model(opt, weights_path, device):
    """创建生成器并严格加载指定权重。"""
    opt['dist'] = False
    opt['num_gpu'] = 0 if device.type == 'cpu' else max(
        1, opt.get('num_gpu', 1))
    model = create_model(opt).net_g
    model.load_state_dict(load_state_dict(weights_path), strict=True)
    model.to(device).eval()
    return model


def infer_one(model, image_rgb, device, factor):
    """单次前向推理；只做反射补边，不缩放图像。"""
    image = image_rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)
    _, _, height, width = tensor.shape
    pad_height = (factor - height % factor) % factor
    pad_width = (factor - width % factor) % factor
    if pad_height or pad_width:
        tensor = F.pad(
            tensor, (0, pad_width, 0, pad_height), mode='reflect')
    with torch.inference_mode():
        restored = model(tensor)
        if isinstance(restored, list):
            restored = restored[-1]
    restored = restored[:, :, :height, :width]
    restored = restored.clamp(0, 1).squeeze(0).permute(1, 2, 0)
    return np.round(restored.cpu().numpy() * 255.0).astype(np.uint8)


def create_lpips_metric(device):
    """创建固定 alex/version=0.1 的 LPIPS 模型。"""
    try:
        import lpips
    except ImportError as exc:
        raise ImportError(
            'LPIPS evaluation requires the lpips package.') from exc
    metric = lpips.LPIPS(net='alex', version='0.1').to(device).eval()
    metric.requires_grad_(False)
    return metric


def calculate_lpips(restored, gt, metric, device):
    """在 RGB [-1, 1] 输入上计算单图 LPIPS。"""
    def to_tensor(image):
        tensor = torch.from_numpy(image.astype(np.float32) / 255.0)
        return tensor.permute(2, 0, 1).unsqueeze(0).to(device).mul(2).sub(1)

    with torch.inference_mode():
        value = metric(to_tensor(restored), to_tensor(gt))
    return float(value.squeeze().item())


def analyze_model_complexity(model):
    """使用 THOP 统计 1x3x256x256 下的 Params、GMACs 与 GFLOPs。"""
    try:
        from thop import profile
    except ImportError as exc:
        raise ImportError(
            'Complexity evaluation requires the thop package.') from exc
    device = next(model.parameters()).device
    dummy_input = torch.zeros(1, 3, 256, 256, device=device)
    model.eval()
    with torch.inference_mode():
        macs, _ = profile(model, inputs=(dummy_input,), verbose=False)
    params = sum(parameter.numel() for parameter in model.parameters())
    return {
        'params_m': float(params) / 1e6,
        'gmacs_g': float(macs) / 1e9,
        'gflops_g': 2.0 * float(macs) / 1e9,
        'input_size': '1x3x256x256',
        'tool': 'THOP',
        'note': 'GFLOPs=2*MACs; unsupported custom operations may be omitted',
    }


def infer_dataset_name(option_name):
    """从配置实验名推断规范数据集名。"""
    normalized = option_name.lower().replace('_', '-')
    if 'v2-real' in normalized:
        return 'LOL-v2-real'
    if 'v2-syn' in normalized:
        return 'LOL-v2-syn'
    if 'lol' in normalized:
        return 'LOL-v1'
    raise ValueError(
        f'Cannot infer dataset from option name {option_name!r}; use --dataset.')


def write_csv(path, fieldnames, rows):
    """以 UTF-8 写入结构固定的 CSV。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    """运行统一 LOL 测试并输出增强图、指标表和日志。"""
    parser = argparse.ArgumentParser(description='Test BioIR on LOL datasets.')
    parser.add_argument('--opt', required=True, help='Path to option YAML.')
    parser.add_argument('--weights', required=True, help='Generator *.pth.')
    parser.add_argument('--dataset', choices=sorted(DATASET_SPLITS))
    parser.add_argument('--name', default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        '--output-root',
        default=None,
        help=('测试产物目录；默认使用 '
              'test_result/<实验名>/<数据集名>。'))
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--factor', type=int, default=32)
    parser.add_argument('--save_comparison', action='store_true')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable.')

    opt = parse(args.opt, is_train=False)
    legacy_dataset = args.name if args.name in DATASET_SPLITS else None
    dataset_name = args.dataset or legacy_dataset or infer_dataset_name(
        opt['name'])
    # 默认按实验名隔离测试产物，防止同一服务器切换分支后覆盖历史结果。
    output_root = resolve_output_root(
        args.output_root, opt['name'], dataset_name)
    enhanced_dir = output_root / 'enhanced'
    comparison_dir = output_root / 'comparison'
    output_root.mkdir(parents=True, exist_ok=True)
    logger = get_file_logger('basicsr', str(output_root / 'test.log'))

    model = load_model(opt, args.weights, device)
    lpips_metric = create_lpips_metric(device)
    complexity = analyze_model_complexity(model)
    lq_dir = Path(opt['datasets']['val']['dataroot_lq'])
    gt_dir = Path(opt['datasets']['val']['dataroot_gt'])
    pairs = make_pairs(lq_dir, gt_dir)
    logger.info(
        f'[{opt["name"]}][TEST] dataset={dataset_name}, pairs={len(pairs)}, '
        f'device={device}, checkpoint={Path(args.weights).resolve()}')

    per_image_rows = []
    metric_values = {'psnr': [], 'ssim': [], 'lpips': []}
    for lq_path, gt_path in tqdm(pairs, unit='image'):
        low = load_rgb(lq_path)
        gt = load_rgb(gt_path)
        restored = infer_one(model, low, device, args.factor)
        if restored.shape != gt.shape:
            raise ValueError(
                f'Restored/GT shape mismatch for {lq_path}: '
                f'{restored.shape} vs {gt.shape}')
        relative_path = lq_path.relative_to(lq_dir)
        save_rgb(enhanced_dir / relative_path.with_suffix('.png'), restored)
        if args.save_comparison:
            save_comparison(
                comparison_dir / relative_path.with_suffix('.png'),
                low, restored, gt)

        psnr = calculate_psnr(
            restored, gt, crop_border=0, input_order='HWC',
            test_y_channel=False)
        ssim = calculate_ssim(
            restored, gt, crop_border=0, input_order='HWC',
            test_y_channel=False, ssim3d=False)
        lpips_value = calculate_lpips(restored, gt, lpips_metric, device)
        per_image_rows.append({
            'image': relative_path.as_posix(),
            'psnr': f'{psnr:.6f}',
            'ssim': f'{ssim:.6f}',
            'lpips': f'{lpips_value:.6f}',
        })
        metric_values['psnr'].append(psnr)
        metric_values['ssim'].append(ssim)
        metric_values['lpips'].append(lpips_value)

    average = {
        key: float(np.mean(values))
        for key, values in metric_values.items()
    }
    train_split, test_split = DATASET_SPLITS[dataset_name]
    metric_row = {
        'experiment': opt['name'],
        'dataset': dataset_name,
        'train_split': train_split,
        'test_split': test_split,
        'psnr': f'{average["psnr"]:.6f}',
        'psnr_mode': 'BasicSR RGB crop_border=0 test_y_channel=False',
        'ssim': f'{average["ssim"]:.6f}',
        'ssim_mode': 'BasicSR RGB crop_border=0 test_y_channel=False',
        'lpips': f'{average["lpips"]:.6f}',
        'lpips_backbone': 'alex',
        'lpips_version': '0.1',
        'lpips_input': 'RGB [-1,1]',
        'params_m': f'{complexity["params_m"]:.6f}',
        'gmacs_g': f'{complexity["gmacs_g"]:.6f}',
        'gflops_g': f'{complexity["gflops_g"]:.6f}',
        'input_size': complexity['input_size'],
        'checkpoint': str(Path(args.weights).resolve()),
        'enhanced_images': len(per_image_rows),
        'metric_source': 'per-image arithmetic mean',
        'complexity_tool': complexity['tool'],
        'complexity_note': complexity['note'],
    }
    write_csv(
        output_root / 'per_image_metrics.csv',
        ['image', 'psnr', 'ssim', 'lpips'], per_image_rows)
    write_csv(output_root / 'metric.csv', list(metric_row), [metric_row])
    logger.info(
        f'[{opt["name"]}][TEST] [metric: psnr={average["psnr"]:.4f}, '
        f'rgb_ssim={average["ssim"]:.4f}, lpips={average["lpips"]:.4f}] '
        f'[complexity: params_m={complexity["params_m"]:.4f}, '
        f'gmacs_g={complexity["gmacs_g"]:.4f}, '
        f'gflops_g={complexity["gflops_g"]:.4f}]')
    logger.info(f'Enhanced images: {enhanced_dir.resolve()}')
    logger.info(f'Metrics CSV: {(output_root / "metric.csv").resolve()}')


if __name__ == '__main__':
    main()
