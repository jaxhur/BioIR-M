"""BioIR 训练入口，包含可复核日志与可靠断点恢复。"""

import argparse
import logging
import math
import os
import pickle
import random
import re
import time
from os import path as osp

import numpy as np
import torch

from basicsr.data import create_dataloader, create_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import create_model
from basicsr.utils import (MessageLogger, check_resume, get_env_info,
                           get_file_logger, init_tb_logger,
                           init_wandb_logger, set_random_seed)
from basicsr.utils.dist_util import get_dist_info, init_dist
from basicsr.utils.options import dict2str, parse


TRAIN_LOG_INTERVAL = 20
STATE_FILE_PATTERN = re.compile(r'^(\d+)\.state$')
NEW_WEIGHT_PATTERN = re.compile(r'^(\d+)_G\.pth$')
LEGACY_WEIGHT_PATTERN = re.compile(r'^net_g_(\d+)\.pth$')


def parse_options(is_train=True):
    """解析训练参数、分布式环境与随机种子。"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--input_path', type=str, required=False)
    parser.add_argument('--output_path', type=str, required=False)
    args = parser.parse_args()
    opt = parse(args.opt, is_train=is_train)

    if args.launcher == 'none':
        opt['dist'] = False
        print('Disable distributed.', flush=True)
    else:
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)
            print('Initialize distributed training.', flush=True)
    opt['rank'], opt['world_size'] = get_dist_info()

    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])
    if args.input_path is not None and args.output_path is not None:
        opt['img_path'] = {
            'input_img': args.input_path,
            'output_img': args.output_path
        }
    return opt


def ensure_experiment_dirs(opt):
    """创建实验目录且不重命名已有目录，以支持日志追加与自动续训。"""
    directory_keys = (
        'experiments_root', 'models', 'training_state', 'logs',
        'visualization', 'tb_logger')
    for key in directory_keys:
        path = opt['path'].get(key)
        if path:
            os.makedirs(path, exist_ok=True)


def _matching_weight_path(models_dir, current_iter):
    """查找与状态迭代数严格匹配的新式或旧式权重。"""
    for filename in (f'{current_iter}_G.pth', f'net_g_{current_iter}.pth'):
        candidate = osp.join(models_dir, filename)
        if osp.isfile(candidate):
            return candidate
    return None


def _load_state(path):
    """读取并校验训练状态的必要字段。"""
    try:
        # 训练状态包含随机数状态等非张量对象，PyTorch 2.6+ 需显式关闭
        # weights_only；旧版 PyTorch 不支持该参数，因此保留兼容分支。
        state = torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        state = torch.load(path, map_location='cpu')
    required_keys = {'epoch', 'iter', 'optimizers', 'schedulers'}
    if not isinstance(state, dict) or not required_keys.issubset(state):
        raise ValueError(
            f'Invalid training state {path}; required keys: {required_keys}')
    return state


def discover_resume_state(opt):
    """优先读取显式状态，否则从新旧目录倒序寻找完整断点。"""
    explicit_path = opt['path'].get('resume_state')
    models_dir = opt['path']['models']
    if explicit_path:
        state = _load_state(explicit_path)
        if _matching_weight_path(models_dir, int(state['iter'])) is None:
            raise FileNotFoundError(
                f'Explicit resume state has no matching generator weight: '
                f'{explicit_path}')
        return state, explicit_path, []

    state_dirs = [
        opt['path']['training_state'],
        osp.join(opt['path']['experiments_root'], 'training_states')
    ]
    candidates = []
    for state_dir in dict.fromkeys(state_dirs):
        if not osp.isdir(state_dir):
            continue
        for filename in os.listdir(state_dir):
            match = STATE_FILE_PATTERN.fullmatch(filename)
            if match:
                candidates.append(
                    (int(match.group(1)), osp.join(state_dir, filename)))

    diagnostics = []
    for _, state_path in sorted(candidates, reverse=True):
        try:
            state = _load_state(state_path)
            current_iter = int(state['iter'])
            if _matching_weight_path(models_dir, current_iter) is None:
                raise FileNotFoundError(
                    f'no matching weight for iter {current_iter}')
            return state, state_path, diagnostics
        except (OSError, RuntimeError, ValueError, EOFError,
                pickle.UnpicklingError) as error:
            diagnostics.append(
                f'Skipped unusable state {state_path}: {error}')
    return None, None, diagnostics


def discover_weight_only_recovery(models_dir):
    """状态缺失时选择最高数字权重，最后才退回 latest 权重。"""
    if not osp.isdir(models_dir):
        return None
    numbered_weights = []
    for filename in os.listdir(models_dir):
        match = NEW_WEIGHT_PATTERN.fullmatch(filename)
        if match is None:
            match = LEGACY_WEIGHT_PATTERN.fullmatch(filename)
        if match:
            numbered_weights.append(
                (int(match.group(1)), osp.join(models_dir, filename)))
    if numbered_weights:
        return max(numbered_weights)[1]
    for filename in ('latest_G.pth', 'net_g_latest.pth'):
        candidate = osp.join(models_dir, filename)
        if osp.isfile(candidate):
            return candidate
    return None


def init_loggers(opt):
    """初始化相互独立的 train/val 日志及实验内 TensorBoard。"""
    train_logger = get_file_logger(
        'basicsr', opt['path']['train_log'], logging.INFO)
    val_logger = get_file_logger(
        'basicsr_val', opt['path']['val_log'], logging.INFO)
    train_logger.info(get_env_info())
    train_logger.info(dict2str(opt))

    if (opt['logger'].get('wandb') is not None
            and opt['logger']['wandb'].get('project') is not None
            and 'debug' not in opt['name']):
        assert opt['logger'].get('use_tb_logger') is True, (
            'TensorBoard must be enabled when using wandb.')
        init_wandb_logger(opt)
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir=opt['path']['tb_logger'])
    return train_logger, val_logger, tb_logger


def create_train_val_dataloader(opt, logger):
    """创建数据加载器，并以实际 DataLoader 长度统计每 epoch 迭代数。"""
    train_loader, train_sampler, val_loader = None, None, None
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = create_dataset(dataset_opt)
            train_sampler = EnlargedSampler(
                train_set, opt['world_size'], opt['rank'], enlarge_ratio)
            train_loader = create_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])
            steps_per_epoch = len(train_loader)
            if steps_per_epoch <= 0:
                raise RuntimeError(
                    'Training DataLoader has zero batches. Check batch size '
                    'and drop_last settings.')
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / steps_per_epoch)
            logger.info(
                'Training statistics:'
                f' images={len(train_set):,}, enlarge_ratio={enlarge_ratio},'
                f' batch_per_gpu={dataset_opt["batch_size_per_gpu"]:,},'
                f' world_size={opt["world_size"]:,},'
                f' steps_per_epoch={steps_per_epoch:,},'
                f' total_epochs={total_epochs:,}, total_iters={total_iters:,}.')
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(
                val_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=None,
                seed=opt['manual_seed'])
            logger.info(
                f'Validation dataset={dataset_opt["name"]}, '
                f'images={len(val_set):,}.')
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')
    if train_loader is None:
        raise RuntimeError('Training dataset is not configured.')
    return (train_loader, train_sampler, val_loader, total_epochs,
            total_iters, steps_per_epoch)


def capture_runtime_state():
    """捕获主进程随机数状态，减少恢复后的随机轨迹偏移。"""
    state = {
        'python_rng_state': random.getstate(),
        'numpy_rng_state': np.random.get_state(),
        'torch_rng_state': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['cuda_rng_state_all'] = torch.cuda.get_rng_state_all()
    return state


def restore_runtime_state(resume_state):
    """恢复断点中存在的主进程随机数状态，兼容旧状态文件。"""
    if 'python_rng_state' in resume_state:
        random.setstate(resume_state['python_rng_state'])
    if 'numpy_rng_state' in resume_state:
        np.random.set_state(resume_state['numpy_rng_state'])
    if 'torch_rng_state' in resume_state:
        torch.set_rng_state(resume_state['torch_rng_state'])
    if torch.cuda.is_available() and 'cuda_rng_state_all' in resume_state:
        torch.cuda.set_rng_state_all(resume_state['cuda_rng_state_all'])


def validation_data_name(opt):
    """返回规范数据集/测试划分名称。"""
    normalized = opt['name'].lower().replace('_', '-')
    if 'v2-real' in normalized:
        return 'LOL-v2-real/Real_captured/Test'
    if 'v2-syn' in normalized:
        return 'LOL-v2-syn/Synthetic/Test'
    if 'lol' in normalized:
        return 'LOL-v1/eval15'
    return opt['datasets']['val']['name']


def format_best_rgb_ssim(best_rgb_ssim):
    """将最佳 PSNR 对应的 RGB SSIM 格式化为日志字段。"""
    if best_rgb_ssim is None:
        return 'unknown'
    return f'{best_rgb_ssim:.4f}'


def run_validation(model, val_loader, opt, current_iter, epoch, total_epochs,
                   tb_logger, val_logger, best_psnr, best_rgb_ssim):
    """执行验证并记录最佳 PSNR 及其对应的 RGB SSIM。"""
    if val_loader is None:
        return best_psnr, best_rgb_ssim
    rgb2bgr = opt['val'].get('rgb2bgr', True)
    use_image = opt['val'].get('use_image', True)
    metrics = model.validation(
        val_loader, current_iter, tb_logger, opt['val']['save_img'],
        rgb2bgr, use_image)
    if metrics is None:
        return best_psnr, best_rgb_ssim
    psnr = float(metrics['psnr'])
    ssim = float(metrics['ssim'])
    updated = psnr > best_psnr
    if updated:
        best_psnr = psnr
        best_rgb_ssim = ssim
        model.save_best()
    val_logger.info(
        f'[{opt["name"]}][VAL] '
        f'[progress: epoch={epoch:,}/{total_epochs:,}, '
        f'iter={current_iter:,}/{int(opt["train"]["total_iter"]):,}] '
        f'[data: name={validation_data_name(opt)}] '
        f'[metric: psnr={psnr:.4f}, rgb_ssim={ssim:.4f}] '
        f'[best: key=psnr, value={best_psnr:.4f}, '
        f'rgb_ssim={format_best_rgb_ssim(best_rgb_ssim)}, '
        f'updated={"yes" if updated else "no"}]')
    return best_psnr, best_rgb_ssim


def build_extra_state(epoch, step, elapsed_time, best_psnr, best_rgb_ssim):
    """构建包含最佳 PSNR 及其对应 RGB SSIM 的复现状态。"""
    extra_state = {
        'state_format_version': 2,
        'epoch': epoch,
        'step': step,
        'elapsed_time': float(elapsed_time),
        'best_psnr': float(best_psnr),
        'best_rgb_ssim': (None if best_rgb_ssim is None else
                          float(best_rgb_ssim)),
    }
    extra_state.update(capture_runtime_state())
    return extra_state


def main():
    """运行训练、验证、保存与自动恢复流程。"""
    opt = parse_options(is_train=True)
    torch.backends.cudnn.benchmark = True

    resume_state, resume_path, diagnostics = discover_resume_state(opt)
    weight_only_path = None
    if resume_state is not None:
        opt['path']['resume_state'] = resume_path
    else:
        weight_only_path = discover_weight_only_recovery(opt['path']['models'])
        if weight_only_path is not None:
            opt['path']['pretrain_network_g'] = weight_only_path

    ensure_experiment_dirs(opt)
    logger, val_logger, tb_logger = init_loggers(opt)
    for diagnostic in diagnostics:
        logger.warning(diagnostic)

    loaders = create_train_val_dataloader(opt, logger)
    (train_loader, train_sampler, val_loader, total_epochs, total_iters,
     steps_per_epoch) = loaders

    if resume_state is not None:
        check_resume(opt, int(resume_state['iter']))
        model = create_model(opt)
        model.resume_training(resume_state)
        current_iter = int(resume_state['iter'])
        elapsed_offset = float(resume_state.get('elapsed_time', 0.0))
        best_psnr = float(resume_state.get('best_psnr', float('-inf')))
        best_rgb_ssim = resume_state.get('best_rgb_ssim')
        if best_rgb_ssim is not None:
            best_rgb_ssim = float(best_rgb_ssim)
        if resume_state.get('state_format_version') == 2:
            epoch = max(1, int(resume_state['epoch']))
            resume_step = max(0, int(resume_state.get('step', 0)))
        else:
            # 历史 BasicSR 状态保存 0-based epoch 且不记录 epoch 内 step。
            epoch = max(1, int(resume_state['epoch']) + 1)
            resume_step = 0
        restore_runtime_state(resume_state)
        logger.info(
            f'Resuming full training state from {resume_path}: '
            f'epoch={epoch:,}, iter={current_iter:,}, step={resume_step:,}.')
    else:
        model = create_model(opt)
        epoch, current_iter, resume_step = 1, 0, 0
        elapsed_offset, best_psnr, best_rgb_ssim = (
            0.0, float('-inf'), None)
        if weight_only_path is not None:
            logger.warning(
                f'Weight-only recovery from {weight_only_path}. Optimizer, '
                'scheduler, iteration, best metric and elapsed time were not '
                'recoverable and restart from iter=0.')

    while resume_step >= steps_per_epoch:
        resume_step -= steps_per_epoch
        epoch += 1
    msg_logger = MessageLogger(
        opt,
        start_iter=current_iter,
        tb_logger=tb_logger,
        total_epochs=total_epochs,
        steps_per_epoch=steps_per_epoch,
        elapsed_offset=elapsed_offset)

    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info('Use CUDA prefetch dataloader.')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        raise ValueError(
            f'Wrong prefetch_mode {prefetch_mode}; expected cpu/cuda/None.')

    logger.info(
        f'Start training: epoch={epoch:,}/{total_epochs:,}, '
        f'iter={current_iter:,}/{total_iters:,}.')
    data_time_start = time.time()
    iter_time_start = time.time()
    last_val_iter = -1
    first_epoch = True
    current_step = resume_step

    while current_iter < total_iters:
        train_sampler.set_epoch(epoch - 1)
        prefetcher.reset()
        train_data = prefetcher.next()
        current_step = 0
        skip_steps = resume_step if first_epoch else 0

        while train_data is not None and current_iter < total_iters:
            current_step += 1
            if current_step <= skip_steps:
                train_data = prefetcher.next()
                data_time_start = time.time()
                iter_time_start = time.time()
                continue

            data_time = time.time() - data_time_start
            current_iter += 1
            model.update_learning_rate(
                current_iter,
                warmup_iter=opt['train'].get('warmup_iter', -1))
            model.feed_data(train_data, is_val=False)
            model.optimize_parameters(current_iter, tb_logger)
            iter_time = time.time() - iter_time_start

            if current_iter % TRAIN_LOG_INTERVAL == 0:
                log_vars = {
                    'epoch': epoch,
                    'total_epochs': total_epochs,
                    'iter': current_iter,
                    'total_iter': total_iters,
                    'step': current_step,
                    'steps_per_epoch': steps_per_epoch,
                    'lrs': model.get_current_learning_rate(),
                    'time': iter_time,
                    'data_time': data_time,
                }
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)

            should_validate = (
                opt.get('val') is not None
                and current_iter % int(opt['val']['val_freq']) == 0)
            if should_validate:
                best_psnr, best_rgb_ssim = run_validation(
                    model, val_loader, opt, current_iter, epoch,
                    total_epochs, tb_logger, val_logger, best_psnr,
                    best_rgb_ssim)
                last_val_iter = current_iter

            if current_iter % int(
                    opt['logger']['save_checkpoint_freq']) == 0:
                logger.info(
                    f'Saving checkpoint at iter={current_iter:,}.')
                extra_state = build_extra_state(
                    epoch, current_step, msg_logger.elapsed_seconds(),
                    best_psnr, best_rgb_ssim)
                model.save(epoch, current_iter, extra_state=extra_state)

            data_time_start = time.time()
            iter_time_start = time.time()
            train_data = prefetcher.next()

        first_epoch = False
        resume_step = 0
        if current_iter < total_iters:
            epoch += 1

    if opt.get('val') is not None and last_val_iter != current_iter:
        best_psnr, best_rgb_ssim = run_validation(
            model, val_loader, opt, current_iter, epoch, total_epochs,
            tb_logger, val_logger, best_psnr, best_rgb_ssim)
    extra_state = build_extra_state(
        epoch, current_step, msg_logger.elapsed_seconds(), best_psnr,
        best_rgb_ssim)
    model.save(epoch, current_iter, extra_state=extra_state)
    logger.info(
        f'End of training: elapsed_seconds={msg_logger.elapsed_seconds():.0f}, '
        f'best_psnr={best_psnr:.4f}.')
    if tb_logger:
        tb_logger.close()


if __name__ == '__main__':
    os.environ['GRPC_POLL_STRATEGY'] = 'epoll1'
    main()
