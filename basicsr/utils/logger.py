import logging
import time
from datetime import datetime, timedelta, timezone

from .dist_util import get_dist_info, master_only


BEIJING_TIMEZONE = timezone(timedelta(hours=8))


class BeijingFormatter(logging.Formatter):
    """使用固定 UTC+8 时区生成可复核的日志时间戳。"""

    def formatTime(self, record, datefmt=None):
        """将日志记录时间转换为北京时间。"""
        date_time = datetime.fromtimestamp(
            record.created, timezone.utc).astimezone(BEIJING_TIMEZONE)
        return date_time.strftime(datefmt or '%Y-%m-%d %H:%M:%S')


def _format_duration(seconds):
    """将秒数格式化为不受 24 小时上限约束的 HH:MM:SS。"""
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


class MessageLogger():
    """Message logger for printing.

    Args:
        opt (dict): Config. It contains the following keys:
            name (str): Exp name.
            logger (dict): Contains 'print_freq' (str) for logger interval.
            train (dict): Contains 'total_iter' (int) for total iters.
            use_tb_logger (bool): Use tensorboard logger.
        start_iter (int): Start iter. Default: 1.
        tb_logger (obj:`tb_logger`): Tensorboard logger. Default： None.
    """

    def __init__(self,
                 opt,
                 start_iter=0,
                 tb_logger=None,
                 total_epochs=None,
                 steps_per_epoch=None,
                 elapsed_offset=0.0):
        self.exp_name = opt['name']
        self.interval = opt['logger']['print_freq']
        self.start_iter = start_iter
        self.max_iters = opt['train']['total_iter']
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.use_tb_logger = opt['logger']['use_tb_logger']
        self.tb_logger = tb_logger
        self.start_time = time.time()
        self.elapsed_offset = float(elapsed_offset)
        self.logger = get_root_logger()

    def elapsed_seconds(self):
        """返回包含续训前历史时长的累计训练秒数。"""
        return self.elapsed_offset + time.time() - self.start_time

    def _use_tensorboard(self):
        return (self.use_tb_logger and self.tb_logger is not None
                and 'debug' not in self.exp_name)

    @master_only
    def __call__(self, log_vars):
        """Format logging message.

        Args:
            log_vars (dict): It contains the following keys:
                epoch (int): Epoch number.
                iter (int): Current iter.
                lrs (list): List for learning rates.

                time (float): Iter time.
                data_time (float): Data time for each iter.
        """
        # epoch、step 使用面向用户的 1-based 口径。
        epoch = log_vars.pop('epoch')
        current_iter = log_vars.pop('iter')
        total_iter = log_vars.pop('total_iter', self.max_iters)
        step = log_vars.pop('step', 1)
        steps_per_epoch = log_vars.pop('steps_per_epoch',
                                       self.steps_per_epoch or 1)
        total_epochs = log_vars.pop('total_epochs', self.total_epochs or 1)
        lrs = log_vars.pop('lrs')
        learning_rate = lrs[0]
        if self._use_tensorboard():
            for i, lr in enumerate(lrs):
                self.tb_logger.add_scalar(f'train/lr_g_{i}', lr, current_iter)

        # ETA 只使用本次进程实际完成的迭代，避免续训历史时长扭曲估计。
        session_elapsed = time.time() - self.start_time
        session_iters = max(1, current_iter - self.start_iter)
        eta_seconds = (session_elapsed / session_iters
                       * max(0, total_iter - current_iter))
        if 'time' in log_vars.keys():
            iter_time = log_vars.pop('time')
            data_time = log_vars.pop('data_time')
            if self._use_tensorboard():
                self.tb_logger.add_scalar('time/iter', iter_time, current_iter)
                self.tb_logger.add_scalar('time/data', data_time, current_iter)

        loss_items = [(key, float(value)) for key, value in log_vars.items()
                      if key.startswith('l_')]
        total_loss = dict(loss_items).get(
            'l_total', sum(value for _, value in loss_items))
        component_losses = [(key, value) for key, value in loss_items
                            if key != 'l_total'] or loss_items
        loss_text = ', '.join(
            f'{key}={value:.4f}' for key, value in component_losses)
        message = (
            f'[{self.exp_name}][TRAIN] '
            f'[progress: epoch={epoch:,}/{total_epochs:,}, '
            f'iter={current_iter:,}/{total_iter:,}, '
            f'step={step:,}/{steps_per_epoch:,}] '
            f'[time: elapsed={_format_duration(self.elapsed_seconds())}, '
            f'eta={_format_duration(eta_seconds)}] '
            f'[optim: lr={learning_rate:.3e}] '
            f'[total_loss: {total_loss:.4f}] [loss: {loss_text}]')

        for key, value in log_vars.items():
            if self._use_tensorboard():
                if key.startswith('l_'):
                    self.tb_logger.add_scalar(
                        f'losses/{key}', value, current_iter)
                elif key.startswith('m_'):
                    self.tb_logger.add_scalar(
                        f'metrics/{key}', value, current_iter)
        self.logger.info(message)


@master_only
def init_tb_logger(log_dir):
    from torch.utils.tensorboard import SummaryWriter
    tb_logger = SummaryWriter(log_dir=log_dir)
    return tb_logger


@master_only
def init_wandb_logger(opt):
    """We now only use wandb to sync tensorboard log."""
    import wandb
    logger = logging.getLogger('basicsr')

    project = opt['logger']['wandb']['project']
    resume_id = opt['logger']['wandb'].get('resume_id')
    if resume_id:
        wandb_id = resume_id
        resume = 'allow'
        logger.warning(f'Resume wandb logger with id={wandb_id}.')
    else:
        wandb_id = wandb.util.generate_id()
        resume = 'never'

    wandb.init(
        id=wandb_id,
        resume=resume,
        name=opt['name'],
        config=opt,
        project=project,
        sync_tensorboard=True)

    logger.info(f'Use wandb logger with id={wandb_id}; project={project}.')


def get_root_logger(logger_name='basicsr',
                    log_level=logging.INFO,
                    log_file=None,
                    file_mode='a'):
    """Get the root logger.

    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added.

    Args:
        logger_name (str): root logger name. Default: 'basicsr'.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.

    Returns:
        logging.Logger: The root logger.
    """
    logger = logging.getLogger(logger_name)
    # 同一进程内复用已配置实例，防止重复输出同一行。
    if getattr(logger, '_basicsr_initialized', False):
        return logger

    format_str = '%(asctime)s %(levelname)s: %(message)s'
    formatter = BeijingFormatter(format_str, datefmt='%Y-%m-%d %H:%M:%S')
    logger.handlers.clear()
    logger.propagate = False
    rank, _ = get_dist_info()
    if rank != 0:
        logger.setLevel('ERROR')
    else:
        logger.setLevel(log_level)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(log_level)
        logger.addHandler(stream_handler)
        if log_file is not None:
            file_handler = logging.FileHandler(
                log_file, mode=file_mode, encoding='utf-8')
            file_handler.setFormatter(formatter)
            file_handler.setLevel(log_level)
            logger.addHandler(file_handler)
    logger._basicsr_initialized = True

    return logger


def get_file_logger(logger_name, log_file, log_level=logging.INFO):
    """创建终端/文件完全同格式且以追加模式写入的独立日志器。"""
    return get_root_logger(
        logger_name=logger_name,
        log_level=log_level,
        log_file=log_file,
        file_mode='a')


def get_env_info():
    """Get environment information.

    Currently, only log the software version.
    """
    import torch
    import torchvision

    from basicsr.version import __version__
    msg = r"""
                ____                _       _____  ____
               / __ ) ____ _ _____ (_)_____/ ___/ / __ \
              / __  |/ __ `// ___// // ___/\__ \ / /_/ /
             / /_/ // /_/ /(__  )/ // /__ ___/ // _, _/
            /_____/ \__,_//____//_/ \___//____//_/ |_|
     ______                   __   __                 __      __
    / ____/____   ____   ____/ /  / /   __  __ _____ / /__   / /
   / / __ / __ \ / __ \ / __  /  / /   / / / // ___// //_/  / /
  / /_/ // /_/ // /_/ // /_/ /  / /___/ /_/ // /__ / /<    /_/
  \____/ \____/ \____/ \____/  /_____/\____/ \___//_/|_|  (_)
    """
    msg += ('\nVersion Information: '
            f'\n\tBasicSR: {__version__}'
            f'\n\tPyTorch: {torch.__version__}'
            f'\n\tTorchVision: {torchvision.__version__}')
    return msg
