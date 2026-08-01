import yaml
from collections import OrderedDict
from os import path as osp


def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def _test_result_dataset_name(experiment_name):
    """从实验名推断规范测试结果目录名，未知实验保留原名。"""
    normalized_name = experiment_name.lower().replace('_', '-')
    if 'v2-real' in normalized_name:
        return 'LOL-v2-real'
    if 'v2-syn' in normalized_name:
        return 'LOL-v2-syn'
    if 'lol' in normalized_name:
        return 'LOL-v1'
    return experiment_name


def parse(opt_path, is_train=True):
    """Parse option file.

    Args:
        opt_path (str): Option file path.
        is_train (str): Indicate whether in training or not. Default: True.

    Returns:
        (dict): Options.
    """
    with open(opt_path, mode='r') as f:
        Loader, _ = ordered_yaml()
        opt = yaml.load(f, Loader=Loader)

    opt['is_train'] = is_train

    # datasets
    if 'datasets' in opt:
        for phase, dataset in opt['datasets'].items():
            # for several datasets, e.g., test_1, test_2
            phase = phase.split('_')[0]
            dataset['phase'] = phase
            if 'scale' in opt:
                dataset['scale'] = opt['scale']
            if dataset.get('dataroot_gt') is not None:
                dataset['dataroot_gt'] = osp.expanduser(dataset['dataroot_gt'])
            if dataset.get('dataroot_lq') is not None:
                dataset['dataroot_lq'] = osp.expanduser(dataset['dataroot_lq'])

    # paths
    for key, val in opt['path'].items():
        if (val is not None) and ('resume_state' in key
                                  or 'pretrain_network' in key):
            opt['path'][key] = osp.expanduser(val)
    opt['path']['root'] = osp.abspath(
        osp.join(__file__, osp.pardir, osp.pardir, osp.pardir))
    if is_train:
        experiments_root = osp.join(opt['path']['root'], 'experiments',
                                    opt['name'])
        opt['path']['experiments_root'] = experiments_root
        opt['path']['models'] = osp.join(experiments_root, 'models')
        training_state = osp.join(experiments_root, 'training_state')
        # 同时保留旧键名，避免已有模型类访问路径时失效。
        opt['path']['training_state'] = training_state
        opt['path']['training_states'] = training_state
        opt['path']['logs'] = osp.join(experiments_root, 'logs')
        opt['path']['train_log'] = osp.join(
            experiments_root, 'logs', 'train.log')
        opt['path']['val_log'] = osp.join(
            experiments_root, 'logs', 'val.log')
        opt['path']['tb_logger'] = osp.join(experiments_root, 'tb_looger')
        opt['path']['log'] = opt['path']['logs']
        opt['path']['visualization'] = osp.join(experiments_root,
                                                'visualization')

        # change some options for debug mode
        if 'debug' in opt['name']:
            if 'val' in opt:
                opt['val']['val_freq'] = 8
            opt['logger']['print_freq'] = 20
            opt['logger']['save_checkpoint_freq'] = 8
    else:  # test
        dataset_name = _test_result_dataset_name(opt['name'])
        results_root = osp.join(opt['path']['root'], 'test_result',
                                dataset_name)
        opt['path']['results_root'] = results_root
        opt['path']['log'] = results_root
        opt['path']['visualization'] = osp.join(results_root, 'enhanced')

    return opt


def dict2str(opt, indent_level=1):
    """dict to string for printing options.

    Args:
        opt (dict): Option dict.
        indent_level (int): Indent level. Default: 1.

    Return:
        (str): Option string for printing.
    """
    msg = '\n'
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_level * 2) + k + ':['
            msg += dict2str(v, indent_level + 1)
            msg += ' ' * (indent_level * 2) + ']\n'
        else:
            msg += ' ' * (indent_level * 2) + k + ': ' + str(v) + '\n'
    return msg
