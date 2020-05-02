import random
import torch
import labvision
from .utils import MissingKeyException
from .controller import AutoControl


def __update__(config, default):
    for k, v in default.items():
        if k not in config:
            config[k] = v
        elif type(v) is dict and type(config[k]) is dict:
            config[k] = __update__(config[k], v)
    return config


def default_config():
    random.seed()
    config = {
        'seed': random.randint(0, 65536),
        'model': None,
        'default_optimizer': torch.optim.SGD,
        'optimizer': {
            'lr': 1e-3,
            'weight_decay': 5e-4,
        },
        'datasets': {
            'dataset_class': None,
            'args': {
                'transform': labvision.transforms.resize_rdcrop_flip(224, (224, 224)),
            },
            'trainset': None,
            'testset': None,
            'valset': None,
            'batch_size': 64,
            'num_workers': 2,
        },
        'frozen_status': None,
        'log_file_path': 'rec.log',
        'build_dir': 'build',
    }
    return config


def check_config(config):
    if config['model'] is None:
        raise MissingKeyException('model', 'A model must be specified in config.')
    datasets = config['datasets']
    if datasets['dataset_class'] is None:
        if datasets['trainset'] is None:
            raise MissingKeyException('datasets.trainset',
                                      'You must specify both trainset and testset, or set use_simple_def=True')
        if datasets['testset'] is None:
            raise MissingKeyException('datasets.testset',
                                      'You must specify both trainset and testset, or set use_simple_def=True')


def compile(config=None, read_frozen_status=False, **kwargs):
    if config is None:
        config = kwargs  # kwargs mode

    # fill default values
    config = __update__(config, default_config())
    check_config(config)

    # init optimizer
    model = config['model']
    optimizer = config['optimizer']
    if type(optimizer) is dict:
        default_optim = config['default_optimizer']
        optimizer = default_optim(model.parameters(), **optimizer)
        config['optimizer'] = optimizer

    # init datasets
    datasets = config['datasets']
    if datasets['dataset_class'] is not None:
        trainset = datasets['dataset_class'](train=True, **datasets['args'])
        testset = datasets['dataset_class'](train=False, **datasets['args'])
        datasets['trainset'] = trainset
        datasets['testset'] = testset
    num_workers = datasets['num_workers']
    num_workers = min(num_workers, labvision.io.backends.device.cpu_num_workers_limit)
    datasets['num_workers'] = num_workers

    # load frozen status
    if not read_frozen_status:
        config['frozen_status'] = None

    return AutoControl(config)


def resume(fp):
    config = torch.load(fp)
    ac = AutoControl(config)
    ac.__log__(f'continue from: {fp}', pure_log=True)
    return ac
