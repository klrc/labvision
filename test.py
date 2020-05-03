import labvision
from labvision import auto
from labvision.transforms import ChainedTransform

if __name__ == "__main__":
    # remote.set_server('172.20.46.235', target_location='/home/sh/Desktop/remote_deploy')
    # remote.set_user('sh', 'sh')
    # remote.deploy('/home/sh/Desktop/Research/src', run='main.py', conda_env='dl')
    # remote.close(latency=1)

    import torch.nn as nn
    import torchvision

    # ac = auto.resume('build/test.freeze')
    # ac.check()
    c = auto.default_config
    model = torchvision.models.resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    c.model = model.cuda()
    c.datasets = torchvision.datasets.MNIST
    c.datasets.root = '/home/sh/Desktop/Research/external/MNIST/'
    c.datasets.transform = ChainedTransform().ToTensor()
    ac = auto.compile(c)

    ac.check()
    ac.freeze('build/test.freeze')