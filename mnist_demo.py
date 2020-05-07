from labvision import auto
from labvision.transforms import ChainedTransform

if __name__ == "__main__":
    import torch.nn as nn
    import torchvision

    config = auto.default_config
    model = torchvision.models.resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    config.model = model.cuda()
    config.datasets = torchvision.datasets.MNIST
    config.datasets.root = '/home/sh/Desktop/Research/external/MNIST/'
    config.datasets.transform = ChainedTransform().ToTensor()
    ac = auto.compile(config)
    ac.check()

    for model, status in ac.step(interval=100, val_interval=16):
        if status.epoch_finished():
            ac.metrics('acc@top1')

            if status.epoch > 3:
                ac.freeze('build/mnist_demo.freeze')
                break

    ac = auto.resume('build/mnist_demo.freeze')
    ac.metrics('acc@top1')
