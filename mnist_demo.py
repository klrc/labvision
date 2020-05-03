import labvision
import torchvision

transforms = labvision.transforms.utils.ChainedTransform()
transforms = transforms.ToTensor()

config = {
    'model': torchvision.models.resnet34(pretrained=True),
    'datasets': {
        'dataset_class': torchvision.datasets.MNIST,
        'args': {
            'root': '/home/sh/Desktop/Research/external/MNIST/',
            'transform': transforms},
    }
}

ac = labvision.auto.compile(config)
ac.check()
