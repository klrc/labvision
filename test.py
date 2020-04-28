import labvision
import torchvision


labvision.io.cpu.limit(10)
core = labvision.auto.compile(model=torchvision.models.resnet18().cuda(),
                       datasets=dict(dataset_class=labvision.datasets.FI,
                                     args={'root': '/home/sh/Desktop/Research/external/FI'}))

for x in core.step():
    print(x)