import labvision
# import torchvision


# labvision.io.cpu.limit(10)
# core = labvision.auto.compile(model=torchvision.models.resnet18().cuda(),
#                               datasets=dict(dataset_class=labvision.datasets.FI,
#                                             args={'root': '/home/sh/Desktop/Research/external/FI'}))
# core = labvision.auto.resume('NovhVIyW.freeze')
# for model, status in core.step():
#     core.metrics(['acc@top1', 'acc@top2'])

# core.check()

from labvision.visualize import Graph
g = Graph('/home/sh/Desktop/Research/rec.log')
g.curve('<gNViC48z>', 'val_loss', alpha=0.5).curve('<gNViC48z>', 'val_loss', smooth=0.99)
g.curve('<gNViC48z>', 'train_loss', alpha=0.5).curve('<gNViC48z>', 'train_loss', smooth=0.99)
g.curve('<gNViC48z>', 'acc@top1')
g.save('test.png')
