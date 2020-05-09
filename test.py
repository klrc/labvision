from labvision import auto
import torchvision
import labvision
import torch

labvision.init('my-project')

transform = labvision.transforms.to_tensor
model = torchvision.models.resnet18()
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = torch.nn.Linear(512, 10)

c = auto.config
c.model = model
c.datasets.trainset = torchvision.datasets.MNIST(root='/home/sh/Desktop/Research/external/MNIST', transform=transform, train=True)
c.datasets.testset = torchvision.datasets.MNIST(root='/home/sh/Desktop/Research/external/MNIST', transform=transform, train=False)
c.whatever = 20
core = auto.compile()
# core = auto.AutoCore(config=c)

core.check()

best_acc = 0
for status in core.steps(iters=24):
    if status.epoch_finished():
        acc = core.eval('acc')
        print(acc)
        if acc > best_acc:
            best_acc = acc
            # torch.save(core.model.state_dict(), 'build/best_model.pt')
