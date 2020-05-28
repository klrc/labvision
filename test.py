import time
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from labvision import mods
from torchvision.models import resnet18
from torchvision.datasets import MNIST


transform_train = transforms.Compose([
    transforms.Resize(size=(10, 10)),
    transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.Resize(size=(10, 10)),
    transforms.ToTensor(),
])

trainloader, valloader, testloader = mods.DataLoaders('/home/sh/datasets/MNIST', MNIST, transform_train, transform_test, 64, pin_memory=True)

net = resnet18(pretrained=True)  # 模型定义-ResNet
net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
net.fc = nn.Linear(512, 10)
net = net.cuda()

criterion = nn.CrossEntropyLoss()  # 损失函数为交叉熵，多用于多分类问题
optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）

trainer = mods.Train(net, criterion, optimizer)
accmeasure = mods.Accuracy(testloader, net, topk=3)
# sample = mods.Sample(trainloader)

total = len(trainloader)
for i, batch in enumerate(trainloader):
    loss = trainer(batch)
    print(time.asctime(time.localtime(time.time())))
    print(f'({i}/{total}) loss={loss}')
    if i > 10:
        break

print(time.asctime(time.localtime(time.time())))
print(accmeasure())
print(time.asctime(time.localtime(time.time())))

# trainloop = mods.TrainLooper(trainloader, net, criterion, optimizer)
# valloop = mods.ValLooper(valloader, net, criterion)
# trainloop.hook(valloop, epoch=0.2, position='end')


# trainloop.hook(accmeasure, epoch=1, position='end')
# trainloop.hook(..., position='inputs')
# accmeasure.hook(savebest, condition='best')
# accmeasure.patience(3)

# trainloop.loop()

# for loss in trainloop.steps(iteration=5):
#     print(trainloop.epoch, trainloop.iteration)
#     print(loss)
#     print(valloop.step())
#     print()
#     if loss < 1.4:
#         break
# print(accmeasure())
