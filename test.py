

from labvision import auto, transforms, datasets
import torch
import torchvision
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

core = auto.Core('.')
auto.manual_seed(114514)
core.seed = 114514

transform = transforms.resize_centercrop_flip(112, (112, 112))
model = torchvision.models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(512, 8)

core.model = model
core.set_datasets(datasets.FI, root='/home/sh/Desktop/Research/external/FI', transform=transform)

core = core.compile()
core.check()

best_acc = 0
for _ in core.steps(iters=0.2):
    if core.epoch_finished():
        acc = core.eval('acc@top3')
        if acc > best_acc:
            best_acc = acc
            torch.save(core.model.state_dict(), 'build/best_model.pt')
    if core.epoch > 3:
        break
