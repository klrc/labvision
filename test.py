



# from labvision import server

# server.init('my-project')
# server.set_server('172.20.46.235', '/home/sh/Desktop/remote_deploy')
# server.set_user('sh', 'sh')
# server.exec_command('ls')
# server.close()


import torchvision
import torch
from labvision import auto, transforms

core = auto.Core('build/test-project')
transform = transforms.to_tensor
model = torchvision.models.resnet18()
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = torch.nn.Linear(512, 10)

core.model = model
core.set_datasets(torchvision.datasets.MNIST, root='/home/sh/Desktop/Research/external/MNIST', transform=transform)

slave = core.compile()
slave.check()

# best_acc = 0
# for status in core.steps(iters=24):
#     if status.epoch_finished():
#         acc = core.eval('acc@top3')
#         print(acc)
#         if acc > best_acc:
#             best_acc = acc
#     if status.epoch > 3:
#         break
#         # torch.save(core.model.state_dict(), 'build/best_model.pt')
