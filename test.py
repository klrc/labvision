
from labvision.visualize import Graph
g = Graph('/home/sh/Desktop/Research/rec.log')

g.curve('<gNViC48z>', 'val_loss', alpha=0.3).curve('<gNViC48z>', 'val_loss', smooth=0.99)
g.curve('<gNViC48z>', 'train_loss', alpha=0.3).curve('<gNViC48z>', 'train_loss', smooth=0.99)
g.curve('<gNViC48z>', 'acc@top1', alpha=0.3).curve('<gNViC48z>', 'acc@top1', smooth=0.6)
g.save('/home/sh/Desktop/Research/gNViC48z.png')


g.refresh()
g.curve('<w2IRuSvg>', 'val_loss', alpha=0.3).curve('<w2IRuSvg>', 'val_loss', smooth=0.99)
g.curve('<w2IRuSvg>', 'train_loss', alpha=0.3).curve('<w2IRuSvg>', 'train_loss', smooth=0.99)
g.curve('<w2IRuSvg>', 'acc@top1', alpha=0.3).curve('<w2IRuSvg>', 'acc@top1', smooth=0.6)
g.save('/home/sh/Desktop/Research/w2IRuSvg.png')
