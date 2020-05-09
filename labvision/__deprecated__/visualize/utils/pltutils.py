import matplotlib.pyplot as plt


def twinx(xlabel='epoches', ylabel1='accuracy', ylabel2='loss'):
    refresh()
    _, acc_ax = plt.subplots()
    loss_ax = acc_ax.twinx()
    acc_ax.set_xlabel(xlabel)
    acc_ax.set_ylabel(ylabel1)
    loss_ax.set_ylabel(ylabel2)
    return acc_ax, loss_ax


def refresh():
    plt.close('all')
