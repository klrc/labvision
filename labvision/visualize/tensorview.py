import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


def tensor2img(x, imtype=np.uint8):
    """"将tensor的数据类型转成numpy类型，并反归一化.

    Parameters:
        input_image (tensor) --  输入的图像tensor数组
        imtype (type)        --  转换后的numpy的数据类型
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if not isinstance(x, np.ndarray):
        if isinstance(x, torch.Tensor):  # get the data from a variable
            image_tensor = x.data
        else:
            return x
        image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        for i in range(len(mean)):
            image_numpy[i] = image_numpy[i] * std[i] + mean[i]
        image_numpy = image_numpy * 255
        image_numpy = np.transpose(image_numpy, (1, 2, 0))  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = x
    return image_numpy.astype(imtype)


def save(x, path, size=(448, 448), annotation=''):
    fig = plt.gcf()  # generate outputs
    plt.imshow(x, aspect='equal'), plt.axis('off'), fig.set_size_inches(size[1]*12/300.0, size[0]*12/300.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator()), plt.gca().yaxis.set_major_locator(plt.NullLocator()), plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0), plt.margins(0, 0)
    plt.text(0, 0, annotation, color='white', size=4, ha="left", va="top", bbox=dict(boxstyle="square", ec='black', fc='black'))
    plt.savefig(path, dpi=300, pad_inches=0)    # visualize masked image


def heatmap(x):
    heatmap = x[0].cpu().numpy()
    heatmap = heatmap/np.max(heatmap)
    # must convert to type unit8
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return heatmap


def show(x):
    if len(x.shape) > 3:
        x = x[0]
    if x.shape[0] == 1:
        x = heatmap(x)
    elif x.shape[0] == 3:
        x = tensor2img(x)
    else:
        raise NotImplementedError
    plt.imshow(x)
    plt.show()
    return x


if __name__ == "__main__":
    show(torch.rand(3, 224, 224))
