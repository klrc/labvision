import matplotlib.pyplot as plt
import shutil
import os
import cv2
import numpy as np
import torch

import sys
sys.path.append('.')


class Visualize():
    def __init__(self, log_path='external/rec.log'):
        """
            simple visualization utils for dirty logs,
            Args:
                log_path: path to .log file.
        """
        self._hash = 'unknown'
        self.figure_type = 'curve'
        self.xs = []
        self.clear()
        self.legends = {}
        self.log_path = log_path

    def __initas__(self, figure_type='curve'):
        if figure_type == 'curve':
            fig, acc_ax = plt.subplots()
            self.acc_ax = acc_ax
            self.loss_ax = acc_ax.twinx()
            self.acc_ax.set_xlabel("epoches")
            self.acc_ax.set_ylabel("accuracy")
            self.loss_ax.set_ylabel("loss")
        elif figure_type == 'image':
            pass

    def clear(self):
        """
            clear plt canvas.
        """
        plt.close('all')

    def import_hash(self, _hash):
        """
            import autocontrol record hash for visualization,
            Args:
                _hash:
        """
        self._hash = _hash.replace('<', '').replace('>', '')
        return self

    def curve(self, _type='acc', smooth=0, legend=None, color=None, alpha=1):
        """
            add acc/loss curve to canvas,
            Args:
        """
        x = self.__loadcurve__(self._hash, _type)
        self.xs.append({'type': _type, 'data': self.__smooth__(x, smooth), 'color': color, 'alpha': alpha})
        self.legend(self._hash if not legend else legend, _type)
        return self

    @staticmethod
    def __smooth__(x, weight):
        scalar = [_y for (_x, _y) in x]
        last = scalar[0]
        smoothed = []
        for point in scalar:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        smoothed = [(x[i][0], _y) for i, _y in enumerate(smoothed)]
        return smoothed

    def plot(self, display_loss_legend=True):
        """
            build the image from canvas,
            Args:
                display_loss_legend:
        """
        if self.figure_type == 'curve':
            def __read__(data, dim=0):
                return [_x[dim] for _x in data]
            self.__initas__('curve')
            for x in self.xs:
                _type = x['type']
                data = x['data']
                color = x['color']
                alpha = x['alpha']
                if 'loss' in _type:
                    self.loss_ax.plot(__read__(data, 0), __read__(data, 1), color=color, alpha=alpha)
                else:
                    self.acc_ax.plot(__read__(data, 0), __read__(data, 1), color=color, alpha=alpha)
            self.__legend__(display_loss_legend)
            return self
        elif self.figure_type == 'image':
            self.__initas__('image')
            x = self.xs[0]
            self.__imshow__(x, size=x.shape[:2])
            return self

    @staticmethod
    def __imshow__(x, size=(448, 448), annotation=''):
        fig = plt.gcf()  # generate outputs
        plt.imshow(x, aspect='equal'), plt.axis('off'), fig.set_size_inches(size[1]*12/300.0, size[0]*12/300.0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator()), plt.gca().yaxis.set_major_locator(plt.NullLocator()), plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0), plt.margins(0, 0)
        plt.text(0, 0, annotation, color='white', size=4, ha="left", va="top", bbox=dict(boxstyle="square", ec='black', fc='black'))
        # plt.savefig(path, dpi=300, pad_inches=0)    # visualize masked image

    def __select__(self, _hash):
        with open(self.log_path, 'r') as f:
            for line in f.readlines():
                if 'testing#' not in line and _hash in line:
                    yield line

    def __readepoch__(self, line):
        """
            read float epoch from line.
            Args:
                line: (str).
        """
        epoch, step = line.split('[')[-1].split(']')[0].split(',')
        if '-' in step:
            return int(epoch) + 1
        else:
            step, total_step = step.strip().split('/')
            return int(epoch) + float(step)/float(total_step)

    def __loadcurve__(self, _hash, loss_type='train_loss'):
        x = []
        for line in self.__select__(_hash):
            if f'{loss_type}:' in line:
                data = float(line.split(':')[-1].strip())
                x.append((self.__readepoch__(line), data))
        return self.__cleancurve__(x)

    def __cleancurve__(self, x):
        last_epoch = -1
        for idx, (epoch, data) in enumerate(x):
            if epoch < last_epoch:
                x = self.__rmdupcurve__(x, idx)
                return self.__cleancurve__(x)
            else:
                last_epoch = epoch
        return x

    def __rmdupcurve__(self, x, idx):
        new_x = []
        dup_start = x[idx][0]
        for i, (epoch, data) in enumerate(x):
            if epoch >= dup_start and i < idx:
                continue
            new_x.append((epoch, data))
        return new_x

    def legend(self, labels, ax_type='acc'):
        """
            add legends for acc/loss curve,
            Args:
                labels: str / list[str]
                ax_type:
        """
        if ax_type not in self.legends.keys():
            self.legends[ax_type] = []
        if type(labels) is str:
            labels = [labels]
        self.legends[ax_type].extend(labels)
        return self

    def __legend__(self, display_loss_legend):
        for ax_type in self.legends:
            if ax_type == 'acc':
                self.acc_ax.legend(self.legends[ax_type])
            elif 'loss' in ax_type and display_loss_legend:
                self.loss_ax.legend(self.legends[ax_type])

    def save(self, path=None):
        """
            save image to path.
            Args:
                path:
        """
        if path is None:
            path = f'build/{self._hash}-{self.figure_type}.png'
        plt.show()
        plt.savefig(path, pad_inches=0)
        print(f'saved as {path}')
        return self

    def from_tensor(self, x):
        """
            read tensor as image,
            1-dim tensors are displayed as heatmap, see __heatmap__().
            Args:
                x: input tensor.
        """
        if self.figure_type != 'image':
            self.xs = []
            self.figure_type = 'image'
        if x.shape[0] == 1:
            self.xs.append(self.__heatmap__(x))
        else:
            self.xs.append(self.__tensor2img__(x))
        return self

    @staticmethod
    def __heatmap__(x):
        heatmap = x[0].cpu().numpy()
        heatmap = heatmap/np.max(heatmap)
        # must convert to type unit8
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        return heatmap

    @staticmethod
    def __tensor2img__(x, imtype=np.uint8):
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

    def __makegif__(self, from_dir='build/cache', fp='build/test.gif'):
        """
            not implemented yet.
        """
        # gif_images = [imageio.imread(f'{from_dir}/{img_file}') for img_file in files]
        # imageio.mimsave(fp, gif_images, fps=8)
        raise NotImplementedError

    def _clean_cache(self, target_dir='build/cache'):
        """
            clean cache dir. (can be dangerous)
            Args:
                target_dir:
        """
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        os.mkdir(target_dir)
        print(f'cache dir {target_dir} cleaned.')
