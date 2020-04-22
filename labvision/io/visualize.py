import re
import matplotlib.pyplot as plt
import shutil
import os
import cv2
import numpy as np
import torch

import sys
sys.path.append('.')


class Visualize():
    def __init__(self, log_fp='rec.log'):
        self.log_fp = log_fp
        self.data = []
        self.axis_type = 'plt'
        self.legends = {}

    @staticmethod
    def __read__(fp, _hash, target=None):
        with open(fp, 'r') as f:
            for line in f.readlines():
                if 'testing#' in line:
                    continue
                if _hash in line:
                    found = Visualize.__re__(line)
                    if not found:
                        continue
                    if not target:
                        yield found
                    if target == found['metrics_name']:
                        yield found

    @staticmethod
    def __re__(line):
        try:
            res = re.search(r'\[(?P<epoch>\d+)\,\ +(?P<iter>\d+)\/(?P<total_iter>\d+)]\ +(?P<metrics_name>.*)\:\ +(?P<value>.*)', line)
            return res.groupdict()
        except Exception:
            return None

    def __curve__(self, _hash, metrics_name):
        for _dict in self.__read__(fp=self.log_fp, _hash=_hash, target=metrics_name):
            x = float(_dict['epoch'])+float(_dict['iter'])/float(_dict['total_iter'])
            y = float(_dict['value'])
            yield x, y

    @staticmethod
    def __smooth__(scalar, weight=0.7):
        last = scalar[0]
        smoothed = []
        for point in scalar:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed

    def curve(self, _hash, metrics_name, legend=None, smooth=0):
        data = [(x, y) for x, y in self.__curve__(_hash, metrics_name)]
        data_x, data_y = zip(*data)
        if legend is None:
            legend = f'{metrics_name}@{_hash}'
        description = {
            'metrics_name': metrics_name,
            '_hash': _hash,
            'legend': legend,
        }
        self.data.append({
            'data_x': data_x,
            'data_y': self.__smooth__(data_y, weight=smooth),
            'description': description,
        })
        return self

    def __plotcurve__(self, axis_type, **kwargs):
        if axis_type == 'plt':
            if 'plt' not in self.legends:
                self.legends['plt'] = []
            for d in self.data:
                data_x = d['data_x']
                data_y = d['data_y']
                description = d['description']
                self.legends.append(description['legend'])
                plt.plot(data_x, data_y, **kwargs)
        elif axis_type == 'twinx':
            if 'acc_ax' not in self.legends:
                self.legends['acc_ax'] = []
                self.legends['loss_ax'] = []
            for d in self.data:
                data_x = d['data_x']
                data_y = d['data_y']
                description = d['description']
                if 'loss' in description['metrics_name']:
                    self.legends['loss_ax'].append(description['legend'])
                    self.loss_ax.plot(data_x, data_y, **kwargs)
                elif 'acc' in description['metrics_name']:
                    self.legends['acc_ax'].append(description['legend'])
                    self.acc_ax.plot(data_x, data_y, **kwargs)

    def plot(self, **kwargs):
        if type(self.data) is list:
            self.__plotcurve__(axis_type=self.axis_type, **kwargs)
        self.data = []
        return self

    def save(self, path):
        """
            save image to path.
            Args:
                path:
        """
        if self.axis_type == 'plt':
            plt.legend(self.legends['plt'])
        elif self.axis_type == 'twinx':
            self.acc_ax.legend(self.legends['acc_ax'])
            self.loss_ax.legend(self.legends['loss_ax'])
        self.legends = {}
        plt.show()
        plt.savefig(path, pad_inches=0)
        print(f'saved as {path}')
        return self

    def clear(self):
        """
            clear plt canvas.
        """
        plt.close('all')

    def twinx(self):
        self.axis_type = 'twinx'
        _, acc_ax = plt.subplots()
        loss_ax = acc_ax.twinx()
        acc_ax.set_xlabel("epoches")
        acc_ax.set_ylabel("accuracy")
        loss_ax.set_ylabel("loss")
        self.acc_ax = acc_ax
        self.loss_ax = loss_ax
        return self

    # --------------------------------unimplemented below --------------------------------------------

    @staticmethod
    def __imshow__(x, size=(448, 448), annotation=''):
        fig = plt.gcf()  # generate outputs
        plt.imshow(x, aspect='equal'), plt.axis('off'), fig.set_size_inches(size[1]*12/300.0, size[0]*12/300.0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator()), plt.gca().yaxis.set_major_locator(plt.NullLocator()), plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0), plt.margins(0, 0)
        plt.text(0, 0, annotation, color='white', size=4, ha="left", va="top", bbox=dict(boxstyle="square", ec='black', fc='black'))
        # plt.savefig(path, dpi=300, pad_inches=0)    # visualize masked image

    @staticmethod
    def __rmdupcurve__(x, idx):
        new_x = []
        dup_start = x[idx][0]
        for i, (epoch, data) in enumerate(x):
            if epoch >= dup_start and i < idx:
                continue
            new_x.append((epoch, data))
        return new_x

    def from_tensor(self, x):
        """
            read tensor as image,
            1-dim tensors are displayed as heatmap, see __heatmap__().
            Args:
                x: input tensor.
        """
        if x.shape[0] == 1:
            self.data.append(self.__heatmap__(x))
        else:
            self.data.append(self.__tensor2img__(x))
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

    @staticmethod
    def _clean_cache(target_dir='build/cache'):
        """
            clean cache dir. (can be dangerous)
            Args:
                target_dir:
        """
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        os.mkdir(target_dir)
        print(f'cache dir {target_dir} cleaned.')
