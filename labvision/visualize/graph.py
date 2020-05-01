from .utils import readlines, twinx, refresh
import matplotlib.pyplot as plt


class Graph():
    def __init__(self, log_fp):
        refresh()
        self.log_fp = log_fp
        self.data = []

    def refresh(self):
        self.data = []
        refresh()

    @staticmethod
    def __smooth__(xs, weight=0.7):
        for i in range(int(10*weight)+1):
            ret = []
            for idx, x in enumerate(xs):
                if idx == 0 or idx == len(xs)-1:
                    ret.append(x)
                else:
                    ret.append((xs[idx-1]+x+xs[idx+1])/3*weight+x*(1-weight))
            xs = ret
        return ret

    def _read_curve(self, _hash, metrics_name):
        for _dict in readlines(fp=self.log_fp, _hash=_hash, target=metrics_name):
            x = float(_dict['epoch'])+float(_dict['iter'])/float(_dict['total_iter'])
            y = float(_dict['value'])
            yield x, y

    def _plot_single_axis(self):
        refresh()
        legends = []
        for d in self.data:
            data_x = d['data_x']
            data_y = d['data_y']
            description = d['description']
            kwargs = d['kwargs']
            legends.append(description['legend'])
            plt.plot(data_x, data_y, **kwargs)
        plt.legend(legends)

    def _plot_twin_axis(self, **kwargs):
        acc_ax, loss_ax = twinx()
        legends = {'acc_ax': [], 'loss_ax': []}
        for d in self.data:
            data_x = d['data_x']
            data_y = d['data_y']
            description = d['description']
            kwargs = d['kwargs']
            if 'loss' in description['metrics_name']:
                legends['loss_ax'].append(description['legend'])
                loss_ax.plot(data_x, data_y, **kwargs)
            elif 'acc' in description['metrics_name']:
                legends['acc_ax'].append(description['legend'])
                acc_ax.plot(data_x, data_y, **kwargs)
        acc_ax.legend(legends['acc_ax'])
        loss_ax.legend(legends['loss_ax'])

    def _plot(self, axis_type):
        if axis_type == 'plt':
            return self._plot_single_axis()
        if axis_type == 'twinx':
            return self._plot_twin_axis()

    def curve(self, _hash, metrics_name, legend=None, smooth=0, **kwargs):
        data = [(x, y) for x, y in self._read_curve(_hash, metrics_name)]
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
            'kwargs': kwargs,
        })
        return self

    def save(self, path, axis_type='twinx', show=True):
        assert axis_type in ['plt', 'twinx']
        self._plot(axis_type)
        if show:
            plt.show()
        plt.savefig(path, pad_inches=0)
        print(f'saved as {path}')
