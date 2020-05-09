
import time


class PrintLogger():
    def __call__(self, msg):
        print(msg)


class TimerLogger():
    def __call__(self, msg, save_fp=None):
        msg_head = f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}]'
        line = f'{msg_head}{msg}'
        if save_fp:
            with open(save_fp, 'a') as f:
                f.write(f'{line}\n')
        print(line)
        return line


log = TimerLogger()
