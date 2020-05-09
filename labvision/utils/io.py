
import time


class PrintLogger():
    def __call__(self, msg):
        print(msg)


class TimerLogger():
    def __call__(self, msg):
        msg_head = f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}]'
        line = f'{msg_head}\t{msg}'
        # with open(f'{utils.config.project}/server.log', 'a') as f:
        #     f.write(f'{line}\n')
        print(line)
        return line
