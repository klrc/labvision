import time


class Time():
    def __init__(self, format="[%Y-%m-%d %H:%M:%S]"):
        self.format = format

    def asctime(self):
        return time.asctime(time.localtime())

    def now(self):
        return time.strftime(self.format, time.localtime())
