

class Logger():
    def __init__(self, fp=None):
        self.fp = fp
        self.test_mode = False

    def __call__(self, msg):
        if self.test_mode:
            msg = f'testing# {msg}'
        print(msg)
        if self.fp is not None:
            with open(self.fp, 'a') as f:
                f.write(f'{msg}\n')
