class ConfigCheckFailedException(Exception):
    def __init__(self, msg):
        super().__init__()
        self.msg = msg

    def __str__(self):
        return f'{self.msg}'


class MissingKeyException(ConfigCheckFailedException):
    def __init__(self, key, msg):
        msg = f'Missing key \'{key}\' in config\n({msg})'
        super().__init__(msg)
