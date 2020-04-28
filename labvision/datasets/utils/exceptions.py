class DatasetNotFoundException(Exception):
    def __init__(self, fp):
        super().__init__()
        self.fp = fp

    def __str__(self):
        # TODO: add download src
        return f'Dataset not found in \'{self.fp}\', you can download from ?.'
