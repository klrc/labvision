from . import functional


class Smooth():
    def __init__(self, weight):
        self.weight = weight

    def __call__(self, inputs):
        return functional.smooth(
            inputs=inputs,
            weight=self.weight,
        )
