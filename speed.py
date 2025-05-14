import numpy as np


class SpeedRatio:
    def __init__(self, step: float = 0.001):
        self.index = 0.0
        self.step = step
        self.paused = False

    @property
    def value(self) -> float:
        if self.paused:
            return 0.0
        return sigmoid(self.index) * 2

    def reset(self) -> None:
        self.index = 0.0

    def update(self) -> None:
        self.index += self.step

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False


class Speed:
    def __init__(self, initial_value: float, ratio: SpeedRatio):
        self.initial_value = initial_value
        self.ratio = ratio

    @property
    def value(self) -> float:
        return self.initial_value * self.ratio.value


def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))
