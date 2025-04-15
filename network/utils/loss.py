import numpy as np
from typing import Callable

class Loss:
    def __init__(self, loss_function: Callable):
        self.loss_function = loss_function

    def __call__(self, predicted: np.ndarray, actual: np.ndarray) -> float:
        return self.loss_function(predicted, actual)