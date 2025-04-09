import numpy as np

class Metrics:
    @staticmethod
    def accuracy(predicted: np.ndarray, actual: np.ndarray) -> float:
        return np.mean(np.round(predicted) == actual)
    
    # Future metrics to be implemented...