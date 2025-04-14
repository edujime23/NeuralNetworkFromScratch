from .base import Optimizer
import numpy as np


class SGD(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.0, nesterov=False):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.nesterov = nesterov
        
    def _initialize_param_state(self, param: np.ndarray, name: str, param_id: int):
        if 'velocity' not in self.param_states[param_id]:
            self.param_states[param_id]['velocity'] = np.zeros_like(param)
            
    def _update_single_param(self, param: np.ndarray, grad: np.ndarray, lr: float):
        param_id = id(param)
        state = self.param_states[param_id]
        velocity = state['velocity']
        
        # Update velocity with momentum and current gradient
        new_velocity = self.momentum * velocity - lr * grad
        state['velocity'] = new_velocity
        
        if self.nesterov:
            # Nesterov momentum: Apply momentum correction to update
            # Uses the formula: param += momentum * momentum * velocity - (1 + momentum) * lr * grad
            param += self.momentum * new_velocity - lr * grad
        else:
            # Standard momentum update
            param += new_velocity