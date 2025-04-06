from .base import Optimizer
import numpy as np

class RMSprop(Optimizer):
    def __init__(self, learning_rate=0.001, rho=0.9, epsilon=1e-8, weight_decay=0.0, gradient_clip=None):
        super().__init__(learning_rate, gradient_clip)
        self.rho = rho
        self.epsilon = epsilon
        self.weight_decay = weight_decay

    def _initialize_param_state(self, param: np.ndarray, name: str, param_id: int):
        if 'square_avg' not in self.param_states[param_id]:
            self.param_states[param_id]['square_avg'] = np.zeros_like(param, dtype=np.float64)

    def _update_single_param(self, param: np.ndarray, grad: np.ndarray, lr: float):
        param_id = id(param)
        state = self.param_states[param_id]
        square_avg = state['square_avg']

        # Apply gradient clipping if specified
        if self.gradient_clip is not None:
            grad = np.clip(grad, -self.gradient_clip, self.gradient_clip)

        # Update the exponentially weighted average of squared gradients
        square_avg_new = self.rho * square_avg + (1 - self.rho) * np.square(grad)
        state['square_avg'] = square_avg_new

        # Calculate the update
        denom = np.sqrt(square_avg_new) + self.epsilon
        update = lr * grad / denom

        # Apply weight decay if specified
        if self.weight_decay != 0:
            param -= update + self.weight_decay * param
        else:
            param -= update