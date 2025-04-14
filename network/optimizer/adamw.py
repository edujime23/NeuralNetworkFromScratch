from .base import Optimizer
import numpy as np

class AdamW(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 weight_decay=0.0, gradient_clip=None):
        super().__init__(learning_rate, gradient_clip)
        self.beta1, self.beta2 = beta1, beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay

    def _initialize_param_state(self, param: np.ndarray, name: str, param_id: int):
        if 'm' not in self.param_states[param_id]:
            self.param_states[param_id]['m'] = np.zeros_like(param)
            self.param_states[param_id]['v'] = np.zeros_like(param)

    def _update_single_param(self, param: np.ndarray, grad: np.ndarray, lr: float):
        param_id = id(param)
        state = self.param_states[param_id]
        m = state['m']
        v = state['v']

        # Apply weight decay directly to parameters
        if self.weight_decay != 0:
            param -= lr * self.weight_decay * param

        # Update biased first moment estimate
        m_new = self.beta1 * m + (1 - self.beta1) * grad
        # Update biased second raw moment estimate
        v_new = self.beta2 * v + (1 - self.beta2) * (np.power(grad, 2))

        # Store updated moments
        state['m'] = m_new
        state['v'] = v_new

        # Compute bias-corrected estimates
        m_hat = m_new / (1 - np.power(self.beta1, self.t))
        v_hat = v_new / (1 - np.power(self.beta2, self.t))

        # Calculate update
        update = m_hat / (np.sqrt(v_hat) + self.epsilon)

        # Update parameters
        param -= lr * update