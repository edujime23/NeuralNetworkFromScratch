import numpy as np
from typing import Tuple, Dict, List

class Optimizer:
    def __init__(self, learning_rate=0.001, gradient_clip=1):
        self.learning_rate = self._resolve_learning_rate(learning_rate)
        self.t = 0
        self.param_states: Dict[int, Dict[str, np.ndarray]] = {}
        self.gradient_clip = gradient_clip

    def _resolve_learning_rate(self, learning_rate):
        return learning_rate() if callable(learning_rate) else learning_rate


    def register_parameter(self, param: np.ndarray, name: str):
        """Registers a parameter with the optimizer."""
        param_id = id(param)
        if param_id not in self.param_states:
            self.param_states[param_id] = {}
        self._initialize_param_state(param, name, param_id)

    def _initialize_param_state(self, param: np.ndarray, name: str, param_id: int):
        """Initializes optimizer-specific values for a parameter."""
        raise NotImplementedError("Subclasses must implement _initialize_param_state")

    def update(self, params_and_grads: List[Tuple[np.ndarray, np.ndarray]]):
        """Updates the parameters based on their gradients."""
        self.t += 1
        lr = self.learning_rate
        for param, grad in params_and_grads:
            self._update_single_param(param, grad, lr)

    def _update_single_param(self, param: np.ndarray, grad: np.ndarray, lr: float):
        """Updates a single parameter."""
        raise NotImplementedError("Subclasses must implement _update_single_param")

class AdamOptimizer(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-3, weight_decay=0.0):
        super().__init__(learning_rate)
        self.beta1, self.beta2 = beta1, beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay

    def _initialize_param_state(self, param: np.ndarray, name: str, param_id: int):
        if 'm' not in self.param_states[param_id]:
            self.param_states[param_id]['m'] = np.zeros_like(param, dtype=np.float32)
            self.param_states[param_id]['v'] = np.zeros_like(param, dtype=np.float32)

    def _update_single_param(self, param: np.ndarray, grad: np.ndarray, lr: float):
        param_id = id(param)
        state = self.param_states[param_id]
        m = state['m']
        v = state['v']
        
        
        if self.gradient_clip is not None:
            grad = np.clip(grad, -self.gradient_clip, self.gradient_clip)

        grad = grad + self.weight_decay * param if self.weight_decay != 0 else grad

        m_new = self.beta1 * m + (1 - self.beta1) * grad
        v_new = self.beta2 * v + (1 - self.beta2) * (grad ** 2)

        m_hat = m_new / (1 - self.beta1 ** self.t)
        v_hat = v_new / (1 - self.beta2 ** self.t)

        state['m'] = m_new
        state['v'] = v_new

        param -= lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

class SGD(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.0, nesterov=False):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.nesterov = nesterov

    def _initialize_param_state(self, param: np.ndarray, name: str, param_id: int):
        if 'velocity' not in self.param_states[param_id]:
            self.param_states[param_id]['velocity'] = np.zeros_like(param, dtype=np.float32)

    def _update_single_param(self, param: np.ndarray, grad: np.ndarray, lr: float):
        param_id = id(param)
        state = self.param_states[param_id]
        velocity = state['velocity']
        new_velocity = self.momentum * velocity - lr * grad

        if self.nesterov:
            param += self.momentum * new_velocity - lr * grad
        else:
            param += new_velocity
        state['velocity'] = new_velocity