import numpy as np
from typing import Tuple, Dict, List, Callable, Union

class Optimizer:
    def __init__(self, learning_rate=0.001, gradient_clip=None):
        self.learning_rate = self._resolve_learning_rate(learning_rate)
        self.t = 0
        self.param_states: Dict[int, Dict[str, np.ndarray]] = {}
        self.gradient_clip = gradient_clip

    def _resolve_learning_rate(self, learning_rate: Union[float, Callable[[], float]]) -> Union[float, Callable[[], float]]:
        return learning_rate() if callable(learning_rate) else learning_rate

    def register_parameter(self, param: np.ndarray, name: str):
        """Registers a parameter with the optimizer."""
        if (param_id := id(param)) not in self.param_states:
            self.param_states[param_id] = {}
            self._initialize_param_state(param, name, param_id)

    def _initialize_param_state(self, param: np.ndarray, name: str, param_id: int):
        """Initializes optimizer-specific values for a parameter."""
        raise NotImplementedError("Subclasses must implement _initialize_param_state")

    def update(self, params_and_grads: List[Tuple[np.ndarray, np.ndarray]]):
        """Updates the parameters based on their gradients."""
        self.t += 1
        lr = self.learning_rate() if callable(self.learning_rate) else self.learning_rate

        for param, grad in params_and_grads:
            param_id = id(param)
            # Check if parameter is registered
            if param_id not in self.param_states:
                self.register_parameter(param, f'param_{param_id}')

            if self.gradient_clip is not None:
                grad = np.clip(grad, -self.gradient_clip, self.gradient_clip)
            self._update_single_param(param, grad, lr)

    def _update_single_param(self, param: np.ndarray, grad: np.ndarray, lr: float):
        """Updates a single parameter."""
        raise NotImplementedError("Subclasses must implement _update_single_param")