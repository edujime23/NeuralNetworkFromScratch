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

class AdamOptimizer(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 weight_decay=0.0, use_adamw=True, gradient_clip=None, amsgrad=True,
                 noise_factor=0.0, cyclical_lr=False, lr_cycle_steps=100, 
                 lr_max_factor=2, restart_every=None):
        super().__init__(learning_rate)
        self.beta1, self.beta2 = beta1, beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.use_adamw = use_adamw
        self.gradient_clip = gradient_clip
        self.amsgrad = amsgrad  # AMSGrad variant
        
        # Local minima avoidance parameters
        self.noise_factor = noise_factor  # Add gradient noise to escape flat regions
        self.cyclical_lr = cyclical_lr    # Use cyclical learning rates
        self.lr_cycle_steps = lr_cycle_steps  # Steps per cycle
        self.lr_max_factor = lr_max_factor  # Max multiplier for learning rate
        self.restart_every = restart_every  # Steps for momentum/variance reset
        
        # Tracking for cyclical learning rate
        self.base_lr = learning_rate
        self.cycle_count = 0

    def _initialize_param_state(self, param: np.ndarray, name: str, param_id: int):
        if 'm' not in self.param_states[param_id]:
            self.param_states[param_id]['m'] = np.zeros_like(param, dtype=np.float64)
            self.param_states[param_id]['v'] = np.zeros_like(param, dtype=np.float64)
            if self.amsgrad:
                self.param_states[param_id]['v_max'] = np.zeros_like(param, dtype=np.float64)
            self.param_states[param_id]['last_update'] = np.zeros_like(param, dtype=np.float64)
    
    def _get_cyclical_learning_rate(self, base_lr):
        """Cyclical learning rate calculation"""
        if not self.cyclical_lr:
            return base_lr
        
        # Triangular CLR
        cycle = np.floor(1 + self.t / (2 * self.lr_cycle_steps))
        x = np.abs(self.t / self.lr_cycle_steps - 2 * cycle + 1)
        
        # Scale learning rate between base_lr and base_lr * max_factor
        return base_lr + (self.lr_max_factor * base_lr - base_lr) * np.maximum(0, (1 - x))
    
    def _add_noise_to_gradient(self, grad):
        """Add Gaussian noise to gradient to escape flat regions"""
        if self.noise_factor <= 0:
            return grad
        
        # Calculate noise variance based on current step (decreasing over time)
        noise_variance = self.noise_factor / (1 + self.t)
        noise = np.random.normal(0, noise_variance, grad.shape)
        return grad + noise

    def _update_single_param(self, param: np.ndarray, grad: np.ndarray, lr: float):
        param_id = id(param)
        state = self.param_states[param_id]
        m = state['m']
        v = state['v']

        # Apply gradient clipping if specified
        if self.gradient_clip is not None:
            grad = np.clip(grad, -self.gradient_clip, self.gradient_clip)

        # Add noise to gradient to help escape local minima
        grad = self._add_noise_to_gradient(grad)

        # Get cyclical learning rate
        current_lr = self._get_cyclical_learning_rate(lr)

        # Check if we should restart the optimizer
        if self.restart_every is not None and self.t % self.restart_every == 0 and self.t > 0:
            # Reset momentum and variance estimates
            m = np.zeros_like(param, dtype=np.float64)
            v = np.zeros_like(param, dtype=np.float64)
            if self.amsgrad:
                state['v_max'] = np.zeros_like(param, dtype=np.float64)
            state['m'] = m
            state['v'] = v

        if self.use_adamw:
            # AdamW: Apply weight decay directly to parameters
            m_new = self.beta1 * m + (1 - self.beta1) * grad
            v_new = self.beta2 * v + (1 - self.beta2) * (np.power(grad, 2))

            # Update biased first and second moment estimates
            state['m'] = m_new
            state['v'] = v_new

            # Compute bias-corrected estimates
            m_hat = m_new / (1 - np.power(self.beta1, self.t))
            v_hat = v_new / (1 - np.power(self.beta2, self.t))

            # AMSGrad variant
            denom = self._amsgrad(state, v_hat)

            # Calculate update
            update = m_hat / denom + self.weight_decay * param

        else:
            # Standard Adam with separate weight decay
            if self.weight_decay != 0:
                grad_with_wd = grad + self.weight_decay * param
            else:
                grad_with_wd = grad

            # Update biased first moment estimate
            m_new = self.beta1 * m + (1 - self.beta1) * grad_with_wd
            # Update biased second raw moment estimate
            v_new = self.beta2 * v + (1 - self.beta2) * (np.power(grad_with_wd, 2))

            # Store updated moments
            state['m'] = m_new
            state['v'] = v_new

            # Compute bias-corrected estimates
            m_hat = m_new / (1 - np.power(self.beta1, self.t))
            v_hat = v_new / (1 - np.power(self.beta2, self.t))

            # AMSGrad variant
            denom = self._amsgrad(state, v_hat)

            # Calculate update
            update = m_hat / denom


        # Store the last update for debugging purposes
        state['last_update'] = current_lr * update

        # AdamW update
        param -= current_lr * update

    def _amsgrad(self, state, v_hat):
        if not self.amsgrad:
            return np.sqrt(v_hat) + self.epsilon

        v_max = np.maximum(state.get('v_max', np.zeros_like(v_hat)), v_hat)
        state['v_max'] = v_max
        return np.sqrt(v_max) + self.epsilon

class SGD(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.0, nesterov=False):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.nesterov = nesterov
        
    def _initialize_param_state(self, param: np.ndarray, name: str, param_id: int):
        if 'velocity' not in self.param_states[param_id]:
            self.param_states[param_id]['velocity'] = np.zeros_like(param, dtype=np.float64)
            
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