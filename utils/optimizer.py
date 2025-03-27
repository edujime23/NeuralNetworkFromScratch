import numpy as np

class Optimizer:
    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate
        self.t = 0

    def get_learning_rate(self):
        return self.learning_rate() if callable(self.learning_rate) else self.learning_rate

    def initialize(self, obj, param_name):
        """Initialize optimizer-specific values for parameters."""
        param = getattr(obj, param_name)
        
        def create_moment_arrays(p):
            return np.zeros_like(p, dtype=np.float32)
        
        if isinstance(param, np.ndarray):
            setattr(obj, f'm_{param_name}', create_moment_arrays(param))
            setattr(obj, f'v_{param_name}', create_moment_arrays(param))
        elif isinstance(param, (int, float)):
            setattr(obj, f'm_{param_name}', 0.0)
            setattr(obj, f'v_{param_name}', 0.0)
        elif isinstance(param, list):
            for i, p in enumerate(param):
                if not isinstance(p, np.ndarray):
                    raise TypeError(f"List parameter '{param_name}' at index {i} must be a NumPy array.")
                setattr(obj, f'm_{param_name}_{i}', create_moment_arrays(p))
                setattr(obj, f'v_{param_name}_{i}', create_moment_arrays(p))
        else:
            raise TypeError(f"Unsupported parameter type for '{param_name}': {type(param)}")

class AdamOptimizer(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0):
        super().__init__(learning_rate)
        self.beta1, self.beta2 = beta1, beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay

    def update(self, obj, param_name, grads):
        self.t += 1
        lr = self.get_learning_rate()
        param = getattr(obj, param_name)

        def _update_single_param(p, grad, m_name, v_name):
            m = getattr(obj, m_name)
            v = getattr(obj, v_name)

            grad = grad + self.weight_decay * p if self.weight_decay > 0 else grad

            m_new = self.beta1 * m + (1 - self.beta1) * grad
            v_new = self.beta2 * v + (1 - self.beta2) * (grad ** 2)

            m_hat = m_new / (1 - self.beta1 ** self.t)
            v_hat = v_new / (1 - self.beta2 ** self.t)

            setattr(obj, m_name, m_new)
            setattr(obj, v_name, v_new)

            return p - lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

        if isinstance(param, np.ndarray):
            m_name, v_name = f'm_{param_name}', f'v_{param_name}'
            updated_param = _update_single_param(param, grads, m_name, v_name)
            setattr(obj, param_name, updated_param.astype(param.dtype, copy=False))
            return updated_param

        elif isinstance(param, (int, float)):
            m_name, v_name = f'm_{param_name}', f'v_{param_name}'
            updated_param = _update_single_param(param, grads, m_name, v_name)
            setattr(obj, param_name, updated_param)
            return updated_param

        elif isinstance(param, list):
            updated_params = []
            for i, p in enumerate(param):
                grad = grads[i]
                m_name, v_name = f'm_{param_name}_{i}', f'v_{param_name}_{i}'
                updated_param = _update_single_param(p, grad, m_name, v_name)
                updated_params.append(updated_param.astype(p.dtype, copy=False))
            setattr(obj, param_name, updated_params)
            return updated_params

        raise TypeError(f"Unsupported parameter type for update: {type(param)}")

class SGD(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.0, nesterov=False):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.nesterov = nesterov

    def initialize(self, obj, param_name):
        super().initialize(obj, param_name)
        param = getattr(obj, param_name)
        
        def create_velocity_arrays(p):
            return np.zeros_like(p, dtype=np.float32)
        
        if isinstance(param, np.ndarray):
            setattr(obj, f'velocity_{param_name}', create_velocity_arrays(param))
        elif isinstance(param, (int, float)):
            setattr(obj, f'velocity_{param_name}', 0.0)
        elif isinstance(param, list):
            for i, p in enumerate(param):
                setattr(obj, f'velocity_{param_name}_{i}', create_velocity_arrays(p))

    def update(self, obj, param_name, grads):
        lr = self.get_learning_rate()
        param = getattr(obj, param_name)

        def _update_single_param(p, grad, velocity_name):
            velocity = getattr(obj, velocity_name)
            new_velocity = self.momentum * velocity - lr * grad
            
            if self.nesterov:
                updated_param = p + self.momentum * new_velocity - lr * grad
            else:
                updated_param = p + new_velocity

            setattr(obj, velocity_name, new_velocity)
            return updated_param

        if isinstance(param, np.ndarray):
            velocity_name = f'velocity_{param_name}'
            updated_param = _update_single_param(param, grads, velocity_name)
            setattr(obj, param_name, updated_param.astype(param.dtype, copy=False))
            return updated_param

        elif isinstance(param, (int, float)):
            velocity_name = f'velocity_{param_name}'
            updated_param = _update_single_param(param, grads, velocity_name)
            setattr(obj, param_name, updated_param)
            return updated_param

        elif isinstance(param, list):
            updated_params = []
            for i, p in enumerate(param):
                grad = grads[i]
                velocity_name = f'velocity_{param_name}_{i}'
                updated_param = _update_single_param(p, grad, velocity_name)
                updated_params.append(updated_param.astype(p.dtype, copy=False))
            setattr(obj, param_name, updated_params)
            return updated_params

        raise TypeError(f"Unsupported parameter type for update: {type(param)}")