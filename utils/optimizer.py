import numpy as np

class Optimizer:
    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate
        self.t = 0

    def initialize(self, obj, param_name):
        """Initialize optimizer-specific values for the parameter in the object."""
        if not hasattr(obj, param_name):
            raise AttributeError(f"Object '{obj.__class__.__name__}' does not have attribute '{param_name}'")
        param = getattr(obj, param_name)
        if isinstance(param, np.ndarray):
            setattr(obj, f'm_{param_name}', np.zeros_like(param, dtype=np.float32))
            setattr(obj, f'v_{param_name}', np.zeros_like(param, dtype=np.float32))
        elif isinstance(param, (int, float)):
            setattr(obj, f'm_{param_name}', 0.0)
            setattr(obj, f'v_{param_name}', 0.0)
        elif isinstance(param, list): # For biases which might be lists of NumPy arrays
            for i, p in enumerate(param):
                if not isinstance(p, np.ndarray):
                    raise TypeError(f"List parameter '{param_name}' at index {i} is not a NumPy array.")
                setattr(obj, f'm_{param_name}_{i}', np.zeros_like(p, dtype=np.float32))
                setattr(obj, f'v_{param_name}_{i}', np.zeros_like(p, dtype=np.float32))
        else:
            raise TypeError(f"Unsupported parameter type for '{param_name}': {type(param)}")

    def update(self, obj, param_name, grads):
        """Update the parameters based on gradients."""
        raise NotImplementedError("Subclasses must implement update method")

class AdamOptimizer(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon # Small constant for numerical stability

    def update(self, obj, param_name, grads):
        self.t += 1
        param = getattr(obj, param_name)

        if param is None:
            raise ValueError(f"Parameter '{param_name}' in object '{obj.__class__.__name__}' is None.")

        if isinstance(param, np.ndarray):
            m_name = f'm_{param_name}'
            v_name = f'v_{param_name}'
            m = getattr(obj, m_name)
            v = getattr(obj, v_name)

            if grads.shape != param.shape:
                raise ValueError(f"Shape mismatch for gradients of '{param_name}' in '{obj.__class__.__name__}'. Expected {param.shape}, got {grads.shape}.")

            m_new = self.beta1 * m + (1 - self.beta1) * grads
            v_new = self.beta2 * v + (1 - self.beta2) * (grads ** 2)

            m_hat = m_new / (1 - self.beta1 ** self.t)
            v_hat = v_new / (1 - self.beta2 ** self.t)

            setattr(obj, m_name, m_new)
            setattr(obj, v_name, v_new)
            updated_param = param - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            setattr(obj, param_name, updated_param.astype(param.dtype, copy=False)) # Ensure updated param has the same dtype
            return updated_param
        elif isinstance(param, (int, float)):
            m_name = f'm_{param_name}'
            v_name = f'v_{param_name}'
            m = getattr(obj, m_name)
            v = getattr(obj, v_name)

            m_new = self.beta1 * m + (1 - self.beta1) * grads
            v_new = self.beta2 * v + (1 - self.beta2) * (grads ** 2)

            m_hat = m_new / (1 - self.beta1 ** self.t)
            v_hat = v_new / (1 - self.beta2 ** self.t)

            setattr(obj, m_name, m_new)
            setattr(obj, v_name, v_new)
            updated_param = param - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            setattr(obj, param_name, updated_param)
            return updated_param
        elif isinstance(param, list): # Assuming biases might be lists of NumPy arrays
            updated_params = []
            for i, p in enumerate(param):
                if not isinstance(grads, list) or i >= len(grads):
                    raise ValueError(f"Gradients for list parameter '{param_name}' are not provided as a list or are shorter than the parameter list.")
                grad = grads[i]
                m_name = f'm_{param_name}_{i}'
                v_name = f'v_{param_name}_{i}'
                m = getattr(obj, m_name)
                v = getattr(obj, v_name)

                if grad.shape != p.shape:
                    raise ValueError(f"Shape mismatch for gradient at index {i} of '{param_name}' in '{obj.__class__.__name__}'. Expected {p.shape}, got {grad.shape}.")

                m_new = self.beta1 * m + (1 - self.beta1) * grad
                v_new = self.beta2 * v + (1 - self.beta2) * (grad ** 2)

                m_hat = m_new / (1 - self.beta1 ** self.t)
                v_hat = v_new / (1 - self.beta2 ** self.t)

                setattr(obj, m_name, m_new)
                setattr(obj, v_name, v_new)
                updated_param = p - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
                updated_params.append(updated_param.astype(p.dtype, copy=False))
            setattr(obj, param_name, updated_params)
            return updated_params
        else:
            raise TypeError(f"Unsupported parameter type for update: {type(param)}")