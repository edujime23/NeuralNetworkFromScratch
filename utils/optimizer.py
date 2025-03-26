import numpy as np

class Optimizer:
    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate

    def initialize(self, neuron):
        """Initialize optimizer-specific values in the neuron."""
        raise NotImplementedError("Subclasses must implement initialize method")

    def update(self, neuron, grads):
        """Update the parameters based on gradients."""
        raise NotImplementedError("Subclasses must implement update method")

class AdamOptimizer(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0

    def initialize(self, neuron):
        """Initialize optimizer-specific values in the neuron."""
        neuron.m_weights = np.zeros_like(neuron.weights)
        neuron.v_weights = np.zeros_like(neuron.weights)
        neuron.m_bias = 0.0
        neuron.v_bias = 0.0

    def update(self, neuron, grads):
        if not hasattr(self, 't'):
            self.t = 0
        self.t += 1

        weight_grads = grads[0]
        bias_grad = grads[1]

        # Update weights
        m_w = neuron.m_weights
        v_w = neuron.v_weights
        neuron.m_weights = self.beta1 * m_w + (1 - self.beta1) * weight_grads
        neuron.v_weights = self.beta2 * v_w + (1 - self.beta2) * (weight_grads ** 2)
        m_hat_w = neuron.m_weights / (1 - self.beta1 ** self.t)
        v_hat_w = neuron.v_weights / (1 - self.beta2 ** self.t)
        neuron.weights = neuron.weights - self.learning_rate * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)

        # Update bias
        m_b = neuron.m_bias
        v_b = neuron.v_bias
        neuron.m_bias = self.beta1 * m_b + (1 - self.beta1) * bias_grad
        neuron.v_bias = self.beta2 * v_b + (1 - self.beta2) * (bias_grad ** 2)
        m_hat_b = neuron.m_bias / (1 - self.beta1 ** self.t)
        v_hat_b = neuron.v_bias / (1 - self.beta2 ** self.t)
        neuron.bias = neuron.bias - self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)