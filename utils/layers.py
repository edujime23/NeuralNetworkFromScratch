import math
import random
from typing import List, Optional

from . import Neuron

class Layer:
    def __init__(self):
        """Base class for all layers."""
        pass

    def activate(self, inputs: List[float], training: bool = False) -> List[float]:
        """Activate the layer with the given inputs."""
        raise NotImplementedError("Subclasses must implement activate method")

    def _update_weights_and_biases(self, prev_layer, learning_rate: float, beta1: float, beta2: float, epsilon: float, lambda_l1: float, lambda_l2: float, gradient_clip: float):
        """Update weights and biases of the layer."""
        raise NotImplementedError("Subclasses must implement _update_weights_and_biases method")

class InputLayer(Layer):
    def __init__(self, num_neurons: int):
        """Initialize an input layer with the given number of neurons."""
        super().__init__()
        self.neurons = [Neuron(0) for _ in range(num_neurons)]

class DenseLayer(Layer):
    def __init__(self, num_neurons: int, num_inputs: int, activation_function, dropout_rate: float = 0.0, batch_norm: bool = False, threshold: float = 1.0):
        """Initialize a dense layer with the given parameters."""
        super().__init__()
        self.threshold = threshold
        self.neurons = [Neuron(num_inputs, activation_function) for _ in range(num_neurons)]
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.gamma = [1.0] * num_neurons if batch_norm else None
        self.beta = [0.0] * num_neurons if batch_norm else None
        self.mean = [0.0] * num_neurons if batch_norm else None
        self.variance = [1.0] * num_neurons if batch_norm else None
        self.epsilon = 1e-5
        self.signals = []

    def activate(self, inputs: List[float], training: bool = False) -> List[float]:
        """Activate the dense layer with the given inputs."""
        self.signals = [neuron.activate(inputs, self.threshold) for neuron in self.neurons]
        for i, signal in enumerate(self.signals):
            if signal is None:
                self.signals[i] = 0.0
        if training:
            self.signals = self._dropout(self.signals)
            if self.batch_norm:
                self.signals = self._batch_norm(self.signals)
        return self.signals

    def _dropout(self, signals: List[float]) -> List[float]:
        """Apply dropout to the signals."""
        if self.dropout_rate > 0:
            self.mask = [int(random.random() <= self.dropout_rate) for _ in signals]
        else:
            self.mask = [1] * len(signals)
        return [signal * mask_val for signal, mask_val in zip(signals, self.mask)]

    def _batch_norm(self, signals: List[float]) -> List[float]:
        """Apply batch normalization to the signals."""
        if self.mean is None:
            self.mean = [sum(signals) / len(signals)] * len(signals)
            variance_sum = sum((signal - mean) ** 2 for signal, mean in zip(signals, self.mean))
            self.variance = [variance_sum / len(signals)] * len(signals)

        self.normalized = [(signal - mean) / math.sqrt(variance + self.epsilon) for signal, mean, variance in zip(signals, self.mean, self.variance)]

        for i, norm in enumerate(self.normalized):
            if norm is None or math.isnan(norm):
                self.normalized[i] = 0.0

        return [gamma * norm + beta for gamma, norm, beta in zip(self.gamma, self.normalized, self.beta)]

    def _update_weights_and_biases(self, prev_layer_neurons: List[Neuron], learning_rate: float, beta1: float, beta2: float, epsilon: float, lambda_l1: float, lambda_l2: float, gradient_clip: float):
        """Update weights and biases of the dense layer."""
        for neuron in self.neurons:
            neuron.time_step += 1
            gradients = [neuron.error_signal * prev_neuron.signal for prev_neuron in prev_layer_neurons]
            gradients = [max(min(grad, gradient_clip), -gradient_clip) for grad in gradients]

            for k in range(len(neuron.weights)):
                neuron.first_moment[k] = beta1 * neuron.first_moment[k] + (1 - beta1) * gradients[k]
                neuron.second_moment[k] = beta2 * neuron.second_moment[k] + (1 - beta2) * (gradients[k] ** 2)
                corrected_first = neuron.first_moment[k] / (1 - (beta1 ** neuron.time_step))
                corrected_second = neuron.second_moment[k] / (1 - (beta2 ** neuron.time_step))
                adam_update = learning_rate * corrected_first / (math.sqrt(corrected_second) + epsilon)
                l1_reg = lambda_l1 * math.copysign(1, neuron.weights[k])
                l2_reg = lambda_l2 * neuron.weights[k]
                neuron.weights[k] -= adam_update + l1_reg + l2_reg

            neuron.first_moment_bias = beta1 * neuron.first_moment_bias + (1 - beta1) * neuron.error_signal
            neuron.second_moment_bias = beta2 * neuron.second_moment_bias + (1 - beta2) * (neuron.error_signal ** 2)
            corrected_first_bias = neuron.first_moment_bias / (1 - (beta1 ** neuron.time_step))
            corrected_second_bias = neuron.second_moment_bias / (1 - (beta2 ** neuron.time_step))
            adam_bias_update = learning_rate * corrected_first_bias / (math.sqrt(corrected_second_bias) + epsilon)
            l1_bias_reg = lambda_l1 * math.copysign(1, neuron.bias)
            l2_bias_reg = lambda_l2 * neuron.bias
            neuron.bias -= adam_bias_update + l1_bias_reg + l2_bias_reg
            neuron.error_signal = max(min(neuron.error_signal, gradient_clip), -gradient_clip)

        if self.batch_norm:
            self.normalized = [(neuron.signal - self.mean[i]) / math.sqrt(self.variance[i] + self.epsilon) for i, neuron in enumerate(self.neurons)]
            gamma_gradients = [sum(neuron.error_signal * norm for neuron, norm in zip(self.neurons, normalized)) / len(self.neurons) for normalized in [self.normalized]]
            beta_gradients = [sum(neuron.error_signal for neuron in self.neurons) / len(self.neurons)]
            self.gamma = [gamma - learning_rate * gamma_grad for gamma, gamma_grad in zip(self.gamma, gamma_gradients)]
            self.beta = [beta - learning_rate * beta_grad for beta, beta_grad in zip(self.beta, beta_gradients)]