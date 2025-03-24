import math
import random
from typing import List

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
        self.signals = [result if (result := neuron.activate(inputs, self.threshold)) is not None else 0.0 for neuron in self.neurons]
        
        if training:
            self.signals = self._dropout(self.signals)
            if self.batch_norm:
                self.signals = self._batch_norm(self.signals)
                
        return self.signals

    def _dropout(self, signals: List[float]) -> List[float]:
        """Apply dropout to the signals."""
        if self.dropout_rate > 0:
            # Generate the mask in one step, then apply
            mask = [1 if random.random() <= self.dropout_rate else 0 for _ in signals]
            return [signal * mask_val for signal, mask_val in zip(signals, mask)]
        return signals

    def _batch_norm(self, signals: List[float]) -> List[float]:
        """Apply batch normalization to the signals."""
        if self.mean is None:
            self.mean = [sum(signals) / len(signals)] * len(signals)
            variance_sum = sum((signal - self.mean[0]) ** 2 for signal in signals)
            self.variance = [variance_sum / len(signals)] * len(signals)
        
        normalized = [(signal - self.mean[0]) / math.sqrt(self.variance[0] + self.epsilon) for signal in signals]
        
        return [gamma * norm + beta for gamma, norm, beta in zip(self.gamma, normalized, self.beta)]

    def _update_weights_and_biases(self, prev_layer_neurons: List[Neuron], learning_rate: float, beta1: float, beta2: float, epsilon: float, lambda_l1: float, lambda_l2: float, gradient_clip: float):
        for neuron in self.neurons:
            neuron.time_step += 1
            gradients = [neuron.error_signal * prev_neuron.signal for prev_neuron in prev_layer_neurons]
            gradients = [max(min(grad, gradient_clip), -gradient_clip) for grad in gradients]
            neuron.gradients = gradients

            for k in range(len(neuron.weights)):
                # Update using Adam optimizer
                neuron.first_moment[k] = beta1 * neuron.first_moment[k] + (1 - beta1) * gradients[k]
                neuron.second_moment[k] = beta2 * neuron.second_moment[k] + (1 - beta2) * (gradients[k] ** 2)
                
                corrected_first = neuron.first_moment[k] / (1 - (beta1 ** neuron.time_step))
                corrected_second = neuron.second_moment[k] / (1 - (beta2 ** neuron.time_step))
                
                adam_update = learning_rate * corrected_first / (math.sqrt(corrected_second) + epsilon)
                
                # L1 and L2 regularization
                l1_reg = lambda_l1 * math.copysign(1, neuron.weights[k])
                l2_reg = lambda_l2 * neuron.weights[k]
                neuron.weights[k] -= adam_update + l1_reg + l2_reg

            # Update bias using Adam optimizer
            neuron.first_moment_bias = beta1 * neuron.first_moment_bias + (1 - beta1) * neuron.error_signal
            neuron.second_moment_bias = beta2 * neuron.second_moment_bias + (1 - beta2) * (neuron.error_signal ** 2)
            corrected_first_bias = neuron.first_moment_bias / (1 - (beta1 ** neuron.time_step))
            corrected_second_bias = neuron.second_moment_bias / (1 - (beta2 ** neuron.time_step))
            adam_bias_update = learning_rate * corrected_first_bias / (math.sqrt(corrected_second_bias) + epsilon)
            
            # L1 and L2 regularization on bias
            l1_bias_reg = lambda_l1 * math.copysign(1, neuron.bias)
            l2_bias_reg = lambda_l2 * neuron.bias
            neuron.bias -= adam_bias_update + l1_bias_reg + l2_bias_reg

            # Clip error signal
            neuron.error_signal = max(min(neuron.error_signal, gradient_clip), -gradient_clip)


class OutputLayer(DenseLayer):
    def __init__(self, num_neurons: int, num_inputs: int, activation_function):
        """Initialize an output layer with the given parameters."""
        self.neurons = [Neuron(num_inputs, activation_function) for _ in range(num_neurons)]

    def activate(self, inputs: List[float], *args, **kwargs) -> List[float]:
        """Activate the output layer with the given inputs."""
        self.signals = [result if (result := neuron.activate(inputs, 1)) is not None else 0.0 for neuron in self.neurons]

        return self.signals