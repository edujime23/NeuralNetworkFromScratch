import math
import random
import pickle

# Cost Functions
class CostFunctions:
    @staticmethod
    def mean_squared_error(predicted, actual):
        return sum((p - a) ** 2 for p, a in zip(predicted, actual)) / len(predicted)

    @staticmethod
    def cross_entropy(predicted, actual):
        epsilon = 1e-8  # Prevents log(0) issues
        return -sum(a * math.log(p) + (1 - a) * math.log(1 - p) for a, p in zip(actual, predicted)) / len(predicted)


# Activation Functions and Their Derivatives
class ActivationFunctions:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def relu(x):
        return max(0, x)

    @staticmethod
    def leaky_relu(x):
        return x if x > 0 else 0.01 * x


# Neuron class with Adam optimizer parameters
class Neuron:
    def __init__(self, num_inputs, activation_function):
        self.activation_function = activation_function
        self.delta = 0
        self.weights = []
        self.bias = 0
        self.signal = 0
        self.error_signal = 0

        # Adam optimizer values
        self.first_moment = []
        self.second_moment = []
        self.first_moment_bias = 0
        self.second_moment_bias = 0
        self.time_step = 0

        if num_inputs >= 1:
            # Improved weight initialization
            if activation_function in [
                ActivationFunctions.relu,
                ActivationFunctions.leaky_relu,
            ]:
                std = math.sqrt(2.0 / num_inputs)  # He initialization for ReLU
            else:
                std = math.sqrt(1.0 / num_inputs)  # Xavier initialization for sigmoid/tanh

            self.weights = [random.gauss(0, std) for _ in range(num_inputs)]
            self.bias = random.gauss(0, std)
            self.first_moment = [0.0 for _ in range(num_inputs)]
            self.second_moment = [0.0 for _ in range(num_inputs)]

    def activate(self, inputs, threshold):
        signal_sum = self.bias + sum(w * i for w, i in zip(self.weights, inputs))
        self.signal = self.activation_function(signal_sum / threshold)
        if math.isnan(self.signal): # Added check to prevent NaN and None values.
            self.signal = 0.0

class Layer:
    def __init__(self, num_neurons, num_inputs, activation_function, dropout_rate=0.0, batch_norm=False, threshold=1.0):
        self.threshold = threshold
        self.neurons = [Neuron(num_inputs, activation_function) for _ in range(num_neurons)]
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.gamma = [1.0] * num_neurons if batch_norm else None
        self.beta = [0.0] * num_neurons if batch_norm else None
        self.mean = [0.0] * num_neurons if batch_norm else None
        self.variance = [1.0] * num_neurons if batch_norm else None
        self.epsilon = 1e-5
        self.signals = [] # store the activations

    def activate(self, inputs, training=False):
        self.signals = [neuron.activate(inputs, self.threshold) for neuron in self.neurons]
        for i, signal in enumerate(self.signals): # added check to prevent None values.
            if signal is None:
                self.signals[i] = 0.0
        if training:
            self.signals = self._dropout(self.signals)
            if self.batch_norm:
                self.signals = self._batch_norm(self.signals)
        return self.signals

    def _dropout(self, signals):
        if self.dropout_rate > 0:
            self.mask = [int(random.random() > self.dropout_rate) for _ in signals]
        else:
            self.mask = [1] * len(signals)
        return [signal * mask_val for signal, mask_val in zip(signals, self.mask)]

    def _batch_norm(self, signals):
        if self.mean is None:  # Initialize during the first pass
            self.mean = [sum(signals) / len(signals)] * len(signals)
            variance_sum = sum((signal - mean) ** 2 for signal, mean in zip(signals, self.mean))
            self.variance = [variance_sum / len(signals)] * len(signals)

        normalized = [(signal - mean) / math.sqrt(variance + self.epsilon) for signal, mean, variance in zip(signals, self.mean, self.variance)]

        # Check for None or NaN values in normalized
        for i, norm in enumerate(normalized):
            if norm is None or math.isnan(norm):
                normalized[i] = 0.0  # Replace with 0.0

        return [gamma * norm + beta for gamma, norm, beta in zip(self.gamma, normalized, self.beta)]

    def _update_weights_and_biases(self, prev_layer_neurons, learning_rate, beta1, beta2, epsilon, lambda_l1, lambda_l2):
        for neuron in self.neurons:
            neuron.time_step += 1
            gradients = [neuron.error_signal * prev_neuron.signal for prev_neuron in prev_layer_neurons]

            for k in range(len(neuron.weights)):
                # Adam update
                neuron.first_moment[k] = beta1 * neuron.first_moment[k] + (1 - beta1) * gradients[k]
                neuron.second_moment[k] = beta2 * neuron.second_moment[k] + (1 - beta2) * (gradients[k] ** 2)
                corrected_first = neuron.first_moment[k] / (1 - (beta1 ** neuron.time_step))
                corrected_second = neuron.second_moment[k] / (1 - (beta2 ** neuron.time_step))
                adam_update = learning_rate * corrected_first / (math.sqrt(corrected_second) + epsilon)

                # Regularization
                l1_reg = lambda_l1 * math.copysign(1, neuron.weights[k])
                l2_reg = lambda_l2 * neuron.weights[k]

                # Combined update
                neuron.weights[k] -= adam_update + l1_reg + l2_reg

            # Bias update (similar to weights)
            neuron.first_moment_bias = beta1 * neuron.first_moment_bias + (1 - beta1) * neuron.error_signal
            neuron.second_moment_bias = beta2 * neuron.second_moment_bias + (1 - beta2) * (neuron.error_signal ** 2)
            corrected_first_bias = neuron.first_moment_bias / (1 - (beta1 ** neuron.time_step))
            corrected_second_bias = neuron.second_moment_bias / (1 - (beta2 ** neuron.time_step))
            adam_bias_update = learning_rate * corrected_first_bias / (math.sqrt(corrected_second_bias) + epsilon)

            l1_bias_reg = lambda_l1 * math.copysign(1, neuron.bias)
            l2_bias_reg = lambda_l2 * neuron.bias

            neuron.bias -= adam_bias_update + l1_bias_reg + l2_bias_reg

        if self.batch_norm:
            # Calculate gradients for gamma and beta
            normalized = [(neuron.signal - self.mean[i]) / math.sqrt(self.variance[i] + self.epsilon) for i, neuron in enumerate(self.neurons)] # Recalculate normalized.
            gamma_gradients = [sum(neuron.error_signal * norm for neuron, norm in zip(self.neurons, normalized)) / len(self.neurons) for normalized in [normalized]]
            beta_gradients = [sum(neuron.error_signal for neuron in self.neurons) / len(self.neurons)]

            # Update gamma and beta
            self.gamma = [gamma - learning_rate * gamma_grad for gamma, gamma_grad in zip(self.gamma, gamma_gradients)]
            self.beta = [beta - learning_rate * beta_grad for beta, beta_grad in zip(self.beta, beta_gradients)]

# Neural Network class
class NeuralNetwork:
    def __init__(self, sizes, activation_functions, cost_function, threshold=1.0, dropout_rate=1/2, batch_norm=False):
        self.input_size = sizes[0]
        self.hidden_layers = sizes[1:-1]
        self.activation_functions = activation_functions
        self.cost_function = cost_function
        self.threshold = threshold
        self.layers: list[Layer] = []

        self.layers.append(Layer(self.input_size, 0, activation_functions[0]))

        for i in range(1, len(sizes)):
            act_func = activation_functions[(i - 1) % len(activation_functions)]
            self.layers.append(Layer(sizes[i], sizes[i - 1], act_func, dropout_rate, batch_norm, threshold))

    def activate(self, inputs, training=True):
        for i, neuron in enumerate(self.layers[0].neurons):
            neuron.signal = inputs[i]

        for i in range(1, len(self.layers)):
            prev_layer_signals = [neuron.signal for neuron in self.layers[i - 1].neurons]
            for neuron in self.layers[i].neurons:
                neuron.activate(prev_layer_signals, self.threshold)
            self.layers[i].activate(prev_layer_signals, training) # Pass training argument

        return [neuron.signal for neuron in self.layers[-1].neurons]

    def __limit_derivative(self, func, x, epsilon=1e-5):
        """Hidden method to compute the derivative using the limit definition"""
        return (func(x + epsilon) - func(x)) / epsilon

    def backpropagate(self, inputs, expected_outputs, learning_rate=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8, lambda_l1=0.0, lambda_l2=0.0):
        """
        Backpropagates the error, updates weights/biases with Adam, and applies L1/L2 regularization for a single training example.

        Args:
            inputs (list): Input values for a single example.
            expected_outputs (list): Expected output values for a single example.
            learning_rate (float): Learning rate for the Adam optimizer.
            beta1 (float): Beta1 parameter for Adam.
            beta2 (float): Beta2 parameter for Adam.
            epsilon (float): Epsilon parameter for Adam.
            lambda_l1 (float): L1 regularization parameter.
            lambda_l2 (float): L2 regularization parameter.
        """
        self.activate(inputs)
        self._calculate_output_layer_error(expected_outputs)
        self._backpropagate_hidden_layers_error()
        self._update_weights_and_biases(learning_rate, beta1, beta2, epsilon, lambda_l1, lambda_l2)

    def _calculate_output_layer_error(self, expected_outputs):
        """Calculates the error signal for the output layer."""
        output_layer = self.layers[-1]
        for i, neuron in enumerate(output_layer.neurons):
            error = neuron.signal - expected_outputs[i]
            neuron.error_signal = error * self.__limit_derivative(neuron.activation_function, neuron.signal)

    def _backpropagate_hidden_layers_error(self):
        """Backpropagates the error signal through the hidden layers."""
        for layer_idx in range(len(self.layers) - 2, 0, -1):
            current_layer = self.layers[layer_idx]
            next_layer = self.layers[layer_idx + 1]
            for i, neuron in enumerate(current_layer.neurons):
                error_signal = sum(next_neuron.error_signal * next_neuron.weights[i] for next_neuron in next_layer.neurons if i < len(next_neuron.weights))
                neuron.error_signal = error_signal * self.__limit_derivative(neuron.activation_function, neuron.signal)

    def _update_weights_and_biases(self, learning_rate, beta1, beta2, epsilon, lambda_l1, lambda_l2):
        for layer_idx in range(1, len(self.layers)):
          self.layers[layer_idx]._update_weights_and_biases(self.layers[layer_idx - 1].neurons, learning_rate, beta1, beta2, epsilon, lambda_l1, lambda_l2)

    def train(self, inputs_list, outputs_list, validation_inputs=None, validation_outputs=None, epochs=10000, batch_size=32):
        for epoch in range(1, epochs + 1):
            epoch_loss = 0
            for i in range(0, len(inputs_list), batch_size):
                batch_inputs = inputs_list[i:i + batch_size]
                batch_outputs = outputs_list[i:i + batch_size]
                for inputs, outputs in zip(batch_inputs, batch_outputs):
                    self.backpropagate(inputs, outputs, epoch / epochs)
            
            # Compute training loss
            epoch_loss = sum(self.cost_function(self.activate(inputs), outputs) for inputs, outputs in zip(inputs_list, outputs_list)) / len(inputs_list)

            # Compute validation loss if provided
            if validation_inputs and validation_outputs:
                validation_loss = sum(self.cost_function(self.activate(inputs), outputs) for inputs, outputs in zip(validation_inputs, validation_outputs)) / len(validation_inputs)
                print(f"Epoch {epoch+1}/{epochs}, Training Loss: {epoch_loss:.4f}, Validation Loss: {validation_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs}, Training Loss: {epoch_loss:.4f}")

    def save_model(self, filename):
        # Save the entire neural network object (including weights, biases, etc.) to a file
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {filename}")
        
    @staticmethod
    def load_model(filename):
        # Load the entire neural network object from a file
        with open(filename, 'rb') as f:
            print(f"Model loaded from {filename}")
            return pickle.load(f)
        



if __name__ == "__main__":
    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    outputs = [[0], [1], [1], [0]]

    # Example validation set (you can modify this)
    validation_inputs = [[0, 0], [1, 1]]
    validation_outputs = [[0], [0]]

    nn = NeuralNetwork([2, 3, 1], [ActivationFunctions.leaky_relu, ActivationFunctions.sigmoid],
                       CostFunctions.cross_entropy, threshold=1.0, dropout_rate=1/2, batch_norm=True)

    # Train the model
    nn.train(inputs, outputs, validation_inputs, validation_outputs, epochs=1000, batch_size=2)

    # Save the model
    nn.save_model('model.pkl')
    
    del nn

    # Load the model
    nn = NeuralNetwork.load_model('model.pkl')

    # Test the loaded model
    for inp in inputs:
        print("Input:", inp, "Output:", nn.activate(inp))
