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


# Layer class to hold multiple neurons
class Layer:
    def __init__(self, num_neurons, num_inputs, activation_function):
        self.neurons = [Neuron(num_inputs, activation_function) for _ in range(num_neurons)]


# Neural Network class
class NeuralNetwork:
    def __init__(self, sizes, activation_functions, cost_function, threshold=1.0):
        self.input_size = sizes[0]
        self.hidden_layers = sizes[1:-1]
        self.activation_functions = activation_functions
        self.cost_function = cost_function
        self.threshold = threshold
        self.layers = []

        self.layers.append(Layer(self.input_size, 0, activation_functions[0]))

        for i in range(1, len(sizes)):
            act_func = activation_functions[(i - 1) % len(activation_functions)]
            self.layers.append(Layer(sizes[i], sizes[i - 1], act_func))

    def activate(self, inputs):
        for i, neuron in enumerate(self.layers[0].neurons):
            neuron.signal = inputs[i]

        for i in range(1, len(self.layers)):
            prev_layer_signals = [neuron.signal for neuron in self.layers[i - 1].neurons]
            for neuron in self.layers[i].neurons:
                neuron.activate(prev_layer_signals, self.threshold)

        return [neuron.signal for neuron in self.layers[-1].neurons]

    def __limit_derivative(self, func, x, epsilon=1e-5):
        """Hidden method to compute the derivative using the limit definition"""
        return (func(x + epsilon) - func(x)) / epsilon

    def backpropagate(self, inputs, expected_outputs, learning_rate=1e-3):
        outputs = self.activate(inputs)

        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8  

        # Calculate error signal for output layer
        output_layer = self.layers[-1]
        for i, neuron in enumerate(output_layer.neurons):
            error = neuron.signal - expected_outputs[i]
            neuron.error_signal = error * self.__limit_derivative(neuron.activation_function, neuron.signal)

        # Backpropagate error for hidden layers
        for layer_idx in range(len(self.layers) - 2, 0, -1):
            current_layer = self.layers[layer_idx]
            next_layer = self.layers[layer_idx + 1]
            for i, neuron in enumerate(current_layer.neurons):
                error_signal = sum(next_neuron.error_signal * next_neuron.weights[i] for next_neuron in next_layer.neurons if i < len(next_neuron.weights))
                neuron.error_signal = error_signal * self.__limit_derivative(neuron.activation_function, neuron.signal)

        # Update weights and biases using Adam optimizer
        for layer_idx in range(1, len(self.layers)):
            current_layer = self.layers[layer_idx]
            prev_layer = self.layers[layer_idx - 1]
            for neuron in current_layer.neurons:
                neuron.time_step += 1

                gradients = [neuron.error_signal * prev_neuron.signal for prev_neuron in prev_layer.neurons]

                for k in range(len(neuron.weights)):
                    neuron.first_moment[k] = beta1 * neuron.first_moment[k] + (1 - beta1) * gradients[k]
                    neuron.second_moment[k] = beta2 * neuron.second_moment[k] + (1 - beta2) * (gradients[k] ** 2)

                    corrected_first = neuron.first_moment[k] / (1 - (beta1 ** neuron.time_step))
                    corrected_second = neuron.second_moment[k] / (1 - (beta2 ** neuron.time_step))
                    neuron.weights[k] -= learning_rate * corrected_first / (math.sqrt(corrected_second) + epsilon)

                neuron.first_moment_bias = beta1 * neuron.first_moment_bias + (1 - beta1) * neuron.error_signal
                neuron.second_moment_bias = beta2 * neuron.second_moment_bias + (1 - beta2) * (neuron.error_signal ** 2)
                corrected_first_bias = neuron.first_moment_bias / (1 - (beta1 ** neuron.time_step))
                corrected_second_bias = neuron.second_moment_bias / (1 - (beta2 ** neuron.time_step))
                neuron.bias -= learning_rate * corrected_first_bias / (math.sqrt(corrected_second_bias) + epsilon)

    def train(self, inputs_list, outputs_list, validation_inputs=None, validation_outputs=None, learning_rate=1e-3, epochs=10000, log_interval=100):
        for epoch in range(epochs):
            epoch_loss = 0
            for inputs, outputs in zip(inputs_list, outputs_list):
                self.backpropagate(inputs, outputs, learning_rate)
            
            # Compute training loss
            epoch_loss = sum(self.cost_function(self.activate(inputs), outputs) for inputs, outputs in zip(inputs_list, outputs_list)) / len(inputs_list)

            # Compute validation loss if provided
            if validation_inputs and validation_outputs:
                validation_loss = sum(self.cost_function(self.activate(inputs), outputs) for inputs, outputs in zip(validation_inputs, validation_outputs)) / len(validation_inputs)
                print(f"Epoch {epoch+1}/{epochs}, Training Loss: {epoch_loss:.4f}, Validation Loss: {validation_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs}, Training Loss: {epoch_loss:.4f}")

            # Print every log_interval iterations
            if (epoch + 1) % log_interval == 0:
                print(f"Epoch {epoch+1} completed with loss: {epoch_loss:.4f}")

    def save_model(self, filename):
        # Save the entire neural network object (including weights, biases, etc.) to a file
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {filename}")
        
    @staticmethod
    def load_model(filename):
        # Load the entire neural network object from a file
        with open(filename, 'rb') as f:
            return pickle.load(f)
        # Now the loaded_model is a complete NeuralNetwork instance
        print(f"Model loaded from {filename}")



if __name__ == "__main__":
    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    outputs = [[0], [1], [1], [0]]

    # Example validation set (you can modify this)
    validation_inputs = [[0, 0], [1, 1]]
    validation_outputs = [[0], [0]]

    nn = NeuralNetwork([2, 6, 1], [ActivationFunctions.leaky_relu, ActivationFunctions.sigmoid],
                       CostFunctions.cross_entropy)

    # Train the model
    nn.train(inputs, outputs, validation_inputs, validation_outputs, learning_rate=0.1, epochs=1000, log_interval=100)

    # Save the model
    nn.save_model('model.pkl')
    
    del nn

    # Load the model
    nn = NeuralNetwork.load_model('model.pkl')

    # Test the loaded model
    for inp in inputs:
        print("Input:", inp, "Output:", nn.activate(inp))
