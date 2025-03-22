import pickle
from utils import *

# Neural Network class
class NeuralNetwork:
    def __init__(self, sizes, layers_types, activation_functions, cost_function, threshold=1.0, dropout_rate=1/2, batch_norm=False, gradient_clip=1.0):
        self.input_size = sizes[0]
        self.hidden_layers = sizes[1:-1]
        self.layers_types = layers_types
        self.activation_functions = activation_functions
        self.cost_function = cost_function
        self.threshold = threshold
        self.layers: list[Layer] = []
        self.gradient_clip = gradient_clip

        self.layers.append(InputLayer(self.input_size))

        for i in range(1, len(sizes)):
            act_func = self.activation_functions[(i - 1) % len(self.activation_functions)]
            self.layers.append(self.layers_types[(i - 1) % len(self.layers_types)](sizes[i], sizes[i - 1], act_func, dropout_rate, batch_norm, threshold))

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
          self.layers[layer_idx]._update_weights_and_biases(self.layers[layer_idx - 1].neurons, learning_rate, beta1, beta2, epsilon, lambda_l1, lambda_l2, self.gradient_clip)

    def train(self, inputs_list, outputs_list, validation_inputs=None, validation_outputs=None, epochs=10000, batch_size=32):
        for epoch in range(1, epochs + 1):
            epoch_loss = 0
            for i in range(0, len(inputs_list), batch_size):
                batch_inputs = inputs_list[i:i + batch_size]
                batch_outputs = outputs_list[i:i + batch_size]
                for inputs, outputs in zip(batch_inputs, batch_outputs):
                    self.backpropagate(inputs, outputs, 10/epochs)
            
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

    nn = NeuralNetwork([2, 8, 4, 2, 1], [DenseLayer], [ActivationFunctions.leaky_relu, ActivationFunctions.sigmoid],
                       CostFunctions.cross_entropy, threshold=1.0, dropout_rate=1/4, batch_norm=True)

    # Train the model
    nn.train(inputs, outputs, validation_inputs, validation_outputs, epochs=10000, batch_size=2)

    # Save the model
    nn.save_model('model.pkl')
    
    del nn

    # Load the model
    nn = NeuralNetwork.load_model('model.pkl')

    # Test the loaded model
    for inp in inputs:
        print("Input:", inp, "Output:", nn.activate(inp))
