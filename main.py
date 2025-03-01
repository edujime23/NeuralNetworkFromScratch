from typing import Optional, Tuple, Callable, List, override, Self, Union
import numpy as np
import itertools
from multiprocessing import Pool, cpu_count

class Neuron:
    def __init__(self, num_inputs: int, activation_function: Callable[[np.float64], np.float64]):
        """
        Initializes a Neuron with the specified number of inputs and activation function.
        
        Args:
            num_inputs (int): The number of input connections to the neuron.
            activation_function (Callable[[np.float64], np.float64]): The activation function to be used by the neuron.
        """
        self.delta: np.float64 = np.float64(0)  # Error term for the neuron
        self.weights: np.ndarray = np.random.rand(num_inputs)  # Weights for the inputs
        self.bias: np.float64 = np.random.rand()  # Bias for the neuron
        self.signal: np.float64 = np.float64(0)  # Output signal of the neuron
        self.activation_function = activation_function  # Activation function for the neuron

    def activate(self, inputs: List[Union[float, np.float64]], threshold: Union[float, np.float64]) -> None:
        """
        Activates the neuron using the given inputs and threshold.
        
        Args:
            inputs (List[Union[float, np.float64]]): The input values to the neuron.
            threshold (Union[float, np.float64]): The threshold value for activation.
        """
        signal_sum: np.float64 = self.bias + np.dot(self.weights, np.array(inputs, dtype=np.float64))  # Weighted sum of inputs and bias
        self.signal = self.activation_function(signal_sum / np.float64(threshold))  # Apply activation function

class Layer:
    def __init__(self, num_neurons: int, num_inputs: int, activation_function: Optional[Callable[[np.float64], np.float64]]):
        self.neurons: List[Neuron] = [Neuron(num_inputs, activation_function) for _ in range(num_neurons)]  # List of neurons in the layer
        self.activation_function = activation_function  # Activation function for the layer
        self.num_neurons = num_neurons  # Number of neurons in the layer
        self.num_inputs = num_inputs  # Number of inputs to each neuron in the layer

class NeuralNetwork:
    def sigmoid(self, x: np.float64) -> np.float64:
        return np.float64(1) / (np.float64(1) + np.exp(-np.clip(x, -500, 500)))

    def mean_squared_error(self, output: np.float64, expected: np.float64) -> np.float64:
        output = np.float64(output)
        expected = np.float64(expected)
        return (output - expected) ** 2

    def derivative(self, activation: Callable[[np.float64], np.float64], x: np.float64, *, dx: np.float64 = np.float64(1e-8)) -> np.float64:
        a1 = np.float64(activation(x + dx))
        a2 = np.float64(activation(x - dx))
        return (a1 - a2) / dx

    def __init__(self, input_size: Union[int, np.uint16], hidden_size: Tuple[Union[int, np.uint16]], output_size: Union[int, np.uint16], 
                 learning_rate: Union[float, np.float16], threshold: Union[float, np.float32],
                 activation_functions: Optional[Tuple[Callable[[np.float64], np.float64], ...]] = None,
                 cost_function: Optional[Callable[[np.float64, np.float64], np.float64]] = None):
        # Type checks and initialization
        if not isinstance(hidden_size, tuple):
            raise TypeError("hidden_size must be a tuple of ints")
        if not all(isinstance(x, (int, np.uint16)) for x in hidden_size):
            raise TypeError("All elements in hidden_size must be ints")
        if not isinstance(input_size, (int, np.uint16)) or not isinstance(output_size, int):
            raise TypeError("input_size and output_size must be ints")
        if input_size <= 0 or output_size <= 0 or any(x <= 0 for x in hidden_size):
            raise ValueError("input_size, output_size, and all elements in hidden_size must be positive integers")
        if activation_functions is not None:
            if not all(callable(func) for func in activation_functions):
                raise TypeError("All elements in activation_functions must be callables")
            self.activation_functions = itertools.cycle(activation_functions)
        else:
            self.activation_functions = itertools.cycle([self.sigmoid])
        if cost_function is not None and not callable(cost_function):
            raise TypeError("cost_function must be a callable")
        self.input_size: np.uint16 = np.uint16(input_size)
        self.hidden_size: Tuple[np.uint16] = tuple(np.uint16(x) for x in hidden_size)
        self.output_size: np.uint16 = np.uint16(output_size)
        self.learning_rate: np.float16 = np.float16(learning_rate)
        self.threshold: np.float32 = np.float32(threshold)
        self.cost: Optional[Callable[[np.float64, np.float64], np.float64]] = cost_function if cost_function else self.mean_squared_error
        
        # Build layers (assumes Layer and Neuron classes are defined elsewhere)
        self.layers: List[Layer] = [Layer(input_size, input_size, next(self.activation_functions))]
        for i in range(len(hidden_size)):
            self.layers.append(Layer(hidden_size[i], hidden_size[i-1] if i > 0 else input_size, next(self.activation_functions)))
        self.layers.append(Layer(output_size, hidden_size[-1] if hidden_size else input_size, next(self.activation_functions)))

    def activate(self, inputs: List[Union[float, np.float64]]) -> List[np.float64]:
        # Set input layer signals
        for i in range(len(inputs)):
            self.layers[0].neurons[i].signal = np.float64(inputs[i])
        # Forward propagation
        for i in range(1, len(self.layers)):
            prev_layer_signals = np.array([neuron.signal for neuron in self.layers[i-1].neurons], dtype=np.float64)
            for neuron in self.layers[i].neurons:
                neuron.activate(prev_layer_signals, self.threshold)
        return [neuron.signal for neuron in self.layers[-1].neurons]

    def backpropagate(self, inputs: List[Union[float, np.float64]], outputs: List[Union[float, np.float64]]) -> None:
        self.activate(inputs)
        outputs = np.array(outputs, dtype=np.float64)
        output_layer = self.layers[-1]
        output_signals = np.array([neuron.signal for neuron in output_layer.neurons])
        output_deltas = (outputs - output_signals) * self.derivative(output_layer.activation_function, output_signals)
        for i, neuron in enumerate(output_layer.neurons):
            neuron.delta = output_deltas[i]
        for i in range(len(self.layers) - 2, 0, -1):
            layer = self.layers[i]
            next_layer = self.layers[i + 1]
            layer_signals = np.array([neuron.signal for neuron in layer.neurons])
            next_weights = np.array([neuron.weights for neuron in next_layer.neurons])
            next_deltas = np.array([neuron.delta for neuron in next_layer.neurons])
            layer_deltas = layer_signals * self.derivative(layer.activation_function, layer_signals) * np.dot(next_weights.T, next_deltas)
            for j, neuron in enumerate(layer.neurons):
                neuron.delta = layer_deltas[j]
        for i in range(1, len(self.layers)):
            layer = self.layers[i]
            prev_layer_signals = np.array([neuron.signal for neuron in self.layers[i - 1].neurons])
            for neuron in layer.neurons:
                neuron.weights += self.learning_rate * neuron.delta * prev_layer_signals
                neuron.bias += self.learning_rate * neuron.delta

    def train(self, inputs: List[Tuple[Union[float, np.float64]]], outputs: List[Tuple[Union[float, np.float64]]], epochs: int) -> None:
        num_processes = cpu_count()
        epochs_per_process = epochs // num_processes
        with Pool(num_processes) as pool:
            trained_networks = pool.starmap(
                self._train_worker,
                [(self, inputs, outputs, epochs_per_process) for _ in range(num_processes)]
            )
        self._aggregate_networks(trained_networks)

    def _train_worker(self, network: Self, inputs: List[Tuple[Union[float, np.float64]]], outputs: List[Tuple[Union[float, np.float64]]], epochs: int) -> Self:
        for epoch in range(epochs):
            for input_data, output_data in zip(inputs, outputs):
                network.backpropagate(input_data, output_data)
        return network

    def _aggregate_networks(self, networks: List[Self]) -> None:
        for i in range(1, len(self.layers)):
            for j, neuron in enumerate(self.layers[i].neurons):
                neuron.weights = np.mean([net.layers[i].neurons[j].weights for net in networks], axis=0)
                neuron.bias = np.mean([net.layers[i].neurons[j].bias for net in networks])

# Example usage
if __name__ == "__main__":
    # XOR Neural Network
    net = NeuralNetwork(2, (12,), 1, learning_rate=1/6, threshold=1, activation_functions=None, cost_function=None)

    # XOR inputs and expected outputs
    xor_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    xor_expected = [(0,), (1,), (1,), (0,)]

    epochs = 100_000
    
    # Train the network
    net.train(xor_inputs, xor_expected, epochs)

    # Test the network
    for inputs in xor_inputs:
        output = net.activate(inputs)
        print(f"Input: {inputs}, Output: {output}")
    
    for layer in range(len(net.layers)):
        for neuron in net.layers[layer].neurons:
            print(f"Layer: {layer}, Weights: {neuron.weights}, Bias: {neuron.bias}")