import json
from neuralNetwork import NeuralNetwork
from utils.functions import ActivationFunctions, CostFunctions
from utils.layers import *
from typing import Type
from typing import overload

@overload
def save(network: NeuralNetwork) -> str:
    """Save the neural network to a JSON string."""
    network_data = {
        "sizes": [layer.num_neurons for layer in network.layers],
        "layers_types": [type(layer).__name__ for layer in network.layers],
        "activation_functions": [func.__name__ for func in network.activation_functions],
        "cost_function": network.cost_function.__name__,
        "threshold": network.threshold,
        "dropout_rate": network.dropout_rate,
        "batch_norm": network.batch_norm,
        "gradient_clip": network.gradient_clip,
        "weights": [[neuron.weights for neuron in layer.neurons] for layer in network.layers if hasattr(layer, 'neurons')],
        "biases": [[neuron.bias for neuron in layer.neurons] for layer in network.layers if hasattr(layer, 'neurons')]
    }
    
    return json.dumps(network_data, indent=4)

def save(network: NeuralNetwork, filepath: str = None) -> None:
    """Save the neural network to a JSON file."""
    network_data = {
        "sizes": [layer.num_neurons for layer in network.layers],
        "layers_types": [type(layer).__name__ for layer in network.layers],
        "activation_functions": [func.__name__ for func in network.activation_functions],
        "cost_function": network.cost_function.__name__,
        "threshold": network.threshold,
        "dropout_rate": network.dropout_rate,
        "batch_norm": network.batch_norm,
        "gradient_clip": network.gradient_clip,
        "weights": [[neuron.weights for neuron in layer.neurons] for layer in network.layers if hasattr(layer, 'neurons')],
        "biases": [[neuron.bias for neuron in layer.neurons] for layer in network.layers if hasattr(layer, 'neurons')]
    }
    with open(filepath, 'w') as f:
        json.dump(network_data, f, indent=4)

def load(filepath: str) -> NeuralNetwork:
    """Load the neural network from a JSON file."""
    try:
        with open(filepath, 'r') as f:
            network_data = json.load(f)
        
        sizes = network_data["sizes"]
        layers_types = [_get_layer_class(name) for name in network_data["layers_types"]]
        activation_functions = [getattr(ActivationFunctions, name) for name in network_data["activation_functions"]]
        cost_function = getattr(CostFunctions, network_data["cost_function"])
        threshold = network_data["threshold"]
        dropout_rate = network_data["dropout_rate"]
        batch_norm = network_data["batch_norm"]
        gradient_clip = network_data["gradient_clip"]

        layers = [layers_types[i](sizes[i], sizes[i - 1] if i > 0 else sizes[0], activation_functions[i % len(activation_functions)], dropout_rate, batch_norm, threshold) for i in range(len(sizes))]
        network = NeuralNetwork(layers, cost_function, threshold, gradient_clip)
        
        for layer, weights, biases in zip(network.layers, network_data["weights"], network_data["biases"]):
            if hasattr(layer, 'neurons'):
                for neuron, w, b in zip(layer.neurons, weights, biases):
                    neuron.weights = w
                    neuron.bias = b

        return network
    except Exception as e:
        print(f"Error loading network: {e}")
        return None

def _get_layer_class(layer_name: str) -> Type:
    """Get the layer class by its name, searching recursively through subclasses."""
    
    def get_all_subclasses(cls):
        subclasses = set(cls.__subclasses__())
        for subclass in list(subclasses):
            subclasses.update(get_all_subclasses(subclass))
        return subclasses

    # Collect all subclasses recursively
    all_layer_classes = {cls.__name__: cls for cls in get_all_subclasses(Layer)}
    
    if layer_name not in all_layer_classes:
        raise ValueError(f"Unknown layer type: {layer_name}")
    
    return all_layer_classes[layer_name]
