from utils.functions import CostFunctions, ActivationFunctions
from utils.optimizer import *
from layer import * # Import all layers from the layer module
from neuralNetwork import NeuralNetwork
from utils.callbacks import Callback
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Callable

class EpochEndCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {logs['loss']:.4f}, Val Loss: {logs.get('val_loss', 'N/A'):.4f}")

# Generate a simple sequence dataset (e.g., a basic pattern recognition)
def generate_sequence_data(num_samples, time_steps, input_dim):
    """
    Generates a simple sequence dataset where the goal is to
    predict the sum of the input sequence. This is just an example.
    You can replace it with any sequence task.
    """
    inputs = np.random.rand(num_samples, time_steps, input_dim)
    outputs = np.sum(inputs, axis=1)  # Sum along the time axis
    return inputs, outputs

num_samples = 1000
time_steps = 5
input_dim = 2

recurrent_layers = [
    {"layer": SimpleRNNLayer, "kwargs": {"units": 20, "return_sequences": False, "activation": ActivationFunctions.tanh}},
    {"layer": LSTMLayer, "kwargs": {"units": 20, "return_sequences": False}},
    {"layer": GRULayer, "kwargs": {"units": 20, "return_sequences": False}},
    {"layer": BidirectionalRNNLayer, "kwargs": {"forward_layer": SimpleRNNLayer(units=10, return_sequences=False),
                                                "backward_layer": SimpleRNNLayer(units=10, return_sequences=False),
                                                "merge_mode": 'concat'}},
    {"layer": IndRNNLayer, "kwargs": {"units": 20, "return_sequences": False}},
    {"layer": CTRNNLayer, "kwargs": {"units": 20}}
]

original_input_dim = input_dim  # Store the original input_dim

# Modified mean_squared_error function
def mean_squared_error(predicted, target):
    if predicted.ndim == 3 and target.ndim == 2 and predicted.shape[0] == target.shape[0] and predicted.shape[2] == target.shape[1]:
        # Assume target corresponds to the prediction at the last time step
        return np.mean((predicted[:, -1, :] - target) ** 2)
    elif predicted.shape == target.shape:
        return np.mean((predicted - target) ** 2)
    elif predicted.ndim == 2 and target.ndim == 2 and predicted.shape[1] == target.shape[1] and predicted.shape[0] > target.shape[0]:
        # Special handling for CTRNNLayer output shape (assuming output at each time step is flattened)
        batch_size = target.shape[0]
        time_steps_local = predicted.shape[0] // batch_size
        last_time_step_predictions = predicted[np.arange(batch_size) * time_steps_local + (time_steps_local - 1)]
        return np.mean((last_time_step_predictions - target) ** 2)
    else:
        raise ValueError(f"Shapes of predicted {predicted.shape} and target {target.shape} are incompatible for MSE. Predicted: {predicted.shape}, Target: {target.shape}")

# Modified mean_squared_error_prime function
def mean_squared_error_prime(predicted, target):
    if predicted.ndim == 3 and target.ndim == 2 and predicted.shape[0] == target.shape[0] and predicted.shape[2] == target.shape[1]:
        # Derivative with respect to the last time step of predicted
        grad_predicted = np.zeros_like(predicted)
        grad_predicted[:, -1, :] = 2 * (predicted[:, -1, :] - target) / target.shape[0]
        return grad_predicted
    elif predicted.shape == target.shape:
        return 2 * (predicted - target) / target.shape[0]
    elif predicted.ndim == 2 and target.ndim == 2 and predicted.shape[1] == target.shape[1] and predicted.shape[0] > target.shape[0]:
        return _extracted_from_mean_squared_error_prime_11(target, predicted)
    else:
        raise ValueError(f"Shapes of predicted {predicted.shape} and target {target.shape} are incompatible for MSE derivative. Predicted: {predicted.shape}, Target: {target.shape}")


# TODO Rename this here and in `mean_squared_error_prime`
def _extracted_from_mean_squared_error_prime_11(target, predicted):
    # Derivative for the special CTRNNLayer case
    batch_size = target.shape[0]
    time_steps_local = predicted.shape[0] // batch_size
    grad_predicted = np.zeros_like(predicted)
    indices = np.arange(batch_size) * time_steps_local + (time_steps_local - 1)
    grad_predicted[indices] = 2 * (predicted[indices] - target) / target.shape[0]
    return grad_predicted

# Replace the original functions
CostFunctions.mean_squared_error = mean_squared_error
CostFunctions.mean_squared_error_prime = mean_squared_error_prime

for rnn_layer in recurrent_layers:
    layer_class = rnn_layer["layer"]
    layer_kwargs = rnn_layer["kwargs"]

    print(f"\n--- Testing {layer_class.__name__} ---")

    # Adjust input_dim for CTRNNLayer
    if layer_class == CTRNNLayer:
        input_dim = layer_kwargs.get("units", original_input_dim)
    else:
        input_dim = original_input_dim

    # Generate data
    inputs, outputs = generate_sequence_data(num_samples, time_steps, input_dim)
    outputs = outputs.mean(axis=1).reshape(num_samples, 1)

    # Shuffle data
    indices = np.arange(inputs.shape[0])
    np.random.shuffle(indices)
    inputs = inputs[indices]
    outputs = outputs[indices]

    # Generate validation data
    val_inputs, val_outputs = generate_sequence_data(200, time_steps, input_dim)
    val_outputs = val_outputs.mean(axis=1).reshape(200, 1)

    recurrent_layer = layer_class(**layer_kwargs)
    if layer_class == BidirectionalRNNLayer:
        rnn_output_dim = recurrent_layer.forward_layer.units * 2 # Bidirectional layer's output dim
    else:
        rnn_output_dim = recurrent_layer.units

    # Create the network
    layers = [
        recurrent_layer,
        DenseLayer(1, num_inputs=rnn_output_dim,
                   activation_function=ActivationFunctions.relu)
    ]
    nn = NeuralNetwork(layers, CostFunctions.mean_squared_error,
                       l1_lambda=1e-5, l2_lambda=1e-4)
    nn.compile(optimizer=AdamOptimizer(learning_rate=0.001,
                                        weight_decay=1e-2))

    # Train the network
    nn.fit(inputs, outputs, epochs=20, batch_size=16,
            validation_data=(val_inputs, val_outputs),
            callbacks=[EpochEndCallback()], restore_best_weights=True)

    # Test the trained network (simple example)
    test_inputs, test_outputs = generate_sequence_data(4, time_steps, input_dim)
    test_outputs = test_outputs.mean(axis=1).reshape(4, 1)

    print("\nTesting trained network:")
    for i in range(len(test_inputs)):
        prediction = nn.predict(
            test_inputs[i].reshape(1, time_steps, input_dim))
        print(
            f"Input: {test_inputs[i]}, Predicted: {prediction[0, 0]:.4f}, True: {test_outputs[i, 0]:.4f}")

input_dim = original_input_dim # Restore the original input_dim if needed for further tests