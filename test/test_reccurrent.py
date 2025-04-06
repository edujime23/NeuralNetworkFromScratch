import sys
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.__str__())

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Callable

from network.utils.functions import CostFunctions, ActivationFunctions
from network.utils.optimizer import AdamOptimizer
from network.layer import (
    SimpleRNNLayer,
    LSTMLayer,
    GRULayer,
    BidirectionalRNNLayer,
    IndRNNLayer,
    CTRNNLayer,
    DenseLayer
)
from network.neuralNetwork import NeuralNetwork
from network.utils.callbacks import Callback

class EpochEndCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 1000 == 0:
            loss = logs.get('loss', None)
            val_loss = logs.get('val_loss', None)
            log_str = f"Epoch {epoch+1}, Loss: {loss:.4f}"
            if val_loss is not None:
                log_str += f", Val Loss: {val_loss:.4f}"
            print(log_str)

def generate_arange_sequence_data(start: int, end: int, step: int, time_steps: int, input_dim: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates a sequence dataset from a range using np.arange.
    It creates subsequences of length time_steps and their corresponding sums.
    """
    sequence = np.arange(start, end, step)
    num_samples = len(sequence) - time_steps
    if num_samples < 1:
        return np.empty((0, time_steps, input_dim)), np.empty((0,))
    inputs = np.zeros((num_samples, time_steps, input_dim))
    outputs = np.zeros((num_samples, 1))
    for i in range(num_samples):
        sub_sequence = sequence[i : i + time_steps]
        inputs[i, :, 0] = sub_sequence
        outputs[i, 0] = np.sum(sub_sequence)
    return inputs, outputs

time_steps = 1
input_dim = 1
units = 128
epochs = 10000
batch_size = 8
learning_rate = 1e-4
esp=1000
tol=1e-4

recurrent_layer_configs = [
    {"layer": SimpleRNNLayer, "kwargs": {"units": units, "return_sequences": False, "activation": ActivationFunctions.tanh}},
    {"layer": LSTMLayer, "kwargs": {"units": units, "return_sequences": False}},
    {"layer": GRULayer, "kwargs": {"units": units, "return_sequences": False}},
    {
        "layer": BidirectionalRNNLayer,
        "kwargs": {
            "forward_layer": SimpleRNNLayer(units=units // 2, return_sequences=False),
            "backward_layer": SimpleRNNLayer(units=units // 2, return_sequences=False),
            "merge_mode": 'concat'
        }
    },
    {"layer": IndRNNLayer, "kwargs": {"units": units, "return_sequences": False}},
    {"layer": CTRNNLayer, "kwargs": {"units": units}}
]

for rnn_config in recurrent_layer_configs:
    layer_class = rnn_config["layer"]
    layer_kwargs = rnn_config["kwargs"]
    layer_name = layer_class.__name__

    print(f"\n--- Testing {layer_name} ---")

    # Generate training data (0 to 50, step 2)
    train_inputs, train_outputs = generate_arange_sequence_data(0, 50, 2, time_steps, input_dim)

    # Generate validation data (50 to 100, step 2)
    val_inputs, val_outputs = generate_arange_sequence_data(50, 100, 2, time_steps, input_dim)

    if train_inputs.shape[0] == 0 or val_inputs.shape[0] == 0:
        print(f"Not enough data for {layer_name} with time_steps={time_steps}. Skipping.")
        continue

    recurrent_layer = layer_class(**layer_kwargs)
    if layer_class == BidirectionalRNNLayer:
        rnn_output_dim = recurrent_layer.forward_layer.units * 2  # Bidirectional layer's output dim
    else:
        rnn_output_dim = recurrent_layer.units

    # Create the network
    layers = [
        recurrent_layer,
        DenseLayer(1, activation_function=ActivationFunctions.relu)
    ]
    nn = NeuralNetwork(layers, CostFunctions.mean_absolute_error,
                       l1_lambda=1e-5, l2_lambda=1e-4)
    nn.compile(optimizer=AdamOptimizer(learning_rate=learning_rate, weight_decay=1e-2))

    # Train the network
    nn.fit(train_inputs, train_outputs, epochs=epochs, batch_size=batch_size,
           validation_data=(val_inputs, val_outputs),
           callbacks=[EpochEndCallback()], restore_best_weights=True, early_stopping_patience=esp, tol=tol)

    # Test the trained network (on a single example from the validation set)
    if val_inputs.shape[0] > 0:
        test_input = val_inputs[0].reshape(1, time_steps, input_dim)
        true_output = val_outputs[0, 0]
        prediction = nn.predict(test_input)[0, 0]
        print("\nTesting trained network on validation data:")
        print(f"Input: {test_input.flatten()}, Predicted sum: {prediction:.4f}, True sum: {true_output:.4f}")