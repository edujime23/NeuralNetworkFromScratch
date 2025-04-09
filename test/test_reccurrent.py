import sys
from pathlib import Path
import time as time_module
import queue
import multiprocessing as mp
import contextlib
import numpy as np
import matplotlib.pyplot as plt

# Improved way to add the project root to the path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from network.functions import *
from network.optimizer import *
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

# --- Configuration for Live Plotting ---
PLOT_QUEUE_MAXSIZE = 5
PLOT_UPDATE_INTERVAL = 0.1  # Minimum seconds between plot updates

class EpochEndCallback(Callback):
    def __init__(self, log_frequency=1000):
        self.log_frequency = log_frequency

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.log_frequency == 0:
            loss = logs.get('loss')
            val_loss = logs.get('val_loss')
            log_str = f"Epoch {epoch+1}, Loss: {loss:.4f}"
            if val_loss is not None:
                log_str += f", Val Loss: {val_loss:.4f}"
            print(log_str)

class LiveLossPlotCallback(Callback):
    """Sends loss data to a separate process for live plotting."""
    def __init__(self, plot_queue: mp.Queue):
        self.plot_queue = plot_queue
        self.train_losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        if loss is not None:
            self.train_losses.append(loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)
        if loss is not None:
            plot_data = {'epoch': epoch + 1, 'train_loss': loss, 'val_loss': val_loss}
            try:
                self.plot_queue.put_nowait(plot_data)
            except queue.Full:
                pass # Skip if plot process is lagging

def plot_loss_process(plot_queue: mp.Queue, num_epochs: int):
    """Target function for the plotting process."""
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Live Training and Validation Loss')
    ax.grid(True)

    line_train, = ax.plot([], [], label='Training Loss', color='blue')
    line_val, = ax.plot([], [], label='Validation Loss', color='orange')
    ax.legend()

    epochs_data = []
    train_loss_data = []
    val_loss_data = []

    plt.show(block=False)
    last_update_time = time_module.time()

    while True:
        try:
            plot_data = plot_queue.get(timeout=0.1)
            if plot_data is None:
                break

            current_time = time_module.time()
            if current_time - last_update_time < PLOT_UPDATE_INTERVAL and plot_data['epoch'] < num_epochs:
                continue

            epoch = plot_data.get('epoch')
            train_loss = plot_data.get('train_loss')
            val_loss = plot_data.get('val_loss')

            if epoch is not None and train_loss is not None:
                epochs_data.append(epoch)
                train_loss_data.append(train_loss)
                line_train.set_xdata(epochs_data)
                line_train.set_ydata(train_loss_data)

            if epoch is not None and val_loss is not None:
                val_loss_data.append(val_loss)
                line_val.set_xdata(epochs_data)
                line_val.set_ydata(val_loss_data)

            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            last_update_time = current_time

        except queue.Empty:
            plt.pause(0.01)
        except Exception as e:
            print(f"Error in plot process: {e}")
            import traceback
            traceback.print_exc()
            break

    plt.ioff()
    print("Loss plotting process finished. Close the plot window manually.")
    plt.show(block=True)
    plt.close(fig)

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

# --- Configuration ---
TIME_STEPS = 1
INPUT_DIM = 16
UNITS = 128
EPOCHS = 10000
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EARLY_STOPPING_PATIENCE = 1000
TOLERANCE = 1e-4

RECURRENT_LAYER_CONFIGS = [
    {"layer": SimpleRNNLayer, "kwargs": {"units": UNITS, "return_sequences": False, "activation": tanh}},
    {"layer": LSTMLayer, "kwargs": {"units": UNITS, "return_sequences": False}},
    {"layer": GRULayer, "kwargs": {"units": UNITS, "return_sequences": False}},
    {
        "layer": BidirectionalRNNLayer,
        "kwargs": {
            "forward_layer": SimpleRNNLayer(units=UNITS // 2, return_sequences=False),
            "backward_layer": SimpleRNNLayer(units=UNITS // 2, return_sequences=False),
            "merge_mode": 'concat'
        }
    },
    {"layer": IndRNNLayer, "kwargs": {"units": UNITS, "return_sequences": False}},
    {"layer": CTRNNLayer, "kwargs": {"units": UNITS}}
]

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    plot_queue = mp.Queue(maxsize=PLOT_QUEUE_MAXSIZE)
    live_loss_plot_callback = LiveLossPlotCallback(plot_queue)
    plot_process_instance = mp.Process(target=plot_loss_process, args=(plot_queue, EPOCHS), daemon=True)
    print("Starting loss plotting process...")
    plot_process_instance.start()

    for rnn_config in RECURRENT_LAYER_CONFIGS:
        layer_class = rnn_config["layer"]
        layer_kwargs = rnn_config["kwargs"]
        layer_name = layer_class.__name__

        print(f"\n--- Testing {layer_name} ---")

        # Generate training data (0 to 50, step 2)
        train_inputs, train_outputs = generate_arange_sequence_data(0, 50, 2, TIME_STEPS, INPUT_DIM)

        # Generate validation data (50 to 100, step 2)
        val_inputs, val_outputs = generate_arange_sequence_data(50, 100, 2, TIME_STEPS, INPUT_DIM)

        if train_inputs.shape[0] == 0 or val_inputs.shape[0] == 0:
            print(f"Not enough data for {layer_name} with time_steps={TIME_STEPS}. Skipping.")
            continue

        recurrent_layer = layer_class(**layer_kwargs)
        if layer_class == BidirectionalRNNLayer:
            rnn_output_dim = recurrent_layer.forward_layer.units * 2  # Bidirectional layer's output dim
        else:
            rnn_output_dim = recurrent_layer.units

        # Create the network
        layers = [
            recurrent_layer,
            DenseLayer(1, activation_function=None) # Consider no activation for summation
        ]
        nn = NeuralNetwork(layers, mean_absolute_error,
                            l1_lambda=1e-5, l2_lambda=1e-4)
        nn.compile(optimizer=Adam(learning_rate=LEARNING_RATE, weight_decay=1e-2))

        # Train the network
        nn.fit(train_inputs, train_outputs, epochs=EPOCHS, batch_size=BATCH_SIZE,
                validation_data=(val_inputs, val_outputs),
                callbacks=[EpochEndCallback(), live_loss_plot_callback], restore_best_weights=True,
                early_stopping_patience=EARLY_STOPPING_PATIENCE, tol=TOLERANCE)

        # Test the trained network (on a single example from the validation set)
        if val_inputs.shape[0] > 0:
            test_input = val_inputs[0].reshape(1, TIME_STEPS, INPUT_DIM)
            true_output = val_outputs[0, 0]
            prediction = nn.predict(test_input)[0, 0]
            print("\nTesting trained network on validation data:")
            print(f"Input: {test_input.flatten()}, Predicted sum: {prediction:.4f}, True sum: {true_output:.4f}")

    # Cleanup plotting process
    print("Cleaning up plotting process...")
    try:
        plot_queue.put_nowait(None)
    except queue.Full:
        with contextlib.suppress(queue.Empty):
            while True:
                plot_queue.get_nowait()
        try:
            plot_queue.put(None, timeout=1.0)
        except queue.Full:
            print("Warning: Could not send termination signal.")

    plot_process_instance.join(timeout=2.0)
    if plot_process_instance.is_alive():
        print("Plot process did not exit gracefully, terminating...")
        plot_process_instance.terminate()
        plot_process_instance.join(timeout=1.0)

    plot_queue.close()
    plot_queue.join_thread()
    print("Cleanup complete.")