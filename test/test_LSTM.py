import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import contextlib
from network.utils.functions import CostFunctions, ActivationFunctions
import scipy
from network.utils.optimizer import *
from inspect import signature
from network.layer import LSTMLayer, DenseLayer
from network.neuralNetwork import NeuralNetwork
from network.utils.callbacks import Callback
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import time as time_module # Renamed the time module import
import queue

class LossCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 1 == 0:
            print(f"Epoch {epoch+1}, Loss: {logs['loss']:.4f}, Val Loss: {logs.get('val_loss', 'N/A')}")

def plot_process(plot_queue, time_steps, original_series, train_outputs, sequence_length):
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    line_objective, = ax.plot(time_steps, original_series, label='Original Time Series', color='blue')
    scatter_train = ax.scatter(time_steps, train_outputs, label='Training Data (Target)', s=10, color='green', alpha=0.5)
    line_predictions, = ax.plot(time_steps, np.zeros_like(train_outputs), label='LSTM Predictions', color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Value')
    ax.set_title('Time Series Prediction using LSTM')
    ax.legend()
    ax.grid(True)

    plt.show(block=False)

    last_update_time = time_module.time()
    update_interval = 1/120  # Seconds between visual updates to prevent excessive redrawing

    while True:
        try:
            # Non-blocking queue get with short timeout
            predictions = plot_queue.get(timeout=1/120)
            if predictions is None:  # Signal to stop
                break

            current_time = time_module.time()
            if current_time - last_update_time > update_interval:
                line_predictions.set_ydata(predictions)
                fig.canvas.draw_idle()  # More efficient than draw()
                fig.canvas.flush_events()
                last_update_time = current_time

        except queue.Empty:
            # No new data, just update the plot if needed
            plt.pause(1/120)  # Small pause to allow GUI events to process
        except Exception as e:
            print(f"Error in plot process: {e}")
            break

    plt.close()

class LivePlotCallback(Callback):
    def __init__(self, plot_queue, model, sequence_length, original_series, update_frequency=10):
        self.plot_queue = plot_queue
        self.model = model
        self.sequence_length = sequence_length
        self.original_series = original_series
        self.update_frequency = update_frequency
        self.last_put_time = 0

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.update_frequency == 0:
            current_time = time_module.time()
            # Only update if queue isn't full and enough time has passed
            with contextlib.suppress(queue.Full):
                num_points = len(self.original_series)
                if num_points <= self.sequence_length:
                    print(f"Warning: Not enough data points ({num_points}) for sequence length ({self.sequence_length})")
                    return

                # Predict the next value based on the last sequence_length values
                last_sequence = self.original_series[-self.sequence_length:].reshape(1, self.sequence_length, 1)
                prediction = self.model.predict(last_sequence)

                # For live plotting, we'll predict for the training data range
                predictions = np.zeros(len(self.original_series) - self.sequence_length)
                for i in range(len(self.original_series) - self.sequence_length):
                    sequence = self.original_series[i : i + self.sequence_length].reshape(1, self.sequence_length, 1)
                    pred = self.model.predict(sequence)
                    predictions[i] = pred.flatten()[0] # Extract the scalar value

                # Use put_nowait to avoid blocking if queue is full
                self.plot_queue.put_nowait(predictions)
                self.last_put_time = current_time

def generate_time_series(num_samples):
    time_steps = np.linspace(0, 10 * np.pi, num_samples)
    series = np.sin(time_steps) + 0.1 * np.random.randn(num_samples)
    return series.reshape(-1, 1), time_steps

def prepare_data(time_series, sequence_length):
    inputs = []
    targets = []
    for i in range(len(time_series) - sequence_length):
        inputs.append(time_series[i : i + sequence_length])
        targets.append(time_series[i + sequence_length])
    return np.array(inputs), np.array(targets)

if __name__ == "__main__":
    # Using context manager to ensure clean process termination
    mp.set_start_method('spawn', force=True)  # More stable for GUI applications

    num_samples = 1000
    original_series, time_steps = generate_time_series(num_samples)

    sequence_length = 30 # Length of the input sequence for the LSTM

    # Prepare data for LSTM
    train_inputs, train_outputs = prepare_data(original_series, sequence_length)

    input_dims = 1
    out_dims = 1

    layers = [
        LSTMLayer(16, return_sequences=False),
        DenseLayer(out_dims, activation_function=ActivationFunctions.linear)
    ]

    adam = AdamOptimizer(learning_rate=1e-2,
                         weight_decay=0,
                         use_adamw=True,
                         amsgrad=True,
                         gradient_clip=None,
                         noise_factor=1e-4,
                         cyclical_lr=True,
                         lr_max_factor=4,
                         lr_cycle_steps=10
    )
    sgd = SGD(learning_rate=1e-3, momentum=1e-2, nesterov=True)

    nn = NeuralNetwork(layers, CostFunctions.mean_squared_error, gradient_clip=None, l1_lambda=0, l2_lambda=0)
    nn.compile(optimizer=adam)

    # Create a multiprocessing Queue with limited size to avoid memory issues
    plot_queue = mp.Queue()  # Only keep latest few predictions

    # Instantiate the LivePlotCallback with the queue
    live_plot_callback = LivePlotCallback(plot_queue, nn, sequence_length, original_series, update_frequency=1)
    loss_callback = LossCallback()

    # Create and start the plotting process
    plot_process_instance = mp.Process(
        target=plot_process,
        args=(plot_queue, time_steps[sequence_length:], original_series[sequence_length:], train_outputs, sequence_length) # Passed sequence_length
    )
    plot_process_instance.daemon = True  # Automatically terminate if main process ends
    plot_process_instance.start()

    try:
        nn.fit(
            train_inputs,
            train_outputs,
            epochs=2**64,
            batch_size=256,
            callbacks=[loss_callback, live_plot_callback],
            restore_best_weights=True
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    finally:
        # Send a signal to the plotting process to stop
        try:
            plot_queue.put_nowait(None)
        except queue.Full:
            # If queue is full, try to empty it first
            while not plot_queue.empty():
                try:
                    plot_queue.get_nowait()
                except queue.Empty:
                    break
            plot_queue.put(None)

        # Wait for the plotting process to finish with timeout
        plot_process_instance.join(timeout=1.0)

        # If process is still alive, terminate it
        if plot_process_instance.is_alive():
            plot_process_instance.terminate()
            plot_process_instance.join()
            plot_process_instance.close()

        print("Training completed.")