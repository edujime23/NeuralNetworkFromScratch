import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import contextlib
from network.functions import *
from network.optimizer import *
from inspect import signature
from network.layer import DenseLayer
from network.neuralNetwork import NeuralNetwork
from network.utils.callbacks import Callback
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import time
import queue
import cProfile  # Import the cProfile module
import pstats  # For analyzing the profile data

# --- Configuration for Live Plotting ---
OUTPUT_PLOT_QUEUE_MAXSIZE = 5
LOSS_PLOT_QUEUE_MAXSIZE = 5
PLOT_UPDATE_INTERVAL = 0.1  # Minimum seconds between plot updates

class LossCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 1 == 0:
            print(f"Epoch {epoch+1}, Loss: {logs['loss']:.4f}, Val Loss: {logs.get('val_loss', 'N/A')}")

def plot_process(plot_queue, inputs, target_func, train_inputs, train_outputs):
    plt.ion()
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))  # 2 subplots for real and imaginary

    # Plotting real and imaginary parts
    line_objective_real, = axs[0].plot(inputs.real, target_func(inputs).real, label='Objective Real', color='blue')
    line_predictions_real, = axs[0].plot(inputs.real, np.zeros_like(inputs.real), label='Prediction Real', color='red', linestyle='--', linewidth=2)
    scatter_train_real = axs[0].scatter(train_inputs.real, train_outputs.real, label='Train Real', s=10, color='green', alpha=0.5)
    axs[0].set_xlabel('Re(x)')
    axs[0].set_ylabel('Re(y)')
    axs[0].set_title('Real Part')
    axs[0].legend()
    axs[0].grid(True)

    line_objective_imag, = axs[1].plot(inputs.real, target_func(inputs).imag, label='Objective Imag', color='blue')
    line_predictions_imag, = axs[1].plot(inputs.real, np.zeros_like(inputs.real), label='Prediction Imag', color='red', linestyle='--', linewidth=2)
    scatter_train_imag = axs[1].scatter(train_inputs.real, train_outputs.imag, label='Train Imag', s=10, color='green', alpha=0.5)
    axs[1].set_xlabel('Re(x)')
    axs[1].set_ylabel('Im(y)')
    axs[1].set_title('Imaginary Part')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show(block=False)

    last_update_time = time.time()
    update_interval = 1 / 120  # Seconds between visual updates

    while True:
        try:
            item = plot_queue.get(timeout=update_interval)
            if item is None:  # Signal to stop
                break
            predictions = item

            current_time = time.time()
            if current_time - last_update_time > update_interval:
                line_predictions_real.set_ydata(predictions.real)
                line_predictions_imag.set_ydata(predictions.imag)
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                last_update_time = current_time

        except queue.Empty:
            plt.pause(update_interval)
        except Exception as e:
            print(f"Error in plot process (output): {e}")
            break

    plt.close()

class LivePlotCallback(Callback):
    def __init__(self, plot_queue, inputs, model, update_frequency=10):
        self.plot_queue = plot_queue
        self.inputs = inputs
        self.model = model
        self.update_frequency = update_frequency
        self.last_put_time = 0

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.update_frequency == 0:
            current_time = time.time()
            with contextlib.suppress(queue.Full):
                predictions = self.model.predict(self.inputs)
                self.plot_queue.put_nowait(predictions)
                self.last_put_time = current_time

class LiveLossPlotCallback(Callback):
    """Sends loss data to a separate process for live plotting."""
    def __init__(self, plot_queue: mp.Queue):
        self.plot_queue = plot_queue
        self.train_losses_real = []
        self.train_losses_imag = []
        self.val_losses_real = []
        self.val_losses_imag = []

    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get('loss')
        val_loss = logs.get('val_loss')

        if loss is not None:
            self.train_losses_real.append(np.real(loss))
            self.train_losses_imag.append(np.imag(loss))
        if val_loss is not None:
            self.val_losses_real.append(np.real(val_loss))
            self.val_losses_imag.append(np.imag(val_loss))

        if loss is not None:
            plot_data = {
                'epoch': epoch + 1,
                'train_loss_real': np.real(loss),
                'train_loss_imag': np.imag(loss),
                'val_loss_real': np.real(val_loss) if val_loss is not None else None,
                'val_loss_imag': np.imag(val_loss) if val_loss is not None else None
            }
            try:
                self.plot_queue.put_nowait(plot_data)
            except queue.Full:
                pass # Skip if plot process is lagging

def plot_loss_process(plot_queue: mp.Queue, num_epochs: int):
    """Target function for the plotting process."""
    plt.ion()
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))  # 2 subplots for real and imaginary loss

    # Real part of the loss
    ax_real = axs[0]
    ax_real.set_xlabel('Epoch')
    ax_real.set_ylabel('Real Loss')
    ax_real.set_title('Live Training and Validation Loss (Real)')
    ax_real.grid(True)
    line_train_real, = ax_real.plot([], [], label='Training Loss (Real)', color='blue')
    line_val_real, = ax_real.plot([], [], label='Validation Loss (Real)', color='orange')
    ax_real.legend()

    # Imaginary part of the loss
    ax_imag = axs[1]
    ax_imag.set_xlabel('Epoch')
    ax_imag.set_ylabel('Imaginary Loss')
    ax_imag.set_title('Live Training and Validation Loss (Imaginary)')
    ax_imag.grid(True)
    line_train_imag, = ax_imag.plot([], [], label='Training Loss (Imaginary)', color='blue', linestyle='--')
    line_val_imag, = ax_imag.plot([], [], label='Validation Loss (Imaginary)', color='orange', linestyle='--')
    ax_imag.legend()

    epochs_data = []
    train_loss_real_data = []
    train_loss_imag_data = []
    val_loss_real_data = []
    val_loss_imag_data = []

    plt.tight_layout()
    plt.show(block=False)
    last_update_time = time.time()

    while True:
        try:
            plot_data = plot_queue.get(timeout=0.1)
            if plot_data is None:
                break

            current_time = time.time()
            if current_time - last_update_time < PLOT_UPDATE_INTERVAL and plot_data.get('epoch', 0) < num_epochs:
                continue

            epoch = plot_data.get('epoch')
            train_loss_real = plot_data.get('train_loss_real')
            train_loss_imag = plot_data.get('train_loss_imag')
            val_loss_real = plot_data.get('val_loss_real')
            val_loss_imag = plot_data.get('val_loss_imag')

            if epoch is not None and train_loss_real is not None and train_loss_imag is not None:
                epochs_data.append(epoch)
                train_loss_real_data.append(train_loss_real)
                train_loss_imag_data.append(train_loss_imag)
                line_train_real.set_xdata(epochs_data)
                line_train_real.set_ydata(train_loss_real_data)
                line_train_imag.set_xdata(epochs_data)
                line_train_imag.set_ydata(train_loss_imag_data)

            if epoch is not None and val_loss_real is not None and val_loss_imag is not None:
                val_loss_real_data.append(val_loss_real)
                val_loss_imag_data.append(val_loss_imag)
                line_val_real.set_xdata(epochs_data)
                line_val_real.set_ydata(val_loss_real_data)
                line_val_imag.set_xdata(epochs_data)
                line_val_imag.set_ydata(val_loss_imag_data)

            ax_real.relim()
            ax_real.autoscale_view()
            ax_imag.relim()
            ax_imag.autoscale_view()
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            last_update_time = current_time

        except queue.Empty:
            plt.pause(0.01)
        except Exception as e:
            print(f"Error in plot process (loss): {e}")
            import traceback
            traceback.print_exc()
            break

    plt.ioff()
    print("Loss plotting process finished. Close the plot window manually.")
    plt.show(block=True)
    plt.close(fig)

def func(x):
    return x * 1j

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    num_samples = 5000
    inputs = np.linspace(0, 10, num_samples).reshape(-1, 1)
    outputs = func(inputs)

    input_dims = signature(func).parameters.__len__()
    out_dims = np.array(func(*np.random.randn(input_dims))).size

    train_inputs = inputs
    train_outputs = outputs

    layers = [DenseLayer(16, activation_function=leaky_relu) for _ in range(1)]
    layers.extend([DenseLayer(out_dims, activation_function=linear)])

    adam = Adam(learning_rate=1e-2, gradient_clip=None)
    adamW = AdamW(learning_rate=1e-6, weight_decay=0, gradient_clip=None)
    amsgrad = AMSGrad(learning_rate=1e-6, weight_decay=0, gradient_clip=None)
    sgd = SGD(learning_rate=1e-12, momentum=1/2, nesterov=True)
    rmsprop = RMSprop(learning_rate=1e-6, rho=0.9, gradient_clip=None)

    nn = NeuralNetwork(layers, gradient_clip=None, l1_lambda=0, l2_lambda=0, use_complex=True)

    nn.compile(optimizer=adam, loss=mean_squared_error)

    output_plot_queue = mp.Queue(maxsize=OUTPUT_PLOT_QUEUE_MAXSIZE)
    loss_plot_queue = mp.Queue(maxsize=LOSS_PLOT_QUEUE_MAXSIZE)

    live_output_plot_callback = LivePlotCallback(output_plot_queue, inputs, nn, update_frequency=1)
    live_loss_plot_callback = LiveLossPlotCallback(loss_plot_queue)
    loss_callback = LossCallback()

    output_plot_process = mp.Process(
        target=plot_process,
        args=(output_plot_queue, inputs, func, train_inputs, train_outputs),
        daemon=True
    )
    print("Starting output plotting process...")
    output_plot_process.start()

    loss_plot_process = mp.Process(
        target=plot_loss_process,
        args=(loss_plot_queue, 1000),  # Use the number of epochs from fit
        daemon=True
    )
    print("Starting loss plotting process...")
    loss_plot_process.start()

    profiler = cProfile.Profile()
    profiler.enable()

    try:
        nn.fit(
            train_inputs,
            train_outputs,
            epochs=1000,  # Reduce epochs for profiling to get a manageable output
            batch_size=num_samples // 64,
            callbacks=[loss_callback, live_output_plot_callback, live_loss_plot_callback],
            restore_best_weights=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    finally:
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('tottime')
        print("\n--- cProfile Analysis ---")
        stats.print_stats(20)  # Print the top 20 functions by total time

        # Cleanup output plotting process
        print("Cleaning up output plotting process...")
        try:
            output_plot_queue.put_nowait(None)
        except queue.Full:
            pass
        output_plot_process.join(timeout=5.0)
        if output_plot_process.is_alive():
            print("Output plot process did not exit gracefully, terminating...")
            output_plot_process.terminate()
            output_plot_process.join(timeout=1.0)
        output_plot_queue.close()
        output_plot_queue.join_thread()

        # Cleanup loss plotting process
        print("Cleaning up loss plotting process...")
        try:
            loss_plot_queue.put_nowait(None)
        except queue.Full:
            pass
        loss_plot_process.join(timeout=5.0)
        if loss_plot_process.is_alive():
            print("Loss plot process did not exit gracefully, terminating...")
            loss_plot_process.terminate()
            loss_plot_process.join(timeout=1.0)
        loss_plot_queue.close()
        loss_plot_queue.join_thread()

        print("Cleanup complete.")