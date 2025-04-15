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
            predictions = plot_queue.get(timeout=1 / 120)
            if predictions is None:  # Signal to stop
                break

            current_time = time.time()
            if current_time - last_update_time > update_interval:
                line_predictions_real.set_ydata(predictions.real)
                line_predictions_imag.set_ydata(predictions.imag)
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                last_update_time = current_time

        except queue.Empty:
            plt.pause(1 / 120)
        except Exception as e:
            print(f"Error in plot process: {e}")
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

def func(x):
    return 1j * x + x

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    num_samples = 5000
    inputs = np.linspace(0, 10, num_samples).astype(np.complex64).reshape(-1, 1)  # Complex input
    outputs = func(inputs)

    input_dims = signature(func).parameters.__len__()
    out_dims = np.array(func(*np.random.randn(input_dims).astype(np.complex64))).size

    train_inputs = inputs
    train_outputs = outputs

    layers = [DenseLayer(8, activation_function=leaky_relu) for _ in range(4)]
    layers.extend([DenseLayer(out_dims, activation_function=linear)])

    adam = Adam(learning_rate=1e-4, weight_decay=0, gradient_clip=None)
    adamW = AdamW(learning_rate=1e-4, weight_decay=0, gradient_clip=None)
    amsgrad = AMSGrad(learning_rate=1e-4, weight_decay=0, gradient_clip=None)
    sgd = SGD(learning_rate=1e-6, momentum=1 / 2, nesterov=True)
    rmsprop = RMSprop(learning_rate=1e-3, rho=0.9, gradient_clip=1)

    nn = NeuralNetwork(layers, mean_squared_error, gradient_clip=None, l1_lambda=0, l2_lambda=0, use_complex=True)
    nn.compile(optimizer=adam)

    plot_queue = mp.Queue()

    live_plot_callback = LivePlotCallback(plot_queue, inputs, nn, update_frequency=1)
    loss_callback = LossCallback()

    plot_process_instance = mp.Process(
        target=plot_process,
        args=(plot_queue, inputs, func, train_inputs, train_outputs),
    )
    plot_process_instance.daemon = True
    plot_process_instance.start()

    try:
        nn.fit(
            train_inputs,
            train_outputs,
            epochs=2 ** 64,
            batch_size=num_samples // 256,
            callbacks=[loss_callback, live_plot_callback],
            restore_best_weights=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    finally:
        try:
            plot_queue.put_nowait(None)
        except queue.Full:
            while not plot_queue.empty():
                try:
                    plot_queue.get_nowait()
                except queue.Empty:
                    break
            plot_queue.put(None)

        plot_process_instance.join(timeout=1.0)

        if plot_process_instance.is_alive():
            plot_process_instance.terminate()
            plot_process_instance.join()
            plot_process_instance.close()

        print("Training completed.")