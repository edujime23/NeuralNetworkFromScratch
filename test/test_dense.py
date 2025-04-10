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
        if epoch % 100 == 0:
            print(f"Epoch {epoch+1}, Loss: {logs['loss']:.4f}, Val Loss: {logs.get('val_loss', 'N/A')}")

def plot_process(plot_queue, inputs, target_func, train_inputs, train_outputs):
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    line_objective, = ax.plot(inputs, target_func(inputs), label='Objective Function (func())', color='blue')
    scatter_train = ax.scatter(train_inputs, train_outputs, label='Training Data', s=10, color='green', alpha=0.5)
    line_predictions, = ax.plot(inputs, np.zeros_like(inputs), label='Neural Network Predictions', color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Approximation of the funky function')
    ax.legend()
    ax.grid(True)
    
    plt.show(block=False)
    
    last_update_time = time.time()
    update_interval = 1/120  # Seconds between visual updates to prevent excessive redrawing
    
    while True:
        try:
            # Non-blocking queue get with short timeout
            predictions = plot_queue.get(timeout=1/120)
            if predictions is None:  # Signal to stop
                break
                
            current_time = time.time()
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
    def __init__(self, plot_queue, inputs, model, update_frequency=10):
        self.plot_queue = plot_queue
        self.inputs = inputs
        self.model = model
        self.update_frequency = update_frequency
        self.last_put_time = 0
        
    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.update_frequency == 0:
            current_time = time.time()
            # Only update if queue isn't full and enough time has passed
            with contextlib.suppress(queue.Full):
                predictions = self.model.predict(self.inputs)
                # Use put_nowait to avoid blocking if queue is full
                self.plot_queue.put_nowait(predictions)
                self.last_put_time = current_time

def func(x):
    return np.sqrt(abs(x))**6

if __name__ == "__main__":
    # Using context manager to ensure clean process termination
    mp.set_start_method('spawn', force=True)  # More stable for GUI applications
    
    num_samples = 5000
    inputs = np.linspace(0, 10, num_samples).reshape(-1, 1)
    outputs = func(inputs)
    
    input_dims = signature(func).parameters.__len__()
    out_dims = np.array(func(*np.random.randn(input_dims))).size
    
    train_inputs = inputs
    train_outputs = outputs
    
    layers = [
        DenseLayer(64, activation_function=leaky_relu) for _ in range(8)
    ]
    
    layers.extend([
        DenseLayer(64, activation_function=linear),
        DenseLayer(64, activation_function=linear),
        DenseLayer(out_dims, activation_function=linear)
    ])
    
    adam = Adam(learning_rate=1e-4, 
                        weight_decay=0, 
                        gradient_clip=None
    )
    adamW = AdamW(learning_rate=1e-4, 
                        weight_decay=0, 
                        gradient_clip=None
    )
    amsgrad = AMSGrad(learning_rate=1e-4, weight_decay=0, gradient_clip=None)
    sgd = SGD(learning_rate=1e-4, momentum=1/2, nesterov=True)
    rmsprop = RMSprop(learning_rate=1e-3, rho=0.9, gradient_clip=1)
    
    nn = NeuralNetwork(layers, huber_loss, gradient_clip=None, l1_lambda=0, l2_lambda=0)
    nn.compile(optimizer=adamW)
    
    # Create a multiprocessing Queue with limited size to avoid memory issues
    plot_queue = mp.Queue()  # Only keep latest few predictions
    
    # Instantiate the LivePlotCallback with the queue
    live_plot_callback = LivePlotCallback(plot_queue, inputs, nn, update_frequency=1)
    loss_callback = LossCallback()
    
    # Create and start the plotting process
    plot_process_instance = mp.Process(
        target=plot_process, 
        args=(plot_queue, inputs, func, train_inputs, train_outputs)
    )
    plot_process_instance.daemon = True  # Automatically terminate if main process ends
    plot_process_instance.start()
    
    try:
        nn.fit(
            train_inputs, 
            train_outputs, 
            epochs=2**64, 
            batch_size=num_samples//256, 
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