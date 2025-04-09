# --- Imports ---
import sys
import os
import time as time_module # Renamed to avoid conflict
import queue
import multiprocessing as mp
import contextlib
import numpy as np
import matplotlib.pyplot as plt

# --- Add project root to path ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Ensure these are correctly implemented in your library
from network.functions import linear, mean_squared_error
from network.optimizer import Adam, SGD
from network.layer import LSTMLayer, DenseLayer
from network.neuralNetwork import NeuralNetwork
from network.utils.callbacks import Callback

# --- Configuration ---
NUM_SAMPLES = 5000    # Reduced for faster testing, increase if needed
SEQUENCE_LENGTH = 10    # Input sequence length
TRAIN_SPLIT_RATIO = 0.8
FORECAST_STEPS = 100     # Number of future steps to forecast autoregressively
LSTM_UNITS = 32         # Number of units in the LSTM layer
LEARNING_RATE = 1e-4    # Adjusted learning rate might be needed
BATCH_SIZE = 64
EPOCHS = 10000            # Set max epochs (can be interrupted)
PLOT_UPDATE_FREQ = 1    # Update plot every N epochs

# --- Callbacks ---
class PrintLossCallback(Callback):
    """Prints loss information periodically."""
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 == 0 or epoch == EPOCHS - 1: # Print less frequently
            loss = logs.get('loss', 'N/A')
            val_loss = logs.get('val_loss', 'N/A') # If validation is used in fit
            loss_str = f"{loss:.4f}" if isinstance(loss, (int, float)) else str(loss)
            val_loss_str = f"{val_loss:.4f}" if isinstance(val_loss, (int, float)) else str(val_loss)
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss_str}, Val Loss: {val_loss_str}")

class LivePlotCallback(Callback):
    """Generates predictions and forecast, sends data to the plotting process."""
    def __init__(self, plot_queue, model, train_inputs, test_inputs, sequence_length, forecast_steps, update_frequency=1):
        self.plot_queue = plot_queue
        self.model = model
        self.train_inputs = train_inputs # Needed for starting forecast
        self.test_inputs = test_inputs   # Needed for test predictions
        self.sequence_length = sequence_length
        self.forecast_steps = forecast_steps
        self.update_frequency = update_frequency

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.update_frequency == 0 or epoch == EPOCHS - 1:
            try:
                # 1. Predict on the test set
                if self.test_inputs.size > 0:
                    test_predictions = self.model.predict(self.test_inputs).flatten()
                else:
                    test_predictions = np.array([])

                # 2. Generate autoregressive forecast
                forecast = []
                # Start forecasting from the last sequence in the training data
                current_sequence = self.train_inputs[-1].reshape(1, self.sequence_length, 1)

                for _ in range(self.forecast_steps):
                    next_pred = self.model.predict(current_sequence).flatten()[0]
                    forecast.append(next_pred)
                    # Update sequence: remove first element, append prediction
                    new_sequence_data = np.append(current_sequence[0, 1:, 0], next_pred)
                    current_sequence = new_sequence_data.reshape(1, self.sequence_length, 1)

                # Prepare data for the queue
                plot_data = {
                    'epoch': epoch + 1,
                    'test_preds': test_predictions,
                    'forecast': np.array(forecast)
                }
                self.plot_queue.put_nowait(plot_data)

            except queue.Full:
                pass # Skip if plot process is lagging
            except Exception as e:
                print(f"\nError in LivePlotCallback: {e}")
                import traceback
                traceback.print_exc()

# --- Plotting Process ---
def plot_process(plot_queue, time_steps, original_series, train_targets, test_targets,
                 sequence_length, split_index, forecast_steps, num_epochs):
    """Target function for the plotting process."""
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 7))

    # Time vectors for plotting different segments
    full_time = time_steps
    train_time_orig = time_steps[:split_index]
    test_time_orig = time_steps[split_index:]
    # Time for targets/predictions (shifted by sequence_length)
    train_time_pred = time_steps[sequence_length:split_index]
    test_time_pred = time_steps[split_index + sequence_length:]
    # Time for forecast starts after the last training point
    forecast_time = time_steps[split_index : split_index + forecast_steps]

    # Ensure target lengths match time vectors
    train_targets = train_targets[:len(train_time_pred)]
    test_targets = test_targets[:len(test_time_pred)]

    # Plot original data (train/test)
    ax.plot(train_time_orig, original_series[:split_index], label='Original Train Data', color='gray', alpha=0.8)
    ax.plot(test_time_orig, original_series[split_index:], label='Original Test Data', color='black', alpha=0.8)

    # Plot target values (optional, can clutter plot)
    # ax.scatter(train_time_pred, train_targets, label='Train Targets', s=10, color='blue', alpha=0.3)
    # ax.scatter(test_time_pred, test_targets, label='Test Targets', s=15, color='green', alpha=0.5)

    # Initialize prediction/forecast lines
    line_test_preds, = ax.plot(test_time_pred, np.full(len(test_time_pred), np.nan), label='Test Set Predictions', color='red', linestyle='-', linewidth=2)
    line_forecast, = ax.plot(forecast_time, np.full(len(forecast_time), np.nan), label='Autoregressive Forecast', color='orange', linestyle=':', linewidth=2.5)

    ax.set_xlabel('Time Step')
    ax.set_ylabel('Value')
    ax.set_title('LSTM Time Series Prediction and Forecast')
    ax.legend(loc='upper left')
    ax.grid(True)
    epoch_text = ax.text(0.02, 0.02, '', transform=ax.transAxes) # Text for epoch number

    plt.show(block=False)

    last_update_time = time_module.time()
    update_interval = 0.1 # Min seconds between plot updates

    while True:
        try:
            plot_data = plot_queue.get(timeout=0.05)

            if plot_data is None: # Termination signal
                break

            current_time = time_module.time()
            # Throttle updates unless it's the last epoch
            if current_time - last_update_time < update_interval and plot_data['epoch'] < num_epochs:
                 continue

            epoch_text.set_text(f"Epoch: {plot_data['epoch']}")

            # Update Test Predictions
            test_preds = plot_data['test_preds']
            # Ensure length matches time vector before plotting
            len_to_plot_test = min(len(test_time_pred), len(test_preds))
            line_test_preds.set_ydata(test_preds[:len_to_plot_test])

            # Update Forecast
            forecast = plot_data['forecast']
            # Ensure length matches time vector before plotting
            len_to_plot_forecast = min(len(forecast_time), len(forecast))
            line_forecast.set_ydata(forecast[:len_to_plot_forecast])

            # Redraw
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            last_update_time = current_time

        except queue.Empty:
            plt.pause(0.05)
        except Exception as e:
            print(f"Error in plot process: {e}")
            import traceback
            traceback.print_exc()
            break

    plt.ioff()
    print("Plotting process finished. Close the plot window manually.")
    plt.show(block=True)
    plt.close(fig)

# --- Data Generation and Preparation ---
def generate_time_series(num_samples):
    """Generates a sine wave time series with some noise."""
    time_steps = np.linspace(0, 15 * np.pi, num_samples) # Increased range
    series = np.sin(time_steps) + 0.2 * np.cos(time_steps * 0.5) # More complex series
    series += 0.1 * np.random.randn(num_samples)
    return series.reshape(-1, 1), time_steps

def prepare_data(time_series, sequence_length):
    """Creates input sequences and corresponding targets."""
    inputs = []
    targets = []
    if len(time_series) <= sequence_length:
        return np.array(inputs), np.array(targets) # Return empty if not enough data

    for i in range(len(time_series) - sequence_length):
        inputs.append(time_series[i : i + sequence_length])
        targets.append(time_series[i + sequence_length])
    return np.array(inputs), np.array(targets)

# --- Main Execution ---
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    # 1. Generate Data
    original_series, time_steps = generate_time_series(NUM_SAMPLES)

    # 2. Split Data
    split_index = int(len(original_series) * TRAIN_SPLIT_RATIO)
    train_series = original_series[:split_index]
    test_series = original_series[split_index:]

    # 3. Prepare Sequences
    train_inputs, train_targets = prepare_data(train_series, SEQUENCE_LENGTH)
    test_inputs, test_targets = prepare_data(test_series, SEQUENCE_LENGTH)

    print(f"Train shapes: Inputs={train_inputs.shape}, Targets={train_targets.shape}")
    print(f"Test shapes: Inputs={test_inputs.shape}, Targets={test_targets.shape}")

    if train_inputs.size == 0:
        print("Not enough training data to create sequences. Exiting.")
        sys.exit(1)

    # 4. Define Model
    layers = [
        LSTMLayer(LSTM_UNITS, return_sequences=False),
        DenseLayer(1, activation_function=linear) # Predict single next value
    ]

    # 5. Compile Model
    optimizer = Adam(learning_rate=LEARNING_RATE, gradient_clip=1.0) # Added gradient clipping
    loss_function = mean_squared_error
    nn = NeuralNetwork(layers, loss_function)
    nn.compile(optimizer=optimizer)
    print("Model compiled.")

    # 6. Setup Plotting
    plot_queue = mp.Queue(maxsize=5)
    print_loss_callback = PrintLossCallback()
    # Pass necessary data splits to the callback
    live_plot_callback = LivePlotCallback(plot_queue, nn, train_inputs, test_inputs,
                                          SEQUENCE_LENGTH, FORECAST_STEPS,
                                          update_frequency=PLOT_UPDATE_FREQ)

    # Prepare arguments for the plot process
    plot_args = (
        plot_queue, time_steps, original_series,
        train_targets, test_targets, # Pass targets for potential plotting
        SEQUENCE_LENGTH, split_index, FORECAST_STEPS, EPOCHS
    )

    # Create and start the plotting process
    plot_process_instance = mp.Process(target=plot_process, args=plot_args, daemon=True)
    print("Starting plotting process...")
    plot_process_instance.start()

    # 7. Train Model
    print(f"Starting training for {EPOCHS} epochs...")
    try:
        # Note: Validation data is not explicitly passed to fit here,
        # but LivePlotCallback uses test_inputs for plotting evaluation.
        # You could add validation_data=(test_inputs, test_targets) to fit
        # if your fit method supports it and you want validation loss printed.
        nn.fit(
            train_inputs,
            train_targets,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[print_loss_callback, live_plot_callback],
            restore_best_weights=False # Keep False for live plotting
        )
        print("\nTraining finished.")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 8. Cleanup
        print("Cleaning up plotting process...")
        try:
            plot_queue.put_nowait(None)
        except queue.Full:
            while not plot_queue.empty():
                try: plot_queue.get_nowait()
                except queue.Empty: break
            try: plot_queue.put(None, timeout=1.0)
            except queue.Full: print("Warning: Could not send termination signal.")

        plot_process_instance.join(timeout=2.0)
        if plot_process_instance.is_alive():
            print("Plot process did not exit gracefully, terminating...")
            plot_process_instance.terminate()
            plot_process_instance.join(timeout=1.0)

        plot_queue.close()
        plot_queue.join_thread()
        print("Cleanup complete.")