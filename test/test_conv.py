# --- Imports ---
import sys
import os
import time
import queue
import multiprocessing as mp
import contextlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# --- Add project root to path ---
# Ensure the network modules can be found
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Make sure binary_cross_entropy is imported or defined in your network.functions
from network.functions import leaky_relu, sigmoid, binary_cross_entropy, linear
from network.optimizer import Adam
from network.layer import ConvNDLayer, FlattenLayer, DenseLayer
from network.neuralNetwork import NeuralNetwork
from network.utils.callbacks import Callback


# --- Configuration ---
IMG_SIZE = 16
CHANNELS = 1
NUM_CLASSES = 1 # Binary classification (0: circle, 1: square)
NUM_SAMPLES = 1000
BATCH_SIZE = 32
EPOCHS = 10000
LEARNING_RATE = 1e-2
PLOT_UPDATE_FREQ = 1 # Update plot every N epochs
FEATURE_MAPS_TO_SHOW = 2 # How many feature maps to visualize

# --- Data Generation ---
def generate_shape_data(num_samples, img_size, channels):
    """Generates images with circles or squares."""
    images = np.zeros((num_samples, img_size, img_size, channels), dtype=np.float32)
    labels = np.zeros((num_samples, 1), dtype=np.float32)
    padding = max(1, img_size // 10)  # Dynamically adjust padding based on image size

    for i in range(num_samples):
        label = np.random.randint(0, 2)  # 0 for circle, 1 for square
        labels[i] = label
        center_x, center_y = np.random.randint(padding, img_size - padding, size=2)
        size = np.random.randint(img_size // 5, img_size // 2 - padding)  # Adjust size range dynamically

        if label == 0:  # Draw Circle
            radius = size // 2
            y, x = np.ogrid[:img_size, :img_size]
            dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            mask = dist_from_center <= radius
        else:  # Draw Square
            half_size = size // 2
            x_start, x_end = max(0, center_x - half_size), min(img_size, center_x + half_size)
            y_start, y_end = max(0, center_y - half_size), min(img_size, center_y + half_size)
            mask = np.zeros((img_size, img_size), dtype=bool)
            mask[y_start:y_end, x_start:x_end] = True

        images[i, :, :, 0] = mask.astype(np.float32)
        images[i] += np.random.randn(img_size, img_size, channels).astype(np.float32) * 0.05
        images[i] = np.clip(images[i], 0.0, 1.0)

    return images, labels

# --- Callbacks ---
class PrintLossCallback(Callback):
    """Prints loss information periodically."""
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            loss = logs.get('loss', 'N/A')
            val_loss = logs.get('val_loss', 'N/A')
            # Ensure loss is formatted correctly even if it's None or str
            loss_str = f"{loss:.4f}" if isinstance(loss, (int, float)) else str(loss)
            val_loss_str = f"{val_loss:.4f}" if isinstance(val_loss, (int, float)) else str(val_loss)
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss_str}, Val Loss: {val_loss_str}")


class LivePlotCallback(Callback):
    """Sends data to the plotting process."""
    def __init__(self, plot_queue, sample_input, model, update_frequency=1):
        self.plot_queue = plot_queue
        self.sample_input = sample_input
        self.model = model
        self.update_frequency = update_frequency
        self.epoch_losses = []
        self.val_epoch_losses = []

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_losses.append(logs.get('loss'))
        self.val_epoch_losses.append(logs.get('val_loss'))

        if epoch % self.update_frequency == 0 or epoch == EPOCHS - 1:
            try:
                # Get Feature Maps
                feature_maps = None
                if self.model.layers and isinstance(self.model.layers[0], ConvNDLayer):
                    try:
                        first_conv_layer = self.model.layers[0]
                        feature_map_model = NeuralNetwork([first_conv_layer], loss_function=None)
                        if hasattr(first_conv_layer, 'weights') and hasattr(self.model.layers[0], 'weights'):
                           feature_map_model.layers[0].weights = self.model.layers[0].weights
                        if hasattr(first_conv_layer, 'bias') and hasattr(self.model.layers[0], 'bias'):
                           feature_map_model.layers[0].bias = self.model.layers[0].bias

                        feature_maps_raw = feature_map_model.predict(self.sample_input)
                        num_maps = feature_maps_raw.shape[-1]
                        maps_to_plot = min(num_maps, FEATURE_MAPS_TO_SHOW)
                        feature_maps = feature_maps_raw[0, :, :, :maps_to_plot]

                    except Exception as e:
                        # print(f"\nWarning: Could not get feature maps: {e}") # Optional debug
                        dummy_map_shape = (self.sample_input.shape[1], self.sample_input.shape[2], FEATURE_MAPS_TO_SHOW)
                        feature_maps = np.random.rand(*dummy_map_shape) * 0.1

                # Get Prediction
                prediction = self.model.predict(self.sample_input)[0, 0]

                # Prepare Data for Queue
                plot_data = {
                    'epoch': epoch + 1,
                    'losses': list(self.epoch_losses),
                    'val_losses': list(self.val_epoch_losses),
                    'input_sample': self.sample_input[0, :, :, 0],
                    'feature_maps': feature_maps,
                    'prediction': prediction
                }
                self.plot_queue.put_nowait(plot_data)

            except queue.Full:
                pass
            except Exception as e:
                print(f"\nError in LivePlotCallback: {e}")


# --- Plotting Process ---
def plot_process(plot_queue, num_epochs, sample_label):
    """Target function for the plotting process."""
    plt.ion()

    fig = plt.figure(figsize=(12, 8))
    # Corrected GridSpec: Add 1 column for the input/pred plots
    gs = gridspec.GridSpec(3, FEATURE_MAPS_TO_SHOW + 1, figure=fig)

    ax_loss = fig.add_subplot(gs[0, :])      # Loss plot spans top row
    ax_input = fig.add_subplot(gs[1, 0])     # Input image in row 1, col 0
    ax_pred = fig.add_subplot(gs[2, 0])      # Prediction text in row 2, col 0

    # Place feature maps in columns 1 onwards, spanning rows 1 and 2
    ax_features = []
    for i in range(FEATURE_MAPS_TO_SHOW):
        ax = fig.add_subplot(gs[1:, i + 1]) # Corrected: Span rows 1-2, column i+1
        ax_features.append(ax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'Map {i+1}', fontsize=8)

    # Initialize Plots
    line_loss, = ax_loss.plot([], [], 'r-', label='Training Loss')
    line_val_loss, = ax_loss.plot([], [], 'b--', label='Validation Loss')
    ax_loss.set_xlim(0, num_epochs)
    ax_loss.set_ylim(0, 1.0)
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')
    ax_loss.set_title('Training Progress')
    ax_loss.legend()
    ax_loss.grid(True)

    img_input = ax_input.imshow(np.zeros((IMG_SIZE, IMG_SIZE)), cmap='viridis', vmin=0, vmax=1)
    true_label_str = "Circle" if sample_label == 0 else "Square"
    ax_input.set_title(f'Sample Input (True: {true_label_str})')
    ax_input.set_xticks([])
    ax_input.set_yticks([])

    img_features = [ax.imshow(np.zeros((IMG_SIZE, IMG_SIZE)), cmap='viridis') for ax in ax_features]

    pred_text = ax_pred.text(0.5, 0.5, 'Pred: N/A', horizontalalignment='center', verticalalignment='center', fontsize=12)
    ax_pred.axis('off')

    fig.tight_layout(pad=2.0)
    plt.show(block=False)

    # Update Loop
    last_update_time = time.time()
    update_interval = 0.1

    while True:
        try:
            plot_data = plot_queue.get(timeout=0.05)

            if plot_data is None:
                break

            current_time = time.time()
            if current_time - last_update_time < update_interval and plot_data['epoch'] < num_epochs:
                 continue

            # Update Plot Data
            epochs = list(range(1, plot_data['epoch'] + 1))
            losses = [l for l in plot_data['losses'] if l is not None]
            val_losses = [l for l in plot_data['val_losses'] if l is not None]

            if losses:
                 line_loss.set_data(epochs[:len(losses)], losses)
            if val_losses:
                 line_val_loss.set_data(epochs[:len(val_losses)], val_losses)

            if combined_losses := losses + val_losses:
                min_l = min(combined_losses)
                max_l = max(combined_losses)
                y_min = max(0, min_l - 0.1 * (max_l - min_l))
                y_max = max_l + 0.1 * (max_l - min_l)
                ax_loss.set_ylim(y_min, max(y_max, 0.1))
            ax_loss.relim()
            ax_loss.autoscale_view(True, True, True)

            img_input.set_data(plot_data['input_sample'])
            img_input.autoscale()

            feature_maps = plot_data['feature_maps']
            if feature_maps is not None:
                for i, im_ax in enumerate(img_features):
                    if i < feature_maps.shape[-1]:
                        feature_map_data = feature_maps[:, :, i]
                        im_ax.set_data(feature_map_data)
                        im_ax.set_clim(vmin=feature_map_data.min(), vmax=feature_map_data.max())
                    else:
                         im_ax.set_data(np.zeros((IMG_SIZE, IMG_SIZE)))
                         im_ax.set_clim(vmin=0, vmax=1)

            pred_val = plot_data['prediction']
            pred_label = "Square" if pred_val >= 0.5 else "Circle"
            pred_text.set_text(f'Pred: {pred_label} ({pred_val:.2f})')

            # Redraw Canvas
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


# --- Main Execution ---
if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError as e:
         print(f"Note: Could not set multiprocessing start method to 'spawn'. Using default. ({e})")

    print("Generating training data...")
    X_train, y_train = generate_shape_data(NUM_SAMPLES, IMG_SIZE, CHANNELS)
    print("Generating validation data...")
    X_val, y_val = generate_shape_data(NUM_SAMPLES // 4, IMG_SIZE, CHANNELS)

    sample_idx = np.random.randint(0, X_val.shape[0])
    sample_input_vis = X_val[sample_idx:sample_idx+1]
    sample_label_vis = y_val[sample_idx, 0]

    # Define Model
    layers = [
        ConvNDLayer(num_filters=8, kernel_size=(3, 3), stride=1, activation_function=leaky_relu, input_shape=(IMG_SIZE, IMG_SIZE, CHANNELS)),
        ConvNDLayer(num_filters=16, kernel_size=(3, 3), stride=2, activation_function=leaky_relu),
        FlattenLayer(),
        DenseLayer(32, activation_function=leaky_relu),
        DenseLayer(NUM_CLASSES, activation_function=sigmoid) # Sigmoid for binary output
    ]

    # Compile Model
    optimizer = Adam(learning_rate=LEARNING_RATE)
    loss_function = binary_cross_entropy # Corrected: Use binary loss for sigmoid output
    nn = NeuralNetwork(layers)
    nn.compile(optimizer=optimizer, loss=loss_function)
    print("Model compiled.")

    # Setup Plotting
    plot_queue = mp.Queue(maxsize=5)
    print_loss_callback = PrintLossCallback()
    live_plot_callback = LivePlotCallback(plot_queue, sample_input_vis, nn, update_frequency=PLOT_UPDATE_FREQ)

    # Create and start the plotting process
    plot_proc = mp.Process(
        target=plot_process,
        args=(plot_queue, EPOCHS, sample_label_vis),
        daemon=True
    )
    print("Starting plotting process...")
    plot_proc.start()

    # Train Model
    print("Starting training...")
    try:
        history = nn.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_val, y_val),
            callbacks=[print_loss_callback, live_plot_callback],
            restore_best_weights=False
        )
        print("\nTraining finished.")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("Cleaning up plotting process...")
        try:
            plot_queue.put_nowait(None)
        except queue.Full:
            while not plot_queue.empty():
                try: plot_queue.get_nowait()
                except queue.Empty: break
            try: plot_queue.put(None, timeout=1.0)
            except queue.Full: print("Warning: Could not send termination signal.")

        plot_proc.join(timeout=2.0)
        if plot_proc.is_alive():
            print("Plot process did not exit gracefully, terminating...")
            plot_proc.terminate()
            plot_proc.join(timeout=1.0)

        plot_queue.close()
        plot_queue.join_thread()
        print("Cleanup complete.")