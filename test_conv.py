from utils.functions import CostFunctions, ActivationFunctions
from utils.optimizer import *
from layer import *
from neuralNetwork import NeuralNetwork
from utils.callbacks import Callback
import numpy as np
import matplotlib.pyplot as plt

class EpochEndCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch+1}, Loss: {logs['loss']}, Val Loss: {logs.get('val_loss', 'N/A')}")

def generate_sample_data(batch_size: int, spatial_dims: tuple, in_channels: int):
    """Generate sample data for testing convolution layers"""
    shape = (batch_size,) + spatial_dims + (in_channels,)
    return np.random.randn(*shape)

# Test 2D Convolution
print("\n--- Testing 2D Convolution ---")
batch_size = 32
spatial_dims_2d = (28, 28)  # Similar to MNIST dimensions
in_channels = 1
out_channels = 16

# Generate synthetic 2D data
inputs_2d = generate_sample_data(batch_size, spatial_dims_2d, in_channels)
outputs_2d = np.random.randint(0, 2, (batch_size, 1)).astype(np.float32)  # Binary classification

# Create a simple ConvNet
layers_2d = [
    ConvNDLayer(num_filters=16, kernel_size=(3, 3), stride=1, 
                activation_function=ActivationFunctions.relu),
    ConvNDLayer(num_filters=32, kernel_size=(3, 3), stride=2,
                activation_function=ActivationFunctions.relu),
    FlattenLayer(),
    DenseLayer(1, activation_function=ActivationFunctions.sigmoid)
]

nn_2d = NeuralNetwork(layers_2d, CostFunctions.cross_entropy)
nn_2d.compile(optimizer=AdamOptimizer(learning_rate=0.001))

# Train the network
nn_2d.fit(inputs_2d, outputs_2d, epochs=5, batch_size=16, 
          callbacks=[EpochEndCallback()], restore_best_weights=True)

# Test 3D Convolution
print("\n--- Testing 3D Convolution ---")
spatial_dims_3d = (16, 16, 16)  # 3D spatial dimensions
batch_size_3d = 16

# Generate synthetic 3D data
inputs_3d = generate_sample_data(batch_size_3d, spatial_dims_3d, in_channels)
outputs_3d = np.random.randint(0, 2, (batch_size_3d, 1)).astype(np.float32)  # Binary classification

# Create a 3D ConvNet
layers_3d = [
    ConvNDLayer(num_filters=8, kernel_size=(3, 3, 3), stride=1,
                activation_function=ActivationFunctions.relu),
    ConvNDLayer(num_filters=16, kernel_size=(3, 3, 3), stride=2,
                activation_function=ActivationFunctions.relu),
    FlattenLayer(),
    DenseLayer(1, activation_function=ActivationFunctions.sigmoid)
]

nn_3d = NeuralNetwork(layers_3d, CostFunctions.cross_entropy)
nn_3d.compile(optimizer=AdamOptimizer(learning_rate=0.001))

# Train the network
nn_3d.fit(inputs_3d, outputs_3d, epochs=50, batch_size=8,
          callbacks=[EpochEndCallback()], restore_best_weights=True)

# Test predictions
print("\n--- Testing Predictions ---")
# 2D predictions
test_input_2d = generate_sample_data(1, spatial_dims_2d, in_channels)
pred_2d = nn_2d.predict(test_input_2d)
print(f"2D ConvNet prediction: {pred_2d[0,0]}")

# 3D predictions
test_input_3d = generate_sample_data(1, spatial_dims_3d, in_channels)
pred_3d = nn_3d.predict(test_input_3d)
print(f"3D ConvNet prediction: {pred_3d[0,0]}")
