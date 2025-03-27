from utils.functions import CostFunctions, ActivationFunctions
from utils.optimizer import *
from utils.layers import DenseLayer, Flatten, Conv2DLayer
from neuralNetwork import NeuralNetwork
from utils.callbacks import Callback
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy.signal

class back(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 1 == 0: print(f"Epoch {epoch+1}, Loss: {logs['loss']:.4f}, Val Loss: {logs.get('val_loss', 'N/A'):.4f}")

def generate_vertical_line_image(size=16, line_col=8):
    image = np.zeros((size, size, 1), dtype=np.float32)
    image[:, line_col, 0] = 1.0
    return image

# Generate training data
num_samples = 1000
inputs = np.zeros((num_samples, 16, 16, 1), dtype=np.float32)
outputs = np.zeros((num_samples, 16, 16, 1), dtype=np.float32)

for i in range(num_samples):
    line_col = np.random.randint(0, 16)
    input_image = generate_vertical_line_image(size=16, line_col=line_col)
    inputs[i] = input_image
    outputs[i] = input_image # Try to learn identity

# Generate validation data
num_val_samples = 10
validation_inputs = np.zeros((num_val_samples, 16, 16, 1), dtype=np.float32)
validation_outputs = np.zeros((num_val_samples, 16, 16, 1), dtype=np.float32)

for i in range(num_val_samples):
    line_col = np.random.randint(0, 16)
    input_image = generate_vertical_line_image(size=16, line_col=line_col)
    validation_inputs[i] = input_image
    validation_outputs[i] = input_image

# Create a very simple convolutional network
layers = [
    Conv2DLayer(4, (3, 3), input_shape=(16, 16, 1), activation_function=ActivationFunctions.sigmoid, padding='same') # Increased filters to 3
]
nn = NeuralNetwork(layers, CostFunctions.mean_squared_error, threshold=1, gradient_clip=1.0)
nn.compile(optimizer=AdamOptimizer(1e-1)) # Reduced learning rate

# Train the network
nn.fit(inputs, outputs, epochs=100, batch_size=64, validation_data=(validation_inputs, validation_outputs), callbacks=[back()]) # Increased epochs

# Test the trained network
print("\nTesting trained network:")
test_input = generate_vertical_line_image(size=16, line_col=7).reshape(1, 16, 16, 1)
predicted_output = nn.predict(test_input)

# Visualize the results
plt.figure(figsize=(6, 3))

plt.subplot(1, 2, 1)
plt.imshow(test_input[0, :, :, 0], cmap='gray')
plt.title("Input (Vertical Line)")

plt.subplot(1, 2, 2)
print(f"Min predicted_output: {np.min(predicted_output)}")
print(f"Max predicted_output: {np.max(predicted_output)}")
plt.imshow(predicted_output[0, :, :, 0], cmap='gray')
plt.title("Predicted Output")

plt.tight_layout()
plt.show()