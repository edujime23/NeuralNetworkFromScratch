from utils.functions import CostFunctions, ActivationFunctions
from utils.optimizer import AdamOptimizer
from utils.layers import DenseLayer, Flatten, Conv2DLayer
from neuralNetwork import NeuralNetwork
from utils.callbacks import Callback
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy.signal
from utils.metrics import Metrics # Ensure this import is correct

class back(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 1 == 0: print(f"Epoch {epoch+1}, Loss: {logs['loss']:.4f}, Val Loss: {logs.get('val_loss', 'N/A'):.4f}")

def generate_vertical_line_image(size=16, line_col=8):
    image = np.zeros((size, size, 1), dtype=np.float32)
    image[:, line_col, 0] = 1.0
    return image

def generate_horizontal_line_image(size=16, line_row=8):
    image = np.zeros((size, size, 1), dtype=np.float32)
    image[line_row, :, 0] = 1.0
    return image

# Generate training data for classifying horizontal vs. vertical lines
num_samples = 2000
inputs = np.zeros((num_samples, 16, 16, 1), dtype=np.float32)
outputs = np.zeros((num_samples, 1), dtype=np.float32) # Output is 0 for horizontal, 1 for vertical

for i in range(num_samples // 2):
    line_pos = np.random.randint(0, 16)
    inputs[i] = generate_horizontal_line_image(size=16, line_row=line_pos)
    outputs[i] = 0.0 # Horizontal

for i in range(num_samples // 2, num_samples):
    line_pos = np.random.randint(0, 16)
    inputs[i] = generate_vertical_line_image(size=16, line_col=line_pos)
    outputs[i] = 1.0 # Vertical

# Shuffle the training data
indices = np.arange(inputs.shape[0])
np.random.shuffle(indices)
inputs = inputs[indices]
outputs = outputs[indices]

# Generate validation data
num_val_samples = 200
validation_inputs = np.zeros((num_val_samples, 16, 16, 1), dtype=np.float32)
validation_outputs = np.zeros((num_val_samples, 1), dtype=np.float32)

for i in range(num_val_samples // 2):
    line_pos = np.random.randint(0, 16)
    validation_inputs[i] = generate_horizontal_line_image(size=16, line_row=line_pos)
    validation_outputs[i] = 0.0 # Horizontal

for i in range(num_val_samples // 2, num_val_samples):
    line_pos = np.random.randint(0, 16)
    validation_inputs[i] = generate_vertical_line_image(size=16, line_col=line_pos)
    validation_outputs[i] = 1.0 # Vertical

# Create a simple convolutional network for binary classification
layers = [
    Conv2DLayer(8, (3, 3), input_shape=(16, 16, 1), activation_function=ActivationFunctions.leaky_relu, padding='same'),
    Conv2DLayer(16, (3, 3), activation_function=ActivationFunctions.leaky_relu, padding='same'),
    Flatten(),
    DenseLayer(32, num_inputs=16 * 16 * 16, activation_function=ActivationFunctions.leaky_relu),
    DenseLayer(1, num_inputs=32, activation_function=ActivationFunctions.sigmoid)
]
nn = NeuralNetwork(layers, CostFunctions.binary_cross_entropy, threshold=0.5, gradient_clip=1)
nn.compile(optimizer=AdamOptimizer(learning_rate=0.0001))

# Train the network
nn.fit(inputs, outputs, epochs=25, batch_size=4, validation_data=(validation_inputs, validation_outputs), callbacks=[back()])

# Print initial weights and biases of the last layer
last_layer = nn.layers[-1]
print("\nInitial Weights of the Last Layer (first neuron):", last_layer.neurons[0].weights)
print("Initial Bias of the Last Layer (first neuron):", last_layer.neurons[0].bias)

# Print initial predictions on a few training samples
print("Initial Predictions (untrained network):")
for i in range(5):
    prediction = nn.predict(inputs[i].reshape(1, 16, 16, 1))
    print(f"Input shape: {inputs[i].shape}, Predicted probability (vertical): {prediction[0, 0]:.4f}, True label (vertical): {outputs[i, 0]:.1f}")

# Test the trained network
print("\nTesting trained network:")
# Test with a horizontal line
test_input_hor = generate_horizontal_line_image(size=16, line_row=7).reshape(1, 16, 16, 1)
predicted_output_hor = nn.predict(test_input_hor)
prediction_hor = "Vertical" if predicted_output_hor[0, 0] > 0.5 else "Horizontal"
print(f"Horizontal Input Prediction (Probability of being vertical: {predicted_output_hor[0, 0]:.4f}): {prediction_hor}")

# Test with a vertical line
test_input_ver = generate_vertical_line_image(size=16, line_col=7).reshape(1, 16, 16, 1)
predicted_output_ver = nn.predict(test_input_ver)
prediction_ver = "Vertical" if predicted_output_ver[0, 0] > 0.5 else "Horizontal"
print(f"Vertical Input Prediction (Probability of being vertical: {predicted_output_ver[0, 0]:.4f}): {prediction_ver}")

# Visualize the results
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(test_input_hor[0, :, :, 0], cmap='gray')
plt.title(f"Input: Horizontal\nPredicted: {prediction_hor}")

plt.subplot(1, 2, 2)
plt.imshow(test_input_ver[0, :, :, 0], cmap='gray')
plt.title(f"Input: Vertical\nPredicted: {prediction_ver}")

plt.tight_layout()
plt.show()