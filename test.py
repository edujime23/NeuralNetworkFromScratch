from utils.functions import CostFunctions, ActivationFunctions
from utils.optimizer import *
from utils.layers import DenseLayer, Flatten, Conv2DLayer
from neuralNetwork import NeuralNetwork
from utils.callbacks import Callback
import numpy as np

class back(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 100 == 0:print(f"Epoch {epoch+1}, Loss: {logs['loss']:.4f}, Val Loss: {logs.get('val_loss', 'N/A'):.4f}")

# XOR dataset
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([[0], [1], [1], [0]])

validation_inputs = np.array([[0, 0], [1, 1]])
validation_outputs = np.array([[0], [0]])

# Reshape inputs for Conv2D layer (batch_size, height, width, channels)
inputs = inputs.reshape(inputs.shape[0], 2, 1, 1)
validation_inputs = validation_inputs.reshape(validation_inputs.shape[0], 2, 1, 1)

# Create the neural network with a Conv2D layer
layers = [
    Conv2DLayer(4, (2, 1), input_shape=(2, 1, 1), activation_function=ActivationFunctions.leaky_relu),
    Flatten(),
    DenseLayer(8, 4 * 1 * 1, ActivationFunctions.leaky_relu, dropout_rate=0.25, batch_norm=True), # Input size is num_filters * output_height * output_width = 4 * 1 * 1 = 4
    DenseLayer(1, 8, ActivationFunctions.sigmoid)
]
nn = NeuralNetwork(layers, CostFunctions.cross_entropy, threshold=1.0, gradient_clip=1.0)

nn.compile(optimizer=AdamOptimizer(0.005))

# Train the network
nn.fit(inputs, outputs, epochs=2000, batch_size=4, validation_data=(validation_inputs, validation_outputs), callbacks=[back()])

# Test the trained network
print("Testing trained network:")
test_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape(4, 2, 1, 1)
for input_data in test_inputs:
    output = nn.predict([input_data])
    print(f"Input: {input_data.flatten()}, Output: {output[0]}")