from utils.functions import CostFunctions, ActivationFunctions
from utils.optimizer import *
from utils.layers import DenseLayer
from neuralNetwork import NeuralNetwork
from utils.callbacks import Callback
import numpy as np

class back(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 100 == 0:print(f"Epoch {epoch+1}, Loss: {logs['loss']:.4f}, Val Loss: {logs.get('val_loss', 'N/A'):.4f}")

def universal_callback(epoch, logs):
    print(f"Epoch {epoch}: Loss = {logs['loss']}, Val Loss = {logs.get('val_loss', 'N/A')}")

# XOR dataset
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([[0], [1], [1], [0]])

validation_inputs = np.array([[0, 0], [1, 1]])
validation_outputs = np.array([[0], [0]])

# Create the neural network (2 inputs, 1 hidden layer of 4 neurons, and 1 output)
layers = [
    DenseLayer(4, 2, ActivationFunctions.leaky_relu, dropout_rate=0.25, batch_norm=True),
    DenseLayer(1, 4, ActivationFunctions.sigmoid)
]
nn = NeuralNetwork(layers, CostFunctions.cross_entropy, threshold=1.0, gradient_clip=1.0)

nn.compile(optimizer=AdamOptimizer(0.01))

# Train the network
nn.fit(inputs, outputs, epochs=2000, batch_size=4, validation_data=(validation_inputs, validation_outputs), callbacks=[back()])

# Test the trained network
print("Testing trained network:")
for input_data in inputs:
    output = nn.predict([input_data])
    print(f"Input: {input_data}, Output: {output[0]}")
