from utils.functions import CostFunctions, ActivationFunctions
from utils.optimizer import *
from utils.layers import DenseLayer
from neuralNetwork import NeuralNetwork
from utils.callbacks import Callback
import numpy as np
import matplotlib.pyplot as plt
from utils.metrics import Metrics

class back(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0: print(f"Epoch {epoch+1}, Loss: {logs['loss']:.4f}, Val Loss: {logs.get('val_loss', 'N/A'):.4f}")

# Generate XOR training data
num_samples = 4000
inputs = np.random.randint(0, 2, size=(num_samples, 2)).astype(np.float32)
outputs = np.logical_xor(inputs[:, 0], inputs[:, 1]).astype(np.float32).reshape(-1, 1)

# Shuffle the training data
indices = np.arange(inputs.shape[0])
np.random.shuffle(indices)
inputs = inputs[indices]
outputs = outputs[indices]

# Generate XOR validation data
num_val_samples = 400
validation_inputs = np.random.randint(0, 2, size=(num_val_samples, 2)).astype(np.float32)
validation_outputs = np.logical_xor(validation_inputs[:, 0], validation_inputs[:, 1]).astype(np.float32).reshape(-1, 1)

# Create a simple neural network with only dense layers for XOR
layers = [
    DenseLayer(5, num_inputs=2, activation_function=ActivationFunctions.leaky_relu),
    DenseLayer(1, num_inputs=5, activation_function=ActivationFunctions.sigmoid)
]
nn = NeuralNetwork(layers, CostFunctions.mean_squared_error)
nn.compile(optimizer=AdamOptimizer(learning_rate=0.1))

# Train the network
nn.fit(inputs, outputs, epochs=1000, batch_size=16, validation_data=(validation_inputs, validation_outputs), callbacks=[back()])

# Print initial weights and biases of the last layer
last_layer = nn.layers[-1]
print("\nInitial Weights of the Last Layer (first neuron):", last_layer.neurons[0].weights)
print("Initial Bias of the Last Layer (first neuron):", last_layer.neurons[0].bias)

# Print initial predictions on a few training samples
print("Initial Predictions (untrained network):")
for i in range(5):
    prediction = nn.predict(inputs[i].reshape(1, 2))
    print(f"Input: {inputs[i]}, Predicted output: {prediction[0, 0]:.4f}, True label: {outputs[i, 0]:.1f}")

# Test the trained network
print("\nTesting trained network:")
test_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
true_outputs = np.array([[0], [1], [1], [0]], dtype=np.float32)

for i in range(len(test_inputs)):
    prediction = nn.predict(test_inputs[i].reshape(1, 2))
    predicted_output = 1 if prediction[0, 0] > 0.5 else 0
    print(f"Input: {test_inputs[i]}, Predicted: {predicted_output}, True: {int(true_outputs[i, 0])}, Probability: {prediction[0, 0]:.4f}")

# Visualize the decision boundary (optional for 2D input)
h = 0.01
x_min, x_max = inputs[:, 0].min() - 0.1, inputs[:, 0].max() + 0.1
y_min, y_max = inputs[:, 1].min() - 0.1, inputs[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = nn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = (Z > 0.5).reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.8)

# Plot the training data points
plt.scatter(inputs[:, 0], inputs[:, 1], c=outputs[:, 0], cmap=plt.cm.RdBu, edgecolors='k')
plt.title('XOR Decision Boundary')
plt.xlabel('Input 1')
plt.ylabel('Input 2')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks([0, 1])
plt.yticks([0, 1])
plt.show()