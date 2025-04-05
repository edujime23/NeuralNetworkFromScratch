import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils.functions import CostFunctions, ActivationFunctions
from utils.optimizer import AdamOptimizer
from utils.callbacks import Callback
from layer import LSTMLayer, DenseLayer
from neuralNetwork import NeuralNetwork
from scipy.ndimage import gaussian_filter1d

class TrainingCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 == 0:
            print(f"Epoch {epoch+1}, Loss: {logs['loss']:.4f}, Val Loss: {logs.get('val_loss', 'N/A'):.4f}")

# Generate synthetic movement data
def generate_data(num_samples=1000):
    t = np.linspace(0, 10 * np.pi, num_samples)
    x = np.cos(t)
    y = np.sin(t)
    z = -t / (10 * np.pi) * 5
    noise_level = 0.05
    x_noisy = x + gaussian_filter1d(np.random.normal(0, noise_level, num_samples), sigma=5)
    y_noisy = y + gaussian_filter1d(np.random.normal(0, noise_level, num_samples), sigma=5)
    z_noisy = z + gaussian_filter1d(np.random.normal(0, noise_level, num_samples), sigma=5)
    data = np.column_stack((x_noisy, y_noisy, z_noisy))
    inputs = data[:-1]
    targets = data[1:]
    return inputs, targets

# Prepare dataset
epochs = 50
num_samples = 1000
X, y = generate_data(num_samples)
train_size = int(len(X) * 0.8)
X_train = X[:train_size].reshape(-1, 1, 3) # Reshape for LSTM (batch_size, time_steps=1, input_dim=3)
X_test = X[train_size:].reshape(-1, 1, 3)
y_train = y[:train_size]
y_test = y[train_size:]

# Define model
layers = [
    LSTMLayer(64, return_sequences=False), # Output the next point directly
    DenseLayer(3, activation_function=ActivationFunctions.linear)
]

model = NeuralNetwork(layers, CostFunctions.mean_squared_error)
model.compile(optimizer=AdamOptimizer(learning_rate=1e-2, weight_decay=1e-2))

# Train model
model.fit(X_train, y_train, epochs=epochs, batch_size=16, validation_data=(X_test, y_test), callbacks=[TrainingCallback()], restore_best_weights=True)

# Predict the trajectory step by step
predicted_trajectory = []
current_point = X_test[0].reshape(1, 1, 3) # Start with the first point of the test set

for _ in range(len(y_test)):
    next_prediction = model.predict(current_point)
    predicted_trajectory.append(next_prediction[0])
    current_point = next_prediction.reshape(1, 1, 3) # The prediction becomes the next input

predicted_trajectory = np.array(predicted_trajectory)

# Visualize in 3D with lines
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')

# Plot true trajectory
ax.plot(y_test[:, 0], y_test[:, 1], y_test[:, 2], c='blue', label='True Trajectory')

# Plot predicted trajectory
ax.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], predicted_trajectory[:, 2], c='red', label='Predicted Trajectory')

# Plot the starting point
ax.scatter(X_test[0][0][0], X_test[0][0][1], X_test[0][0][2], c='green', marker='o', s=50, label='Start Point')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('True vs. Predicted Movement Trajectory (Point-by-Point Prediction)')
ax.legend()
plt.show()