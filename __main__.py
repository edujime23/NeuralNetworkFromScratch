import matplotlib.pyplot as plt
from utils.functions import CostFunctions, ActivationFunctions
from utils.layers import Layer, InputLayer, DenseLayer
from neuralNetwork import NeuralNetwork
import sys
import threading
import multiprocessing

def universal_callback(epoch, epochs, training_loss, validation_loss, costs, gradients, shared_data):
    shared_data['epoch'] = epoch
    shared_data['epochs'] = epochs
    shared_data['training_loss'] = training_loss
    shared_data['validation_loss'] = validation_loss
    shared_data['costs'] = costs
    shared_data['gradients'] = gradients

def train_network(nn, inputs, outputs, validation_inputs, validation_outputs, shared_data):
    nn.train(inputs, outputs, validation_inputs, validation_outputs, epochs=10000, batch_size=2, callback=lambda *args: universal_callback(*args, shared_data))

if __name__ == "__main__":
    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    outputs = [[0], [1], [1], [0]]

    validation_inputs = [[0, 0], [1, 1]]
    validation_outputs = [[0], [0]]

    nn = NeuralNetwork([2, 4, 1], [DenseLayer], [ActivationFunctions.leaky_relu, ActivationFunctions.sigmoid],
                       CostFunctions.cross_entropy, threshold=1.0, dropout_rate=1/4, batch_norm=True)

    # Set up the plot window close event to kill the program
    def on_close(event):
        plt.ioff()
        sys.exit(0)

    fig = plt.figure()
    fig.canvas.mpl_connect('close_event', on_close)

    # Create shared memory for communication between threads
    manager = multiprocessing.Manager()
    shared_data = manager.dict()
    shared_data['epoch'] = 0
    shared_data['epochs'] = 0
    shared_data['training_loss'] = 0
    shared_data['validation_loss'] = 0
    shared_data['costs'] = []
    shared_data['gradients'] = []

    # Train the model with a universal callback for real-time plotting and logging
    training_thread = threading.Thread(target=train_network, args=(nn, inputs, outputs, validation_inputs, validation_outputs, shared_data))
    training_thread.daemon = True
    training_thread.start()

    plt.ion()
    while True:
        if shared_data['epoch'] > 0:
            plt.subplot(2, 1, 1)
            plt.plot(shared_data['costs'])
            plt.xlabel('Epochs')
            plt.ylabel('Cost')
            plt.title('Gradient Descent Progress')

            plt.subplot(2, 1, 2)
            plt.plot(shared_data['gradients'])
            plt.xlabel('Epochs')
            plt.ylabel('Gradient')
            plt.title('Gradient Magnitude')

            plt.draw()
            plt.pause(0.01)
            plt.clf()
            if shared_data['validation_loss'] is not None:
                print(f"Epoch {shared_data['epoch']}/{shared_data['epochs']}, Training Loss: {shared_data['training_loss']:.4f}, Validation Loss: {shared_data['validation_loss']:.4f}")
            else:
                print(f"Epoch {shared_data['epoch']}/{shared_data['epochs']}, Training Loss: {shared_data['training_loss']:.4f}")
        if not training_thread.is_alive():
            break

    plt.ioff()
    plt.show()
