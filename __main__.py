import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Necesario para gráficos 3D
from utils.functions import CostFunctions, ActivationFunctions
from utils.layers import DenseLayer
from neuralNetwork import NeuralNetwork
from utils.callbacks import EpochEndCallback
import sys
import threading
import multiprocessing
import numpy as np

def universal_callback(epoch, logs, shared_data):
    # Actualiza los datos compartidos con información del entrenamiento
    shared_data['epoch'] = epoch
    shared_data['epochs'] = logs.get('epochs', 0)
    shared_data['training_loss'] = logs.get('loss', 0)
    shared_data['validation_loss'] = logs.get('val_loss', 0)
    shared_data['costs'] = logs.get('costs', [])

def train_network(nn, inputs, outputs, validation_inputs, validation_outputs, shared_data):
    nn.fit(inputs, outputs, epochs=10000, batch_size=2, validation_data=(validation_inputs, validation_outputs), callbacks=[EpochEndCallback(lambda epoch, logs: universal_callback(epoch, logs, shared_data))])

if __name__ == "__main__":
    # Dataset XOR
    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    outputs = [[0], [1], [1], [0]]

    validation_inputs = [[0, 0], [1, 1]]
    validation_outputs = [[0], [0]]

    # Crear la red neuronal (2 entradas, 1 capa oculta de 4 neuronas y 1 salida)
    layers = [
        DenseLayer(4, 2, ActivationFunctions.leaky_relu, dropout_rate=0.25, batch_norm=True),
        DenseLayer(1, 4, ActivationFunctions.sigmoid)
    ]
    nn = NeuralNetwork(layers, CostFunctions.cross_entropy, threshold=1.0, gradient_clip=1.0)

    # Configuración para salir del programa al cerrar la ventana
    def on_close(event):
        plt.ioff()
        sys.exit(0)

    # Creamos la figura y conectamos el evento de cierre
    fig = plt.figure()
    fig.canvas.mpl_connect('close_event', on_close)
    
    # Usaremos un único subplot para la gráfica 3D de la función aprendida
    ax = fig.add_subplot(111, projection='3d')

    # Configuración de memoria compartida entre el thread de entrenamiento y la actualización de la gráfica
    manager = multiprocessing.Manager()
    shared_data = manager.dict()
    shared_data['epoch'] = 0
    shared_data['epochs'] = 0
    shared_data['training_loss'] = 0
    shared_data['validation_loss'] = 0
    shared_data['costs'] = []

    # Inicia el entrenamiento en un thread separado
    training_thread = threading.Thread(target=train_network, 
                                         args=(nn, inputs, outputs, validation_inputs, validation_outputs, shared_data))
    training_thread.daemon = True
    training_thread.start()

    plt.ion()
    
    # Definimos los puntos para la malla (dominio para el XOR, de 0 a 1)
    grid_points = np.linspace(0, 1, 50)
    X1, X2 = np.meshgrid(grid_points, grid_points)
    grid_inputs = np.column_stack((X1.ravel(), X2.ravel()))
    
    # Bucle de actualización en vivo (actualiza cada 100 épocas)
    while True:
        if shared_data['epoch'] > 0 and shared_data['epoch'] % 100 == 0:
            # Actualiza la gráfica 3D con la función aprendida
            ax.clear()
            # Evalúa la red en cada punto de la malla (usando nn.activate o nn.activate)
            # Se asume que nn.forward(x) devuelve la salida de la red para el input x
            Z = np.array([nn.predict([x])[0] for x in grid_inputs])
            Z = Z.reshape(X1.shape)
            
            surf = ax.plot_surface(X1, X2, Z, cmap='viridis', edgecolor='none')
            ax.set_xlabel('Input 1')
            ax.set_ylabel('Input 2')
            ax.set_zlabel('Output')
            ax.set_title(f'Función aprendida (Epoch {shared_data["epoch"]})')
            
            plt.draw()
            plt.pause(0.01)
            
            # Mostrar información básica del entrenamiento en consola
            if shared_data['validation_loss'] is not None:
                print(f"Epoch {shared_data['epoch']}/{shared_data['epochs']} - Training Loss: {shared_data['training_loss']:.4f}, Validation Loss: {shared_data['validation_loss']:.4f}")
            else:
                print(f"Epoch {shared_data['epoch']}/{shared_data['epochs']} - Training Loss: {shared_data['training_loss']:.4f}")
        
        # Si el thread de entrenamiento finaliza, se sale del bucle
        if not training_thread.is_alive():
            break

    plt.ioff()
    plt.show()
