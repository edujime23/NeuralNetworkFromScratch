import pickle
import numpy as np
from typing import List, Optional, Callable, Tuple
from utils.functions import *
from utils.layers import Layer
from utils.loss import Loss
from utils.optimizer import AdamOptimizer
from utils.callbacks import Callback
from utils.metrics import Metrics # Assuming metrics are in utils/metrics.py

# Neural Network class
class NeuralNetwork:
    def __init__(self, layers: List[Layer], cost_function: Callable, threshold: float = 1.0, gradient_clip: Optional[float] = 1.0, learning_rate: float = 0.001):
        self.layers: List[Layer] = layers
        self.cost_function: Callable = cost_function
        self.threshold: float = threshold
        self.gradient_clip: Optional[float] = gradient_clip
        self.loss: Loss = Loss(cost_function)
        self.optimizer: AdamOptimizer = AdamOptimizer(learning_rate=learning_rate)
        self.metrics: List[str] = []

    def compile(self, optimizer: Optional[AdamOptimizer] = None, loss: Optional[Callable] = None, metrics: Optional[List[str]] = None):
        if optimizer:
            self.optimizer = optimizer
        for layer in self.layers:
            layer._init_optimizer(self.optimizer)
            layer.threshold = self.threshold
        if loss:
            self.loss = Loss(loss)
        if metrics:
            self.metrics = metrics

    def _trigger_callbacks_with_info(self, callbacks: Optional[List[Callback]], method_name: str, *args, **kwargs):
        if callbacks:
            for callback in callbacks:
                if isinstance(callback, Callback) and hasattr(callback, method_name) and callable(getattr(callback, method_name)):
                    getattr(callback, method_name)(*args, **kwargs)

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int = 1, batch_size: int = 32, validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None, callbacks: Optional[List[Callback]] = None, early_stopping_patience: int = 1, tolerance: float = 1e-4):
        x, y = np.array(x), np.array(y)
        if validation_data:
            validation_data = (np.array(validation_data[0]), np.array(validation_data[1]))
        self.callbacks = callbacks or [] # Store callbacks in the instance
        consecutive_below_tolerance = 0

        self._trigger_callbacks(self.callbacks, 'on_train_begin')

        for epoch in range(epochs):
            self._trigger_callbacks(self.callbacks, 'on_epoch_begin', epoch)
            self._train_on_batch(x, y, batch_size, self.callbacks)
            logs = self._get_logs(x, y, validation_data)
            self._trigger_callbacks(self.callbacks, 'on_epoch_end', epoch, logs=logs)

            if validation_data and 'val_loss' in logs and 'loss' in logs:
                if logs['val_loss'] < tolerance and logs['loss'] < tolerance:
                    consecutive_below_tolerance += 1
                else:
                    consecutive_below_tolerance = 0

                if consecutive_below_tolerance >= early_stopping_patience:
                    break

        self._trigger_callbacks(self.callbacks, 'on_train_end')

    def _train_on_batch(self, x: np.ndarray, y: np.ndarray, batch_size: int, callbacks: Optional[List[Callback]]):
        for i in range(0, len(x), batch_size):
            batch_x = x[i:i + batch_size]
            batch_y = y[i:i + batch_size]
            batch_index = i // batch_size
            self._trigger_callbacks(callbacks, 'on_batch_begin', batch_index)
            self._train_step(batch_x, batch_y)
            self._trigger_callbacks_with_info(callbacks, 'on_batch_end_with_info', batch_index, network=self) # Pass the network instance
            self._trigger_callbacks(callbacks, 'on_batch_end', batch_index) # Keep the original on_batch_end for other purposes

    def _train_step(self, x: np.ndarray, y: np.ndarray):
        self._trigger_callbacks(self.callbacks, 'on_batch_start')
        outputs = self._forward_pass(x)
        loss_value = np.mean(self.loss(outputs, y))
        self._trigger_callbacks(self.callbacks, 'on_batch_loss', loss=loss_value)
        self._backward_pass(y)
        for layer in self.layers:
            layer.update()
        self._trigger_callbacks(self.callbacks, 'on_batch_end_step')

    def _forward_pass(self, inputs: np.ndarray) -> np.ndarray:
        inputs = inputs.copy()
        self._trigger_callbacks(self.callbacks, 'on_forward_pass_begin', inputs=inputs)
        for layer in self.layers:
            self._trigger_callbacks(self.callbacks, 'on_forward_layer_begin', layer=layer, input_data=inputs)
            inputs = layer.forward(inputs)
            self._trigger_callbacks(self.callbacks, 'on_forward_layer_end', layer=layer, output_data=inputs)
        self._trigger_callbacks(self.callbacks, 'on_forward_pass_end', output=inputs)
        return inputs

    def _backward_pass(self, targets: np.ndarray):
        grad = ActivationFunctions.derivative(self.loss, 0, self.layers[-1].signals, targets)

        if self.gradient_clip is not None:
            grad = np.clip(grad, -self.gradient_clip, self.gradient_clip) # Clip loss gradient

        self._trigger_callbacks(self.callbacks, 'on_backward_pass_begin', targets=targets, output_gradient=grad)
        self._trigger_callbacks(self.callbacks, 'on_backward_output_gradient', gradient=grad)
        for layer in reversed(self.layers):
            self._trigger_callbacks(self.callbacks, 'on_backward_layer_begin', layer=layer, input_gradient=grad)
            grad = layer.backward(grad)
            if self.gradient_clip is not None:
                grad = np.clip(grad, -self.gradient_clip, self.gradient_clip) # Keep the layer-wise clipping
            self._trigger_callbacks(self.callbacks, 'on_backward_layer_end', layer=layer, output_gradient=grad)

    def _get_logs(self, x: np.ndarray, y: np.ndarray, validation_data: Optional[Tuple[np.ndarray, np.ndarray]]) -> dict:
        logs = {'loss': np.mean(self.loss(self.predict(x), y))}
        if validation_data:
            val_x, val_y = validation_data
            logs['val_loss'] = np.mean(self.loss(self.predict(val_x), val_y))
        return logs

    def _trigger_callbacks(self, callbacks: Optional[List[Callback]], method_name: str, *args, **kwargs):
        if callbacks:
            for callback in callbacks:
                if isinstance(callback, Callback) and hasattr(callback, method_name) and callable(getattr(callback, method_name)):
                    getattr(callback, method_name)(*args, **kwargs)

    def _evaluate(self, x: np.ndarray, y: np.ndarray) -> float:
        outputs = self._forward_pass(x)
        total_loss = self.loss(outputs, y)
        return np.mean(total_loss)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._forward_pass(x)

    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath: str) -> 'NeuralNetwork':
        with open(filepath, 'rb') as f:
            return pickle.load(f)