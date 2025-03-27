import pickle
import numpy as np
from typing import List, Optional, Callable, Tuple
from utils import Layer, Loss, AdamOptimizer, Callback, Metrics  # Assuming these are directly in utils

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
        if loss:
            self.loss = Loss(loss)
        if metrics:
            self.metrics = metrics

    def _trigger_callbacks(self, callbacks: Optional[List[Callback]], method_name: str, *args, **kwargs):
        if callbacks:
            for callback in callbacks:
                if isinstance(callback, Callback):
                    getattr(callback, method_name)(*args, **kwargs)

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int = 1, batch_size: int = 32, validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None, callbacks: Optional[List[Callback]] = None):
        x, y = np.array(x), np.array(y)
        if validation_data:
            validation_data = (np.array(validation_data[0]), np.array(validation_data[1]))
        callbacks = callbacks or []

        self._trigger_callbacks(callbacks, 'on_train_begin')

        for epoch in range(epochs):
            self._trigger_callbacks(callbacks, 'on_epoch_begin', epoch)
            self._train_on_batch(x, y, batch_size, callbacks)
            logs = self._get_logs(x, y, validation_data)
            self._trigger_callbacks(callbacks, 'on_epoch_end', epoch, logs=logs)

        self._trigger_callbacks(callbacks, 'on_train_end')

    def _train_on_batch(self, x: np.ndarray, y: np.ndarray, batch_size: int, callbacks: Optional[List[Callback]]):
        num_batches = len(x) // batch_size + (1 if len(x) % batch_size != 0 else 0)
        for i in range(0, len(x), batch_size):
            batch_x = x[i:i + batch_size]
            batch_y = y[i:i + batch_size]
            batch_index = i // batch_size
            self._trigger_callbacks(callbacks, 'on_batch_begin', batch_index)
            self._train_step(batch_x, batch_y)
            self._trigger_callbacks(callbacks, 'on_batch_end', batch_index)

    def _train_step(self, x: np.ndarray, y: np.ndarray):
        self._forward_pass(x)
        self._backward_pass(y)
        self._update_weights()

    def _forward_pass(self, inputs: np.ndarray) -> np.ndarray:
        res = inputs
        for layer in self.layers:
            res = layer.forward(res)
        return res

    def _backward_pass(self, targets: np.ndarray):
        grad = self.loss.backward(self.layers[-1].signals, targets)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def _update_weights(self):
        all_grads = []
        for layer in self.layers:
            if hasattr(layer, 'd_filters') and layer.d_filters is not None:
                all_grads.append(layer.d_filters.flatten())
            if hasattr(layer, 'd_biases') and layer.d_biases is not None:
                if isinstance(layer.d_biases, list):
                    all_grads.extend(bias_grad.flatten() for bias_grad in layer.d_biases)
                else:
                    all_grads.append(layer.d_biases.flatten())

        if self.gradient_clip is not None and all_grads:
            global_norm = np.linalg.norm(np.concatenate(all_grads))
            if global_norm > self.gradient_clip:
                clip_factor = self.gradient_clip / global_norm
                for layer in self.layers:
                    if hasattr(layer, 'd_filters') and layer.d_filters is not None:
                        layer.d_filters *= clip_factor
                    if hasattr(layer, 'd_biases') and layer.d_biases is not None:
                        if isinstance(layer.d_biases, list):
                            for bias_grad in layer.d_biases:
                                bias_grad *= clip_factor
                        else:
                            layer.d_biases *= clip_factor

        for layer in self.layers:
            if hasattr(layer, 'optimizer'):
                if hasattr(layer, 'd_filters') and layer.d_filters is not None:
                    layer.optimizer.update(layer, 'filters', layer.d_filters)
                if hasattr(layer, 'd_biases') and layer.d_biases is not None:
                    layer.optimizer.update(layer, 'biases', layer.d_biases)

    def _get_logs(self, x: np.ndarray, y: np.ndarray, validation_data: Optional[Tuple[np.ndarray, np.ndarray]]) -> dict:
        logs = {'loss': self._evaluate(x, y)}
        if validation_data:
            val_x, val_y = validation_data
            logs['val_loss'] = self._evaluate(val_x, val_y)
        for metric in self.metrics:
            metric_fn = getattr(Metrics, metric)
            logs[metric] = metric_fn(self.predict(x), y)
            if validation_data:
                logs[f'val_{metric}'] = metric_fn(self.predict(val_x), val_y)
        return logs

    def _evaluate(self, x: np.ndarray, y: np.ndarray) -> float:
        x = np.array(x)
        outputs = self._forward_pass(x)
        total_loss = self.loss(outputs, y)
        return total_loss / len(x)

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.array(x)
        return self._forward_pass(x)

    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath: str) -> 'NeuralNetwork':
        with open(filepath, 'rb') as f:
            return pickle.load(f)