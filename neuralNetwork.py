import pickle
import numpy as np
from typing import List, Optional, Callable, Tuple, Type
from utils.functions import *
from utils.layers import Layer
from utils.loss import Loss
from utils.optimizer import AdamOptimizer, Optimizer  # Import the base Optimizer class
from utils.callbacks import Callback
from utils.metrics import Metrics # Assuming metrics are in utils/metrics.py

# Neural Network class
class NeuralNetwork:
    def __init__(self, layers: List[Layer], cost_function: Callable, threshold: float = 1.0, gradient_clip: Optional[float] = 1.0, learning_rate: float = 0.001, optimizer_type: Type[Optimizer] = AdamOptimizer, optimizer_params: Optional[dict] = None):
        self.layers: List[Layer] = layers
        self.cost_function: Callable = cost_function
        self.threshold: float = threshold
        self.gradient_clip: Optional[float] = gradient_clip
        self.loss: Loss = Loss(cost_function)
        self.optimizer_type: Type[Optimizer] = optimizer_type
        self.optimizer_params: dict = optimizer_params if optimizer_params is not None else {'learning_rate': learning_rate}
        self.optimizer: Optional[Optimizer] = None
        self.metrics: List[str] = []
        self._is_compiled = False

    def compile(self, optimizer: Optional[Optimizer] = None, loss: Optional[Callable] = None, metrics: Optional[List[str]] = None):
        if optimizer:
            self.optimizer = optimizer
        else:
            self.optimizer = self.optimizer_type(**self.optimizer_params)
        for layer in self.layers:
            layer._init_optimizer(self.optimizer)
            if hasattr(layer, 'threshold'): # Apply threshold only if the layer has this attribute
                layer.threshold = self.threshold
        if loss:
            self.loss = Loss(loss)
        if metrics:
            self.metrics = metrics
        self._is_compiled = True

    def _trigger_callbacks_with_info(self, callbacks: Optional[List[Callback]], method_name: str, *args, **kwargs):
        if callbacks:
            for callback in callbacks:
                if isinstance(callback, Callback) and hasattr(callback, method_name) and callable(getattr(callback, method_name)):
                    getattr(callback, method_name)(*args, **kwargs)

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int = 1, batch_size: int = 32, validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None, callbacks: Optional[List[Callback]] = None, early_stopping_patience: int = 10, tol: float = 1e-4, restore_best_weights: bool = False):
        if not self._is_compiled:
            raise RuntimeError("The model must be compiled before training.")
        x, y = np.array(x), np.array(y)
        best_val_loss = float('inf')
        best_model_weights = None
        epochs_no_improvement = 0
        if validation_data:
            validation_data = (np.array(validation_data[0]), np.array(validation_data[1]))
        self.callbacks = callbacks or [] # Store callbacks in the instance

        self._trigger_callbacks(self.callbacks, 'on_train_begin')

        for epoch in range(epochs):
            self._trigger_callbacks(self.callbacks, 'on_epoch_begin', epoch)
            self._train_on_batch(x, y, batch_size, self.callbacks)
            logs = self._get_logs(x, y, validation_data)
            self._trigger_callbacks(self.callbacks, 'on_epoch_end', epoch, logs=logs)

            if validation_data and 'val_loss' in logs:
                current_val_loss = logs['val_loss']

                # Check for tolerance
                if current_val_loss < tol:
                    print(f"Validation loss reached tolerance ({tol}) at epoch {epoch+1}. Stopping training.")
                    break

                # Check for improvement (using a small epsilon for comparison)
                if current_val_loss < best_val_loss - 1e-6:
                    best_val_loss = current_val_loss
                    best_model_weights = self._get_model_weights()
                    epochs_no_improvement = 0
                else:
                    epochs_no_improvement += 1
                    if epochs_no_improvement >= early_stopping_patience:
                        print(f"Early stopping triggered at epoch {epoch+1} (no improvement in validation loss for {early_stopping_patience} epochs).")
                        if restore_best_weights and best_model_weights is not None:
                            self._set_model_weights(best_model_weights)
                            print("Restored model weights from the best epoch.")
                        break

        self._trigger_callbacks(self.callbacks, 'on_train_end')

    def _get_model_weights(self) -> List[np.ndarray]:
        weights = []
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                weights.append(layer.weights.copy())
            if hasattr(layer, 'biases'):
                weights.append(layer.biases.copy())
        return weights

    def _set_model_weights(self, weights: List[np.ndarray]):
        weight_index = 0
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                layer.weights = weights[weight_index]
                weight_index += 1
            if hasattr(layer, 'biases'):
                layer.biases = weights[weight_index]
                weight_index += 1

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
        gradients = self._backward_pass(y) # Get gradients from backward pass
        for layer in self.layers:
            if hasattr(layer, 'weights') or hasattr(layer, 'biases'):
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

    def _backward_pass(self, targets: np.ndarray) -> None:
        grad = ActivationFunctions.derivative(self.loss, 0, self.layers[-1].signals, targets)

        self._trigger_callbacks(self.callbacks, 'on_backward_pass_begin', targets=targets, output_gradient=grad)
        self._trigger_callbacks(self.callbacks, 'on_backward_output_gradient', gradient=grad)
        for layer in reversed(self.layers):
            self._trigger_callbacks(self.callbacks, 'on_backward_layer_begin', layer=layer, input_gradient=grad)
            grad = layer.backward(grad)
            self._trigger_callbacks(self.callbacks, 'on_backward_layer_end', layer=layer, output_gradient=grad)

    def _get_logs(self, x: np.ndarray, y: np.ndarray, validation_data: Optional[Tuple[np.ndarray, np.ndarray]]) -> dict:
        logs = {'loss': np.mean(self.loss(self.predict(x), y))}
        if validation_data:
            val_x, val_y = validation_data
            logs['val_loss'] = np.mean(self.loss(self.predict(val_x), val_y))
            if self.metrics:
                val_preds = self.predict(val_x)
                for metric_name in self.metrics:
                    if metric_fn := getattr(Metrics, metric_name, None):
                        logs[f'val_{metric_name}'] = metric_fn(val_y, val_preds)
        if self.metrics:
            train_preds = self.predict(x)
            for metric_name in self.metrics:
                if metric_fn := getattr(Metrics, metric_name, None):
                    logs[metric_name] = metric_fn(y, train_preds)
        return logs

    def _trigger_callbacks(self, callbacks: Optional[List[Callback]], method_name: str, *args, **kwargs):
        if callbacks:
            for callback in callbacks:
                if isinstance(callback, Callback) and hasattr(callback, method_name) and callable(getattr(callback, method_name)):
                    getattr(callback, method_name)(*args, **kwargs)

    def _evaluate(self, x: np.ndarray, y: np.ndarray) -> dict:
        outputs = self._forward_pass(x)
        logs = {'loss': np.mean(self.loss(outputs, y))}
        if self.metrics:
            preds = outputs
            for metric_name in self.metrics:
                if metric_fn := getattr(Metrics, metric_name, None):
                    logs[metric_name] = metric_fn(y, preds)
        return logs

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._forward_pass(x)

    def save(self, filepath: str):
        params = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'weights'):
                params[f'layer_{i}_weights'] = layer.weights
            if hasattr(layer, 'biases'):
                params[f'layer_{i}_biases'] = layer.biases
        with open(filepath, 'wb') as f:
            pickle.dump(params, f)

    def load(self, filepath: str):
        with open(filepath, 'rb') as f:
            params = pickle.load(f)
        param_index = 0
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'weights') and f'layer_{i}_weights' in params:
                layer.weights = params[f'layer_{i}_weights']
            if hasattr(layer, 'biases') and f'layer_{i}_biases' in params:
                layer.biases = params[f'layer_{i}_biases']