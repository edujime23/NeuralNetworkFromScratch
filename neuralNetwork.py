import pickle
import numpy as np
import utils

# Neural Network class
class NeuralNetwork:
    def __init__(self, layers, cost_function, threshold=1.0, gradient_clip=1.0, learning_rate=0.001):
        self.layers: list[utils.Layer] = layers
        self.cost_function = cost_function
        self.threshold = threshold
        self.gradient_clip = gradient_clip
        self.loss = utils.Loss(cost_function)  # Initialize loss attribute with Loss class
        self.optimizer = utils.AdamOptimizer(learning_rate=learning_rate)  # Initialize optimizer attribute with learning rate
        self.metrics = []  # Initialize metrics attribute

    def compile(self, optimizer=None, loss=None, metrics=None):
        if optimizer:
            self.optimizer = optimizer
        for layer in self.layers:
            layer._init_optimizer(self.optimizer)
        if loss:
            self.loss = utils.Loss(loss)
        if metrics:
            self.metrics = metrics

    def fit(self, x, y, epochs=1, batch_size=32, validation_data=None, callbacks=None):
        x, y = np.array(x), np.array(y)
        if validation_data:
            validation_data = (np.array(validation_data[0]), np.array(validation_data[1]))
        callbacks = callbacks or []
        for callback in callbacks:
            if isinstance(callback, utils.Callback):
                callback.on_train_begin()
        for epoch in range(epochs):
            for callback in callbacks:
                if isinstance(callback, utils.Callback):
                    callback.on_epoch_begin(epoch)
            self._train_on_batch(x, y, batch_size, callbacks)
            logs = self._get_logs(x, y, validation_data)
            for callback in callbacks:
                if isinstance(callback, utils.Callback):
                    callback.on_epoch_end(epoch, logs=logs)
        for callback in callbacks:
            if isinstance(callback, utils.Callback):
                callback.on_train_end()

    def _train_on_batch(self, x, y, batch_size, callbacks):
        for i in range(0, len(x), batch_size):
            batch_x = x[i:i + batch_size]
            batch_y = y[i:i + batch_size]
            for callback in callbacks:
                if isinstance(callback, utils.Callback):
                    callback.on_batch_begin(i // batch_size)
            self._train_step(batch_x, batch_y)
            for callback in callbacks:
                if isinstance(callback, utils.Callback):
                    callback.on_batch_end(i // batch_size)

    def _train_step(self, x, y):
        for inputs, targets in zip(x, y):
            self._forward_pass(inputs)
            self._backward_pass(targets)
            self._update_weights()

    def _forward_pass(self, inputs):
        res = inputs
        for layer in self.layers:
            res = layer.forward(res)
        return res


    def _backward_pass(self, targets):
        grad = self.loss.backward(self.layers[-1].signals, targets)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def _update_weights(self):
        for layer in self.layers:
            layer.update()

    def _get_logs(self, x, y, validation_data):
        logs = {'loss': self._evaluate(x, y)}
        if validation_data:
            val_x, val_y = validation_data
            logs['val_loss'] = self._evaluate(val_x, val_y)
        for metric in self.metrics:
            metric_fn = getattr(utils.Metrics, metric)
            logs[metric] = metric_fn(self.predict(x), y)
            if validation_data:
                logs[f'val_{metric}'] = metric_fn(self.predict(val_x), val_y)
        return logs

    def _evaluate(self, x, y):
        total_loss = 0
        for inputs, targets in zip(x, y):
            outputs = self._forward_pass(inputs)
            total_loss += self.loss(outputs, targets)
        return total_loss / len(x)

    def predict(self, x):
        x = np.array(x)
        return [self._forward_pass(inputs) for inputs in x]

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)