class Callback:
    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

    def on_batch_end_with_info(self, batch, network=None):
        pass

    def on_forward_pass_begin(self, inputs=None):
        pass

    def on_forward_layer_begin(self, layer=None, input_data=None):
        pass

    def on_forward_layer_end(self, layer=None, output_data=None):
        pass

    def on_forward_pass_end(self, output=None):
        pass

    def on_backward_pass_begin(self, targets=None, output_gradient=None):
        pass

    def on_backward_output_gradient(self, gradient=None):
        pass

    def on_backward_layer_begin(self, layer=None, input_gradient=None):
        pass

    def on_backward_layer_end(self, layer=None, output_gradient=None):
        pass

    def on_batch_start(self):
        pass

    def on_batch_loss(self, loss):
        pass

    def on_batch_end_step(self):
        pass