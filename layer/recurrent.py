import numpy as np
from typing import Optional, Callable, List, Tuple
from layer.base import RecurrentLayer
from utils import Optimizer
from utils.functions import derivative, ActivationFunctions

class SimpleRNNLayer(RecurrentLayer):
    """
    Basic recurrent layer (Vanilla RNN).
    """

    def __init__(self, units: int, activation: Callable[[np.ndarray], np.ndarray] = np.tanh,
                 return_sequences: bool = False, trainable: bool = True, input_shape: Optional[Tuple[int, int]] = None):  # Added input_shape
        super().__init__(units, activation, return_sequences, trainable)
        self.Wh = None  # Weights for hidden-to-hidden connections
        self.Wx = None  # Weights for input-to-hidden connections
        self.b = None  # Bias
        self.dWh = None
        self.dWx = None
        self.db = None
        self.prev_hidden_state = None
        self.last_pre_activation = None
        self.input_shape = input_shape  # Store the input shape

    def _initialize_weights(self, input_shape):
        if self.input_shape is None:  # Use stored input_shape if available
            input_dim = input_shape[-1]
        else:
            input_dim = self.input_shape[-1]
        scale = np.sqrt(1.0 / self.units)
        self.Wh = np.random.randn(self.units, self.units) * scale
        self.Wx = np.random.randn(input_dim, self.units) * scale
        self.b = np.zeros(self.units, dtype=np.float64)
        self.dWh = np.zeros_like(self.Wh)
        self.dWx = np.zeros_like(self.Wx)
        self.db = np.zeros_like(self.b)
        if self.optimizer:
            self.optimizer.register_parameter(self.Wh, 'Wh')
            self.optimizer.register_parameter(self.Wx, 'Wx')
            self.optimizer.register_parameter(self.b, 'b')

    def _step(self, input_at_t: np.ndarray) -> np.ndarray:
        if self.Wh is None:
            self._initialize_weights(self.inputs.shape)
        self.prev_hidden_state = self.hidden_state.copy()
        self.last_pre_activation = np.dot(input_at_t, self.Wx) + \
            np.dot(self.hidden_state, self.Wh) + self.b
        self.hidden_state = self.activation_function(self.last_pre_activation)
        return self.hidden_state

    def backward(self, grad: np.ndarray) -> np.ndarray:
        if self.Wh is None:
            return np.zeros_like(self.inputs)

        d_input = np.zeros_like(self.inputs)
        batch_size, time_steps, input_dim = self.inputs.shape
        d_hidden_next = np.zeros_like(self.hidden_state)

        self.dWh = np.zeros_like(self.Wh)
        self.dWx = np.zeros_like(self.Wx)
        self.db = np.zeros_like(self.b)

        for t in reversed(range(time_steps)):
            input_t = self.inputs[:, t, :]
            hidden_prev = self.prev_hidden_state if t > 0 else np.zeros(
                (batch_size, self.units))
            pre_activation = self.last_pre_activation[t] if self.return_sequences else self.last_pre_activation

            # Calculate the derivative of the activation function
            activation_derivative = derivative(
                self.activation_function, 0, pre_activation)

            # Gradient from the next time step
            d_h = grad[:, t, :] + \
                d_hidden_next if self.return_sequences else grad + d_hidden_next

            # Gradient through the activation
            d_h_raw = d_h * activation_derivative

            # Gradients for weights and bias
            self.dWh += np.dot(hidden_prev.T, d_h_raw)
            self.dWx += np.dot(input_t.T, d_h_raw)
            self.db += np.sum(d_h_raw, axis=0)

            # Gradient for the previous hidden state
            d_hidden_prev = np.dot(d_h_raw, self.Wh.T)

            # Gradient for the input at this time step
            d_input[:, t, :] = np.dot(d_h_raw, self.Wx.T)

            d_hidden_next = d_hidden_prev

        return d_input

    def _get_params_and_grads(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        return [
            (self.Wh, self.dWh),
            (self.Wx, self.dWx),
            (self.b, self.db)
        ]

    def _init_optimizer(self, optimizer: Optimizer):
        super()._init_optimizer(optimizer)
        if optimizer:
            optimizer.register_parameter(self.Wh, 'Wh')
            optimizer.register_parameter(self.Wx, 'Wx')
            optimizer.register_parameter(self.b, 'b')

    def update(self):
        if self.optimizer and self.trainable:
            self.Wh = self.optimizer.update(self.Wh, self.dWh, 'Wh')
            self.Wx = self.optimizer.update(self.Wx, self.dWx, 'Wx')
            self.b = self.optimizer.update(self.b, self.db, 'b')
            # Reset gradients after update (important for the next batch)
            self.dWh = np.zeros_like(self.Wh)
            
class LSTMLayer(RecurrentLayer):
    """
    Long Short-Term Memory (LSTM) layer.
    """
    def __init__(self, units: int, return_sequences: bool = False, trainable: bool = True):
        super().__init__(units, return_sequences=return_sequences, trainable=trainable)
        self.Wf = None  # Weights for forget gate (input to hidden)
        self.Wi = None  # Weights for input gate (input to hidden)
        self.Wo = None  # Weights for output gate (input to hidden)
        self.Wc = None  # Weights for cell state update (input to hidden)
        self.Uf = None  # Recurrent weights for forget gate (hidden to hidden)
        self.Ui = None  # Recurrent weights for input gate (hidden to hidden)
        self.Uo = None  # Recurrent weights for output gate (hidden to hidden)
        self.Uc = None  # Recurrent weights for cell state update (hidden to hidden)
        self.bf = None  # Bias for forget gate
        self.bi = None  # Bias for input gate
        self.bo = None  # Bias for output gate
        self.bc = None  # Bias for cell state update
        self.cell_state = None
        self.prev_cell_state = None
        self.input_gate_output = None
        self.forget_gate_output = None
        self.output_gate_output = None
        self.cell_state_candidate = None
        self.last_input = None

        self.dWf = None
        self.dWi = None
        self.dWo = None
        self.dWc = None
        self.dUf = None
        self.dUi = None
        self.dUo = None
        self.dUc = None
        self.dbf = None
        self.dbi = None
        self.dbo = None
        self.dbc = None

    def _initialize_weights(self, input_shape):
        input_dim = input_shape[-1]
        scale = np.sqrt(1.0 / self.units)
        # Input weights (W)
        self.Wf = np.random.randn(input_dim, self.units) * scale
        self.Wi = np.random.randn(input_dim, self.units) * scale
        self.Wo = np.random.randn(input_dim, self.units) * scale
        self.Wc = np.random.randn(input_dim, self.units) * scale
        # Recurrent weights (U)
        self.Uf = np.random.randn(self.units, self.units) * scale
        self.Ui = np.random.randn(self.units, self.units) * scale
        self.Uo = np.random.randn(self.units, self.units) * scale
        self.Uc = np.random.randn(self.units, self.units) * scale
        # Biases (b)
        self.bf = np.zeros(self.units)
        self.bi = np.zeros(self.units)
        self.bo = np.zeros(self.units)
        self.bc = np.zeros(self.units)

        # Initialize gradients
        self.dWf = np.zeros_like(self.Wf)
        self.dWi = np.zeros_like(self.Wi)
        self.dWo = np.zeros_like(self.Wo)
        self.dWc = np.zeros_like(self.Wc)
        self.dUf = np.zeros_like(self.Uf)
        self.dUi = np.zeros_like(self.Ui)
        self.dUo = np.zeros_like(self.Uo)
        self.dUc = np.zeros_like(self.Uc)
        self.dbf = np.zeros_like(self.bf)
        self.dbi = np.zeros_like(self.bi)
        self.dbo = np.zeros_like(self.bo)
        self.dbc = np.zeros_like(self.bc)

        # Initialize hidden state and cell state
        self.hidden_state = np.zeros((input_shape[0], self.units))  # Initialize hidden state
        self.cell_state = np.zeros((input_shape[0], self.units))     # Initialize cell state

        if self.optimizer:
            self._reg_params()

    def _reg_params(self):
        self.optimizer.register_parameter(self.Wf, 'Wf')
        self.optimizer.register_parameter(self.Wi, 'Wi')
        self.optimizer.register_parameter(self.Wo, 'Wo')
        self.optimizer.register_parameter(self.Wc, 'Wc')
        self.optimizer.register_parameter(self.Uf, 'Uf')
        self.optimizer.register_parameter(self.Ui, 'Ui')
        self.optimizer.register_parameter(self.Uo, 'Uo')
        self.optimizer.register_parameter(self.Uc, 'Uc')
        self.optimizer.register_parameter(self.bf, 'bf')
        self.optimizer.register_parameter(self.bi, 'bi')
        self.optimizer.register_parameter(self.bo, 'bo')
        self.optimizer.register_parameter(self.bc, 'bc')
        
    def update(self):
        if self.optimizer and self.trainable:
            self.Wf = self.optimizer.update(self.Wf, self.dWf, 'Wf')
            self.Wi = self.optimizer.update(self.Wi, self.dWi, 'Wi')
            self.Wo = self.optimizer.update(self.Wo, self.dWo, 'Wo')
            self.Wc = self.optimizer.update(self.Wc, self.dWc, 'Wc')
            self.Uf = self.optimizer.update(self.Uf, self.dUf, 'Uf')
            self.Ui = self.optimizer.update(self.Ui, self.dUi, 'Ui')
            self.Uo = self.optimizer.update(self.Uo, self.dUo, 'Uo')
            self.Uc = self.optimizer.update(self.Uc, self.dUc, 'Uc')
            self.bf = self.optimizer.update(self.bf, self.dbf, 'bf')
            self.bi = self.optimizer.update(self.bi, self.dbi, 'bi')
            self.bo = self.optimizer.update(self.bo, self.dbo, 'bo')
            self.bc = self.optimizer.update(self.bc, self.dbc, 'bc')
            # Reset gradients
            self.dWf = np.zeros_like(self.Wf)
            self.dWi = np.zeros_like(self.Wi)
            self.dWo = np.zeros_like(self.Wo)
            self.dWc = np.zeros_like(self.Wc)
            self.dUf = np.zeros_like(self.Uf)
            self.dUi = np.zeros_like(self.Ui)
            self.dUo = np.zeros_like(self.Uo)
            self.dUc = np.zeros_like(self.Uc)
            self.dbf = np.zeros_like(self.bf)
            self.dbi = np.zeros_like(self.bi)
            self.dbo = np.zeros_like(self.bo)
            self.dbc = np.zeros_like(self.dbc)
            
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        batch_size, time_steps, input_dim = inputs.shape
        self.outputs = np.zeros((batch_size, time_steps, self.units)) if self.return_sequences else np.zeros((batch_size, self.units))
        self.hidden_state = np.zeros((batch_size, self.units)) # Initialize hidden state for the batch
        self.cell_state = np.zeros((batch_size, self.units))   # Initialize cell state for the batch
        for t in range(time_steps):
            input_at_t = inputs[:, t, :]
            output_at_t = self._step(input_at_t)
            if self.return_sequences:
                self.outputs[:, t, :] = output_at_t
            else:
                self.outputs = output_at_t
        return self.outputs

    def _step(self, input_at_t: np.ndarray) -> np.ndarray:
        if self.Wf is None:
            self._initialize_weights(self.inputs.shape)

        self.last_input = input_at_t
        self.prev_cell_state = self.cell_state.copy()

        # Forget gate
        forget_gate_input = np.dot(input_at_t, self.Wf) + np.dot(self.hidden_state, self.Uf) + self.bf
        self.forget_gate_output = ActivationFunctions.sigmoid(forget_gate_input)

        # Input gate
        input_gate_input = np.dot(input_at_t, self.Wi) + np.dot(self.hidden_state, self.Ui) + self.bi
        self.input_gate_output = ActivationFunctions.sigmoid(input_gate_input)

        # Candidate cell state
        cell_state_candidate_input = np.dot(input_at_t, self.Wc) + np.dot(self.hidden_state, self.Uc) + self.bc
        self.cell_state_candidate = np.tanh(cell_state_candidate_input)

        # Update cell state
        self.cell_state = self.forget_gate_output * self.cell_state + self.input_gate_output * self.cell_state_candidate

        # Output gate
        output_gate_input = np.dot(input_at_t, self.Wo) + np.dot(self.hidden_state, self.Uo) + self.bo
        self.output_gate_output = ActivationFunctions.sigmoid(output_gate_input)

        # Current hidden state
        self.hidden_state = self.output_gate_output * np.tanh(self.cell_state)

        return self.hidden_state

    def backward(self, grad: np.ndarray) -> np.ndarray:
        if self.Wf is None:
            return np.zeros_like(self.inputs)

        d_input = np.zeros_like(self.inputs)
        batch_size, time_steps, input_dim = self.inputs.shape
        d_hidden_next = np.zeros_like(self.hidden_state)
        d_cell_next = np.zeros_like(self.cell_state)

        # Initialize gradients (already done in __init__, but ensuring)
        self.dWf = np.zeros_like(self.Wf)
        self.dWi = np.zeros_like(self.Wi)
        self.dWo = np.zeros_like(self.Wo)
        self.dWc = np.zeros_like(self.Wc)
        self.dUf = np.zeros_like(self.Uf)
        self.dUi = np.zeros_like(self.Ui)
        self.dUo = np.zeros_like(self.Uo)
        self.dUc = np.zeros_like(self.Uc)
        self.dbf = np.zeros_like(self.bf)
        self.dbi = np.zeros_like(self.bi)
        self.dbo = np.zeros_like(self.bo)
        self.dbc = np.zeros_like(self.bc)

        for t in reversed(range(time_steps)):
            input_t = self.inputs[:, t, :]
            hidden_prev = self.outputs[:, t - 1, :] if t > 0 and self.return_sequences else np.zeros((batch_size, self.units))
            cell_prev = self.prev_cell_state if t > 0 else np.zeros((batch_size, self.units))

            # Get activations at this time step
            ft = self.forget_gate_output[:, t, :] if self.return_sequences else self.forget_gate_output
            it = self.input_gate_output[:, t, :] if self.return_sequences else self.input_gate_output
            ot = self.output_gate_output[:, t, :] if self.return_sequences else self.output_gate_output
            ct_candidate = self.cell_state_candidate[:, t, :] if self.return_sequences else self.cell_state_candidate
            ct = self.cell_state[:, t, :] if self.return_sequences else self.cell_state
            ht = self.hidden_state[:, t, :] if self.return_sequences else self.hidden_state

            # Gradient from output
            dht = grad[:, t, :] + d_hidden_next if self.return_sequences else grad + d_hidden_next

            # Gradient for output gate
            dot = dht * np.tanh(ct)
            doutput_gate_input = dot * derivative(ActivationFunctions.sigmoid, 0, np.dot(input_t, self.Wo) + np.dot(hidden_prev, self.Uo) + self.bo)
            self.dWo += np.dot(input_t.T, doutput_gate_input)
            self.dUo += np.dot(hidden_prev.T, doutput_gate_input)
            self.dbo += np.sum(doutput_gate_input, axis=0)

            # Gradient for cell state
            dct = dht * ot * derivative(np.tanh, 0, ct) + d_cell_next

            # Gradient for candidate cell state
            dct_candidate = dct * it
            dcell_state_candidate_input = dct_candidate * derivative(np.tanh, 0, np.dot(input_t, self.Wc) + np.dot(hidden_prev, self.Uc) + self.bc)
            self.dWc += np.dot(input_t.T, dcell_state_candidate_input)
            self.dUc += np.dot(hidden_prev.T, dcell_state_candidate_input)
            self.dbc += np.sum(dcell_state_candidate_input, axis=0)

            # Gradient for input gate
            dit = dct * ct_candidate
            dinput_gate_input = dit * derivative(ActivationFunctions.sigmoid, 0, np.dot(input_t, self.Wi) + np.dot(hidden_prev, self.Ui) + self.bi)
            self.dWi += np.dot(input_t.T, dinput_gate_input)
            self.dUi += np.dot(hidden_prev.T, dinput_gate_input)
            self.dbi += np.sum(dinput_gate_input, axis=0)

            # Gradient for forget gate
            dft = dct * cell_prev
            dforget_gate_input = dft * derivative(ActivationFunctions.sigmoid, 0, np.dot(input_t, self.Wf) + np.dot(hidden_prev, self.Uf) + self.bf)
            self.dWf += np.dot(input_t.T, dforget_gate_input)
            self.dUf += np.dot(hidden_prev.T, dforget_gate_input)
            self.dbf += np.sum(dforget_gate_input, axis=0)

            # Gradient for input at this time step
            d_input_t = (np.dot(dforget_gate_input, self.Wf.T) +
                         np.dot(dinput_gate_input, self.Wi.T) +
                         np.dot(dcell_state_candidate_input, self.Wc.T) +
                         np.dot(doutput_gate_input, self.Wo.T))
            d_input[:, t, :] = d_input_t

            # Gradient for previous hidden state
            d_hidden_prev = (np.dot(dforget_gate_input, self.Uf.T) +
                             np.dot(dinput_gate_input, self.Ui.T) +
                             np.dot(dcell_state_candidate_input, self.Uc.T) +
                             np.dot(doutput_gate_input, self.Uo.T))
            d_hidden_next = d_hidden_prev
            d_cell_next = dct * ft

        return d_input

    def _get_params_and_grads(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        return [
            (self.Wf, self.dWf), (self.Wi, self.dWi), (self.Wo, self.dWo), (self.Wc, self.dWc),
            (self.Uf, self.dUf), (self.Ui, self.dUi), (self.Uo, self.dUo), (self.Uc, self.dUc),
            (self.bf, self.dbf), (self.bi, self.dbi), (self.bo, self.dbo), (self.bc, self.dbc)
        ]

    def _init_optimizer(self, optimizer: Optimizer):
        super()._init_optimizer(optimizer)
        if optimizer:
            optimizer.register_parameter(self.Wf, 'Wf')
            optimizer.register_parameter(self.Wi, 'Wi')
            optimizer.register_parameter(self.Wo, 'Wo')
            optimizer.register_parameter(self.Wc, 'Wc')
            optimizer.register_parameter(self.Uf, 'Uf')
            optimizer.register_parameter(self.Ui, 'Ui')
            optimizer.register_parameter(self.Uo, 'Uo')
            optimizer.register_parameter(self.Uc, 'Uc')
            optimizer.register_parameter(self.bf, 'bf')
            optimizer.register_parameter(self.bi, 'bi')
            optimizer.register_parameter(self.bo, 'bo')
            optimizer.register_parameter(self.bc, 'bc')

class GRULayer(RecurrentLayer):
    """
    Gated Recurrent Unit (GRU) layer.
    """
    def __init__(self, units: int, return_sequences: bool = False, trainable: bool = True):
        super().__init__(units, return_sequences=return_sequences, trainable=trainable)
        self.Wz = None  # Weights for update gate (input to hidden)
        self.Wr = None  # Weights for reset gate (input to hidden)
        self.Wh = None  # Weights for current memory content (input to hidden)
        self.Uz = None  # Recurrent weights for update gate (hidden to hidden)
        self.Ur = None  # Recurrent weights for reset gate (hidden to hidden)
        self.Uh = None  # Recurrent weights for current memory content (hidden to hidden)
        self.bz = None  # Bias for update gate
        self.br = None  # Bias for reset gate
        self.bh = None  # Bias for current memory content
        self.prev_hidden_state = None
        self.update_gate_output = None
        self.reset_gate_output = None
        self.current_memory = None
        self.last_input = None

        self.dWz = None
        self.dWr = None
        self.dWh = None
        self.dUz = None
        self.dUr = None
        self.dUh = None
        self.dbz = None
        self.dbr = None
        self.dbh = None

    def _initialize_weights(self, input_shape):
        input_dim = input_shape[-1]
        scale = np.sqrt(1.0 / self.units)
        # Input weights (W)
        self.Wz = np.random.randn(input_dim, self.units) * scale
        self.Wr = np.random.randn(input_dim, self.units) * scale
        self.Wh = np.random.randn(input_dim, self.units) * scale
        # Recurrent weights (U)
        self.Uz = np.random.randn(self.units, self.units) * scale
        self.Ur = np.random.randn(self.units, self.units) * scale
        self.Uh = np.random.randn(self.units, self.units) * scale
        # Biases (b)
        self.bz = np.zeros(self.units)
        self.br = np.zeros(self.units)
        self.bh = np.zeros(self.units)

        # Initialize gradients
        self.dWz = np.zeros_like(self.Wz)
        self.dWr = np.zeros_like(self.Wr)
        self.dWh = np.zeros_like(self.Wh)
        self.dUz = np.zeros_like(self.Uz)
        self.dUr = np.zeros_like(self.Ur)
        self.dUh = np.zeros_like(self.Uh)
        self.dbz = np.zeros_like(self.bz)
        self.dbr = np.zeros_like(self.br)
        self.dbh = np.zeros_like(self.bh)

        if self.optimizer:
            self._reg_params()

    def _reg_params(self):
        self.optimizer.register_parameter(self.Wz, 'Wz')
        self.optimizer.register_parameter(self.Wr, 'Wr')
        self.optimizer.register_parameter(self.Wh, 'Wh')
        self.optimizer.register_parameter(self.Uz, 'Uz')
        self.optimizer.register_parameter(self.Ur, 'Ur')
        self.optimizer.register_parameter(self.Uh, 'Uh')
        self.optimizer.register_parameter(self.bz, 'bz')
        self.optimizer.register_parameter(self.br, 'br')
        self.optimizer.register_parameter(self.bh, 'bh')
        
    def update(self):
        if self.optimizer and self.trainable:
            self.Wz = self.optimizer.update(self.Wz, self.dWz, 'Wz')
            self.Wr = self.optimizer.update(self.Wr, self.dWr, 'Wr')
            self.Wh = self.optimizer.update(self.Wh, self.dWh, 'Wh')
            self.Uz = self.optimizer.update(self.Uz, self.dUz, 'Uz')
            self.Ur = self.optimizer.update(self.Ur, self.dUr, 'Ur')
            self.Uh = self.optimizer.update(self.Uh, self.dUh, 'Uh')
            self.bz = self.optimizer.update(self.bz, self.dbz, 'bz')
            self.br = self.optimizer.update(self.br, self.dbr, 'br')
            self.bh = self.optimizer.update(self.bh, self.dbh, 'bh')
            # Reset gradients
            self.dWz = np.zeros_like(self.Wz)
            self.dWr = np.zeros_like(self.Wr)
            self.dWh = np.zeros_like(self.Wh)
            self.dUz = np.zeros_like(self.Uz)
            self.dUr = np.zeros_like(self.Ur)
            self.dUh = np.zeros_like(self.Uh)
            self.dbz = np.zeros_like(self.bz)
            self.dbr = np.zeros_like(self.br)
            self.dbh = np.zeros_like(self.bh)

    def _step(self, input_at_t: np.ndarray) -> np.ndarray:
        if self.Wz is None:
            self._initialize_weights(self.inputs.shape)

        self.last_input = input_at_t
        self.prev_hidden_state = self.hidden_state.copy()

        # Update gate
        update_gate_input = np.dot(input_at_t, self.Wz) + np.dot(self.hidden_state, self.Uz) + self.bz
        self.update_gate_output = ActivationFunctions.sigmoid(update_gate_input)

        # Reset gate
        reset_gate_input = np.dot(input_at_t, self.Wr) + np.dot(self.hidden_state, self.Ur) + self.br
        self.reset_gate_output = ActivationFunctions.sigmoid(reset_gate_input)

        # Current memory content
        h_candidate_input = np.dot(input_at_t, self.Wh) + np.dot(self.reset_gate_output * self.hidden_state, self.Uh) + self.bh
        self.current_memory = np.tanh(h_candidate_input)

        # Final hidden state
        self.hidden_state = (1 - self.update_gate_output) * self.prev_hidden_state + self.update_gate_output * self.current_memory

        return self.hidden_state

    def backward(self, grad: np.ndarray) -> np.ndarray:
        if self.Wz is None:
            return np.zeros_like(self.inputs)

        d_input = np.zeros_like(self.inputs)
        batch_size, time_steps, input_dim = self.inputs.shape
        d_hidden_next = np.zeros_like(self.hidden_state)

        # Initialize gradients
        self.dWz = np.zeros_like(self.Wz)
        self.dWr = np.zeros_like(self.Wr)
        self.dWh = np.zeros_like(self.Wh)
        self.dUz = np.zeros_like(self.Uz)
        self.dUr = np.zeros_like(self.Ur)
        self.dUh = np.zeros_like(self.Uh)
        self.dbz = np.zeros_like(self.bz)
        self.dbr = np.zeros_like(self.br)
        self.dbh = np.zeros_like(self.bh)

        for t in reversed(range(time_steps)):
            input_t = self.inputs[:, t, :]
            hidden_prev = self.outputs[:, t - 1, :] if t > 0 and self.return_sequences else np.zeros((batch_size, self.units))

            # Get activations at this time step
            zt = self.update_gate_output[:, t, :] if self.return_sequences else self.update_gate_output
            rt = self.reset_gate_output[:, t, :] if self.return_sequences else self.reset_gate_output
            ht_candidate = self.current_memory[:, t, :] if self.return_sequences else self.current_memory
            ht = self.hidden_state[:, t, :] if self.return_sequences else self.hidden_state

            # Gradient from output
            dht = grad[:, t, :] + d_hidden_next if self.return_sequences else grad + d_hidden_next

            # Gradient for current memory content
            dcurrent_memory_raw = dht * zt
            dh_candidate_input = dcurrent_memory_raw * derivative(np.tanh, 0, np.dot(input_t, self.Wh) + np.dot(rt * hidden_prev, self.Uh) + self.bh)
            self.dWh += np.dot(input_t.T, dh_candidate_input)
            self.dbh += np.sum(dh_candidate_input, axis=0)

            # Gradient for reset gate
            dreset_gate_raw = np.dot(dh_candidate_input, self.Uh.T) * hidden_prev
            dreset_gate_input = dreset_gate_raw * derivative(ActivationFunctions.sigmoid, 0, np.dot(input_t, self.Wr) + np.dot(hidden_prev, self.Ur) + self.br)
            self.dWr += np.dot(input_t.T, dreset_gate_input)
            self.dUr += np.dot(hidden_prev.T, dreset_gate_input)
            self.dbr += np.sum(dreset_gate_input, axis=0)

            # Gradient for update gate
            dupdate_gate_raw = dht * (ht - hidden_prev)
            dupdate_gate_input = dupdate_gate_raw * derivative(ActivationFunctions.sigmoid, 0, np.dot(input_t, self.Wz) + np.dot(hidden_prev, self.Uz) + self.bz)
            self.dWz += np.dot(input_t.T, dupdate_gate_input)
            self.dUz += np.dot(hidden_prev.T, dupdate_gate_input)
            self.dbz += np.sum(dupdate_gate_input, axis=0)

            # Gradient for input at this time step
            d_input_t = (np.dot(dupdate_gate_input, self.Wz.T) +
                         np.dot(dreset_gate_input, self.Wr.T) +
                         np.dot(dh_candidate_input, self.Wh.T))
            d_input[:, t, :] = d_input_t

            # Gradient for previous hidden state
            d_hidden_prev = (np.dot(dupdate_gate_input, self.Uz.T) +
                             np.dot(dreset_gate_input, self.Ur.T) +
                             np.dot(dh_candidate_input, self.Uh.T) * rt +
                             dht * (1 - zt))
            d_hidden_next = d_hidden_prev

        return d_input

    def _get_params_and_grads(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        return [
            (self.Wz, self.dWz), (self.Wr, self.dWr), (self.Wh, self.dWh),
            (self.Uz, self.dUz), (self.Ur, self.dUr), (self.Uh, self.dUh),
            (self.bz, self.dbz), (self.br, self.dbr), (self.bh, self.dbh)
        ]

    def _init_optimizer(self, optimizer: Optimizer):
        super()._init_optimizer(optimizer)
        if optimizer:
            optimizer.register_parameter(self.Wz, 'Wz')
            optimizer.register_parameter(self.Wr, 'Wr')
            optimizer.register_parameter(self.Wh, 'Wh')
            optimizer.register_parameter(self.Uz, 'Uz')
            optimizer.register_parameter(self.Ur, 'Ur')
            optimizer.register_parameter(self.Uh, 'Uh')
            optimizer.register_parameter(self.bz, 'bz')
            optimizer.register_parameter(self.br, 'br')
            optimizer.register_parameter(self.bh, 'bh')

class BidirectionalRNNLayer(RecurrentLayer):
    """
    Bidirectional Recurrent Neural Network layer.
    """
    def __init__(self, forward_layer: RecurrentLayer, backward_layer: RecurrentLayer, merge_mode: str = 'concat', trainable: bool = True):
        if not isinstance(forward_layer, RecurrentLayer) or not isinstance(backward_layer, RecurrentLayer):
            raise ValueError("forward_layer and backward_layer must be instances of RecurrentLayer.")
        units = forward_layer.units + backward_layer.units if merge_mode == 'concat' else forward_layer.units
        super().__init__(units=units,
                         return_sequences=forward_layer.return_sequences,  # Assuming both layers have the same return_sequences
                         trainable=trainable)
        self.forward_layer = forward_layer
        self.backward_layer = backward_layer
        self.merge_mode = merge_mode.lower()
        if self.merge_mode not in ['concat', 'sum', 'ave', 'mul']:
            raise ValueError(f"Invalid merge_mode: '{merge_mode}'. Must be 'concat', 'sum', 'ave', or 'mul'.")
        # Ensure trainable status is consistent
        self.forward_layer.trainable = trainable
        self.backward_layer.trainable = trainable
        self.forward_inputs = None
        self.backward_inputs = None
        self.forward_outputs = None
        self.backward_outputs = None

    def _initialize_weights(self, input_shape):
        self.forward_layer._initialize_weights(input_shape)
        self.backward_layer._initialize_weights(input_shape)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.forward_inputs = inputs
        self.forward_outputs = self.forward_layer.forward(inputs)
        reversed_inputs = np.flip(inputs, axis=1).copy()
        self.backward_inputs = reversed_inputs
        self.backward_outputs = self.backward_layer.forward(reversed_inputs)
        reversed_backward_outputs = np.flip(self.backward_outputs, axis=1)

        if self.merge_mode == 'concat':
            return np.concatenate([self.forward_outputs, reversed_backward_outputs], axis=-1)
        elif self.merge_mode == 'sum':
            return self.forward_outputs + reversed_backward_outputs
        elif self.merge_mode == 'ave':
            return (self.forward_outputs + reversed_backward_outputs) / 2
        elif self.merge_mode == 'mul':
            return self.forward_outputs * reversed_backward_outputs
        return None

    def update(self):
        self.forward_layer.update()
        self.backward_layer.update()

    def _step(self, input_at_t: np.ndarray) -> np.ndarray:
        raise NotImplementedError("BidirectionalRNNLayer does not use a single _step method.")

    def backward(self, grad: np.ndarray) -> np.ndarray:
        if self.merge_mode == 'concat':
            forward_grad = grad[:, :self.forward_layer.units]
            backward_grad = grad[:, self.forward_layer.units:]
        else:
            forward_grad = backward_grad = grad

        d_forward = self.forward_layer.backward(forward_grad)

        d_backward_reversed_input = self.backward_layer.backward(backward_grad)
        d_backward = np.flip(d_backward_reversed_input, axis=1)

        return d_forward + d_backward

    def _get_params_and_grads(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        params_and_grads = self.forward_layer._get_params_and_grads()
        params_and_grads.extend(self.backward_layer._get_params_and_grads())
        return params_and_grads

    def _init_optimizer(self, optimizer):
        self.forward_layer._init_optimizer(optimizer)
        self.backward_layer._init_optimizer(optimizer)
        
class IndRNNLayer(RecurrentLayer):
    """
    Independently Recurrent Neural Network (IndRNN) layer.
    """
    def __init__(self, units: int, activation: Callable[[np.ndarray], np.ndarray] = ActivationFunctions.leaky_relu, return_sequences: bool = False, trainable: bool = True):
        super().__init__(units, activation, return_sequences, trainable)
        self.Wr = None  # Recurrent weights (vector)
        self.Wx = None  # Input weights
        self.b = None   # Bias
        self.prev_hidden_state = None
        self.last_input = None
        self.last_pre_activation = None

        self.dWr = None
        self.dWx = None
        self.db = None

    def _initialize_weights(self, input_shape):
        input_dim = input_shape[-1]
        scale_input = np.sqrt(1.0 / input_dim)
        scale_recurrent = np.sqrt(1.0) # IndRNN paper suggests larger recurrent weights
        self.Wx = np.random.randn(input_dim, self.units) * scale_input
        self.Wr = np.random.randn(self.units) * scale_recurrent
        self.b = np.zeros(self.units)

        self.dWr = np.zeros_like(self.Wr)
        self.dWx = np.zeros_like(self.Wx)
        self.db = np.zeros_like(self.b)

        if self.optimizer:
            self._reg_params()

    def _reg_params(self):
        self.optimizer.register_parameter(self.Wr, 'Wr')
        self.optimizer.register_parameter(self.Wx, 'Wx')
        self.optimizer.register_parameter(self.b, 'b')
        
    def update(self):
        if self.optimizer and self.trainable:
            self.Wr = self.optimizer.update(self.Wr, self.dWr, 'Wr')
            self.Wx = self.optimizer.update(self.Wx, self.dWx, 'Wx')
            self.b = self.optimizer.update(self.b, self.db, 'b')
            # Reset gradients
            self.dWr = np.zeros_like(self.Wr)
            self.dWx = np.zeros_like(self.Wx)
            self.db = np.zeros_like(self.b)

    def _step(self, input_at_t: np.ndarray) -> np.ndarray:
        if self.Wr is None:
            self._initialize_weights(self.inputs.shape)

        self.last_input = input_at_t
        self.prev_hidden_state = self.hidden_state.copy()

        self.last_pre_activation = np.dot(input_at_t, self.Wx) + self.hidden_state * self.Wr + self.b
        self.hidden_state = self.activation_function(self.last_pre_activation)
        return self.hidden_state

    def backward(self, grad: np.ndarray) -> np.ndarray:
        if self.Wr is None:
            return np.zeros_like(self.inputs)

        d_input = np.zeros_like(self.inputs)
        batch_size, time_steps, input_dim = self.inputs.shape
        d_hidden_next = np.zeros_like(self.hidden_state)

        self.dWr = np.zeros_like(self.Wr)
        self.dWx = np.zeros_like(self.Wx)
        self.db = np.zeros_like(self.b)

        for t in reversed(range(time_steps)):
            input_t = self.inputs[:, t, :]
            hidden_prev = self.prev_hidden_state if t > 0 else np.zeros((batch_size, self.units))
            pre_activation = self.last_pre_activation[t] if self.return_sequences else self.last_pre_activation

            # Calculate the derivative of the activation function
            activation_derivative = derivative(self.activation_function, 0, pre_activation)

            # Gradient from the next time step
            d_h = grad[:, t, :] + d_hidden_next if self.return_sequences else grad + d_hidden_next

            # Gradient through the activation
            d_h_raw = d_h * activation_derivative

            # Gradients for weights and bias
            self.dWx += np.dot(input_t.T, d_h_raw)
            self.dWr += np.sum(d_h_raw * hidden_prev, axis=0)
            self.db += np.sum(d_h_raw, axis=0)

            # Gradient for the previous hidden state
            d_hidden_prev = d_h_raw * self.Wr

            # Gradient for the input at this time step
            d_input[:, t, :] = np.dot(d_h_raw, self.Wx.T)

            d_hidden_next = d_hidden_prev

        return d_input

    def _get_params_and_grads(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        return [
            (self.Wr, self.dWr),
            (self.Wx, self.dWx),
            (self.b, self.db)
        ]

    def _init_optimizer(self, optimizer: Optimizer):
        super()._init_optimizer(optimizer)
        if optimizer:
            optimizer.register_parameter(self.Wr, 'Wr')
            optimizer.register_parameter(self.Wx, 'Wx')
            optimizer.register_parameter(self.b, 'b')

class CTRNNLayer(RecurrentLayer):
    """
    Continuous-time RNN (CTRNN) layer.
    """
    def __init__(self, units: int, time_constants: Optional[np.ndarray] = None, activation: Callable[[np.ndarray], np.ndarray] = np.tanh, trainable: bool = True):
        super().__init__(units, trainable=trainable)
        self.weights = None  # Connection weights
        self.biases = None
        self.time_constants = time_constants if time_constants is not None else np.ones(units) # Tau values for each neuron
        self.activation_function = activation
        self.state = None # Current state of the neurons
        self.last_inputs = None

        self.d_weights = None
        self.d_biases = None

    def _initialize_weights(self, input_shape):
        input_dim = input_shape[-1]
        scale = np.sqrt(1.0 / self.units)
        self.weights = np.random.randn(self.units, self.units) * scale
        self.biases = np.zeros(self.units)

        self.d_weights = np.zeros_like(self.weights)
        self.d_biases = np.zeros_like(self.biases)

        if self.optimizer:
            self._reg_params()

    def _reg_params(self):
        self.optimizer.register_parameter(self.weights, 'weights')
        self.optimizer.register_parameter(self.biases, 'biases')

    def forward(self, inputs: np.ndarray, time_step_size: float = 0.1) -> np.ndarray:
        self.inputs = inputs
        if self.weights is None:
            self._initialize_weights(inputs.shape)

        self.last_inputs = inputs
        batch_size, time_steps, input_dim = inputs.shape
        self.state = np.zeros((batch_size, self.units))  # Initialize state

        for t in range(time_steps):
            external_input = inputs[:, t, :]
            # Euler method for discretization
            d_state_dt = (1.0 / self.time_constants) * (
                -self.state + np.dot(self.activation_function(self.state), self.weights.T) + external_input + self.biases
            )
            self.state = self.state + time_step_size * d_state_dt

        return self.activation_function(self.state)  # Changed line: return only the last state

    def backward(self, grad: np.ndarray) -> np.ndarray:
        if self.weights is None:
            return np.zeros_like(self.inputs)

        batch_size, time_steps, input_dim = self.inputs.shape
        d_weights = np.zeros_like(self.weights)
        d_biases = np.zeros_like(self.biases)

        if self.state is None or self.state.shape[0] != batch_size:
            self.state = np.zeros((batch_size, self.units))

        d_state_next = np.zeros_like(self.state)  # Initialize d_state_next

        for t in reversed(range(time_steps)):
            external_input = self.last_inputs[:, t, :]
            state_t = self.state

            # 1. Derivative of activation
            activation_derivative = derivative(self.activation_function, 0, state_t)

            # 2. Gradient from the next time step
            d_h = grad[:, t, :] + d_state_next if self.return_sequences else grad + d_state_next

            # 3. Calculate intermediate gradients
            d_state_raw = d_h * activation_derivative

            # Gradients for weights and biases (accumulate)
            d_weights += np.dot(d_state_raw.T, state_t).T
            d_biases += np.sum(d_state_raw, axis=0)

            # 4. Gradient for the previous state (simplified Euler-like step)
            d_state_prev = (d_state_raw @ self.weights) - (d_state_raw * (1.0 / self.time_constants))
            d_state_next = d_state_prev

        self.d_weights = d_weights
        self.d_biases = d_biases
        return np.zeros_like(self.inputs)
    
    def update(self):
        if self.optimizer and self.trainable:
            params_and_grads = self._get_params_and_grads()
            self.optimizer.update(params_and_grads)
            # Reset gradients
            self.d_weights = np.zeros_like(self.weights)
            self.d_biases = np.zeros_like(self.biases)

    def _step(self, input_at_t: np.ndarray) -> np.ndarray:
        raise NotImplementedError("CTRNNLayer does not use a discrete _step method.")

    def _get_params_and_grads(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        # Returning None for gradients as the backward pass is not fully implemented
        return [
            (self.weights, self.d_weights),
            (self.biases, self.d_biases)
        ]

    def _init_optimizer(self, optimizer: Optimizer):
        super()._init_optimizer(optimizer)
        if optimizer:
            optimizer.register_parameter(self.weights, 'weights')
            optimizer.register_parameter(self.biases, 'biases')