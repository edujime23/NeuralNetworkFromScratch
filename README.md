# NumPy Neural Network Module

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python->=3.12.7-blue.svg)](https://www.python.org/downloads/)
[![Dependencies](https://img.shields.io/badge/Dependencies-NumPy%20%7C%20SciPy%20%7C%20Matplotlib-brightgreen.svg)](https://numpy.org/ | https://scipy.org/ | https://matplotlib.org/)

A neural network built from scratch using only NumPy and SciPy. This has convolutional, dense, and recurrent layers, with support for mixing them. It also includes a selection of commonly used activation and cost functions, as well as popular optimization algorithms. I tried to make it easily extensible, allowing you to implement their your own custom layers and optimizers.

## Features

* **Layer Types:**
    * Convolutional Layers (Also supports convolutions of 4D+ dimensions)
    * Dense (Fully Connected) Layers
    * Recurrent Layers (e.g., SimpleRNN, LSTM)
    * Utility (Pooling, Reshape, Flatten, Permutate) Layers (**May raise errors**)
* **Layer Mixing:** Ability to combine different layer types to create complex architectures.
* **Activation Functions:**
    * Sigmoid
    * ReLU
    * Tanh
    * Softmax
    * (and more...)
* **Cost Functions:**
    * Mean Squared Error (MSE)
    * Binary Cross-Entropy
    * Categorical Cross-Entropy
    * (and more...)
* **Optimizers:**
    * Adam
    * AdamW
    * AmsGrad
    * SGD
    * SGD with Nesterov Momentum
    * RMSprop
* **Extensibility:**
    * Base classes for creating custom layer types.
    * Base classes for implementing custom optimization algorithms.
* **Dependencies:** Only relies on `NumPy` and `SciPy`, but for running the test files `matplotlib` is needed
