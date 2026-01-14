from abc import abstractmethod
from typing import Union

import numpy as np

from si.neural_networks.layers import Layer


class ActivationLayer(Layer):
    """
    Base class for activation layers.
    """

    def forward_propagation(self, input: np.ndarray, training: bool) -> np.ndarray:
        """
        Perform forward propagation on the given input.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.
        training: bool
            Whether the layer is in training mode or in inference mode.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        """
        self.input = input
        self.output = self.activation_function(self.input)
        return self.output

    def backward_propagation(self, output_error: float) -> Union[float, np.ndarray]:
        """
        Perform backward propagation on the given output error.

        Parameters
        ----------
        output_error: float
            The output error of the layer.

        Returns
        -------
        Union[float, numpy.ndarray]
            The output error of the layer.
        """
        return self.derivative(self.input) * output_error

    @abstractmethod
    def activation_function(self, input: np.ndarray) -> Union[float, np.ndarray]:
        """
        Activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        Union[float, numpy.ndarray]
            The output of the layer.
        """
        raise NotImplementedError

    @abstractmethod
    def derivative(self, input: np.ndarray) -> Union[float, np.ndarray]:
        """
        Derivative of the activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        Union[float, numpy.ndarray]
            The derivative of the activation function.
        """
        raise NotImplementedError

    def output_shape(self) -> tuple:
        """
        Returns the output shape of the layer.

        Returns
        -------
        tuple
            The output shape of the layer.
        """
        return self._input_shape

    def parameters(self) -> int:
        """
        Returns the number of parameters of the layer.

        Returns
        -------
        int
            The number of parameters of the layer.
        """
        return 0
    
class SigmoidActivation(ActivationLayer):
    """
    Sigmoid activation function.
    """

    def activation_function(self, input: np.ndarray):
        """
        Sigmoid activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        """
        return 1 / (1 + np.exp(-input))

    def derivative(self, input: np.ndarray):
        """
        Derivative of the sigmoid activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The derivative of the activation function.
        """
        return self.activation_function(input) * (1 - self.activation_function(input))


class ReLUActivation(ActivationLayer):
    """
    ReLU activation function.
    """

    def activation_function(self, input: np.ndarray):
        """
        ReLU activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        """
        return np.maximum(0, input)

    def derivative(self, input: np.ndarray):
        """
        Derivative of the ReLU activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The derivative of the activation function.
        """
        return np.where(input >= 0, 1, 0)

class TanhActivation(Layer):
    """
    Hyperbolic tangent activation.
    Squashes values into [-1, 1].
    """

    def __init__(self):
        super().__init__()
        self.input = None
        self.output = None

    def forward_propagation(self, input: np.ndarray, training: bool = True) -> np.ndarray:
        self.input = input
        self.output = np.tanh(input)
        return self.output

    def backward_propagation(self, error: np.ndarray) -> np.ndarray:
        # d/dx tanh(x) = 1 - tanh(x)^2
        return error * (1.0 - self.output ** 2)

    def output_shape(self) -> tuple:
        return self._input_shape

    def parameters(self) -> int:
        return 0


class SoftmaxActivation(Layer):
    """
    Softmax activation (stable version).
    Produces a probability distribution per sample (rows sum to 1).
    """

    def __init__(self):
        super().__init__()
        self.input = None
        self.output = None

    def forward_propagation(self, input: np.ndarray, training: bool = True) -> np.ndarray:
        self.input = input

        # stable softmax: subtract max per row
        x = input - np.max(input, axis=1, keepdims=True)

        exp_x = np.exp(x)
        self.output = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.output

    def backward_propagation(self, error: np.ndarray) -> np.ndarray:
        """
        Jacobian-vector product for softmax:
        dL/dx = y * (error - sum(error*y))
        """
        y = self.output
        dot = np.sum(error * y, axis=1, keepdims=True)
        return y * (error - dot)

    def output_shape(self) -> tuple:
        return self._input_shape

    def parameters(self) -> int:
        return 0