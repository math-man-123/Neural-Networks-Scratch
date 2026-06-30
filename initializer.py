# pyright: reportMissingImports=false, reportUndefinedVariable=false

from typing import Any
import numpy as np
from factory import factory, make
import validation as val
from custom_types import * 


class Initializer:
    """
    Create custom Initializers for a neural net with given layer_size.
    For both biases and weights a custom initialization config can be provided.
    """
    _default_bias_config = { "name": "uniform", "low": -1.0, "high": 1.0 }
    _default_weight_config = { "name": "normal", "mu": 0.0, "sigma": 1.0 }


    def __init__(
        self,
        weight_config: dict[str, Any] = _default_weight_config,
        bias_config: dict[str, Any] = _default_bias_config
    ) -> None:
        """
        Create Initializer object for neural network with given layer_sizes. 
        Accepts initialization configs for both biases and weights.
        Defaults are uniform biases in [-1, 1] and standard normal weights.
        
        :param layer_sizes: number of neurons in each layer
        :param bias_init: initialization function for biases
        :param weight_init: initialization function for weights
        """
        self._bias_init = make(bias_config)
        self._weight_init = make(weight_config)


    def get_biases(self, layer_sizes: LayerSize) -> list[Vector]:
        """
        Calculate bias vectors b^(k) for k=1,...,n-1 where
        n = len(self.layer_sizes) is the number of layers.
        Each b^(k) has the size (self.layer_sizes[k], 1).
        First (input) layer gets no bias vector.

        :return: (n-1)-tuple of bias vectors
        """
        # use bias initialization function given in __init__
        return [
            self._bias_init((rows, 1)) 
            for rows in layer_sizes[1:]]


    def get_weights(self, layer_sizes: LayerSize) -> list[Matrix]:
        """
        Calculate weight matrices W^(k) for k=1,...,n-1 where
        n = len(self.layer_sizes) is the number of layers.
        Each W^(k) has the size (self.layer_sizes[k], self.layer_sizes[k-1]).

        :return: (n-1)-tuple of weight matrices
        """
        # use weight initialization function given in __init__
        return [
            self._weight_init((rows, cols)) 
            for rows, cols in zip(layer_sizes[1:], layer_sizes)]


@factory
def make_uniform(low: float = 0.0, high: float = 1.0) -> InitFunct:
    # ensure correct draw range [low, high)
    val.check_condition(low < high, "low must be strictly smaller than high")

    def uniform(size: ArraySize) -> Array:
        """
        Create a size[0] x size[1] matrix with elements 
        uniformly drawn from range [low, high).
        
        :param low: lower bound of draw range
        :param high: higher bound of draw range
        :param size: tuple (rows, cols) matrix size
        :return: size[0] x size[1] uniformly random matrix 
        """
        return np.random.uniform(low, high, size)
    
    return uniform


@factory
def make_normal(mu: float = 0.0, sigma: float = 1.0) -> InitFunct:
    # ensure sigma is greater 0
    val.check_condition(sigma > 0, "sigma must be greater 0")

    def normal(size: ArraySize) -> Array:
        """
        Create a size[0] x size[1] matrix with elements 
        normally drawn around mu with std sigma.
        
        :param mu: mean of normal distribution
        :param sigma: std of normal distribution
        :param size: tuple (rows, cols) matrix size
        :return: size[0] x size[1] normally random matrix
        """
        return np.random.normal(mu, sigma, size)

    return normal


@factory
def make_copy(arrays: list[Array]) -> InitFunct:

    iterator = iter(arrays)
    def copy(size: ArraySize) -> Array:
        """
        Create an copy of the next array in arrays.
        Checks if its shape is size[0] x size[1].
        
        :param size: expected shape
        :return: next array in list
        """
        # get next array and ensure expected size
        array = next(iterator)
        val.check_condition(
            array.shape == size,
            "mismatch in expected array size")

        return array

    return copy
