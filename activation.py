# pyright: reportMissingImports=false, reportUndefinedVariable=false

from typing import Any
import numpy as np
import validation as val
from factory import factory, make
from custom_types import * 


class Activation:
    """
    Create a custom Activation with given config.
    """
    _default_config = { "name": "sigmoid", "T": 1.0 }


    def __init__(self, config: dict[str, Any] = _default_config) -> None:
        """
        Create a new Activation instance from given config.
        Defaults to sigmoid function 1 / (1 + exp(-x)).
        
        :param config: parameter object for activation factory
        """
        self._config = config
        self._eval, self._rate = make(config)


    def eval(self, z: Vector) -> Vector:
        """
        Calculate the post-activation signal sigma(z).
        
        :param z: pre-activation signal
        :return: post-activation signal
        """
        return self._eval(z)
    

    def rate(self, v: Vector) -> Vector:
        """
        Calculate the elementwise rate of change 
        (derivative) of the activation function.

        :param v: (any) input vector
        :return: derivative vector
        """
        return self._rate(v)
    

    @property
    def config(self) -> dict[str, Any]: return self._config
    """Get the config used to create this Activation."""


@factory
def make_sigmoid(T: float = 1.0) -> tuple[ActivFunct, ...]:
    # ensure temperature is greater 0
    val.check_condition(T > 0, "T must be greater 0")

    def eval(x: Vector) -> Vector:
        """
        Elementwise sigmoid function with temperature.
        
        :param x: input vector x
        :param T: temperature parameter (T > 0)
        :return: output vector sigmoid(x)
        """
        return 1 / (1 + np.exp(-x / T))
    
    def rate(x: Vector) -> Vector:
        """
        Elementwise rate of change (derivative) of
        the sigmoid function with temperature.
        
        :param x: Description
        :return: Description
        """
        val = eval(x)
        return val * (1 - val) / T
    
    return eval, rate
