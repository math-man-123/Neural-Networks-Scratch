# pyright: reportMissingImports=false, reportUndefinedVariable=false

from typing import Any
import numpy as np
from activation import Activation
from factory import factory, make
from custom_types import * 


class LossFunction:
    """
    Create a custom Loss function handling gradient calculations.
    """
    _default_loss_config = { "name": "least_squares" }
    _default_activation_config = Activation._default_config


    def __init__(
            self,
            activation_config: dict[str, Any] = _default_activation_config,
            loss_config: dict[str, Any] = _default_loss_config
        ) -> None:
        """
        Create a new LossFunction instance from given config.
        Loss gradients depend on network activations, hence
        its config should be given too.
        
        :param activation_config: parameter object for activation factory
        :param loss_config: parameter object for loss function factory
        """
        self._activation = Activation(activation_config)
        self._loss_eval, self._loss_rate = make(loss_config)


    def eval(self, output: Vector, target: Vector) -> float:
        """
        Calculate the elementwise loss evaluation.
        
        :param output: network output
        :param target: target output
        :return: loss evaluation
        """
        return self._loss_eval(output, target)
    

    def rate(self, output: Vector, target: Vector) -> float:
        """
        Calculate the elementwise rate of change
        (derivative) of the loss function.
        
        :param output: network output
        :param target: target output
        :return: derivative vector
        """
        return self._loss_rate(output, target)
    

    def _gradient(self, weight: Matrix, gradient: Vector, pre_signal: Vector):
        """
        Implements loss gradient backpropagation 
        from layer k to layer k-1 with formula:
        delta^(k-1) = W^(k)^T delta^(k) * sigma^'(z^(k-1))
        
        :param weight: W^(k) weight matrix layer k
        :param gradient: delta^(k) loss gradient layer k
        :param pre_signal: z^(k-1) pre signal layer k-1
        """
        return weight.T @ gradient * self._activation.rate(pre_signal)


    def get_gradients(
            self,
            signals: dict[str, list[Vector]], 
            weights: list[Matrix],
            output: Vector, target: Vector
            ) -> tuple[list[Matrix], list[Vector]]:
        """
        Calculates loss gradients with respect to weights and 
        biases. Returns two lists of gradients for each layer.
        
        :param signals: pre and post activation aignals of all layers
        :param weights: weight matricies of all layers
        :param output: output vector of neural network
        :param target: target vector of neural network
        """
        pre = signals["pre"]; post = signals["post"]
        weight_grads = []; bias_grads = []

        # note input layer has no weight matrix
        last_layer = len(weights)

        # last layer gradient calculation
        loss_rate = self.rate(output, target)
        activation_rate = self._activation.rate(pre[last_layer])
        gradient = loss_rate * activation_rate

        weight_grads.append(gradient @ post[last_layer - 1].T)
        bias_grads.append(gradient)

        # gradient backpropagation through remaining layers
        for layer in range(last_layer - 1, 0, -1):
            # beware of indexshifts when comparing to math
            gradient = self._gradient(weights[layer], gradient, pre[layer])
            
            weight_grads.append(gradient @ post[layer - 1].T)
            bias_grads.append(gradient)

        # reverse order of lists
        weight_grads.reverse()
        bias_grads.reverse()
        return weight_grads, bias_grads

    
@factory
def make_least_squares():

    def eval(output: Vector, target: Vector) -> float:
        """
        Elementwise least square (L2 norm) of input diffrence.
        
        :param output: network output
        :param target: target output
        :return: 1/2 ||target - output||^2
        """
        return 0.5 * np.linalg.norm(target - output) ** 2
    
    def rate(output: Vector, target: Vector) -> float:
        """
        Loss derivative without the activation factor.
        
        :param output: network output
        :param target: target output
        :return: dL(x^(n),t) / dz^(n) / sigma^'(z^(n))
        """
        return output - target

    return eval, rate
