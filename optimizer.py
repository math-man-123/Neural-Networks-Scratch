# pyright: reportMissingImports=false, reportUndefinedVariable=false

from typing import Any
import numpy as np
from factory import factory, make
from custom_types import * 


class Optimizer:
    """
    Create a custom Optimizer with given config.
    """
    _default_config = { 
        "name": "adamw", "alpha": 1e-3, "epsilon": 1e-8,
        "beta0": 0.9, "beta1": 0.999, "phi": 1e-3, }


    def __init__(self, config: dict[str, Any] = _default_config) -> None:
        """
        Create a new instance of Optimizer with given config.
        
        :param config: parameter object for optimizer factory
        """
        self._optimizer = make(config)


    def get_steps(
            self,
            weight_grads: list[Matrix], 
            bias_grads: list[Vector],
            weights: list[Matrix],
            biases: list[Vector]
        ) -> tuple[list[Matrix], list[Vector]]:
        """
        Get all weight and bias steps according to optimizer algorithm.
        
        :param weight_grads: weight gradients
        :param bias_grads: bias gradients
        :param weights: weight matrices
        :param biases: bias vectors
        :return: weight and bias adjustment steps
        """
        return self._optimizer(weight_grads, bias_grads, weights, biases)


@factory
def make_adamw(
    alpha: float = 1e-3, epsilon = 1e-8,
    beta0: float = 0.9, beta1: float = 0.999,
    phi: float = 1e-3
    ) -> Callable[
            [list[Matrix], list[Vector], 
             list[Matrix], list[Vector]], 
            tuple[list[Matrix], list[Vector]]]:

    # moment and correction cache
    moment = []; correction = []

    def adamw(
            weight_grads: list[Matrix], 
            bias_grads: list[Vector],
            weights: list[Matrix],
            biases: list[Vector]
        ) -> tuple[list[Matrix], list[Vector]]:
        

        def calc_steps(
                grads: list[Array],
                params: list[Array],
                idx_offset: int = 0
            ) -> list[Array]:
            """
            Calculate steps of all given params with seperate caching.
            
            :param grads: (batch) gradients of all params
            :param params: current position of all params
            :param idx_offset: use to not overwrite cache
            :return: all parameter adjustment steps
            """
            steps = []
            # calculate param steps with seperate caches
            for idx, (grad, param) in enumerate(zip(grads, params)):
                idx += idx_offset
                # create cache for current param if needed
                if idx >= len(moment): moment.append([0.0, 0.0])
                if idx >= len(correction): correction.append([1.0, 1.0])

                steps.append(step(grad, param, idx))
            
            return steps
        

        def step(grad: Array, param: Array, idx: int) -> Array:
            """
            Implementation of AdamW algorithm. Based on exponential 
            smooting moment estimation. Takes (batch) gradient of 
            current param position and returns next param step.
            
            :param grad: (batch) gradient current position
            :param param: current parameter position
            :return: next parameter adjustment step
            """
            # exponential smoothing moment estimators (biased)
            moment[idx][0] = beta0 * moment[idx][0] + (1.0 - beta0) * grad
            moment[idx][1] = beta1 * moment[idx][1] + (1.0 - beta1) * grad ** 2

            # correction terms to get unbiased estimators
            correction[idx][0] = correction[idx][0] * beta0
            correction[idx][1] = correction[idx][1] * beta1

            # exponential smoothing moment estimators (unbiased)
            unbiased = [moment[idx][i] / (1.0 - correction[idx][i]) for i in range(2)]
            
            # calculate param update step as described by paper
            step = alpha * unbiased[0]
            step /= np.sqrt(unbiased[1]) + epsilon
            
            # decoupled weight decay instead of L2 regulation
            step += phi * param
            
            return step
        

        # get adamw steps for each weight matrix and bias vector
        weight_steps = calc_steps(weight_grads, weights, idx_offset=0)
        bias_steps = calc_steps(bias_grads, biases, idx_offset=len(weights))

        return weight_steps, bias_steps
    
    return adamw
