# pyright: reportMissingImports=false, reportUndefinedVariable=false

"""
Both warmup and decay functions should be of the form
f: [0,1] -> [0,1], be continuous, and obey the rules:
    warmup: f(0) = 0 and f(1) = 1
    decay: f(0) = 1 and f(x) -> 0 for x -> 1
Scaling and shifting is then handled automatically
by the LearningSchedule class based on LSConfig.
"""

from typing import Any
import numpy as np
from factory import factory, make
import validation as val
from custom_types import * 

if __debug__: 
    import matplotlib.pyplot as plt


class LearningSchedule:
    """
    Create a custom LearningSchedule with given config.
    Consists of first a warmup then a decay phase.
    """
    _default_schedule_config = { 
        "low": 0.001, "high": 0.1, "total": 100, "frac": 0.05 }
    _default_warmup_config = { "name": "linear_warmup" }
    _default_decay_config = { "name": "cosine_decay", "pow": 1.0 }


    def __init__(
            self,
            schedule_config: dict[str, Any] = _default_schedule_config,
            warmup_config: dict[str, Any] = _default_warmup_config,
            decay_config: dict[str, Any] = _default_decay_config
        ) -> None:
        """
        Create LearningSchedule with given config
        asx well as warmup and decay functions.

        :param config: global LSConfig
        :param warmup: function to use in warmup phase
        :param decay: function to use in decay phase
        """
        # ensure schedule config contains low, high, total, frac
        val.check_condition("low" in schedule_config, "schedule config needs low")
        val.check_condition("high" in schedule_config, "schedule config needs high")
        val.check_condition("total" in schedule_config, "schedule config needs total")
        val.check_condition("frac" in schedule_config, "schedule config needs frac")

        # max diffrence in learning rates
        delta = schedule_config["high"] - schedule_config["low"]

        # calculate warmup rates
        warmup = make(warmup_config)
        warmup_total = np.ceil(schedule_config["frac"] * schedule_config["total"])
        warmup_ts = np.arange(0, warmup_total, 1) / warmup_total
        warmup_rates = schedule_config["low"] + delta * warmup(warmup_ts)

        # calculate decay rates
        decay = make(decay_config)
        decay_total = schedule_config["total"] - warmup_total
        decay_ts = np.arange(0, decay_total, 1) / decay_total
        decay_rates = schedule_config["low"] + delta * decay(decay_ts)

        # combine warmup and decay rates
        self._schedule = np.concatenate([warmup_rates, decay_rates])

        # setup next rate timer
        self._total = schedule_config["total"]
        self._time = 0

    
    def get_next_rate(self) -> np.floating:
        """
        Get the next learning rate. If called
        after total steps get last rate.

        :return: next learning rate
        """
        if self._time < self._total - 1:
            self._time += 1
        return self._schedule[self._time]


    if __debug__: 
        def show_plot(self) -> None:
            """
            Plot and show the actual learning rate curve.
            """
            ts = np.arange(0, self._total, 1)
            plt.plot(ts, self._schedule); plt.show()


@factory
def make_classic_decay(alpha: float = 2.0, pow: float = 2.0) -> DecayFunct:
    # ensure alpha, pow are strictly larger zero
    val.check_condition(alpha > 0, "alpha must be strictly larger than 0")
    val.check_condition(pow > 0, "pow must be strictly larger than 0")

    def classic_decay(t: Vector) -> Vector:
        """
        Create a decay step vector following 
        the classic 1 / t style curve.

        :param t: time step vector
        :param alpha: decay rate
        :param pow: decay power
        :return: decay step vector
        """
        decay = (1.0 + alpha * t  ) ** pow
        return 1.0 / decay
    
    return classic_decay


@factory
def make_cosine_decay(pow: float = 1.0) -> DecayFunct:
    # ensure pow is strictly larger one
    val.check_condition(pow >= 1, "pow must be at least 1")

    def cosine_decay(t: Vector) -> Vector:
        """
        Create a decay step vector following
        the first half of the cosine curve.

        :param t: time step vector
        :param pow: decay power
        :return: decay step vector
        """
        decay = 1.0 + np.cos(np.pi * t ** pow)
        return 0.5 * decay
    
    return cosine_decay


@factory
def make_linear_warmup() -> WarmupFunct:

    def linear_warmup(t: Vector) -> Vector:
        """
        Create a warmup step vector following
        the identity line (45 deg).

        :param t: time step vector
        :return: warmup step vector
        """
        return t
    
    return linear_warmup
