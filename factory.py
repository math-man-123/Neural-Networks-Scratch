# pyright: reportMissingImports=false, reportUndefinedVariable=false

from typing import Callable, Any
import validation as val


type Factory = Callable[..., tuple[Callable, ...]]
REGISTRY: dict[str, Factory] = {}


def factory(fn: Factory) -> Factory:
    """
    Register a factory function 'make_name'
    with the '@factory' decorator as 'name'.

    :param fn: function to be registered
    :return: the same function (for decorator)
    """
    name = fn.__name__[len("make_"):]
    return REGISTRY.setdefault(name, fn)


def make(config: dict[str, Any]) -> tuple[Callable, ...]:
    """
    Make a function via registered factory with 
    given name where the factory is 'make_name'.

    :param config: parameter object for factory
    :return: function made from chosen factory
    """
    # ensure config has at least name
    val.check_condition(
        "name" in config, 
        "config must contain at least name")

    # copy config so not to change it
    config = dict(config)
    name = config.pop("name")

    return REGISTRY[name](**config)
