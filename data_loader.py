# pyright: reportMissingImports=false, reportUndefinedVariable=false

from typing import Any
from pathlib import Path
import numpy as np
from factory import factory, make
from custom_types import * 


class DataLoader:
    """
    Custom DataLoader handling arrangment of raw data.
    """
    def __init__(self, config: dict[str, Any]) -> None:
        """
        Create new DataLoader instance with custom config.
        
        :param config: parameter object for loader factory
        """
        self._loader = make(config)


    def load(self, file_path: str) -> list[tuple[Vector, Vector]]:
        """
        Loads data via custom loader function and returns it 
        as list of arranged examples i.e (input, target) pairs.
        
        :param file_path: relative file path
        :return: arranged examples pair list
        """
        return self._loader(file_path)


@factory
def make_mnist_csv(
    in_low: float = -0.5, in_high: float = 0.5,
    out_low: float = 0.01, out_high: float = 0.99
    ) -> Callable[[str], list[tuple[Vector, Vector]]]:

    def _represent_target(digit: int) -> Vector:
        """
        Represent target digit (0,...,9) as vector with 10 elements.
        low means megative for that digit, high means positive. Should
        be used with an activation function in the range of (low, high).
        
        :param digit: target digit (0,...,9)
        :return: target digit representation
        """
        representation = np.full((10, 1), out_low)
        representation[int(digit), 0] = out_high

        return representation

    def _represent_grayscale(gray: Vector) -> Vector:
        """
        Rescale grayscale from {0,...,255} to [low, high].
        
        :param gray: grayscale value in {0,...,255}
        :return: rescaled value in [low, high]
        """
        return gray / 255.0 * (in_high - in_low) + in_low

    def _arrange_example(raw_data_row: list[int]) -> tuple[Vector, Vector]:
        """
        Convert single raw csv mnist data row to
        input target vector pair for training.
        
        :param raw_example: raw data row
        :return: arranged example pair
        """
        target = _represent_target(raw_data_row[0])
        input = _represent_grayscale(raw_data_row[1:])
        input = input.reshape(28**2, 1)
        
        return input, target
    
    def mnist_csv(file_path: str) -> list[tuple[Vector, Vector]]:
        """
        Loads MNIST data from .csv files and returns it as
        list of arranged examples i.e (input, target) pairs.
        
        :param file_path: relative file path
        :return: arranged examples pair list
        """
        base_dir = Path(__file__).resolve().parent
        raw_data = np.loadtxt(base_dir / file_path, delimiter=",")

        return [_arrange_example(row) for row in raw_data]

    return mnist_csv
