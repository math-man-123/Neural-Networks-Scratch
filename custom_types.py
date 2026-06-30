# pyright: reportMissingImports=false, reportUndefinedVariable=false

from collections.abc import Callable
from numpy.typing import NDArray
import numpy as np


type Array = NDArray[np.floating]
type Vector = Array   # dimension (m, 1)
type Matrix = Array   # dimension (m, n)

type ArraySize = tuple[int, int]                     # rows, cols
type LayerSize = tuple[int, int, *tuple[int, ...]]   # min 2 layers

type InitFunct = Callable[..., Array]           # dim -> vect / mat
type ActivFunct = Callable[[Vector], Vector]    # pre signal -> post signal

type DecayFunct = Callable[[Vector], Vector]    # time -> learn mult
type WarmupFunct = Callable[[Vector], Vector]   # time -> learn mult
