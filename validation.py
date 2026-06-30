# pyright: reportMissingImports=false, reportUndefinedVariable=false

import warnings


def check_condition(cond: bool, msg: str | None = None) -> None:
    """
    Check if given condition is met, else raise ValueError.
    
    :param cond: condition to check
    :param msg: optional message if condition fails
    """
    # condition met; no changes required
    if cond: return

    # condition failed; raise value error
    error_msg = "condition failed"
    if msg is not None: 
        error_msg += f": {msg}"
    raise ValueError(error_msg)


def clamp_and_warn[T: (int, float)](val: T, *, low: T, high: T) -> T:
    """
    Clamp given value to interval [low, high], 
    emit warning if clamping occurs.
    
    :param val: value to clamp if needed
    :param low: lower bound of clamp range
    :param high: higher bound of clamp range
    :return: original or clamped value
    """
    # ensure correct clamp range [low, high]
    check_condition(low <= high, "low must not be greater than high")
    
    # value within range; return it
    if low <= val <= high: return val

    # value out of range; clamp and warn
    warnings.warn(
        f"value {val} out of range; clamped to [{low}, {high}]",
        UserWarning, stacklevel=2
    )
    return max(low, min(val, high))
