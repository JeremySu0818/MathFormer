from typing import Union, Optional
from .api import MathFormerAPI, MathFormer
from .tokenizer import MathTokenizer


__version__ = "1.2.1"


_default_api = MathFormerAPI()


def add(*args: Union[str, int, float]) -> str:
    """Addition operation, supports integers and decimals."""
    return _default_api.add(*args)


def sub(*args: Union[str, int, float]) -> str:
    """Subtraction operation, supports integers and decimals."""
    return _default_api.sub(*args)


def mul(*args: Union[str, int, float]) -> str:
    """Multiplication operation, supports integers and decimals."""
    return _default_api.mul(*args)


def div(*args: Union[str, int, float], precision: int = 10) -> str:
    """
    Division operation, supports integers and decimals.
    
    Args:
        *args: Division operands (can be an expression string or multiple values).
        precision: Decimal precision, defaults to 10 digits.
    
    Returns:
        String representation of the operation result, returns an integer if divisible, otherwise a decimal.
    """
    return _default_api.div(*args, precision=precision)


def calculate(operation: str, a: Union[str, int, float], b: Union[str, int, float], precision: int = 10) -> str:
    """
    Execute specified operation, supports integers and decimals.
    
    Args:
        operation: Operation type ("add", "sub", "mul", "div").
        a: First operand.
        b: Second operand.
        precision: Decimal precision for division (defaults to 10 digits).
    
    Returns:
        String representation of the operation result.
    """
    return _default_api.calculate(operation, a, b, precision=precision)


def unload_models():
    _default_api.unload_all()


__all__ = [
    "MathFormerAPI",
    "MathFormer",
    "MathTokenizer",
    "add",
    "sub",
    "mul",
    "div",
    "calculate",
    "unload_models",
]
