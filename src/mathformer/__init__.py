from typing import Union, Optional
from .api import MathFormerAPI, MathFormer
from .tokenizer import MathTokenizer


__version__ = "2.0.0"


_default_api = MathFormerAPI()


def add(*args: Union[str, int, float]) -> str:
    """
    Perform addition operation supporting both integers and decimals.
    
    :param args: Operands for addition (can be an expression string or multiple numeric values).
    :type args: Union[str, int, float]
    :return: Formatted string representing the addition result.
    :rtype: str
    """
    return _default_api.add(*args)


def sub(*args: Union[str, int, float]) -> str:
    """
    Perform subtraction operation supporting both integers and decimals.
    
    :param args: Operands for subtraction (can be an expression string or multiple numeric values).
    :type args: Union[str, int, float]
    :return: Formatted string representing the subtraction result.
    :rtype: str
    """
    return _default_api.sub(*args)


def mul(*args: Union[str, int, float]) -> str:
    """
    Perform multiplication operation supporting both integers and decimals.
    
    :param args: Operands for multiplication (can be an expression string or multiple numeric values).
    :type args: Union[str, int, float]
    :return: Formatted string representing the multiplication result.
    :rtype: str
    """
    return _default_api.mul(*args)


def div(*args: Union[str, int, float], precision: int = 10) -> str:
    """
    Perform division operation supporting both integers and decimals.
    
    :param args: Operands for division (can be an expression string or multiple numeric values).
    :type args: Union[str, int, float]
    :param precision: Number of decimal places to calculate for the quotient. Defaults to 10.
    :type precision: int
    :return: Formatted string representing the division result (integer if divisible, otherwise decimal).
    :rtype: str
    """
    return _default_api.div(*args, precision=precision)


def calculate(operation: str, a: Union[str, int, float], b: Union[str, int, float], precision: int = 10) -> str:
    """
    Execute a specified mathematical operation on two operands.
    
    :param operation: The type of operation to perform ("add", "sub", "mul", "div").
    :type operation: str
    :param a: The first operand.
    :type a: Union[str, int, float]
    :param b: The second operand.
    :type b: Union[str, int, float]
    :param precision: Decimal precision exclusively utilized for division operations. Defaults to 10.
    :type precision: int
    :return: Formatted string representation of the operation result.
    :rtype: str
    """
    return _default_api.calculate(operation, a, b, precision=precision)


def unload_models() -> None:
    """
    Unload all loaded default MathFormerAPI models from memory globally.
    
    :return: None
    """
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
