from typing import Union, Optional
from .api import MathFormerAPI, MathFormer
from .tokenizer import MathTokenizer


__version__ = "1.0.3"


_default_api = MathFormerAPI()


def add(*args: Union[str, int]) -> str:
    return _default_api.add(*args)


def sub(*args: Union[str, int]) -> str:
    return _default_api.sub(*args)


def mul(*args: Union[str, int]) -> str:
    return _default_api.mul(*args)


def div(*args: Union[str, int]) -> str:
    return _default_api.div(*args)


def calculate(operation: str, a, b) -> str:
    return _default_api.calculate(operation, a, b)


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
