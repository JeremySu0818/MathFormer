from typing import Union, Optional
from .api import MathFormerAPI, MathFormer
from .tokenizer import MathTokenizer


__version__ = "1.0.0"


_default_api = MathFormerAPI()


def add(a: Union[str, int], b: Optional[Union[int, str]] = None) -> str:
    return _default_api.add(a, b)


def sub(a: Union[str, int], b: Optional[Union[int, str]] = None) -> str:
    return _default_api.sub(a, b)


def mul(a: Union[str, int], b: Optional[Union[int, str]] = None) -> str:
    return _default_api.mul(a, b)


def div(a: Union[str, int], b: Optional[Union[int, str]] = None) -> str:
    return _default_api.div(a, b)


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
