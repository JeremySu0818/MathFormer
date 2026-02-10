from typing import Union, Optional
from .api import MathFormerAPI, MathFormer
from .tokenizer import MathTokenizer


__version__ = "1.2.0"


_default_api = MathFormerAPI()


def add(*args: Union[str, int, float]) -> str:
    """加法運算，支援整數和小數"""
    return _default_api.add(*args)


def sub(*args: Union[str, int, float]) -> str:
    """減法運算，支援整數和小數"""
    return _default_api.sub(*args)


def mul(*args: Union[str, int, float]) -> str:
    """乘法運算，支援整數和小數"""
    return _default_api.mul(*args)


def div(*args: Union[str, int, float], precision: int = 10) -> str:
    """
    除法運算，支援整數和小數
    
    Args:
        *args: 除法運算元（可以是表達式字串或多個數值）
        precision: 小數精度，預設為 10 位
    
    Returns:
        運算結果的字串表示，整除時返回整數，否則返回小數
    """
    return _default_api.div(*args, precision=precision)


def calculate(operation: str, a: Union[str, int, float], b: Union[str, int, float], precision: int = 10) -> str:
    """
    執行指定運算，支援整數和小數
    
    Args:
        operation: 運算類型 ("add", "sub", "mul", "div")
        a: 第一個運算元
        b: 第二個運算元
        precision: 除法的小數精度（預設 10 位）
    
    Returns:
        運算結果的字串表示
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
