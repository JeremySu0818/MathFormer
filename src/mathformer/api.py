import re
import os
import warnings
from decimal import Decimal, getcontext
from typing import Optional, Dict, Any, List, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

getcontext().prec = 50

from .tokenizer import MathTokenizer
from .llama_pure import TinyLlama

_BASE_DIR = Path(__file__).parent
_DEFAULT_MODEL_PATHS = {
    "add": _BASE_DIR / "addformer",
    "sub": _BASE_DIR / "subformer",
    "mul": _BASE_DIR / "mulformer",
    "div": _BASE_DIR / "divformer",
}


_OPERATION_SYMBOLS = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
}


class MathFormer:

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None, 
        max_new_tokens: int = 32,
    ):
        self.model_path = Path(model_path)
        self.max_new_tokens = max_new_tokens

        self._model: Optional[TinyLlama] = None
        self._tokenizer: Optional[MathTokenizer] = None
        self._loaded = False

    def load(self) -> "MathFormer":
        if self._loaded:
            return self

        self._tokenizer = MathTokenizer.from_pretrained(str(self.model_path))
        self._model = TinyLlama(str(self.model_path))
        self._loaded = True
        return self

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def predict(self, expression: str) -> str:
        if not self._loaded:
            self.load()

        if "=" not in expression:
            expression += "="

        inputs = self._tokenizer(expression, padding=False)
        input_ids = inputs["input_ids"][0]
        output_ids = self._model.generate(
            input_ids=input_ids,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self._tokenizer.eos_token_id,
        )

        generated_text = self._tokenizer.decode(output_ids, skip_special_tokens=True)

        if "=" in generated_text:
            answer = generated_text.split("=", 1)[1].strip()
        else:
            answer = generated_text.strip()

        return answer

    def batch_predict(self, expressions: List[str]) -> List[str]:
        """Batch inference for multiple expressions.
        Since we are in pure Python, true batching matrix ops is not implemented.
        We simply loop.
        """
        if not self._loaded:
            self.load()

        results = []
        for expr in expressions:
            results.append(self.predict(expr))
        return results

    def __call__(self, expression: str) -> str:
        return self.predict(expression)

    def __enter__(self) -> "MathFormer":
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.unload()


class MathFormerAPI:

    def __init__(
        self,
        model_paths: Optional[Dict[str, str]] = None,
        device: Optional[str] = None,
        max_new_tokens: int = 32,
        lazy_load: bool = True,
        max_workers: Optional[int] = None,
    ):
        self.max_new_tokens = max_new_tokens
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)

        paths = model_paths or {}
        self._model_paths = {
            op: Path(paths.get(op, _DEFAULT_MODEL_PATHS[op]))
            for op in ["add", "sub", "mul", "div"]
        }

        self.models: Dict[str, MathFormer] = {
            op: MathFormer(
                model_path=str(path),
                max_new_tokens=self.max_new_tokens,
            )
            for op, path in self._model_paths.items()
        }

        if not lazy_load:
            self.load_all()

    def load_all(self) -> "MathFormerAPI":
        for model in self.models.values():
            model.load()
        return self

    def unload_all(self) -> None:
        for model in self.models.values():
            model.unload()

    def load(self, operation: str) -> "MathFormerAPI":
        if operation in self.models:
            self.models[operation].load()
        return self

    def unload(self, operation: str) -> None:
        if operation in self.models:
            self.models[operation].unload()

    def _raw_predict(self, operation: str, expression: str) -> str:
        if operation not in self.models:
            raise ValueError(
                f"Unknown operation type: {operation}. Available: {list(self.models.keys())}"
            )
        return self.models[operation].predict(expression)

    def _batch_raw_predict(self, operation: str, expressions: List[str]) -> List[str]:
        if operation not in self.models:
            raise ValueError(
                f"Unknown operation type: {operation}. Available: {list(self.models.keys())}"
            )
        if len(expressions) == 0:
            return []
        
        return self.models[operation].batch_predict(expressions)

    def _single_add(self, a: int, b: int) -> Tuple[int, int]:
        result_str = self._raw_predict("add", f"{a}+{b}")
        if not result_str: return 0, 0
        try:
            result = int(result_str)
        except ValueError:
            return 0, 0
        return result % 10, result // 10

    def _single_sub(self, a: int, b: int, borrow: int = 0) -> Tuple[int, int]:
        a_actual = a - borrow

        if a_actual >= b:
            result_str = self._raw_predict("sub", f"{a_actual}-{b}")
            try:
                return int(result_str), 0
            except ValueError:
                return 0, 0
        else:
            a_with_borrow = a_actual + 10
            result_str = self._raw_predict("sub", f"{a_with_borrow}-{b}")
            try:
                return int(result_str), 1
            except ValueError:
                return 0, 0

    def _single_mul(self, a: int, b: int) -> int:
        result_str = self._raw_predict("mul", f"{a}*{b}")
        try:
            return int(result_str)
        except ValueError:
            return 0

    def _single_div(self, a: int, b: int) -> Tuple[int, int]:
        result_str = self._raw_predict("div", f"{a}/{b}")
        match = re.match(r"Q(\d+)R(\d+)", result_str)
        if match:
            return int(match.group(1)), int(match.group(2))
        try:
            return int(result_str), 0
        except ValueError:
            return 0, 0

    def _multi_add(self, a: int, b: int) -> int:
        if a < 0 or b < 0:
            if a < 0 and b < 0:
                return -self._multi_add(-a, -b)
            elif a < 0:
                return self._multi_sub(b, -a)
            else:
                return self._multi_sub(a, -b)

        digits_a = [int(d) for d in str(a)[::-1]]
        digits_b = [int(d) for d in str(b)[::-1]]

        max_len = max(len(digits_a), len(digits_b))
        digits_a.extend([0] * (max_len - len(digits_a)))
        digits_b.extend([0] * (max_len - len(digits_b)))

        result = []
        carry = 0

        for i in range(max_len):
            sum_with_carry = digits_a[i] + carry
            carry_from_first = 0

            if sum_with_carry > 9:
                sum_with_carry = sum_with_carry - 10
                carry_from_first = 1

            digit_result, new_carry = self._single_add(sum_with_carry, digits_b[i])

            result.append(digit_result)
            carry = new_carry + carry_from_first

        if carry > 0:
            result.append(carry)

        return int("".join(str(d) for d in result[::-1]))

    def _multi_sub(self, a: int, b: int) -> int:
        if a < 0 and b < 0:
            return self._multi_sub(-b, -a)
        elif a < 0:
            return -self._multi_add(-a, b)
        elif b < 0:
            return self._multi_add(a, -b)

        if a < b:
            return -self._multi_sub(b, a)

        digits_a = [int(d) for d in str(a)[::-1]]
        digits_b = [int(d) for d in str(b)[::-1]]

        digits_b.extend([0] * (len(digits_a) - len(digits_b)))

        result = []
        borrow = 0

        for i in range(len(digits_a)):
            digit_a = digits_a[i]
            digit_b = digits_b[i]

            digit_result, new_borrow = self._single_sub(digit_a, digit_b, borrow)
            result.append(digit_result)
            borrow = new_borrow

        while len(result) > 1 and result[-1] == 0:
            result.pop()

        return int("".join(str(d) for d in result[::-1]))

    def _compute_partial_product(self, i: int, digit_b: int, digits_a: List[int]) -> List[Tuple[int, int]]:
        """Calculate the partial product of a single digit_b with all digits_a (result of one row)."""
        expressions = [f"{digit_a}*{digit_b}" for digit_a in digits_a]
        results = self._batch_raw_predict("mul", expressions)
        products = []
        for r in results:
            try:
                products.append(int(r))
            except ValueError:
                products.append(0)

        partial = []
        carry = 0
        for j, product in enumerate(products):
            total = product + carry
            partial.append((i + j, total % 10))
            carry = total // 10

        k = i + len(digits_a)
        while carry > 0:
            partial.append((k, carry % 10))
            carry = carry // 10
            k += 1

        return partial

    def _multi_mul(self, a: int, b: int) -> int:
        negative = (a < 0) ^ (b < 0)
        a, b = abs(a), abs(b)

        if a == 0 or b == 0:
            return 0

        digits_a = [int(d) for d in str(a)[::-1]]
        digits_b = [int(d) for d in str(b)[::-1]]

        result = [0] * (len(digits_a) + len(digits_b))

        if len(digits_b) >= 2:
            partial_products = {}
            for i, digit_b in enumerate(digits_b):
                 partial_products[i] = self._compute_partial_product(i, digit_b, digits_a)

            for i in range(len(digits_b)):
                for pos, val in partial_products[i]:
                    result[pos] += val

            for pos in range(len(result) - 1):
                if result[pos] >= 10:
                    result[pos + 1] += result[pos] // 10
                    result[pos] = result[pos] % 10
        else:
            for i, digit_b in enumerate(digits_b):
                carry = 0
                for j, digit_a in enumerate(digits_a):
                    product = self._single_mul(digit_a, digit_b)
                    total = product + carry + result[i + j]
                    result[i + j] = total % 10
                    carry = total // 10

                k = i + len(digits_a)
                while carry > 0:
                    total = carry + result[k]
                    result[k] = total % 10
                    carry = total // 10
                    k += 1

        while len(result) > 1 and result[-1] == 0:
            result.pop()

        final_result = int("".join(str(d) for d in result[::-1]))
        return -final_result if negative else final_result

    def _trial_division(self, dividend: int, divisor: int) -> Tuple[int, int]:
        products = {}
        for q in range(10):
            products[q] = self._multi_mul(divisor, q)

        quotient = 0
        for q in range(9, -1, -1):
            if products[q] <= dividend:
                quotient = q
                break

        remainder = self._multi_sub(dividend, products[quotient])

        return quotient, remainder

    def _multi_div(self, a: int, b: int) -> Tuple[int, int]:
        if b == 0:
            raise ZeroDivisionError("Divisor cannot be zero")

        negative = (a < 0) ^ (b < 0)
        a, b = abs(a), abs(b)

        if a < b:
            return 0, a

        if a == 0:
            return 0, 0

        digits_a = [int(d) for d in str(a)]

        quotient_digits = []
        remainder = 0

        for digit in digits_a:
            current = remainder * 10 + digit

            if current < b:
                quotient_digits.append(0)
                remainder = current
            else:
                if b <= 9 and current <= 89:
                    q, r = self._single_div(current, b)
                else:
                    q, r = self._trial_division(current, b)

                quotient_digits.append(q)
                remainder = r

        while len(quotient_digits) > 1 and quotient_digits[0] == 0:
            quotient_digits.pop(0)

        quotient = int("".join(str(d) for d in quotient_digits))

        if negative:
            quotient = -quotient

        return quotient, remainder

    def _parse_decimal(self, value: Union[str, int, float, Decimal]) -> Tuple[int, int]:
        if isinstance(value, (int, float)):
            value = str(value)
        elif isinstance(value, Decimal):
            value = str(value)

        value = value.strip()

        negative = value.startswith("-")
        if negative:
            value = value[1:]

        if "." in value:
            integer_part, decimal_part = value.split(".")
            integer_part = integer_part.lstrip("0") or "0"
            combined = integer_part + decimal_part
            combined = combined.lstrip("0") or "0"
            decimal_places = len(decimal_part)
            int_value = int(combined)
        else:
            int_value = int(value)
            decimal_places = 0

        if negative:
            int_value = -int_value

        return int_value, decimal_places

    def _format_decimal_result(self, int_value: int, decimal_places: int) -> str:
        if decimal_places == 0:
            return str(int_value)

        negative = int_value < 0
        int_value = abs(int_value)

        str_value = str(int_value)

        if len(str_value) <= decimal_places:
            str_value = "0" * (decimal_places - len(str_value) + 1) + str_value

        integer_part = str_value[:-decimal_places]
        decimal_part = str_value[-decimal_places:]

        decimal_part = decimal_part.rstrip("0")

        if decimal_part:
            result = f"{integer_part}.{decimal_part}"
        else:
            result = integer_part

        if negative:
            result = "-" + result

        return result

    def _decimal_add(self, a: Union[str, int, float, Decimal], b: Union[str, int, float, Decimal]) -> str:
        a_val, a_dec = self._parse_decimal(a)
        b_val, b_dec = self._parse_decimal(b)

        max_dec = max(a_dec, b_dec)
        if a_dec < max_dec:
            a_val *= (10 ** (max_dec - a_dec))
        if b_dec < max_dec:
            b_val *= (10 ** (max_dec - b_dec))

        result = self._multi_add(a_val, b_val)

        return self._format_decimal_result(result, max_dec)

    def _decimal_sub(self, a: Union[str, int, float, Decimal], b: Union[str, int, float, Decimal]) -> str:
        a_val, a_dec = self._parse_decimal(a)
        b_val, b_dec = self._parse_decimal(b)

        max_dec = max(a_dec, b_dec)
        if a_dec < max_dec:
            a_val *= (10 ** (max_dec - a_dec))
        if b_dec < max_dec:
            b_val *= (10 ** (max_dec - b_dec))

        result = self._multi_sub(a_val, b_val)

        return self._format_decimal_result(result, max_dec)

    def _decimal_mul(self, a: Union[str, int, float, Decimal], b: Union[str, int, float, Decimal]) -> str:
        a_val, a_dec = self._parse_decimal(a)
        b_val, b_dec = self._parse_decimal(b)

        result = self._multi_mul(a_val, b_val)
        total_dec = a_dec + b_dec

        return self._format_decimal_result(result, total_dec)

    def _decimal_div(self, a: Union[str, int, float, Decimal], b: Union[str, int, float, Decimal], precision: int = 10) -> str:
        a_val, a_dec = self._parse_decimal(a)
        b_val, b_dec = self._parse_decimal(b)

        if b_val == 0:
            raise ZeroDivisionError("Divisor cannot be zero")

        negative = (a_val < 0) ^ (b_val < 0)
        a_val = abs(a_val)
        b_val = abs(b_val)

        if a_dec > b_dec:
            b_val *= (10 ** (a_dec - b_dec))
        elif b_dec > a_dec:
            a_val *= (10 ** (b_dec - a_dec))

        quotient, remainder = self._multi_div(a_val, b_val)

        if remainder == 0:
            result = str(quotient)
            if negative and quotient != 0:
                result = "-" + result
            return result

        decimal_digits = []
        for _ in range(precision):
            remainder = self._multi_mul(remainder, 10)

            digit_quotient, remainder = self._multi_div(remainder, b_val)
            decimal_digits.append(digit_quotient)

            if remainder == 0:
                break

        decimal_part = "".join(str(d) for d in decimal_digits)
        decimal_part = decimal_part.rstrip("0")

        if decimal_part:
            result = f"{quotient}.{decimal_part}"
        else:
            result = str(quotient)

        if negative and (quotient != 0 or decimal_part):
            result = "-" + result

        return result

    def _is_decimal_input(self, value: Union[str, int, float]) -> bool:
        if isinstance(value, float):
            return True
        if isinstance(value, str) and "." in value:
            return True
        return False

    def add(self, *args: Union[str, int, float]) -> str:
        if len(args) == 0:
            raise ValueError("At least one argument is required")

        values = []
        has_decimal = False

        if len(args) == 1 and isinstance(args[0], str) and "+" in args[0]:
            expression = args[0].replace(" ", "").replace("=", "")
            parts = expression.split("+")
            values = parts
            has_decimal = any("." in p for p in parts)
        else:
            values = list(args)
            has_decimal = any(self._is_decimal_input(a) for a in args)

        if not values:
            return "0"

        if has_decimal:
            result = str(values[0])
            for val in values[1:]:
                result = self._decimal_add(result, str(val))
            return result
        else:
            try:
               int_values = [int(v) for v in values]
            except ValueError:
               return "Error"
            result = int_values[0]
            for val in int_values[1:]:
                result = self._multi_add(result, val)
            return str(result)

    def sub(self, *args: Union[str, int, float]) -> str:
        if len(args) == 0:
            raise ValueError("At least one argument is required")

        values = []
        has_decimal = False

        if len(args) == 1 and isinstance(args[0], str) and "-" in args[0].lstrip("-"):
            expression = args[0].replace(" ", "").replace("=", "")
            if expression.startswith("-"):
                temp_expr = expression[1:]
                parts = temp_expr.split("-")
                values = ["-" + parts[0]] + parts[1:]
            else:
                parts = expression.split("-")
                values = parts
            has_decimal = any("." in p.lstrip("-") for p in values)
        else:
            values = list(args)
            has_decimal = any(self._is_decimal_input(a) for a in args)

        if not values:
            return "0"

        if has_decimal:
            result = str(values[0])
            for val in values[1:]:
                result = self._decimal_sub(result, str(val))
            return result
        else:
            try:
                int_values = [int(v) for v in values]
            except ValueError:
                return "Error"
            result = int_values[0]
            for val in int_values[1:]:
                result = self._multi_sub(result, val)
            return str(result)

    def mul(self, *args: Union[str, int, float]) -> str:
        if len(args) == 0:
            raise ValueError("At least one argument is required")

        values = []
        has_decimal = False

        if (
            len(args) == 1
            and isinstance(args[0], str)
            and any(op in args[0] for op in ["*", "×"])
        ):
            expression = args[0].replace(" ", "").replace("=", "").replace("×", "*")
            parts = expression.split("*")
            values = parts
            has_decimal = any("." in p for p in parts)
        else:
            values = list(args)
            has_decimal = any(self._is_decimal_input(a) for a in args)

        if not values:
            return "0"

        if has_decimal:
            result = str(values[0])
            for val in values[1:]:
                result = self._decimal_mul(result, str(val))
            return result
        else:
            try:
                int_values = [int(v) for v in values]
            except ValueError:
                return "Error"
            result = int_values[0]
            for val in int_values[1:]:
                result = self._multi_mul(result, val)
            return str(result)

    def div(self, *args: Union[str, int, float], precision: int = 10) -> str:
        if len(args) == 0:
            raise ValueError("At least one argument is required")

        values = []
        has_decimal = False

        if (
            len(args) == 1
            and isinstance(args[0], str)
            and any(op in args[0] for op in ["/", "÷"])
        ):
            expression = args[0].replace(" ", "").replace("=", "").replace("÷", "/")
            parts = expression.split("/")
            values = parts
            has_decimal = any("." in p for p in parts)
        else:
            values = list(args)
            has_decimal = any(self._is_decimal_input(a) for a in args)

        if not values:
            return "0"

        result = str(values[0])
        for val in values[1:]:
            result = self._decimal_div(result, str(val), precision=precision)
        return result

    def calculate(
        self, operation: str, a: Union[int, float, str], b: Union[int, float, str],
        precision: int = 10
    ) -> str:
        if operation == "add":
            return self.add(a, b)
        elif operation == "sub":
            return self.sub(a, b)
        elif operation == "mul":
            return self.mul(a, b)
        elif operation == "div":
            return self.div(a, b, precision=precision)
        else:
            raise ValueError(f"Unknown operation: {operation}")
