# MathFormer

MathFormer is a Python library that leverages Transformer-based language models to perform mathematical operations. Unlike standard arithmetic libraries, MathFormer uses Llama-architecture models to "predict" the results of arithmetic operations, token by token, demonstrating the capability of small language models to learn arithmetic rules.

It supports basic arithmetic operations: **Addition**, **Subtraction**, **Multiplication**, and **Division**.

## Features

- **Transformer-Powered Arithmetic**: Uses specialized Llama-based models for each arithmetic operation.
- **Large Number Support**: Implements recursive logic to handle multi-digit arithmetic using digit-by-digit prediction (similar to manual calculation).
- **Unified API**: Easy-to-use functions for `add`, `sub`, `mul`, and `div`.
- **Resource Management**: Supports lazy loading of models to save memory, and manual unloading.
- **Custom Tokenizer**: Built-in minimalist tokenizer optimized for mathematical expressions.

## Installation

You can install MathFormer via pip:

```bash
pip install mathformer
```

## Quick Start

The simplest way to use MathFormer is through the top-level convenience functions. These functions automatically handle model loading when needed.

```python
import mathformer

# Addition
result = mathformer.add(123, 456)
print(f"123 + 456 = {result}")  # Output: 579

# Subtraction
result = mathformer.sub(1000, 250)
print(f"1000 - 250 = {result}") # Output: 750

# Multiplication
result = mathformer.mul(12, 12)
print(f"12 * 12 = {result}")    # Output: 144

# Division (returns decimal for non-exact results)
result = mathformer.div(100, 3)
print(f"100 / 3 = {result}")    # Output: 0.3333333333

result = mathformer.div(100, 4)
print(f"100 / 4 = {result}")    # Output: 25 (exact division)
```

You can also pass string expressions:

```python
print(mathformer.add("100 + 200"))
print(mathformer.calculate("mul", 50, 4))
```

## Decimal Support (v1.1.0+)

MathFormer now supports decimal (floating-point) arithmetic while maintaining its algorithm-based approach:

```python
import mathformer

# Decimal addition
result = mathformer.add(1.5, 2.3)
print(f"1.5 + 2.3 = {result}")  # Output: 3.8

# Decimal subtraction (with negative results)
result = mathformer.sub(2.3, 5.5)
print(f"2.3 - 5.5 = {result}")  # Output: -3.2

# Decimal multiplication
result = mathformer.mul(1.5, 2.5)
print(f"1.5 * 2.5 = {result}")  # Output: 3.75

# Division with custom precision
result = mathformer.div(1, 7, precision=5)
print(f"1 / 7 = {result}")      # Output: 0.14285

# Mixed decimal and integer operations
result = mathformer.div(7.5, 2.5)
print(f"7.5 / 2.5 = {result}")  # Output: 3
```

### Division Behavior

- **Exact division**: Returns an integer (e.g., `10 / 2 = 5`)
- **Non-exact division**: Returns a decimal with up to 10 decimal places (configurable via `precision` parameter)
- **No more Q/R format**: Division now outputs decimals instead of `Q{quotient}R{remainder}` format

## Advanced Usage

For more control over resource usage, you can use the `MathFormerAPI` class directly.

### Managing Resources (Load/Unload)

By default, models are lazy-loaded (loaded only when first requested). You can manually load all models at startup or unload them to free CPU memory.

```python
from mathformer import MathFormerAPI

# Initialize API (lazy_load=False to load everything immediately)
api = MathFormerAPI(lazy_load=True)

# Perform operations
print(api.add(50, 50))

# Unload all models to free memory
api.unload_all()
```

### Context Manager

You can use `MathFormerAPI` as a context manager to ensure models are cleaned up after use:

```python
from mathformer import MathFormerAPI

with MathFormerAPI() as api:
    print(api.mul(99, 9))
# Models are automatically unloaded here
```

## How It Works

MathFormer isn't just calling Python's `+` or `-` operators. It actually uses a neural network to predict the result!

1.  **Single-Step Prediction**: For small single-digit operations (e.g., `5+7`), it queries a Transformer model customized for that operation.
2.  **Multi-Digit Logic**: For larger numbers (e.g., `123+456`), the library implements the standard grade-school algorithms (carrying, borrowing, partial products) but delegates the fundamental single-digit arithmetic steps to the Transformer model.

## Training Repositories

The training code and datasets for the models used in this library can be found in the following repositories:

- [JeremySu0818/AddFormer](https://github.com/JeremySu0818/AddFormer)
- [JeremySu0818/SubFormer](https://github.com/JeremySu0818/SubFormer)
- [JeremySu0818/MulFormer](https://github.com/JeremySu0818/MulFormer)
- [JeremySu0818/DivFormer](https://github.com/JeremySu0818/DivFormer)

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- Transformers >= 4.30.0

## License

This project is licensed under the MIT License.
