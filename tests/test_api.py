"""
Unit tests for MathFormer API
"""

import pytest
import mathformer


class TestAddition:
    """Test cases for addition operations"""

    def test_add_two_integers(self):
        """Test adding two integers"""
        result = mathformer.add(1, 2)
        assert result == "3"

    def test_add_multiple_integers(self):
        """Test adding multiple integers"""
        result = mathformer.add(1, 2, 3)
        assert result == "6"

    def test_add_with_strings(self):
        """Test adding numbers passed as strings"""
        result = mathformer.add("10", "20")
        assert result == "30"


class TestSubtraction:
    """Test cases for subtraction operations"""

    def test_sub_two_integers(self):
        """Test subtracting two integers"""
        result = mathformer.sub(5, 3)
        assert result == "2"

    def test_sub_multiple_integers(self):
        """Test subtracting multiple integers"""
        result = mathformer.sub(10, 3, 2)
        assert result == "5"


class TestMultiplication:
    """Test cases for multiplication operations"""

    def test_mul_two_integers(self):
        """Test multiplying two integers"""
        result = mathformer.mul(3, 4)
        assert result == "12"

    def test_mul_multiple_integers(self):
        """Test multiplying multiple integers"""
        result = mathformer.mul(2, 3, 4)
        assert result == "24"


class TestDivision:
    """Test cases for division operations"""

    def test_div_two_integers(self):
        """Test dividing two integers"""
        result = mathformer.div(10, 2)
        assert result == "5"


class TestCalculate:
    """Test cases for the calculate function"""

    def test_calculate_add(self):
        """Test calculate with add operation"""
        result = mathformer.calculate("add", 5, 3)
        assert result == "8"

    def test_calculate_sub(self):
        """Test calculate with sub operation"""
        result = mathformer.calculate("sub", 10, 4)
        assert result == "6"

    def test_calculate_mul(self):
        """Test calculate with mul operation"""
        result = mathformer.calculate("mul", 6, 7)
        assert result == "42"

    def test_calculate_div(self):
        """Test calculate with div operation"""
        result = mathformer.calculate("div", 20, 5)
        assert result == "4"


class TestModuleExports:
    """Test that all expected exports are available"""

    def test_mathformerapi_exists(self):
        """Test MathFormerAPI class is exported"""
        assert hasattr(mathformer, "MathFormerAPI")

    def test_mathformer_exists(self):
        """Test MathFormer class is exported"""
        assert hasattr(mathformer, "MathFormer")

    def test_tokenizer_exists(self):
        """Test MathTokenizer class is exported"""
        assert hasattr(mathformer, "MathTokenizer")

    def test_version_exists(self):
        """Test version is defined"""
        assert hasattr(mathformer, "__version__")
        assert mathformer.__version__ == "1.2.0"


class TestDecimalAddition:
    """Test cases for decimal addition operations"""

    def test_add_decimals(self):
        """Test adding two decimals"""
        result = mathformer.add(1.5, 2.3)
        assert result == "3.8"

    def test_add_decimal_strings(self):
        """Test adding decimal strings"""
        result = mathformer.add("1.5", "2.3")
        assert result == "3.8"

    def test_add_decimal_expression(self):
        """Test adding decimal expression"""
        result = mathformer.add("1.5+2.3")
        assert result == "3.8"

    def test_add_mixed_decimal_integer(self):
        """Test adding mixed decimal and integer"""
        result = mathformer.add(1, 2.5)
        assert result == "3.5"


class TestDecimalSubtraction:
    """Test cases for decimal subtraction operations"""

    def test_sub_decimals(self):
        """Test subtracting two decimals"""
        result = mathformer.sub(5.5, 2.3)
        assert result == "3.2"

    def test_sub_decimals_negative_result(self):
        """Test subtracting decimals with negative result"""
        result = mathformer.sub(2.3, 5.5)
        assert result == "-3.2"


class TestDecimalMultiplication:
    """Test cases for decimal multiplication operations"""

    def test_mul_decimals(self):
        """Test multiplying two decimals"""
        result = mathformer.mul(1.5, 2.5)
        assert result == "3.75"

    def test_mul_decimal_and_integer(self):
        """Test multiplying decimal and integer"""
        result = mathformer.mul(2.5, 4)
        assert result == "10"


class TestDecimalDivision:
    """Test cases for decimal division operations"""

    def test_div_exact(self):
        """Test exact division returns integer"""
        result = mathformer.div(10, 2)
        assert result == "5"

    def test_div_non_exact(self):
        """Test non-exact division returns decimal instead of QxRy"""
        result = mathformer.div(10, 4)
        assert result == "2.5"

    def test_div_repeating_decimal(self):
        """Test repeating decimal with precision"""
        result = mathformer.div(1, 3)
        assert result == "0.3333333333"  # 10 decimal places

    def test_div_custom_precision(self):
        """Test division with custom precision"""
        result = mathformer.div(1, 7, precision=5)
        assert result == "0.14285"

    def test_div_decimal_by_integer(self):
        """Test decimal divided by integer"""
        result = mathformer.div(7.5, 3)
        assert result == "2.5"

    def test_div_integer_by_decimal(self):
        """Test integer divided by decimal"""
        result = mathformer.div(10, 2.5)
        assert result == "4"

    def test_div_decimal_by_decimal(self):
        """Test decimal divided by decimal"""
        result = mathformer.div(7.5, 2.5)
        assert result == "3"


class TestDecimalCalculate:
    """Test cases for calculate function with decimals"""

    def test_calculate_add_decimals(self):
        """Test calculate add with decimals"""
        result = mathformer.calculate("add", 1.5, 2.5)
        assert result == "4"

    def test_calculate_div_with_precision(self):
        """Test calculate div with custom precision"""
        result = mathformer.calculate("div", 10, 3, precision=3)
        assert result == "3.333"

