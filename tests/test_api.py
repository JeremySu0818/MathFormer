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
        assert mathformer.__version__ == "1.0.0"
