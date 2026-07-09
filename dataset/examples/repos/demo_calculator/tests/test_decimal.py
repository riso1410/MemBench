from decimal import Decimal

import pytest

from calculator.decimal_ops import divide, modulo
from calculator.exceptions import CalculatorError


def test_divide_by_zero_raises_calculator_error():
    with pytest.raises(CalculatorError):
        divide("1", "0")


def test_divide_basic():
    assert divide("6", "3") == Decimal("2")


def test_modulo_by_zero_raises_calculator_error():
    with pytest.raises(CalculatorError):
        modulo("1", "0")
