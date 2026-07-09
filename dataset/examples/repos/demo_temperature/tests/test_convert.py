from decimal import Decimal

import pytest

from temperature.convert import c_to_f, f_to_c
from temperature.exceptions import TemperatureError


def test_c_to_f_invalid_raises_temperature_error():
    with pytest.raises(TemperatureError):
        c_to_f("not-a-number")


def test_c_to_f_basic():
    assert c_to_f("100") == Decimal("212")


def test_f_to_c_invalid_raises_temperature_error():
    with pytest.raises(TemperatureError):
        f_to_c("oops")
