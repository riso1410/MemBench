from decimal import Decimal, InvalidOperation

from .exceptions import TemperatureError


def c_to_f(value: str) -> Decimal:
    return Decimal(value) * 9 / 5 + 32


def f_to_c(value: str) -> Decimal:
    try:
        return (Decimal(value) - 32) * 5 / 9
    except (InvalidOperation, ValueError) as exc:
        raise TemperatureError(f"f_to_c failed: {exc}") from exc
