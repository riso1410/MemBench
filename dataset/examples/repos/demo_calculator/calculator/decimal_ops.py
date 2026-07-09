from decimal import Decimal, DivisionByZero, InvalidOperation

from .exceptions import CalculatorError


def modulo(a: str, b: str) -> Decimal:
    try:
        return Decimal(a) % Decimal(b)
    except (DivisionByZero, InvalidOperation, ZeroDivisionError) as exc:
        raise CalculatorError(f"modulo failed: {exc}") from exc


def divide(a: str, b: str) -> Decimal:
    return Decimal(a) / Decimal(b)
