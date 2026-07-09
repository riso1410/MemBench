import pytest
from decimal import Decimal

from serializer.encode import to_json
from serializer.exceptions import SerializerError


def test_decimal_is_serialized_as_string():
    assert to_json({"amount": Decimal("1.50")}) == '{"amount": "1.50"}'


def test_plain_types_unchanged():
    assert to_json({"n": 1}) == '{"n": 1}'


def test_unserializable_raises_serializer_error():
    with pytest.raises(SerializerError):
        to_json({"f": object()})
