import json
from decimal import Decimal

from .exceptions import SerializerError


def to_json(data) -> str:
    return json.dumps(data)
