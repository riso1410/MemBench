# Error handling guide

Public calculator APIs should expose `CalculatorError` for arithmetic backend
failures so callers do not depend on implementation-specific exception classes.
