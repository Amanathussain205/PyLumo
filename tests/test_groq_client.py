from pylumo.groq_client import retry_after_seconds


def test_retry_after_seconds_with_integer():
    result = retry_after_seconds("10")

    assert result == 10.0


def test_retry_after_seconds_with_decimal():
    result = retry_after_seconds("2.5")

    assert result == 2.5


def test_retry_after_seconds_with_invalid_value():
    result = retry_after_seconds("invalid")

    assert result == 0.0


def test_retry_after_seconds_with_empty_value():
    result = retry_after_seconds("")

    assert result == 0.0