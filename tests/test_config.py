from pylumo.config import (
    MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    SUMMARY_EVERY_TURNS,
)


def test_model_is_configured():
    assert MODEL
    assert isinstance(MODEL, str)


def test_default_temperature_is_valid():
    assert 0.0 <= DEFAULT_TEMPERATURE <= 1.0


def test_default_max_tokens_is_positive():
    assert DEFAULT_MAX_TOKENS > 0


def test_summary_interval_is_positive():
    assert SUMMARY_EVERY_TURNS > 0