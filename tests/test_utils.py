from pylumo.utils import valid_api_key, convert_history


def test_valid_api_key_accepts_realistic_key():
    assert valid_api_key("gsk_abcdefghijklmnopqrstuvwxyz") is True


def test_valid_api_key_rejects_empty_value():
    assert valid_api_key("") is False


def test_valid_api_key_rejects_placeholder():
    assert valid_api_key("your_api_key_here") is False


def test_convert_history_keeps_valid_messages():
    messages = [
        {"role": "user", "content": "What is a variable?"},
        {"role": "assistant", "content": "A variable stores a value."},
    ]

    result = convert_history(messages)

    assert result == messages