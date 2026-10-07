"""Small pure/testable helpers used by the Streamlit application."""

from typing import Optional


def valid_api_key(api_key: Optional[str]) -> bool:
    """Return True when a Groq API key exists."""

    return bool(
        api_key
        and api_key.strip()
    )


def convert_history(messages: list[dict]) -> list[dict]:
    """Convert Streamlit message history into Groq chat-message dictionaries."""
    converted = []

    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role not in {"user", "assistant"}:
            raise ValueError(f"Unsupported conversation role: {role!r}")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Conversation messages must contain non-empty text.")
        converted.append({"role": role, "content": content})

    return converted