"""Application configuration and environment validation for PyLumo."""

import logging
import math
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_FILE, override=False)

API_KEY = (os.getenv("GROQ_API_KEY") or "").strip()
MODEL = (os.getenv("GROQ_MODEL") or "openai/gpt-oss-20b").strip()
LOGGER = logging.getLogger("pylumo")


def env_number(name: str, default, minimum: float, maximum: float):
    """Validate optional numeric settings; fail clearly on configuration errors."""
    try:
        value = float(os.getenv(name, str(default)))
        if not math.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError
        if isinstance(default, int):
            if not value.is_integer():
                raise ValueError
            return int(value)
        return value
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"{name} must be a number between {minimum} and {maximum}."
        ) from exc


MIN_REQUEST_INTERVAL = env_number(
    "GROQ_MIN_REQUEST_INTERVAL_SECONDS", 2.1, 0.0, 120.0
)
MAX_API_ATTEMPTS = env_number("GROQ_MAX_ATTEMPTS", 3, 1, 5)
RETRY_WAIT_BUDGET = env_number(
    "GROQ_RETRY_WAIT_BUDGET_SECONDS", 30.0, 0.0, 120.0
)
REQUEST_TIMEOUT_SECONDS = env_number(
    "GROQ_REQUEST_TIMEOUT_SECONDS", 60.0, 5.0, 180.0
)
DEBUG_GROQ = os.getenv("DEBUG_GROQ", "false").lower() in {"1", "true", "yes"}

DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 1024
SUMMARY_EVERY_TURNS = 5
SUMMARY_MAX_TOKENS = 2048
SUMMARY_MAX_CHARS = 6000
MEMORY_DIR = BASE_DIR / "memory"