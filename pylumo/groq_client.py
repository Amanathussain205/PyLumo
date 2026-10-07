"""Groq client creation, request pacing, retries, and API diagnostics."""

import logging
import math
import os
import random
import re
import threading
import time
from email.utils import parsedate_to_datetime
from typing import Optional
from datetime import datetime, timezone
import groq
import streamlit as st
from groq import Groq

from .config import (
    API_KEY,
    DEBUG_GROQ,
    LOGGER,
    MAX_API_ATTEMPTS,
    MIN_REQUEST_INTERVAL,
    MODEL,
    REQUEST_TIMEOUT_SECONDS,
    RETRY_WAIT_BUDGET,
)


@st.cache_resource(show_spinner=False)
def get_client(api_key: str) -> Groq:
    """Create and cache the Groq client.

    The SDK's automatic retries are disabled because generate_with_retry()
    below is the single retry layer for this app.
    """

    return Groq(
        api_key=api_key,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=0,
    )


class GroqRequestError(RuntimeError):
    """A student-safe API error, with private, redacted developer diagnostics."""

    def __init__(self, message: str, *, category: str, code=None, details=""):
        super().__init__(message)
        self.category = category
        self.code = code
        self.details = details


def redact_error(text: str) -> str:
    """Best-effort secret removal; diagnostics must still be treated as private."""
    if API_KEY:
        text = text.replace(API_KEY, "[REDACTED_API_KEY]")
    text = re.sub(r"gsk_[0-9A-Za-z_-]+", "[REDACTED_API_KEY]", text)
    text = re.sub(
        r"(?i)([?&](?:key|api_key)=)[^&\s'\"]+",
        r"\1[REDACTED_API_KEY]", text,
    )
    text = re.sub(r"(?i)(Bearer\s+)[A-Za-z0-9._~+/-]+=*", r"\1[REDACTED]", text)
    return text[:16000]


def error_payload(exc) -> dict:
    """Return the decoded Groq error body when the SDK provides one."""
    data = getattr(exc, "body", None)
    if not isinstance(data, dict):
        response = getattr(exc, "response", None)
        try:
            data = response.json() if response is not None else {}
        except (ValueError, TypeError):
            data = {}
    if not isinstance(data, dict):
        return {}
    nested = data.get("error")
    return nested if isinstance(nested, dict) else data


def positive_seconds(value) -> Optional[float]:
    """Accept numeric seconds or a duration-like string such as '12.3s'."""
    try:
        result = float(str(value).strip().removesuffix("s"))
        return result if math.isfinite(result) and result >= 0 else None
    except (TypeError, ValueError, OverflowError):
        return None


def retry_after_seconds(exc) -> Optional[float]:
    """Honor Groq's Retry-After response header when present."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", {}) or {}
    value = headers.get("Retry-After") or headers.get("retry-after")
    if value is not None:
        delay = positive_seconds(value)
        if delay is None:
            try:
                when = parsedate_to_datetime(str(value))
                if when.tzinfo is None:
                    when = when.replace(tzinfo=timezone.utc)
                delay = max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
            except (ValueError, TypeError, OverflowError):
                delay = None
        if delay is not None:
            return delay

    # Fallback for SDK/API messages that include a human-readable retry hint.
    match = re.search(
        r"retry(?:\s+after|\s+in)?\s+([0-9]+(?:\.[0-9]+)?)\s*s",
        str(exc),
        re.I,
    )
    return positive_seconds(match.group(1)) if match else None


def quota_category(exc) -> str:
    """Distinguish likely daily limits from short-lived Groq rate limits."""
    payload = error_payload(exc)
    text = " ".join(
        str(value)
        for value in (
            payload.get("message", ""),
            payload.get("type", ""),
            payload.get("code", ""),
            str(exc),
        )
    ).lower()

    if re.search(
        r"requests?\s+per\s+day|tokens?\s+per\s+day|\brpd\b|\btpd\b|\bdaily\b",
        text,
    ):
        return "daily_quota"
    return "rate_limit"


class RequestGate:
    """Thread-safe pacing shared by this app's sessions in ONE Python process.

    This is not a distributed or account-wide quota tracker. Other apps,
    other API keys, and other processes are outside this gate.
    """

    def __init__(self, interval: float):
        self.interval = interval
        self.next_allowed = 0.0
        self.lock = threading.Lock()

    def cooldown(self, seconds: float) -> None:
        with self.lock:
            self.next_allowed = max(self.next_allowed, time.monotonic() + seconds)

    def wait_for_slot(self, wait_budget: float) -> float:
        started = time.monotonic()
        while True:
            with self.lock:
                now = time.monotonic()
                delay = max(0.0, self.next_allowed - now)
                if delay <= 0:
                    self.next_allowed = now + self.interval
                    return max(0.0, now - started)
                if now - started + delay > wait_budget + 0.001:
                    raise GroqRequestError(
                        f"API requests are locally paused. Try again in about "
                        f"{math.ceil(delay)} seconds. No request was sent for "
                        "this attempt; this pause does not guarantee Groq quota is available.",
                        category="local_cooldown",
                    )
            # Do not hold the lock while sleeping or making HTTP requests.
            time.sleep(delay)


@st.cache_resource(show_spinner=False)
def get_request_gate(interval: float) -> RequestGate:
    return RequestGate(interval)


def record_api_error(exc, category: str, purpose: str) -> str:
    details = redact_error(str(exc))
    code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    st.session_state.last_api_error = {
        "model": MODEL, "purpose": purpose, "code": code,
        "category": category, "details": details,
    }
    LOGGER.warning(
        "Groq %s | model=%s | code=%s | category=%s | %s",
        purpose, MODEL, code, category, details,
    )
    return details


def generate_with_retry(
    *,
    purpose: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
):
    """One entry point for BOTH tutor and summary requests.

    At most MAX_API_ATTEMPTS physical calls. The Groq SDK's own retries are off.
    Daily limits and non-transient client errors are NOT retried. Temporary
    errors honor Retry-After when possible and use a bounded local backoff.
    """
    gate = get_request_gate(MIN_REQUEST_INTERVAL)
    wait_budget = RETRY_WAIT_BUDGET
    client = get_client(API_KEY)

    for attempt in range(1, MAX_API_ATTEMPTS + 1):
        waited = gate.wait_for_slot(max(0.0, wait_budget))
        wait_budget = max(0.0, wait_budget - waited)

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_tokens,
            )
            st.session_state.last_api_error = None
            return response

        except groq.APIStatusError as exc:
            code = getattr(exc, "status_code", None)
            category = quota_category(exc) if code == 429 else "api_error"
            details = record_api_error(exc, category, purpose)

            if category == "daily_quota":
                raise GroqRequestError(
                    "Groq's daily free-plan limit appears to be exhausted for this model. "
                    "Check your Groq rate-limits dashboard and try again after the limit resets.",
                    category=category,
                    code=code,
                    details=details,
                ) from exc

            transient = code in {408, 409, 429, 500, 502, 503, 504}
            if transient:
                hint = retry_after_seconds(exc)
                backoff = min(2.0 ** attempt, 30.0) + random.uniform(0.0, 1.0)
                delay = max(backoff, hint or 0.0)

                # Avoid an immediate resubmission storm after a final 429.
                if attempt == MAX_API_ATTEMPTS and code == 429 and hint is None:
                    delay = max(delay, 60.0)

                gate.cooldown(delay)

                if attempt < MAX_API_ATTEMPTS and delay <= wait_budget:
                    st.caption(
                        f"Groq returned a temporary error ({code}). "
                        f"Retrying {purpose}: attempt {attempt + 1}/{MAX_API_ATTEMPTS}."
                    )
                    continue

                if code == 429:
                    message = (
                        "Groq is temporarily rate-limiting requests (HTTP 429). "
                        f"Local retries are paused for at least {math.ceil(delay)} seconds. "
                        "Please submit the question again after the cooldown."
                    )
                else:
                    message = (
                        f"Groq returned a temporary service error ({code}). "
                        "The automatic retry limit or wait budget was reached. "
                        "Please submit the question again."
                    )

                raise GroqRequestError(
                    message,
                    category=category,
                    code=code,
                    details=details,
                ) from exc

            if code == 401:
                message = "Groq authentication failed. Check GROQ_API_KEY in your .env file."
            elif code == 403:
                message = (
                    "Groq denied access. Check your API key permissions and model access."
                )
            elif code == 404:
                message = (
                    f"Groq could not access model '{MODEL}'. Check GROQ_MODEL and "
                    "confirm that the model is available to your account."
                )
            elif code == 400:
                message = (
                    "Groq rejected the request (HTTP 400). Check the configured model "
                    "and request settings. Private logs contain the detailed error."
                )
            else:
                message = (
                    f"Groq rejected the request (HTTP {code}). "
                    "Check the private server logs for the detailed error."
                )

            raise GroqRequestError(
                message,
                category=category,
                code=code,
                details=details,
            ) from exc

        except groq.APITimeoutError as exc:
            details = record_api_error(exc, "timeout", purpose)
            raise GroqRequestError(
                "The connection to Groq timed out. Check the server's network "
                "connection and submit the question again.",
                category="timeout",
                details=details,
            ) from exc

        except groq.APIConnectionError as exc:
            # A lost connection can have an unknown server outcome. Do not
            # automatically duplicate that request; let the student choose.
            details = record_api_error(exc, "network_error", purpose)
            raise GroqRequestError(
                "The connection to Groq failed. Check the server's network "
                "connection before submitting again. No automatic network retry was made.",
                category="network_error",
                details=details,
            ) from exc

    raise RuntimeError("Unreachable retry state")


def show_api_status(placeholder) -> None:
    with placeholder.container():
        st.subheader("API Status")
        st.caption(f"Minimum spacing between API calls: {MIN_REQUEST_INTERVAL:g} seconds")
        st.caption(f"Maximum attempts per API operation: {MAX_API_ATTEMPTS}")
        st.caption("Five tutor replies also require one summary request, excluding retries.")
        last = st.session_state.get("last_api_error")
        if last:
            st.warning(f"Last API issue: {last['category']} ({last['purpose']}).")
            st.caption("Private server logs contain the detailed error.")
            if DEBUG_GROQ:
                with st.expander("Developer diagnostics — disable on public apps"):
                    st.json(last)