"""Local rolling conversation memory for PyLumo.

Every function here takes its inputs explicitly and returns its result, with no
Streamlit dependency, so the logic can be unit-tested without a running app.
The Streamlit glue (session state, sidebar) lives in ``pylumo/session.py``.

How it works: after every ``SUMMARY_EVERY_TURNS`` exchanges, the previous
summary plus the new exchanges are condensed into one checkpoint file on disk.
The tutor then sees only that summary, the unsummarized messages, and the
current question, so the prompt stays small however long the chat grows.
"""

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

from .config import (
    MEMORY_DIR, SUMMARY_EVERY_TURNS, SUMMARY_MAX_CHARS, SUMMARY_MAX_TOKENS,
)
from .groq_client import GroqRequestError, generate_with_retry
from .prompts import SUMMARY_PROMPT
from .utils import convert_history

SCHEMA_VERSION = 1
BATCH_SIZE = SUMMARY_EVERY_TURNS * 2  # messages per summary batch (user + tutor)
_CONVERSATION_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")


class MemoryUpdateError(RuntimeError):
    """The compact context could not be safely loaded or updated."""


class Checkpoint(NamedTuple):
    """The remembered summary and how many messages it already covers."""

    summary: str = ""
    covered: int = 0


def _check_conversation_id(conversation_id: str) -> str:
    """Reject ids that could escape the memory folder (e.g. '../x')."""
    if not isinstance(conversation_id, str) or not _CONVERSATION_ID.fullmatch(
        conversation_id
    ):
        raise ValueError(f"Invalid conversation id: {conversation_id!r}")
    return conversation_id


def memory_file(conversation_id: str) -> Path:
    """Every conversation has its own checkpoint; there is no global file."""
    _check_conversation_id(conversation_id)
    return MEMORY_DIR / conversation_id / "latest_summary.json"


def load_memory(
    conversation_id: str, total_messages: int, known_covered: int = 0
) -> Checkpoint:
    """Read and validate this conversation's checkpoint from disk.

    ``known_covered`` is what the caller already believes is summarized. A
    missing file is only acceptable when that is 0; otherwise the app would
    silently resend the whole chat.
    """
    try:
        data = json.loads(memory_file(conversation_id).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        if known_covered:
            raise MemoryUpdateError(
                "The saved memory file is missing. Restore it or clear the "
                "conversation; the app will not silently resend the full chat."
            ) from exc
        return Checkpoint()
    except (OSError, ValueError) as exc:
        raise MemoryUpdateError(
            "The memory file could not be read. Check its permissions/content "
            "or clear this conversation."
        ) from exc

    if not isinstance(data, dict):
        raise MemoryUpdateError("The memory file must contain a JSON object.")
    summary = data.get("summary")
    covered = data.get("summarized_messages")
    if (
        data.get("schema_version") != SCHEMA_VERSION
        or data.get("conversation_id") != conversation_id
        or not isinstance(summary, str)
        or not summary.strip()
        or len(summary) > SUMMARY_MAX_CHARS
        or type(covered) is not int
        or covered <= 0
        or covered % BATCH_SIZE != 0
        or covered > total_messages
        or covered < known_covered
    ):
        raise MemoryUpdateError(
            "The saved memory does not match this conversation. "
            "Restore the correct file or clear the conversation."
        )
    return Checkpoint(summary, covered)


def save_memory(conversation_id: str, summary: str, covered_messages: int) -> None:
    """Replace the checkpoint atomically (write temp file, fsync, rename)."""
    path = memory_file(conversation_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "conversation_id": conversation_id,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "summarized_messages": covered_messages,
        "summarized_turns": covered_messages // 2,
        "summary": summary,
    }
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=".summary-", suffix=".tmp", delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def delete_memory(conversation_id: str) -> bool:
    """Delete the checkpoint. Returns False if the file could not be removed."""
    path = memory_file(conversation_id)
    try:
        path.unlink(missing_ok=True)
    except OSError:
        return False
    try:
        path.parent.rmdir()  # Remove the folder only if it is empty.
    except OSError:
        pass
    return True


def summarize_conversation(previous_summary: str, batch: list[dict]) -> str:
    """One extra Groq request: previous memory plus the new exchanges."""
    if len(batch) != BATCH_SIZE:
        raise ValueError(
            f"A summary batch must contain exactly {SUMMARY_EVERY_TURNS} exchanges."
        )
    for index, message in enumerate(batch):
        expected_role = "user" if index % 2 == 0 else "assistant"
        if (
            message.get("role") != expected_role
            or not isinstance(message.get("content"), str)
            or not message["content"].strip()
        ):
            raise ValueError("Only complete, alternating exchanges are summarized.")

    response = generate_with_retry(
        purpose="summary",
        messages=[
            {"role": "system", "content": SUMMARY_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {"previous_summary": previous_summary, "new_exchanges": batch},
                    ensure_ascii=False,
                ),
            },
        ],
        temperature=0.1,
        max_tokens=SUMMARY_MAX_TOKENS,
    )

    # Do not overwrite good memory with an empty or truncated summary.
    if not response.choices:
        raise ValueError("Groq returned no summary choice.")

    choice = response.choices[0]
    if choice.finish_reason != "stop":
        raise ValueError(
            f"Groq did not finish the summary normally ({choice.finish_reason})."
        )

    summary = (choice.message.content or "").strip()
    if not summary or len(summary) > SUMMARY_MAX_CHARS:
        raise ValueError("Groq returned an empty or oversized summary.")
    return summary


def update_memory_if_needed(
    conversation_id: str, messages: list[dict], known_covered: int = 0
) -> Checkpoint:
    """Summarize every full batch of new exchanges and return the checkpoint.

    All raw messages are kept for display; only the checkpoint advances, and
    only after it has been safely written to disk.
    """
    try:
        summary, covered = load_memory(conversation_id, len(messages), known_covered)
        while len(messages) - covered >= BATCH_SIZE:
            # Never pass the entire archive to the summarizer.
            batch = messages[covered:covered + BATCH_SIZE]
            new_summary = summarize_conversation(summary, batch)
            new_covered = covered + BATCH_SIZE
            save_memory(conversation_id, new_summary, new_covered)
            summary, covered = new_summary, new_covered
        return Checkpoint(summary, covered)
    except MemoryUpdateError:
        raise
    except GroqRequestError as exc:
        raise MemoryUpdateError(
            f"Memory update failed: {exc} The previous summary and completed "
            "exchanges are unchanged. The app will retry the pending summary "
            "before sending a new tutor request."
        ) from exc
    except Exception as exc:
        raise MemoryUpdateError(
            f"Memory update failed ({type(exc).__name__}). The last saved "
            "summary and unsummarized exchanges are unchanged. It will retry "
            "on your next question. Check the Groq quota/network and that "
            "the app can write to its memory folder."
        ) from exc


def build_tutor_messages(messages: list[dict], checkpoint: Checkpoint) -> list[dict]:
    """Use only the latest summary + unsummarized exchanges + current question."""
    summary, covered = checkpoint
    recent_messages = messages[covered:]

    # A request has up to four completed pairs plus the next student question.
    if len(recent_messages) > BATCH_SIZE - 1:
        raise MemoryUpdateError("Update the pending summary before continuing.")
    if not recent_messages or recent_messages[-1]["role"] != "user":
        raise ValueError("A tutor request must end with a student question.")

    chat_messages = convert_history(recent_messages)

    if summary:
        # Keep remembered text at user-data priority, not in system instructions,
        # so a poisoned summary cannot override the tutor's rules.
        memory_json = json.dumps({"conversation_memory": summary}, ensure_ascii=False)
        first_content = chat_messages[0]["content"]
        chat_messages[0] = {
            "role": "user",
            "content": f"{memory_json}\n\n{first_content}",
        }

    return chat_messages