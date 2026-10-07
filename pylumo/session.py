"""Streamlit session-state glue for PyLumo.

Everything that touches ``st.session_state`` lives here, so ``memory.py``
stays pure and testable.
"""

from uuid import uuid4

import streamlit as st

from .config import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, SUMMARY_EVERY_TURNS
from .memory import Checkpoint, delete_memory, update_memory_if_needed


def init_session_state() -> None:
    """Create any missing session keys; existing values are left untouched."""
    defaults = {
        "messages": [],
        # Scoped to this Streamlit session, never shared across students.
        "conversation_id": uuid4().hex,
        "conversation_summary": "",
        "summarized_messages": 0,
        "memory_warning": None,
        "temperature": DEFAULT_TEMPERATURE,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "notice": None,
        "exited": False,
        "pending_prompt": None,
        "last_failed_prompt": None,
        "last_api_error": None,
        "learning_level": "Beginner",
        "active_mode": "Ask anything",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def refresh_memory(messages: list[dict]) -> Checkpoint:
    """Summarize any pending batch, then mirror the checkpoint into the session.

    The checkpoint on disk is authoritative, so the session copy is always
    overwritten with what was actually saved.
    """
    checkpoint = update_memory_if_needed(
        st.session_state.conversation_id,
        messages,
        st.session_state.summarized_messages,
    )
    st.session_state.conversation_summary = checkpoint.summary
    st.session_state.summarized_messages = checkpoint.covered
    st.session_state.memory_warning = None
    return checkpoint


def reset_conversation() -> None:
    """Clear the visible chat and the active memory together."""
    deleted = delete_memory(st.session_state.conversation_id)

    st.session_state.messages = []
    st.session_state.conversation_id = uuid4().hex
    st.session_state.conversation_summary = ""
    st.session_state.summarized_messages = 0
    st.session_state.memory_warning = None
    st.session_state.last_failed_prompt = None
    # Clearing the conversation does NOT reset the shared API cooldown.
    st.session_state.pending_prompt = None
    st.session_state.exited = False
    st.session_state.notice = (
        "✨ Conversation and memory cleared. Starting fresh!"
        if deleted
        else "Conversation and active memory cleared. The previous memory file "
        "could not be deleted from disk; remove it manually."
    )


def show_memory_status(placeholder) -> None:
    """Refresh the sidebar immediately, without resending a chat request."""
    covered = st.session_state.summarized_messages
    pending = (len(st.session_state.messages) - covered) // 2
    with placeholder.container():
        st.subheader("Conversation Memory")
        st.caption(f"Summarized: {covered // 2} exchanges")
        st.caption(f"New exchanges: {pending}/{SUMMARY_EVERY_TURNS}")
        with st.expander("View latest summary"):
            st.text(st.session_state.conversation_summary or "No summary yet.")
            st.caption(
                f"File: memory/{st.session_state.conversation_id}/latest_summary.json"
            )
        if st.session_state.memory_warning:
            st.warning(st.session_state.memory_warning)