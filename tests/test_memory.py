"""Tests for pylumo.memory: pure logic, no Streamlit or network needed."""

import json

import pytest

from pylumo.memory import (
    BATCH_SIZE,
    Checkpoint,
    MemoryUpdateError,
    build_tutor_messages,
    delete_memory,
    load_memory,
    memory_file,
    save_memory,
    update_memory_if_needed,
)

CONVERSATION_ID = "test-conversation"


@pytest.fixture(autouse=True)
def isolated_memory_dir(tmp_path, monkeypatch):
    """Every test writes into its own temporary folder."""
    monkeypatch.setattr("pylumo.memory.MEMORY_DIR", tmp_path)
    return tmp_path


def make_messages(count: int) -> list[dict]:
    """Alternating user/assistant messages, starting with the user."""
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"message {i}"}
        for i in range(count)
    ]


def test_save_and_load_memory():
    save_memory(CONVERSATION_ID, "The user learned Python variables.", BATCH_SIZE)

    checkpoint = load_memory(CONVERSATION_ID, total_messages=BATCH_SIZE)

    assert checkpoint == Checkpoint("The user learned Python variables.", BATCH_SIZE)


def test_load_missing_memory_returns_empty():
    assert load_memory("does-not-exist", total_messages=0) == Checkpoint("", 0)


def test_missing_file_is_an_error_when_memory_was_already_saved():
    with pytest.raises(MemoryUpdateError):
        load_memory("does-not-exist", total_messages=BATCH_SIZE, known_covered=BATCH_SIZE)


def test_delete_memory_removes_file_and_folder():
    save_memory(CONVERSATION_ID, "Test summary", BATCH_SIZE)
    path = memory_file(CONVERSATION_ID)
    assert path.exists()

    assert delete_memory(CONVERSATION_ID) is True

    assert not path.exists()
    assert not path.parent.exists()


def test_corrupt_memory_file_is_reported():
    save_memory(CONVERSATION_ID, "Test summary", BATCH_SIZE)
    memory_file(CONVERSATION_ID).write_text("{not json", encoding="utf-8")

    with pytest.raises(MemoryUpdateError):
        load_memory(CONVERSATION_ID, total_messages=BATCH_SIZE)


def test_memory_belonging_to_another_conversation_is_rejected():
    save_memory(CONVERSATION_ID, "Test summary", BATCH_SIZE)
    path = memory_file(CONVERSATION_ID)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["conversation_id"] = "someone-else"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(MemoryUpdateError):
        load_memory(CONVERSATION_ID, total_messages=BATCH_SIZE)


@pytest.mark.parametrize("bad_id", ["../evil", "a/b", "", "has space"])
def test_unsafe_conversation_ids_are_rejected(bad_id):
    with pytest.raises(ValueError):
        memory_file(bad_id)


def test_update_does_nothing_below_the_batch_size(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("summarizer must not be called")

    monkeypatch.setattr("pylumo.memory.summarize_conversation", fail)

    checkpoint = update_memory_if_needed(CONVERSATION_ID, make_messages(BATCH_SIZE - 2))

    assert checkpoint == Checkpoint("", 0)


def test_update_summarizes_a_full_batch_and_persists_it(monkeypatch):
    monkeypatch.setattr(
        "pylumo.memory.summarize_conversation", lambda previous, batch: "New summary"
    )

    checkpoint = update_memory_if_needed(CONVERSATION_ID, make_messages(BATCH_SIZE))

    assert checkpoint == Checkpoint("New summary", BATCH_SIZE)
    assert load_memory(CONVERSATION_ID, BATCH_SIZE) == checkpoint


def test_failed_summary_keeps_the_previous_memory(monkeypatch):
    save_memory(CONVERSATION_ID, "Old summary", BATCH_SIZE)

    def boom(previous, batch):
        raise ValueError("Groq returned an empty or oversized summary.")

    monkeypatch.setattr("pylumo.memory.summarize_conversation", boom)

    with pytest.raises(MemoryUpdateError):
        update_memory_if_needed(
            CONVERSATION_ID, make_messages(BATCH_SIZE * 2), known_covered=BATCH_SIZE
        )

    assert load_memory(CONVERSATION_ID, BATCH_SIZE * 2) == Checkpoint("Old summary", BATCH_SIZE)


def test_tutor_messages_carry_the_summary_in_the_first_user_message():
    messages = make_messages(BATCH_SIZE) + [{"role": "user", "content": "What is a list?"}]

    result = build_tutor_messages(messages, Checkpoint("Learned variables.", BATCH_SIZE))

    assert result[0]["role"] == "user"
    assert "conversation_memory" in result[0]["content"]
    assert "Learned variables." in result[0]["content"]
    assert "What is a list?" in result[-1]["content"]


def test_tutor_request_must_end_with_a_student_question():
    with pytest.raises(ValueError):
        build_tutor_messages(make_messages(2), Checkpoint())


def test_unsummarized_backlog_blocks_a_new_request():
    with pytest.raises(MemoryUpdateError):
        build_tutor_messages(make_messages(BATCH_SIZE), Checkpoint())