# 🐍 PyLumo: an AI-Powered Python Tutor

PyLumo is a Streamlit tutoring app, powered by Groq, that teaches Python at the
learner's level. It was built as a hands-on project in **prompt and context
engineering**: how to structure instructions, memory, and per-turn context so a
language model stays focused, consistent, and safe over a long conversation.

## Features

- **Guided learning modes:** Learn a Topic, Explain My Code, Debug My Code, Quiz Me, Practice Challenge.
- **Level-aware answers:** Beginner, Intermediate, or Advanced.
- **Rolling conversation memory:** long chats stay cheap and coherent (see below).
- **Resilient API handling:** retries, a visible API status panel, and failed questions kept for a manual retry.
- **Slash commands:** `/clear` resets the chat and its memory, `/exit` ends the session.

## Prompt and context engineering in this project

| Technique | Where | Why |
|---|---|---|
| Separate system prompt and memory instruction | `pylumo/prompts.py` | Tutor behaviour is defined once; memory handling is a distinct, explicit instruction. |
| Per-turn learner context | `ask_tutor()` in `app.py` | The chosen level and mode are attached to each question, so answers adapt without bloating the system prompt. |
| Rolling summarization | `pylumo/memory.py` | Every 5 exchanges the previous summary plus the new exchanges are condensed. The model sees summary + recent messages, never the full archive. |
| Memory kept at user-data priority | `build_tutor_messages()` | Remembered text is JSON-wrapped and sent as user content, not system instructions, so a poisoned summary cannot override the tutor's rules. |
| Low-temperature summarizer | `summarize_conversation()` | Summaries use `temperature=0.1` for consistency; the tutor's temperature is user-adjustable. |
| Output validation | `summarize_conversation()` | A summary is rejected if it is empty, oversized, or cut off (`finish_reason != "stop"`), so good memory is never overwritten by a bad one. |
| Templated task prompts | `GUIDED_PROMPTS` in `app.py` | Each learning mode is a structured prompt with explicit steps (explain, example, practice, hint, wait for the learner). |

## Architecture

```
app.py                 Streamlit UI and request orchestration
pylumo/
  config.py            Settings loaded from the environment
  prompts.py           System, memory, and summary prompts
  groq_client.py       Groq calls with retry/backoff and API status
  memory.py            Pure memory logic: load, save, summarize (no Streamlit)
  session.py           Streamlit session-state glue
  utils.py             Helpers
tests/                 pytest suite
```

`memory.py` deliberately has no Streamlit dependency, which keeps it fully
unit-testable. Checkpoints are written atomically (temp file, fsync, rename) and
validated on every load, and conversation ids are sanitized before being used as
folder names.

## Getting started

```bash
git clone <your-repo-url>
cd <your-repo-folder>
python -m venv .venv
.venv\Scripts\activate          # Windows; use `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt

cp .env.example .env            # Windows: copy .env.example .env
# then edit .env and add your Groq API key

streamlit run app.py
```

Get a free API key at [console.groq.com](https://console.groq.com).

## Running the tests

```bash
pytest
```

## Notes

- PyLumo reviews code but does **not** execute it.
- Conversation memory is stored locally under `memory/` (git-ignored) and is scoped per session.