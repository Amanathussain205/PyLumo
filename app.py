"""PyLumo: Streamlit UI and application orchestration."""

import streamlit as st
import streamlit.components.v1 as components

from pylumo.config import (
    API_KEY,
    MODEL,
    SUMMARY_EVERY_TURNS,
)
from pylumo.groq_client import (
    GroqRequestError,
    generate_with_retry,
    record_api_error,
    show_api_status,
)
from pylumo.memory import MemoryUpdateError, build_tutor_messages
from pylumo.prompts import MEMORY_CONTEXT_INSTRUCTION, SYSTEM_PROMPT
from pylumo.session import (
    init_session_state,
    refresh_memory,
    reset_conversation,
    show_memory_status,
)
from pylumo.utils import valid_api_key

USER_AVATAR = "🧑"
TUTOR_AVATAR = "🐍"


def render_copy_button(text: str, key: str) -> None:
    """Render a compact, polished copy control for a chat message."""
    import html
    import json

    safe_text = json.dumps(text)
    safe_key = html.escape(key, quote=True)
    components.html(
        f"""
        <div class=\"pylumo-copy-control\">
            <button id=\"copy-{safe_key}\" type=\"button\"
                aria-label=\"Copy message\" title=\"Copy message\">
                <svg viewBox=\"0 0 24 24\" aria-hidden=\"true\">
                    <rect x=\"8\" y=\"8\" width=\"11\" height=\"11\" rx=\"2\"></rect>
                    <path d=\"M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2\"></path>
                </svg>
            </button>
        </div>
        <script>
        const button = document.getElementById(\"copy-{safe_key}\");
        const textToCopy = {safe_text};
        button.addEventListener(\"click\", async () => {{
            try {{
                if (navigator.clipboard && window.isSecureContext) {{
                    await navigator.clipboard.writeText(textToCopy);
                }} else {{
                    const area = document.createElement(\"textarea\");
                    area.value = textToCopy;
                    area.style.position = \"fixed\";
                    area.style.opacity = \"0\";
                    document.body.appendChild(area);
                    area.focus();
                    area.select();
                    document.execCommand(\"copy\");
                    area.remove();
                }}
                button.classList.add(\"is-copied\");
                button.innerHTML = '<span aria-hidden=\"true\">✓</span>';
                setTimeout(() => {{
                    button.classList.remove(\"is-copied\");
                    button.innerHTML = '<svg viewBox=\"0 0 24 24\" aria-hidden=\"true\"><rect x=\"8\" y=\"8\" width=\"11\" height=\"11\" rx=\"2\"></rect><path d=\"M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2\"></path></svg>';
                }}, 1200);
            }} catch (error) {{
                button.classList.add(\"is-error\");
                button.innerHTML = '<span aria-hidden=\"true\">!</span>';
                setTimeout(() => {{
                    button.classList.remove(\"is-error\");
                    button.innerHTML = '<svg viewBox=\"0 0 24 24\" aria-hidden=\"true\"><rect x=\"8\" y=\"8\" width=\"11\" height=\"11\" rx=\"2\"></rect><path d=\"M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2\"></path></svg>';
                }}, 1200);
            }}
        }});
        </script>
        """,
        height=30,
        scrolling=False,
    )

def begin_edit(index: int) -> None:
    """Put one user message into inline edit mode."""
    st.session_state.editing_message_index = index
    st.session_state.edit_draft = st.session_state.messages[index]["content"]


def cancel_edit() -> None:
    """Leave inline edit mode without changing the conversation."""
    st.session_state.editing_message_index = None
    st.session_state.edit_draft = ""


def submit_edited_query(index: int, memory_status, api_status) -> None:
    """Generate a replacement answer for an edited user query."""
    new_prompt = st.session_state.get("edit_draft", "").strip()
    if not new_prompt:
        st.session_state.edit_error = "Your edited question cannot be empty."
        return

    original_messages = list(st.session_state.messages)
    edited_context = [dict(message) for message in original_messages[:index]]
    edited_context.append({"role": "user", "content": new_prompt})

    try:
        with st.spinner("PyLumo is updating the response..."):
            new_response = ask_tutor(
                messages=edited_context,
                temperature=st.session_state.temperature,
                max_tokens=st.session_state.max_tokens,
            )
    except MemoryUpdateError as exc:
        st.session_state.edit_error = str(exc)
        show_memory_status(memory_status)
        return
    except GroqRequestError as exc:
        st.session_state.edit_error = str(exc)
        show_api_status(api_status)
        return
    except Exception as exc:
        record_api_error(exc, "unexpected_error", "edited_tutor")
        st.session_state.edit_error = (
            "The edited question could not be completed. Please try again."
        )
        show_api_status(api_status)
        return

    # Replace the edited user turn and its corresponding tutor response.
    # Later turns are retained, so the user's conversation is not silently deleted.
    st.session_state.messages[index] = {"role": "user", "content": new_prompt}
    if index + 1 < len(st.session_state.messages) and st.session_state.messages[index + 1]["role"] == "assistant":
        st.session_state.messages[index + 1] = {
            "role": "assistant",
            "content": new_response,
        }
    else:
        st.session_state.messages.insert(
            index + 1, {"role": "assistant", "content": new_response}
        )

    st.session_state.response_feedback.pop(index + 1, None)
    st.session_state.editing_message_index = None
    st.session_state.edit_draft = ""
    st.session_state.edit_error = None
    st.session_state.last_failed_prompt = None
    show_memory_status(memory_status)
    show_api_status(api_status)
    st.rerun()


def record_response_feedback(message_index: int, value: str) -> None:
    """Store lightweight UI feedback for one tutor response."""
    st.session_state.response_feedback[message_index] = value


PAGE_STYLE = """
<style>
    /* =========================================================
       PyLumo accessible UI theme
       Explicit foreground/background pairs are used for controls so
       Streamlit's Light/Dark theme cannot make text disappear.
       ========================================================= */
    :root {
        --pylumo-bg: var(--background-color, #f7fafc);
        --pylumo-surface: var(--secondary-background-color, #ffffff);
        --pylumo-text: var(--text-color, #102a43);
        --pylumo-muted: #627d98;
        --pylumo-border: rgba(100, 116, 139, .24);
        --pylumo-primary: #0f766e;
        --pylumo-primary-hover: #115e59;
        --pylumo-button-text: #ffffff;
    }

    html, body,
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] > .main,
    [data-testid="stAppViewContainer"] section.main,
    [data-testid="stAppViewContainer"] .main .block-container {
        background-color: var(--pylumo-bg) !important;
        color: var(--pylumo-text) !important;
    }

    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] > .main,
    [data-testid="stAppViewContainer"] section.main {
        background-image: none !important;
    }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #102a43 0%, #163c57 100%) !important;
    }
    [data-testid="stSidebar"] * {
        color: #eef9ff !important;
    }
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] small {
        color: #cfe8f3 !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(238, 249, 255, .16) !important;
    }

    /* ---------- Model / Subject info boxes ---------- */
    /* Streamlit renders the value after Model/Subject as inline <code>.
       Target the actual caption container and every nested element so the
       value cannot inherit the sidebar background/text color. */
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] code,
    [data-testid="stSidebar"] .stCaption code,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] code span {
        display: inline-block !important;
        padding: .28rem .55rem !important;
        margin: .12rem 0 !important;
        border-radius: 7px !important;
        background: #24506a !important;
        border: 1px solid #5aa6c4 !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        font-size: .82rem !important;
        font-weight: 700 !important;
        opacity: 1 !important;
    }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] code *,
    [data-testid="stSidebar"] .stCaption code * {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        background: transparent !important;
    }

    /* ---------- Sidebar expanders ---------- */
    [data-testid="stSidebar"] [data-testid="stExpander"],
    [data-testid="stSidebar"] [data-testid="stExpanderToggle"] {
        background: #1b4058 !important;
        border: 1px solid #397895 !important;
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpanderToggle"] * {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        background: transparent !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpanderDetails"] {
        background: #17384f !important;
        color: #eef9ff !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpanderDetails"] * {
        color: #eef9ff !important;
    }

    /* ---------- Hero ---------- */
    .pylumo-hero {
        padding: 1.05rem 1.2rem;
        margin: .35rem 0 1.2rem;
        border-radius: 16px;
        background: linear-gradient(120deg, #0d9488, #2563eb);
        color: #ffffff !important;
        box-shadow: 0 12px 28px rgba(37, 99, 235, .18);
    }
    .pylumo-hero h3,
    .pylumo-hero p {
        color: #ffffff !important;
    }
    .pylumo-hero h3 { margin: 0 0 .25rem; }
    .pylumo-hero p { margin: 0; opacity: .96; }

    /* ---------- ALL Streamlit buttons ---------- */
    .stButton > button,
    [data-testid="baseButton-secondary"],
    [data-testid="baseButton-primary"] {
        min-height: 2.55rem !important;
        border-radius: 10px !important;
        border: 1px solid #0f766e !important;
        background: #0f766e !important;
        color: #ffffff !important;
        box-shadow: 0 2px 7px rgba(15, 118, 110, .14) !important;
        font-weight: 600 !important;
        transition: transform .15s ease, background-color .15s ease, box-shadow .15s ease;
    }
    .stButton > button * {
        color: #ffffff !important;
    }
    .stButton > button:hover,
    [data-testid="baseButton-secondary"]:hover,
    [data-testid="baseButton-primary"]:hover {
        background: #115e59 !important;
        border-color: #115e59 !important;
        color: #ffffff !important;
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(15, 118, 110, .22) !important;
    }
    .stButton > button:focus-visible {
        outline: 3px solid rgba(45, 212, 191, .55) !important;
        outline-offset: 2px;
    }
    .stButton > button:disabled {
        background: #64748b !important;
        border-color: #64748b !important;
        color: #ffffff !important;
        opacity: .72 !important;
    }

    /* ---------- Sidebar buttons: same guaranteed contrast ---------- */
    [data-testid="stSidebar"] .stButton > button {
        background: #1f9d8f !important;
        border-color: #49c7bb !important;
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] .stButton > button * {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: #2bb3a5 !important;
        border-color: #72ddd3 !important;
    }

    /* ---------- Radio / select / slider labels ---------- */
    [data-testid="stRadio"] label,
    [data-testid="stSlider"] label,
    [data-testid="stSelectbox"] label,
    [data-testid="stTextInput"] label {
        color: var(--pylumo-text) !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label,
    [data-testid="stSidebar"] [data-testid="stSlider"] label {
        color: #eef9ff !important;
    }
    [data-testid="stRadio"] [role="radio"] + div,
    [data-testid="stRadio"] [role="radio"] + div * {
        color: inherit !important;
    }

    /* ---------- Chat messages ---------- */
    [data-testid="stChatMessage"] {
        border-radius: 14px;
        border: 1px solid var(--pylumo-border);
        background: var(--pylumo-surface) !important;
        color: var(--pylumo-text) !important;
    }
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] strong,
    [data-testid="stChatMessage"] em {
        color: var(--pylumo-text);
    }

    /* ---------- Hover-only message actions ---------- */
    /* One compact action rail is revealed only while the cursor is inside the
       corresponding chat bubble. */
    [data-testid="stChatMessage"]:has([class*="st-key-pylumo-actions-"])
        [class*="st-key-pylumo-actions-"] {
        opacity: 0 !important;
        pointer-events: none !important;
        transform: translateY(2px);
        transition: opacity .16s ease, transform .16s ease !important;
    }
    [data-testid="stChatMessage"]:has([class*="st-key-pylumo-actions-"]):hover
        [class*="st-key-pylumo-actions-"],
    [data-testid="stChatMessage"]:has([class*="st-key-pylumo-actions-"]):focus-within
        [class*="st-key-pylumo-actions-"] {
        opacity: 1 !important;
        pointer-events: auto !important;
        transform: translateY(0);
    }

    /* Keep every action slot identical in height and vertically centered. */
    [class*="st-key-pylumo-actions-"] {
        min-height: 30px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    [class*="st-key-pylumo-actions-"] > div,
    [class*="st-key-pylumo-actions-"] [data-testid="stVerticalBlock"],
    [class*="st-key-pylumo-actions-"] [data-testid="element-container"] {
        width: 100% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }

    /* Native Streamlit action buttons: identical geometry, restrained styling,
       and no dashboard-wide button rules leaking into the action rail. */
    [class*="st-key-pylumo-actions-"] .stButton,
    [class*="st-key-pylumo-actions-"] .stButton > div {
        width: 100% !important;
    }
    [class*="st-key-pylumo-actions-"] .stButton > button {
        box-sizing: border-box !important;
        width: 30px !important;
        min-width: 30px !important;
        height: 30px !important;
        min-height: 30px !important;
        margin: 0 auto !important;
        padding: 0 !important;
        border-radius: 8px !important;
        border: 1px solid rgba(71, 113, 132, .72) !important;
        background: rgba(27, 64, 88, .94) !important;
        color: #eef9ff !important;
        box-shadow: 0 1px 3px rgba(15, 23, 42, .12) !important;
        font-size: 13px !important;
        line-height: 1 !important;
        font-weight: 500 !important;
        transform: none !important;
        transition: background-color .14s ease, border-color .14s ease,
                    box-shadow .14s ease, transform .14s ease !important;
    }
    [class*="st-key-pylumo-actions-"] .stButton > button:hover {
        background: #2b607b !important;
        border-color: #72b6ce !important;
        color: #ffffff !important;
        box-shadow: 0 2px 7px rgba(15, 23, 42, .18) !important;
        transform: translateY(-1px) !important;
    }
    [class*="st-key-pylumo-actions-"] .stButton > button:focus-visible {
        outline: 2px solid rgba(45, 212, 191, .55) !important;
        outline-offset: 2px !important;
    }
    [class*="st-key-pylumo-actions-"] .stButton > button * {
        color: #eef9ff !important;
        -webkit-text-fill-color: #eef9ff !important;
    }

    /* Copy is a tiny embedded control, so its iframe must behave exactly like
       the native action buttons rather than looking like a separate widget. */
    [class*="st-key-pylumo-actions-"] iframe {
        display: block !important;
        width: 30px !important;
        height: 30px !important;
        min-width: 30px !important;
        margin: 0 auto !important;
        padding: 0 !important;
        border: 0 !important;
        background: transparent !important;
        overflow: hidden !important;
        vertical-align: top !important;
    }
    .pylumo-copy-control {
        width: 30px;
        height: 30px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto;
    }
    .pylumo-copy-control button {
        box-sizing: border-box;
        width: 30px;
        height: 30px;
        min-width: 30px;
        min-height: 30px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        margin: 0;
        padding: 0;
        border: 1px solid rgba(71, 113, 132, .72);
        border-radius: 8px;
        background: rgba(27, 64, 88, .94);
        color: #eef9ff;
        cursor: pointer;
        box-shadow: 0 1px 3px rgba(15, 23, 42, .12);
        transition: background-color .14s ease, border-color .14s ease,
                    box-shadow .14s ease, transform .14s ease;
    }
    .pylumo-copy-control button:hover {
        background: #2b607b;
        border-color: #72b6ce;
        color: #ffffff;
        box-shadow: 0 2px 7px rgba(15, 23, 42, .18);
        transform: translateY(-1px);
    }
    .pylumo-copy-control button:focus-visible {
        outline: 2px solid rgba(45, 212, 191, .55);
        outline-offset: 2px;
    }
    .pylumo-copy-control button.is-copied {
        background: #0f766e;
        border-color: #2dd4bf;
        color: #ffffff;
    }
    .pylumo-copy-control button.is-error {
        background: #9f1239;
        border-color: #fb7185;
        color: #ffffff;
    }
    .pylumo-copy-control svg {
        width: 15px;
        height: 15px;
        fill: none;
        stroke: currentColor;
        stroke-width: 1.8;
        stroke-linecap: round;
        stroke-linejoin: round;
    }
    .pylumo-copy-control span {
        font-size: 14px;
        font-weight: 700;
        line-height: 1;
    }

    .pylumo-action-anchor {
        display: none !important;
    }
    .pylumo-edit-box {
        margin-top: .15rem;
    }
    .pylumo-edit-box [data-testid="stTextInput"] input {
        border-radius: 9px !important;
    }

    /* ---------- Chat input ---------- */
    [data-testid="stChatInput"] {
        background: transparent !important;
    }
    [data-testid="stChatInput"] > div {
        background: var(--pylumo-surface) !important;
        border: 1px solid var(--pylumo-border) !important;
        border-radius: 14px !important;
    }
    [data-testid="stChatInput"] textarea {
        color: var(--pylumo-text) !important;
        background: transparent !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #64748b !important;
        opacity: 1 !important;
    }

    /* ---------- Info / success / error / warning cards ---------- */
    [data-testid="stAlert"] {
        border-radius: 10px !important;
    }
    [data-testid="stAlert"] p,
    [data-testid="stAlert"] span {
        color: inherit !important;
    }

    /* ---------- Tips / expanders / metrics ---------- */
    .pylumo-tip {
        border: 1px solid var(--pylumo-border);
        border-left: 4px solid #0f766e;
        padding: .7rem .9rem;
        background: var(--pylumo-surface) !important;
        color: var(--pylumo-text) !important;
        border-radius: 10px;
    }
    .pylumo-tip strong,
    .pylumo-tip em {
        color: var(--pylumo-text) !important;
    }
    [data-testid="stExpander"] {
        border: 1px solid #397895 !important;
        border-radius: 10px !important;
        overflow: hidden !important;
        background: var(--pylumo-surface) !important;
    }

    /* Closed expander headers must visibly differ from the sidebar/dashboard. */
    [data-testid="stSidebar"] [data-testid="stExpander"] details > summary {
        background: #24506a !important;
        color: #ffffff !important;
        border-radius: 9px !important;
        min-height: 2.55rem !important;
        padding: .55rem .75rem !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] details > summary:hover {
        background: #2f647f !important;
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] details[open] > summary {
        background: #0f766e !important;
        color: #ffffff !important;
        border-radius: 9px 9px 0 0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] details > summary,
    [data-testid="stSidebar"] [data-testid="stExpander"] details > summary * {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }

    /* Expanded content keeps a readable, distinct surface. */
    [data-testid="stSidebar"] [data-testid="stExpander"] details > div {
        background: #163c57 !important;
        color: #eef9ff !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] details > div * {
        color: #eef9ff !important;
    }

    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary * {
        color: var(--pylumo-text) !important;
    }
    [data-testid="stMetric"] label,
    [data-testid="stMetricValue"],
    [data-testid="stMetricDelta"] {
        color: var(--pylumo-text) !important;
    }

    /* ---------- Code blocks ---------- */
    [data-testid="stCodeBlock"] {
        border-radius: 10px !important;
    }

    /* ---------- Responsive spacing ---------- */
    @media (max-width: 768px) {
        .main .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        .pylumo-hero {
            padding: .9rem 1rem;
        }
    }
</style>
<div class="pylumo-hero">
    <h3>Build Python confidence, one concept at a time.</h3>
    <p>Choose a learning mode in the sidebar, paste code for feedback, or ask your own question.</p>
</div>
"""

QUICK_START_TIP = """
<div class="pylumo-tip">
    <strong>Quick start:</strong> Select <em>Learn a Topic</em> for a guided lesson,
    <em>Debug My Code</em> to investigate an error, or <em>Quiz Me</em> to practise.
</div>
"""

QUICK_PROMPTS = {
    "Variables": "Explain Python variables to a complete beginner.",
    "Loops": "Teach me Python for loops with a simple example.",
    "Functions": "What is a Python function and why do we use it?",
    "Decorators": "Explain Python decorators with an example.",
    "GIL": "Explain Python's Global Interpreter Lock.",
    "Debugging": "Help me understand how to debug Python errors.",
}

GUIDED_PROMPTS = {
    "📚 Learn a Topic": (
        "Teach me Python [topic] at my current level. Start with a clear "
        "explanation, show a small example, then give me one practice question."
    ),
    "🔍 Explain My Code": (
        "Explain this Python code step by step. Tell me what each important "
        "part does and suggest one improvement if appropriate:\n\n"
        "```python\n# Paste your Python code here\n```"
    ),
    "🐞 Debug My Code": (
        "Help me debug this Python code. Identify the likely cause, show the "
        "smallest safe fix, explain why it works, and suggest how to avoid the "
        "error next time.\n\n```python\n# Paste code and traceback here\n```"
    ),
    "🧠 Quiz Me": (
        "Quiz me on Python [topic] at my current level. Ask one question at a "
        "time, wait for my answer, then explain the answer before continuing."
    ),
    "🛠️ Practice Challenge": (
        "Give me a small, practical Python coding challenge for my current level. "
        "State the requirements clearly, provide one hint, and wait for my solution."
    ),
}


# ============================================================
# Tutor request
# ============================================================


def ask_tutor(messages: list[dict], temperature: float, max_tokens: int) -> str:
    """Reply using compact memory rather than the full display history."""
    checkpoint = refresh_memory(messages)
    tutor_messages = [dict(m) for m in build_tutor_messages(messages, checkpoint)]

    # Per-turn context: the learner's level and mode ride along with the question.
    learning_context = (
        f"Learning level: {st.session_state.learning_level}. "
        f"Selected learning mode: {st.session_state.active_mode}."
    )
    tutor_messages[-1]["content"] = (
        f"[{learning_context}]\n\n{tutor_messages[-1]['content']}"
    )

    response = generate_with_retry(
        purpose="tutor",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT + MEMORY_CONTEXT_INSTRUCTION},
            *tutor_messages,
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    if not response.choices:
        raise ValueError("Groq returned no response choice.")

    response_text = (response.choices[0].message.content or "").strip()
    if not response_text:
        raise ValueError("Groq returned an empty response.")
    return response_text


def remove_failed_user_message() -> None:
    """Remove the most recent user message when the request fails."""
    messages = st.session_state.messages
    if messages and messages[-1]["role"] == "user":
        st.session_state.last_failed_prompt = messages.pop()["content"]


def handle_command(text: str) -> bool:
    """Handle PyLumo slash commands and return whether input was handled."""
    command = text.strip().lower()
    if command == "/clear":
        reset_conversation()
        return True
    if command == "/exit":
        st.session_state.exited = True
        st.session_state.notice = "👋 Study session ended."
        return True
    if command.startswith("/"):
        st.session_state.notice = (
            f"Unknown command: `{command}`. "
            "Available commands: `/clear` and `/exit`."
        )
        return True
    return False


# ============================================================
# Sidebar
# ============================================================


def render_sidebar():
    """Draw the sidebar; return the placeholders refreshed after each request."""
    with st.sidebar:
        st.header("Tutor Settings")

        st.session_state.temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state.temperature),
            step=0.1,
            help=(
                "Lower values generally produce more focused and consistent "
                "responses. Higher values allow greater variation."
            ),
        )
        st.session_state.max_tokens = st.slider(
            "Max response tokens",
            min_value=256,
            max_value=2048,
            value=int(st.session_state.max_tokens),
            step=256,
            help="Controls the maximum length of PyLumo's response.",
        )
        st.caption(f"Model: `{MODEL}`")
        st.caption("Subject: `Python`")

        st.divider()
        memory_status = st.empty()
        show_memory_status(memory_status)
        st.divider()
        api_status = st.empty()
        show_api_status(api_status)

        st.divider()
        st.subheader("Try a Topic")
        for label, example_prompt in QUICK_PROMPTS.items():
            if st.button(
                label,
                use_container_width=True,
                disabled=st.session_state.exited,
            ):
                st.session_state.pending_prompt = example_prompt

        st.divider()
        if st.button("Clear Conversation", use_container_width=True):
            reset_conversation()
            st.rerun()

        st.divider()
        st.header("Learning Studio ✨")
        st.caption("Set your level, choose a guided activity, and learn at your pace.")

        st.radio(
            "Your current level",
            options=["Beginner", "Intermediate", "Advanced"],
            key="learning_level",
            horizontal=True,
            help="PyLumo uses this preference to adjust detail, terminology, and examples.",
        )

        st.subheader("Choose a learning mode")
        for label, guided_prompt in GUIDED_PROMPTS.items():
            if st.button(label, use_container_width=True, key=f"learning_mode_{label}"):
                st.session_state.active_mode = label
                st.session_state.pending_prompt = guided_prompt
                st.rerun()

        st.divider()
        st.subheader("Your session")
        first_metric, second_metric = st.columns(2)
        first_metric.metric("Exchanges", len(st.session_state.messages) // 2)
        second_metric.metric("Level", st.session_state.learning_level)
        st.caption(f"Current mode: {st.session_state.active_mode}")

        with st.expander("How to get the best help"):
            st.markdown(
                "- Paste the **full traceback** when debugging.\n"
                "- Tell PyLumo what you already understand.\n"
                "- Ask for a smaller example or another explanation anytime.\n"
                "- PyLumo reviews code but does **not** execute it."
            )

    return memory_status, api_status


# ============================================================
# Chat
# ============================================================


def render_message_actions(role: str, index: int, text: str) -> None:
    """Render tiny hover-only actions for one chat message."""
    st.markdown('<span class="pylumo-action-anchor"></span>', unsafe_allow_html=True)
    if role == "user":
        action_count = 2
    else:
        action_count = 3

    widths = [0.78] + [0.0733] * action_count
    columns = st.columns(widths, vertical_alignment="center")
    with columns[0]:
        st.markdown('<div class="pylumo-message-actions"></div>', unsafe_allow_html=True)

    with columns[1]:
        with st.container(key=f"pylumo-actions-copy-{role}-{index}"):
            st.markdown('<div class="pylumo-message-actions">', unsafe_allow_html=True)
            render_copy_button(text, f"action-{role}-{index}")
            st.markdown('</div>', unsafe_allow_html=True)

    if role == "user":
        with columns[2]:
            with st.container(key=f"pylumo-actions-edit-{index}"):
                if st.button("✎", key=f"edit-message-{index}", help="Edit your question"):
                    begin_edit(index)
                    st.rerun()
    else:
        feedback = st.session_state.response_feedback.get(index)
        with columns[2]:
            with st.container(key=f"pylumo-actions-good-{index}"):
                if st.button("👍", key=f"good-response-{index}", help="Good response"):
                    record_response_feedback(index, "good")
                    st.rerun()
        with columns[3]:
            with st.container(key=f"pylumo-actions-bad-{index}"):
                if st.button("👎", key=f"bad-response-{index}", help="Bad response"):
                    record_response_feedback(index, "bad")
                    st.rerun()

        if feedback:
            label = "Thanks — marked as a good response." if feedback == "good" else "Thanks — marked as a bad response."
            st.caption(label)


def render_edit_mode(index: int, memory_status, api_status) -> None:
    """Render the inline editor; Enter or the return button submits the edit."""
    with st.form(key=f"edit-form-{index}", clear_on_submit=False):
        st.text_input(
            "Edit your question",
            key="edit_draft",
            label_visibility="collapsed",
        )
        edit_columns = st.columns([0.10, 0.10, 0.80])
        with edit_columns[0]:
            submitted = st.form_submit_button("↵", help="Send edited question")
        with edit_columns[1]:
            cancelled = st.form_submit_button("×", help="Cancel editing")

    if submitted:
        submit_edited_query(index, memory_status, api_status)
    elif cancelled:
        cancel_edit()
        st.rerun()

    if st.session_state.edit_error:
        st.error(st.session_state.edit_error)


def handle_prompt(prompt: str, memory_status, api_status) -> None:
    """Run one student question through the tutor and render the result."""
    if prompt.startswith("/") and handle_command(prompt):
        st.rerun()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)
        render_message_actions("user", len(st.session_state.messages) - 1, prompt)

    with st.chat_message("assistant", avatar=TUTOR_AVATAR):
        with st.spinner("PyLumo is thinking..."):
            try:
                response_text = ask_tutor(
                    messages=st.session_state.messages,
                    temperature=st.session_state.temperature,
                    max_tokens=st.session_state.max_tokens,
                )
            except MemoryUpdateError as exc:
                remove_failed_user_message()
                st.session_state.memory_warning = str(exc)
                st.error(
                    f"{exc} No new tutor request was sent. "
                    "Please submit your question again after resolving this."
                )
                show_memory_status(memory_status)
            except GroqRequestError as exc:
                remove_failed_user_message()
                st.error(str(exc))
                st.caption("Your question is retained below for a manual retry.")
            except Exception as exc:
                remove_failed_user_message()
                record_api_error(exc, "unexpected_error", "tutor")
                st.error(
                    "The request could not be completed. Check the private server "
                    "logs for the error; the failed question was retained for retry."
                )
            else:
                st.session_state.last_failed_prompt = None
                st.markdown(response_text)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response_text}
                )
                render_message_actions(
                    "assistant", len(st.session_state.messages) - 1, response_text
                )

                try:
                    pending = (
                        len(st.session_state.messages)
                        - st.session_state.summarized_messages
                    )
                    if pending >= SUMMARY_EVERY_TURNS * 2:
                        with st.spinner("Updating conversation memory..."):
                            refresh_memory(st.session_state.messages)
                except MemoryUpdateError as exc:
                    st.session_state.memory_warning = str(exc)

                show_memory_status(memory_status)

    show_api_status(api_status)


def main() -> None:
    st.set_page_config(page_title="PyLumo", page_icon="🐍", layout="centered")
    st.title("🐍 PyLumo")
    st.caption("Making Python click. Your focused Python tutor powered by Groq.")
    st.markdown(PAGE_STYLE, unsafe_allow_html=True)

    init_session_state()
    if "editing_message_index" not in st.session_state:
        st.session_state.editing_message_index = None
    if "edit_draft" not in st.session_state:
        st.session_state.edit_draft = ""
    if "edit_error" not in st.session_state:
        st.session_state.edit_error = None
    if "response_feedback" not in st.session_state:
        st.session_state.response_feedback = {}

    if not valid_api_key(API_KEY):
        st.error(
            "GROQ_API_KEY was not found. Add it to your `.env` file and restart the app."
        )
        st.code(
            "GROQ_API_KEY=your_api_key_here\nGROQ_MODEL=openai/gpt-oss-20b",
            language="bash",
        )
        st.stop()

    memory_status, api_status = render_sidebar()

    if st.session_state.notice:
        st.info(st.session_state.notice)
        st.session_state.notice = None

    if not st.session_state.messages:
        st.info(
            "Ask me anything about Python, from your first variable to advanced "
            "internals, frameworks, libraries, and language design."
        )
        st.markdown(QUICK_START_TIP, unsafe_allow_html=True)

    for index, message in enumerate(st.session_state.messages):
        avatar = USER_AVATAR if message["role"] == "user" else TUTOR_AVATAR
        with st.chat_message(message["role"], avatar=avatar):
            if (
                message["role"] == "user"
                and st.session_state.editing_message_index == index
            ):
                render_edit_mode(index, memory_status, api_status)
            else:
                st.markdown(message["content"])
                render_message_actions(message["role"], index, message["content"])

    if st.session_state.exited:
        st.success("Your PyLumo session has ended.")
        if st.button("Start a New Session"):
            reset_conversation()
            st.rerun()
        st.stop()

    typed_prompt = st.chat_input("Ask PyLumo anything about Python...")

    # Consume queued quick/retry input once. A typed question takes priority.
    queued_prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None
    prompt = (typed_prompt or queued_prompt or "").strip()

    if prompt:
        handle_prompt(prompt, memory_status, api_status)

    # Rendered AFTER the request so a newly failed question is shown immediately.
    # Retrying requires an explicit click; it is not an automatic retry loop.
    if st.session_state.last_failed_prompt:
        with st.expander("Last unanswered question"):
            st.text(st.session_state.last_failed_prompt)
        if st.button("Retry last question"):
            st.session_state.pending_prompt = st.session_state.last_failed_prompt
            st.rerun()


main()