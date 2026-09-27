"""Phase 9 - the chat page.

    streamlit run src/growbot/ui/app.py

A single page, no auth, and no PII fields anywhere. The whole UI is a
presentation layer: it renders an `AnswerPayload` and nothing else. It never
imports Chroma, the embedding model, or the LLM client, and it holds no
retrieval logic of its own. Every answer comes from `growbot.ask.ask()`, which
is the boundary the architecture draws in §9.

Three things this page is careful about, because each one has bitten a
prototype at least once:

**The disclaimer is shown twice, deliberately.** Once as a persistent banner
that is visible before anyone has asked anything, and once inside every card,
because that text comes from the payload. A disclaimer that only appears after
a user has already read an answer is not much of a disclaimer.

**Example 3 is a question the bot cannot answer.** "How to download a
capital-gains statement?" is the known corpus gap from PRD §12 Q7, so its
button renders a refusal card. That is intentional: it makes the refusal path
demonstrable on demand instead of only when someone types something
unanswerable, and it shows the bot declining rather than inventing.

**A first question is slow.** The sentence-transformer model loads on the first
retrieval. The spinner is honest about that rather than letting the page look
hung for a few seconds.
"""

from __future__ import annotations

import logging

import streamlit as st

# Quieten the libraries before Growbot's own modules import and configure
# logging. Done here rather than in config.py because a Streamlit app is a
# long-running process that logs on every rerun.
for _noisy in (
    "httpx", "httpcore", "urllib3", "chromadb", "sentence_transformers",
    "transformers", "huggingface_hub", "filelock", "onnxruntime", "tokenizers",
):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

from growbot.ask import ask_with_trace, status  # noqa: E402
from growbot.config import DISCLAIMER, MEMORY_TURNS, SCHEMES  # noqa: E402
from growbot.guards.intent import display_name  # noqa: E402

#: The three clickable examples from implementation.md:320-322. Kept verbatim
#: because the spec names them, including the third one that must refuse.
EXAMPLES = [
    "Expense ratio of HDFC Large Cap Fund (Direct Growth)?",
    "Lock-in for HDFC ELSS Tax Saver?",
    "How to download a capital-gains statement?",
]

WELCOME = (
    "Hi - I'm Growbot, a facts-only assistant for five HDFC Mutual Fund "
    "Direct Growth schemes. Ask me about expense ratio, exit load, minimum SIP, "
    "lock-in, benchmark or riskometer level, or click an example below."
)

#: PRD §9's long disclaimer, shown in the expander. The short one is the banner
#: because the long one is too heavy to sit above every answer.
LONG_DISCLAIMER = (
    "This assistant answers **facts only** from public scheme documents "
    "(expense ratio, exit load, SIP minimum, lock-in, riskometer, benchmark, "
    "and document download guides). It is **not** investment advice, a "
    "recommendation to buy or sell, or a substitute for the SID/KIM/factsheet. "
    "Mutual fund investments are subject to market risks. Read all "
    "scheme-related documents carefully."
)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _fact_card(payload) -> None:
    """A grounded answer: the text, one link, and the source date."""
    st.markdown(payload.text)
    if payload.scheme_name:
        st.caption(f"Scheme: {display_name(payload.scheme_name)}")
    if payload.source_url:
        st.markdown(f"[Source: {payload.source_url}]({payload.source_url})")
    if payload.last_updated:
        st.caption(f"Last updated from sources: {payload.last_updated}")


def _refusal_card(payload) -> None:
    """A refusal: polite, one educational link, the same disclaimer."""
    st.markdown(payload.text)
    if payload.source_url:
        label = (
            "Read more on AMFI's investor-education glossary"
            if payload.mode == "refuse"
            else "Source"
        )
        st.markdown(f"[{label}]({payload.source_url})")
    if payload.reason:
        st.caption(f"Declined: {payload.reason.replace('_', ' ')}")


def _answer(payload) -> None:
    if payload.mode == "fact":
        _fact_card(payload)
    else:
        _refusal_card(payload)


def _user_turns(messages: list[dict], limit: int = MEMORY_TURNS) -> list[str]:
    """The user questions from a thread, most recent last, capped to `limit`.

    Split out from `_history` because the interesting part is pure and the
    interesting part is what needs checking. It cannot be reached from outside
    a Streamlit script run - `st.session_state` does not exist between runs -
    so testing it through the session state would mean testing it through a
    whole conversation. As a function it is a three-line claim:

    - only *user* turns are collected. Assistant text is not a question, and
      feeding it back would let a previous answer's wording resolve a scheme;
    - the window is `MEMORY_TURNS`, trimmed here rather than only in
      `growbot.memory`, because the thread must stay complete for display - a
      chat the user can scroll back through should not lose turns just because
      memory only reads the recent ones.
    """
    # `get` rather than `[...]`: the thread is session state this function does
    # not own, and a malformed entry should cost a blank turn rather than
    # crash the chat. The role filter is what keeps answers out.
    turns = [t.get("text", "") for t in messages if t.get("role") == "user"]
    # Offset from the length, not `turns[-limit:]`: for a limit of 0 that form
    # returns the whole list, which is the opposite of "remember nothing".
    return turns[len(turns) - limit:]


def _history() -> list[str]:
    """Earlier user questions for `ask_with_trace`, oldest first.

    Read before the new question is appended, so the question being asked is
    not also offered to memory as a donor of its own scheme.

    The text is *not* filtered for PII here, and should not be: screening
    belongs in the layer that stores it (`growbot.memory.screen`), so that every
    caller gets it rather than only this one.
    """
    return _user_turns(st.session_state.messages)


def _respond(question: str) -> None:
    """Run one question through the pipeline and record it in the thread.

    Nothing is rendered here except the spinner. The thread is rendered from
    `messages` alone, so appending both turns and rerunning is the only way an
    answer reaches the screen. Rendering the assistant card inline as well
    would draw it twice - once now and once from the thread on the next run.
    """
    # Read the thread *before* appending, so the question being asked is not
    # also offered to memory as a donor of its own scheme.
    history = _history()
    st.session_state.messages.append({"role": "user", "text": question})

    first = not st.session_state.get("warmed", False)
    prefix = "Loading the retrieval model for the first time. " if first else ""
    with st.spinner(f"{prefix}Looking for the answer in the source documents..."):
        payload, trace = ask_with_trace(question, history=history)
    st.session_state["warmed"] = True

    st.session_state.messages.append(
        {"role": "assistant", "payload": payload, "trace": trace}
    )


def _draw_turn(turn: dict) -> None:
    """One entry of the thread: a question bubble or an answer card."""
    with st.chat_message(turn["role"]):
        if turn["role"] == "user":
            st.markdown(turn["text"])
            return
        _answer(turn["payload"])

        # Demo aid, not product behaviour. The spec's contract is ask();
        # ask_with_trace() runs the identical pipeline and additionally reports
        # which stage decided, so a guard refusal can be shown to have happened
        # without ever calling the model.
        trace = turn.get("trace")
        if st.session_state.get("show_trace", True) and trace and trace.stages:
            with st.expander("why this answer"):
                st.code(trace.render(), language="text")
                st.caption(
                    f"Model called: {'yes' if trace.model_called else 'no'} "
                    f"| stopped at: {trace.stopped_at}"
                )


def _preflight(info) -> None:
    """Say up front what is and isn't ready, before anyone clicks anything."""
    if info.ready:
        return
    with st.warning("Not fully set up", icon="\N{WARNING SIGN}\N{VARIATION SELECTOR-16}"):
        if not info.index_ready:
            st.markdown(
                "**No index found**, so there is nothing to retrieve from and "
                "every answer would be a guess. Build it first:\n\n"
                "```\npython -m growbot.ingest\n```\n\n"
                "Until then the chat below will refuse rather than invent facts."
            )
        if not info.llm_ready:
            st.markdown(
                "**No LLM key configured.** The guard and retrieval still work, "
                "so advice questions, PII and off-topic questions are still "
                "refused correctly - but a fact question has no answer. Set "
                "`LLM_API_KEY` and `LLM_MODEL` in `.env`."
            )


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Growbot - HDFC scheme facts",
        page_icon="\U0001f4c4",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    # `setdefault` is not dependable on the session_state proxy across
    # Streamlit versions, so the defaults are spelled out.
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "show_trace" not in st.session_state:
        st.session_state["show_trace"] = True
    if "warmed" not in st.session_state:
        st.session_state["warmed"] = False

    st.title("Growbot")
    # Persistent, above everything, before any question is asked. Named escapes
    # rather than \uXXXX: U+26A4 is a dingbat, not a valid Streamlit emoji.
    st.info(f"**{DISCLAIMER}**", icon="\N{INFORMATION SOURCE}\N{VARIATION SELECTOR-16}")

    with st.sidebar:
        st.session_state["show_trace"] = st.checkbox(
            "Show pipeline trace",
            value=bool(st.session_state.get("show_trace", True)),
            help="Demo aid: shows which stage answered or refused, and "
                 "whether the model was called at all. Does not change answers.",
        )

    # One readiness check per rerun, passed to whoever needs it.
    info = status()
    _preflight(info)
    st.write(WELCOME)

    st.markdown("**Try one of these:**")
    for index, example in enumerate(EXAMPLES):
        if st.button(example, key=f"example_{index}", use_container_width=True):
            st.session_state["pending"] = example

    st.caption(
        f"{info.record_count} indexed passages | "
        f"model: {info.provider} / {info.model}"
    )

    for turn in st.session_state.messages:
        _draw_turn(turn)

    # A button click queues the question and reruns; drain it here so the
    # example buttons and the typed input share one code path.
    pending = st.session_state.pop("pending", None)
    typed = st.chat_input("Ask about expense ratio, exit load, SIP, lock-in...")

    # Answer every queued question before rerunning. st.rerun() raises, so a
    # second question must not be left behind if both an example click and a
    # typed question land in the same run.
    queued = [q.strip() for q in (pending, typed) if q and q.strip()]
    if queued:
        for question in queued:
            _respond(question)
        st.rerun()

    with st.expander("Disclaimer and what I will not do"):
        st.markdown(LONG_DISCLAIMER)
        st.markdown(
            f"**Schemes covered:** {', '.join(display_name(s) for s in SCHEMES)}."
        )
    st.caption(DISCLAIMER)


# No `sys.exit()` here. Streamlit execs this file as `__main__`, so raising
# SystemExit at the end of the module unwinds the script runner instead of
# ending a process.
if __name__ == "__main__":
    main()
