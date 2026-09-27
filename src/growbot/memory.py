"""Conversational memory for scheme resolution, and nothing else.

    from growbot.memory import resolve_scheme

## What this is for

The bot answers a follow-up like "What about its exit load?" badly without
memory, and measurably so. Asking it cold, with no history:

| question (asked cold)               | result                              |
|-------------------------------------|-------------------------------------|
| "What about its exit load?"         | answered, citing a **general** page |
| "And the benchmark?"                | refused, `weak_retrieval`           |
| "What is the minimum SIP?"          | refused, `weak_retrieval`           |
| "Is that risky?"                    | refused, `weak_retrieval`           |

The first row is the one that matters, and it is a bug rather than a gap. A
question that names no scheme resolves to a `general` AMFI explainer chunk, and
that chunk's URL is then attached to the answer as *the source* - so the bot
cites a page about loads in general as the source for a fact about one
particular fund. The number may well be right; the citation is not, and a
reader has no way to tell.

## What memory is allowed to do

Resolve **which scheme an underspecified question is about**. That is all.

It deliberately does *not*:

- reach the generator. History never enters the prompt, so the model cannot be
  talked into answering about the previous fund, and the grounding check still
  sees only chunks retrieved for the current question;
- be embedded into the query as a blob of past text. Ten turns of text
  swamped by history would drag the query away from the current question, and
  "context window 10" would then mean "retrieval gets worse the more you talk";
- survive a question that names its own scheme. Memory only ever fills a gap
  that `detect_scheme` left empty.

So the effect is narrow and checkable: a follow-up resolves to the same fund the
conversation was already about, its chunks get the metadata filter they were
missing, and the citation comes from that fund's own chunks by construction.

## The rules

1. **The current question always wins.** If it names a scheme, history is not
   consulted at all.
2. **Only single-scheme turns donate.** A turn naming two funds is ambiguous,
   and a turn that resolved to `general` never identified a fund.
3. **Most recent wins.** Scanning goes backwards and stops at the first
   donor.
4. **Refused turns can donate.** "Should I buy HDFC Small Cap?" is refused, but
   it established that the conversation is about Small Cap. The refusal was
   about the *kind* of question, not the fund.
5. **PII is never remembered.** Every history entry is screened through
   `detect_pii` and dropped if it trips. This is enforced here, in the layer
   that stores it, rather than left to callers - `implementation.md` lists
   "store chat logs that include PII" under what not to do, and a caller who
   forgets to filter should not be able to breach it.
6. **The window is capped.** Only the most recent `config.MEMORY_TURNS`
   entries are considered. Not for privacy - the entries are already in the
   caller's memory - but because an unbounded window makes behaviour depend on
   how long a tab has been open, which is not a property worth having.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from growbot.config import GENERAL_SCHEME, MEMORY_TURNS, detect_scheme
from growbot.guards.intent import detect_pii

log = logging.getLogger("growbot.memory")

__all__ = [
    "screen",
    "resolve_scheme",
    "memory_query",
    "describe",
]


def screen(history: Sequence[str] | None) -> list[str]:
    """Drop PII-bearing entries, then keep only the most recent window.

    Returns prior question texts, oldest first. The cap is applied *after*
    screening so that the window is a window over safe entries rather than over
    raw ones - otherwise a burst of PII could push older, harmless context out
    of range and quietly change an answer for an unrelated reason.
    """
    if not history:
        return []
    clean: list[str] = []
    dropped = 0
    for entry in history:
        text = (entry or "").strip()
        if not text:
            continue
        if detect_pii(text):
            # Category and a count, never the message. Same rule as the guard.
            dropped += 1
            continue
        clean.append(text)
    if dropped:
        log.info("dropped %d history entr(ies) containing PII", dropped)
    # `clean[-MEMORY_TURNS:]` would be wrong for a window of 0: it returns the
    # whole list, which is the exact opposite of "remember nothing". Offsetting
    # from the length instead is correct for every non-negative window, so
    # MEMORY_TURNS=0 is an honest way to switch memory off.
    window = clean[len(clean) - MEMORY_TURNS:]
    if len(clean) > len(window):
        log.info(
            "history window %d -> %d entr(ies); the oldest are not consulted",
            len(clean),
            len(window),
        )
    return window


def resolve_scheme(
    question: str, history: Sequence[str] | None = None
) -> tuple[list[str], str]:
    """Which scheme is this question about, and where that came from.

    Returns ``(schemes, source)`` where `source` is ``"question"`` (the
    question named it), ``"memory"`` (inherited from a prior turn) or ``""``
    (neither - the question is genuinely about no single scheme).

    The source is returned rather than inferred by the caller because the two
    cases must be treated differently downstream and guessing is how a
    follow-up ends up citing the wrong fund.
    """
    own = detect_scheme(question)
    if own:
        # Rule 1: the question wins outright, and history is not consulted.
        return own, "question"

    # Rule 2: only a turn that resolved to exactly one real fund can donate.
    for entry in reversed(screen(history)):
        donor = detect_scheme(entry)
        if len(donor) == 1 and donor[0] != GENERAL_SCHEME:
            log.info(
                "inherited scheme %s from an earlier turn (question named none)",
                donor[0],
            )
            return donor, "memory"
    return [], ""


def memory_query(question: str, schemes: list[str], source: str) -> str:
    """The text actually embedded for retrieval.

    Identical to `question` unless the scheme came from memory, in which case
    the scheme name is appended so the query expresses which fund is meant.

    This is `search(append_scheme=True)` applied *only* to inherited questions.
    The general case stays off, because that was measured and rejected - it
    made the query match chunk prefixes rather than chunk content. Here the
    situation is different: the question says no fund at all, so without this
    the embedding has nothing fund-specific to match, and the "general"
    explainer wins by default. So the trade-off only applies where it helps.
    """
    if source == "memory" and len(schemes) == 1:
        return f"{question} {schemes[0]}"
    return question


def describe(question: str, history: Sequence[str] | None = None) -> str:
    """One line for the trace, so a demo can show memory working."""
    schemes, source = resolve_scheme(question, history)
    if source == "question":
        return "named in the question"
    if source == "memory":
        return f"inherited {schemes[0]} from an earlier turn"
    return "no scheme named or inherited"
