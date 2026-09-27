"""Phase 7 - The generation prompt (architecture §6.4).

A prompt is a request, not a guarantee, so the rules that matter are enforced
in :mod:`growbot.generate.answer` after the model replies. This module states
the policy in the one place a reader will look for it, and the enforcement
layer cites these rules back to the same numbers.

The design decision worth knowing: the model is told to output **prose only**.
It is not asked for a source URL, a date, or a citation, because those come
from the chunk metadata (architecture §6.3, and `implementation.md:258` -
"source_url and date *must* come from assembled metadata, not the model"). A
model that is never asked for a URL cannot invent one, which removes a whole
class of failure instead of trying to detect it afterwards.
"""

from __future__ import annotations

import re

from growbot.config import DISCLAIMER, MAX_ANSWER_SENTENCES

__all__ = [
    "INSUFFICIENT",
    "SYSTEM_PROMPT",
    "build_messages",
    "build_user_message",
    "looks_insufficient",
]

#: Sentinel the model must emit when the context does not answer the question.
#: A word, not a JSON field, because a word needs no parser and cannot be
#: malformed. Detection is a substring check, so the model does not have to get
#: the casing or punctuation right.
INSUFFICIENT = "INSUFFICIENT_CONTEXT"

#: Rules are ordered by priority. The first two are the ones the post-checks in
#: answer.py enforce mechanically; the rest are the model's own discipline,
#: backed by the Phase 5 guard for the advice and returns cases.
SYSTEM_PROMPT = f"""\
You are Growbot. You report published facts about five HDFC Mutual Fund Direct \
Growth schemes, and you report nothing else.

Rules, in priority order:

1. Answer ONLY from the CONTEXT given below. That context is the entire \
universe of facts available to you. You have no other knowledge of these funds.
2. If the context does not contain the answer, reply with exactly \
{INSUFFICIENT} and nothing else. Do not guess. Do not reason from general \
knowledge of mutual funds. Do not fill a gap. "I don't have that" is a correct \
and expected answer.
3. Never state a number that is not written in the context. Copy every figure \
exactly as written, with its unit. If a figure is not there, it does not go in \
the answer.
4. Write at most {MAX_ANSWER_SENTENCES} sentences. Short, declarative, factual. \
No preamble, no sign-off, no restating the question.
5. Give no investment advice. Never say whether to buy, sell, hold, switch, or \
which of these funds is better. Never compare performance between funds. Never \
calculate, estimate, or project returns.
6. Output prose only. Do not output URLs, links, citations, dates, or source \
names - the interface adds the source link and the as-of date itself.

{DISCLAIMER}\
"""


def build_user_message(question: str, context: str) -> str:
    """The user turn: the question, then the numbered context block.

    The question comes first so it is the most recent thing in the model's
    attention, and the context is fenced in a delimiter so an instruction
    cannot arrive looking like part of the source material.
    """
    return (
        f"CONTEXT\n"
        f"--------\n"
        f"{context}\n"
        f"--------\n"
        f"\n"
        f"QUESTION: {question}\n"
        f"\n"
        f"Answer from the context above, following the rules. If the context "
        f"does not answer it, reply with exactly {INSUFFICIENT}."
    )


def build_messages(question: str, context: str) -> list[dict[str, str]]:
    """The full message list for an OpenAI-compatible chat call."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(question, context)},
    ]


#: Phrasings a model reaches for when it has nothing. The sentinel is the
#: contract, but a model that ignores it usually declines in its own words
#: instead, and treating that as an answer would ship "I don't know" to the
#: user labelled ``mode=fact`` - a refusal wearing the label of an answer.
_DECLINE_PATTERNS = (
    r"i\s+don'?t\s+know",
    r"\bno\s+(?:information|data|passage|mention|details?)\b",
    r"not\s+(?:available|mentioned|specified|stated|found|provided|present)\b",
    r"cannot\s+(?:be\s+)?(?:determin|answer|find|provide|tell)",
    r"(?:i\s+)?(?:am|'?m)\s+unable\s+to",
    r"unable\s+to\s+(?:answer|determine|find|provide)",
    r"insufficient",
    r"does\s+not\s+(?:contain|include|mention|provide|answer)",
    r"context\s+does\s+not",
)
_DECLINE_RE = re.compile("|".join(_DECLINE_PATTERNS), re.I)


def looks_insufficient(text: str) -> bool:
    """True when the model declined rather than answered.

    Matching is deliberately loose. Every pattern here is one that a genuine
    factual sentence about a mutual fund is very unlikely to contain, and
    reading an answer as a refusal is the safe direction to be wrong in - it
    refuses rather than invents.
    """
    if not text or not text.strip():
        return True
    return bool(_DECLINE_RE.search(text))
