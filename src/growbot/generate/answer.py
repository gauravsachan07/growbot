"""Phase 7 - LLM call -> `AnswerPayload` (architecture §6.4, §7.3).

Everything the user eventually sees passes through this module, and the
important property is that **the model is not trusted with any of it**. The
model is asked for prose and nothing else. The citation URL, the as-of date and
the scheme chip are copied from the Phase 6 `Assembly`, which took them from
chunk metadata. If the model returns a URL or a date in its prose, it is
stripped rather than shown.

Three checks run after the model replies, each converting a bad reply into a
refusal instead of a defect:

1. **The model declined.** Prompt rule 2 asks for a sentinel when the context
   does not answer the question. Honoured as a refusal.
2. **A number that is not in the context.** The highest-value check in the
   system. The corpus is full of live NAV snapshots, return tables and a
   326-row holdings list, so a plausible-looking ratio is exactly the failure
   this project exists to prevent (architecture §13).
3. **Too many sentences.** Clamped to `MAX_ANSWER_SENTENCES` at a sentence
   boundary, which keeps the answer usable instead of refusing a correct reply
   for being verbose.

None of these are the model's promise. They are this module's.
"""

from __future__ import annotations

import logging
import re
from typing import Callable, Sequence

from growbot.config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MAX_ATTEMPTS,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_PROVIDER,
    LLM_RETRY_BACKOFF,
    LLM_RETRY_BACKOFF_MAX,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
    MAX_ANSWER_SENTENCES,
    llm_is_configured,
)
from growbot.guards.intent import (
    AnswerPayload,
    refuse_not_in_corpus,
    refuse_ungrounded,
    refuse_unusable_answer,
)
from growbot.retrieve.assemble import Assembly
from growbot.generate.prompt import build_messages, looks_insufficient

log = logging.getLogger("growbot.generate.answer")

__all__ = [
    "LLMError",
    "LLMNotConfigured",
    "LLMQuotaExceeded",
    "LLMTruncated",
    "call_openai_compatible",
    "generate",
    "is_prose_free",
    "split_sentences",
    "ungrounded_numbers",
]

#: A `complete(messages) -> str` callable. Injectable so the tests can drive
#: every branch, including the hostile ones, without a key or a network.
Completer = Callable[[Sequence[dict[str, str]]], str]


class LLMError(RuntimeError):
    """The provider was reachable but did not return usable text."""


class LLMNotConfigured(RuntimeError):
    """No usable key, model or base URL in the environment."""


class LLMTruncated(LLMError):
    """The provider stopped mid-answer because the token budget ran out.

    Separate from `LLMError` because the fix is different and the cause is not
    a fault worth reporting to a user as "the model is down". A truncated reply
    is a partial sentence - "The expense ratio of HDFC Large Cap Fund Direct" -
    and passing that off as an answer is worse than failing, so it is never
    returned. Raise `LLM_MAX_TOKENS` or shorten the context.
    """


class LLMQuotaExceeded(LLMError):
    """The provider's allowance for this key is used up.

    Distinct from a transient failure and deliberately **not retried**. A free
    tier returns HTTP 429 for two very different things: a per-minute rate
    limit, which a wait fixes, and a daily quota that is simply spent, which no
    amount of retrying fixes. Measured on a gemini-3.8-flash free key, the quota
    case reads "You exceeded your current quota, please check your plan and
    billing details" - a daily allowance, gone for the day.

    Retrying that four times cost about eight seconds of a demo's time and then
    failed identically, which is strictly worse than saying so immediately.
    """


#: Statuses worth trying again. 429 is here because of the per-minute rate-limit
#: case only; `LLMQuotaExceeded` intercepts the spent-quota case before this
#: set is consulted. Everything else - 400, 401, 403, 404 - is a mistake in the
#: request or the credentials, and retrying it just spends the demo's time.
_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

#: Phrases in a 429 body that mean "your allowance is spent", as opposed to
#: "slow down". Matched against the provider's own message, which is the only
#: place the distinction is available - both cases are HTTP 429.
_QUOTA_PHRASES = (
    "exceeded your current quota",
    "quota exceeded",
    "check your plan and billing",
    "billing details",
    "insufficient_quota",
    "resource_exhausted",
    "out of credits",
)


def _backoff_seconds(attempt: int) -> float:
    """Exponential backoff with jitter, capped.

    Jitter matters more than it looks: without it, every client that hits a
    demand spike retries at the same instant and reproduces the spike.
    """
    import random

    raw = min(
        LLM_RETRY_BACKOFF * (2**attempt), LLM_RETRY_BACKOFF_MAX
    )
    return raw * (0.5 + random.random() / 2)


# ---------------------------------------------------------------------------
# The LLM call
# ---------------------------------------------------------------------------


def _extract_text(response) -> str:
    """Pull the message text out of a chat-completions response.

    Separated from the HTTP work because this is where a real provider's
    response shape varies, and each variation needs its own error. Measured on
    gemini-3.8-flash: when the budget runs out the reply comes back as
    ``{"role": "assistant"}`` with **no content key at all**, so indexing
    straight into it raises a KeyError that says nothing useful.
    """
    try:
        data = response.json()
    except ValueError as exc:
        raise LLMError(
            f"{LLM_PROVIDER} returned a non-JSON body: {response.text[:200]}"
        ) from exc

    try:
        choice = data["choices"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(
            f"{LLM_PROVIDER} returned an unexpected response shape: "
            f"{response.text[:200]}"
        ) from exc

    finish_reason = choice.get("finish_reason")
    message = choice.get("message") or {}

    if finish_reason == "length":
        raise LLMTruncated(
            f"{LLM_PROVIDER} stopped at the {LLM_MAX_TOKENS}-token limit "
            "(finish_reason=length), so the answer is incomplete. Raise "
            "LLM_MAX_TOKENS in .env."
        )
    if finish_reason not in (None, "stop", "end_turn", "STOP"):
        raise LLMError(
            f"{LLM_PROVIDER} stopped for an unexpected reason "
            f"(finish_reason={finish_reason!r})"
        )

    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise LLMError(
            f"{LLM_PROVIDER} returned no message content "
            f"(finish_reason={finish_reason!r}, message keys="
            f"{sorted(message) if isinstance(message, dict) else 'n/a'})"
        )
    return content


def call_openai_compatible(messages: Sequence[dict[str, str]]) -> str:
    """Call an OpenAI-compatible chat endpoint and return the message text.

    One client shape serves OpenAI, Groq, OpenRouter and Gemini's compatibility
    endpoint; `LLM_PROVIDER` only picks the base URL. `httpx` is already a core
    dependency, so this adds no package to the demo install.

    Retries transient failures up to `LLM_MAX_ATTEMPTS` times with exponential
    backoff. This is not defensive padding - a live provider on a free tier
    fails often enough that a single attempt makes the bot look broken during
    a demo.

    The key is only ever placed in an Authorization header. It is never logged
    and never included in an exception message, so an exception can be pasted
    into a bug report.
    """
    if not llm_is_configured():
        missing = [
            name
            for name, value in (
                ("LLM_API_KEY", LLM_API_KEY),
                ("LLM_MODEL", LLM_MODEL),
                ("LLM_BASE_URL", LLM_BASE_URL),
            )
            if not value
        ]
        raise LLMNotConfigured(
            "Cannot generate: " + ", ".join(missing) + " not set. "
            "Copy .env.example to .env and fill in LLM_PROVIDER, LLM_API_KEY "
            "and LLM_MODEL (any free tier works), then restart. "
            "See docs/architecture.md §6.4."
        )

    import time

    import httpx

    url = f"{LLM_BASE_URL.rstrip('/')}/chat/completions"
    payload = {
        "model": LLM_MODEL,
        "messages": list(messages),
        "temperature": LLM_TEMPERATURE,
        "max_tokens": LLM_MAX_TOKENS,
    }
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }

    attempts = max(1, LLM_MAX_ATTEMPTS)
    last_error: LLMError | None = None

    for attempt in range(attempts):
        try:
            with httpx.Client(timeout=LLM_TIMEOUT) as client:
                response = client.post(url, json=payload, headers=headers)
        except httpx.HTTPError as exc:
            # A transport failure (DNS, connect, read timeout) is transient in
            # the same way a 503 is, so it earns the same retry.
            last_error = LLMError(
                f"Could not reach {LLM_PROVIDER} at {LLM_BASE_URL}: "
                f"{exc.__class__.__name__}"
            )
        else:
            if response.status_code == 429 and _is_quota_exhausted(response.text):
                # Fail fast. A spent daily allowance will still be spent in
                # eight seconds' time, and a demo is better served by an honest
                # message than a long wait for the same failure.
                raise LLMQuotaExceeded(
                    f"{LLM_PROVIDER} reports this key's allowance is used up "
                    f"({_short_reason(response.text)}). This is a provider quota, "
                    "not a Growbot failure: it resets on the provider's own "
                    "schedule, or needs a different key. Nothing was retried."
                )
            if response.status_code in _RETRYABLE_STATUS:
                last_error = LLMError(
                    f"{LLM_PROVIDER} returned HTTP {response.status_code} "
                    f"(retried {attempt + 1}/{attempts}): "
                    f"{_short_reason(response.text)}"
                )
            elif response.status_code >= 400:
                # Not transient: a bad key, a bad model id, a malformed body.
                raise LLMError(
                    f"{LLM_PROVIDER} returned HTTP {response.status_code}: "
                    f"{_short_reason(response.text)}"
                )
            else:
                return _extract_text(response)

        if attempt < attempts - 1:
            delay = _backoff_seconds(attempt)
            log.warning(
                "%s; retrying in %.1fs (attempt %d/%d)",
                last_error, delay, attempt + 2, attempts,
            )
            time.sleep(delay)

    raise last_error or LLMError(f"{LLM_PROVIDER} call failed")


def _is_quota_exhausted(body: str) -> bool:
    """True when a 429 means "allowance spent" rather than "slow down".

    Both arrive as HTTP 429 with no machine-readable field to tell them apart,
    so the provider's own wording is the only signal available.
    """
    lowered = body.lower()
    return any(phrase in lowered for phrase in _QUOTA_PHRASES)


def _short_reason(body: str) -> str:
    """The provider's own error message, without the JSON envelope.

    A 503 body is ~250 characters of JSON wrapper around one useful sentence,
    and the wrapper is identical every time, so it is worth unwrapping.

    The shape is not standardised: gemini-3.8-flash wraps the error in a
    *list* - ``[{"error": {...}}]`` - while OpenAI returns a bare object. Both
    are handled, because guessing produced a log line that read
    ``[{ "error": { "code": 503,`` instead of the actual problem.
    """
    try:
        import json

        parsed = json.loads(body)
        if isinstance(parsed, list) and parsed:
            parsed = parsed[0]
        if isinstance(parsed, dict):
            message = parsed.get("error", {}).get("message")
            if isinstance(message, str):
                return " ".join(message.split())[:160]
    except (ValueError, KeyError, TypeError, AttributeError):
        pass
    return " ".join(body.split())[:160]


# ---------------------------------------------------------------------------
# Post-check 1: sentence count
# ---------------------------------------------------------------------------

#: Words that end in a period without ending a sentence. "Rs. 100" is the one
#: that matters most here, because currency is exactly what these answers quote.
_ABBREVIATIONS = frozenset(
    {
        "e.g", "i.e", "etc", "vs", "approx", "no", "fig",
        "rs", "inr", "mr", "mrs", "ms", "dr", "sh",
        "amfi", "sebi", "nav", "ltd", "pvt", "co", "inc", "as on",
    }
)


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences, without breaking on decimals or abbreviations.

    Naive splitting on ``[.!?]`` mangles exactly the text this system produces:
    "The expense ratio is 1.03%." would become two fragments, and "3.5" would
    become a sentence. Both matter here, because the first is a normal answer
    and the second is the numeric check's raw material.
    """
    sentences: list[str] = []
    start = 0
    index = 0
    length = len(text)

    while index < length:
        char = text[index]
        if char not in ".!?":
            index += 1
            continue

        # A period between digits is a decimal point or a version number.
        if (
            char == "."
            and 0 < index < length - 1
            and text[index - 1].isdigit()
            and text[index + 1].isdigit()
        ):
            index += 1
            continue

        # A period preceded by a known abbreviation is not a sentence end.
        if char == ".":
            preceding = re.split(r"[\s(\[]", text[:index])[-1].lower().rstrip(".")
            if preceding in _ABBREVIATIONS:
                index += 1
                continue

        # Sentence end only when followed by whitespace or the end of the text.
        if index + 1 == length or text[index + 1].isspace():
            candidate = text[start:index + 1].strip()
            if candidate:
                sentences.append(candidate)
            start = index + 1
        index += 1

    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def _clamp_sentences(text: str, limit: int) -> str:
    """Keep at most `limit` complete sentences."""
    sentences = split_sentences(text)
    if len(sentences) <= limit:
        return " ".join(sentences).strip()
    return " ".join(sentences[:limit]).strip()


# ---------------------------------------------------------------------------
# Post-check 2: every number in the answer came from the context
# ---------------------------------------------------------------------------

#: A number optionally followed by a unit. The leading lookbehind stops the
#: scanner from picking up digits inside a word or inside another number.
_NUMBER_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(%|percent|bps)?", re.I)
#: A URL together with the connective that introduces it. Removing a URL on
#: its own leaves debris - "See for details" - so the connector is taken with
#: it. The list is short and explicit rather than clever, because guessing at
#: English connective phrases is how a rewriter starts deleting content.
_URL_WITH_CONNECTOR_RE = re.compile(
    r"\b(?:see(?:\s+also)?|refer\s+to|as\s+per|according\s+to|from|at|on|per|in|via|"
    r"source|reference|details?(?:\s+(?:at|on|in|are\s+at))?|"
    r"available\s+(?:at|on|from)|read\s+more\s+(?:at|on))\s*[:\-]?\s*"
    r"(?:https?://|www\.)\S+",
    re.I,
)
#: Any remaining bare URL, with no connective to carry away.
_URL_BARE_RE = re.compile(r"(?:https?://|www\.)\S+", re.I)
#: A connective left stranded *after* a removed span, as in
#: "see <url> for details". Removing only the leading connective handles
#: "see <url>" but leaves "for details" dangling, which reads as broken prose.
_TRAILING_CONNECTOR_RE = re.compile(
    r"\s+(?:for\s+(?:more\s+|further\s+|the\s+)?(?:details?|information|reference|"
    r"source|figures?)|at\s+the\s+source|on\s+the\s+website|in\s+the\s+source)\b",
    re.I,
)
#: A date together with the phrase that introduces it. The prompt forbids dates
#: as well as URLs, and an invented "last updated 1999-01-01" in the prose is
#: the same defect as an invented URL: the payload's date comes from chunk
#: metadata, so any date in the prose is the model's own invention.
_DATE_PHRASE_RE = re.compile(
    r"\b(?:last\s+updated|as\s+of|as\s+on|updated\s+on|data\s+as\s+of|dated|"
    r"retrieved\s+on|fetched\s+(?:on|at))\s*[:\-]?\s*"
    r"(?:\d{4}-\d{2}-\d{2}|\d{1,2}\s+[A-Za-z]+\s+\d{4}|[A-Za-z]+\s+\d{1,2},?\s+\d{4})?",
    re.I,
)
_ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
#: Comma-grouped digits, normalised so "1,030" and "1030" compare equal.
_COMMA_RE = re.compile(r"(?<=\d),(?=\d\d\d)")


def _normalise_numbers(text: str) -> str:
    return _COMMA_RE.sub("", text)


def ungrounded_numbers(answer: str, context: str) -> list[str]:
    """Numeric claims in `answer` that do not appear in `context`.

    Only numbers that could plausibly be a *figure* are enforced:

    * anything with a decimal point - every expense ratio, exit load and
      portfolio weight in this corpus;
    * anything carrying a unit (``%``, percent, bps);
    * bare integers of three digits or more - "NIFTY 100", holdings counts.

    Bare one- and two-digit integers are accepted without checking. In a
    corpus whose chunks are dense with figures, "3" occurs somewhere almost
    always, so enforcing them would add false refusals ("one of the five
    funds") while catching almost nothing.

    **Known limit, stated rather than hidden:** matching is substring-based, so
    a fabricated number whose digits coincidentally occur somewhere in the
    ~2,500 characters of context will pass. The check is strong against
    invented *ratios* - the PRD's stated concern - and weak against invented
    small integers. Tightening it needs an exact-span matcher, which belongs
    with a real evaluation set rather than a guess at a threshold.
    """
    haystack = _normalise_numbers(context).lower()
    found: list[str] = []
    seen: set[str] = set()

    for match in _NUMBER_RE.finditer(answer):
        raw, unit = match.group(1), (match.group(2) or "").lower()
        digits_only = raw.replace(".", "")
        enforce = "." in raw or bool(unit) or len(digits_only) >= 3
        if not enforce:
            continue
        if raw in seen:
            continue
        seen.add(raw)
        if raw.lower() not in haystack:
            # The matched text, not raw+unit: re-joining them would print
            # "2.75percent" in the log.
            found.append(match.group(0).strip())
    return found


def _clean_prose(text: str) -> str:
    """Strip anything the model was told not to emit, without mangling prose.

    The prompt asks for prose only, and the payload supplies the one citation
    URL and the one as-of date. So a URL or a date in the reply is redundant at
    best and the model's own invention at worst - and it would sit in the same
    sentence as a correct fact, where the user has no way to tell the two apart.

    URLs and dates usually arrive with a connective ("see https://...", "last
    updated 2026-01-01"), so the connective is removed with them. A URL copied
    out of the context's own ``source:`` line is a real corpus URL rather than
    a fabrication, which is why this cleans the text rather than refusing an
    otherwise correct answer.
    """
    cleaned = _URL_WITH_CONNECTOR_RE.sub("", text)
    cleaned = _URL_BARE_RE.sub("", cleaned)
    cleaned = _DATE_PHRASE_RE.sub("", cleaned)
    cleaned = _ISO_DATE_RE.sub("", cleaned)
    cleaned = _TRAILING_CONNECTOR_RE.sub("", cleaned)
    # Tidy the punctuation a removed span leaves behind, in dependency order:
    # pull punctuation in tight, then drop one of two adjacent stops, then
    # collapse whatever runs of the same stop that leaves. Doing the collapse
    # earlier misses the doubled period that step two creates.
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"([,;:])\s*(?=[,.;:!?])", "", cleaned)
    cleaned = re.sub(r"([.!?])\1{1,}", r"\1", cleaned)
    cleaned = re.sub(r"\(\s*\)", "", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip(" ,;:-")


#: Appended as an extra user turn when the model replies with a bare figure.
#: Asking again is cheap and specific: the retrieved context is unchanged, so
#: the only thing wrong with the answer is its shape.
RESHAPE_NUDGE = (
    "That reply was only a figure. Answer again as a complete sentence naming "
    "what the figure is, using only the context above. Do not add any other "
    "number, and do not include a link or a date."
)


def is_prose_free(text: str) -> bool:
    """True when the reply contains no words at all - e.g. a bare ``0.77%``.

    Such a reply passes every other post-check: it is short, it holds no stray
    URL, and any number in it is by construction grounded in the context. It is
    also not an answer. Rendered with a citation and an as-of date, ``0.77%``
    reads as a rendering bug, and it was the single answer in the demo pack
    that a reader could not tell was correct.

    One alphabetic character is the test, not a word count: the shortest
    legitimate answers ("Very High risk.", "Benchmark is NIFTY 100 TRI.") are
    fragments, and a word-count rule would throw those away too.
    """
    return not any(ch.isalpha() for ch in text)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def generate(
    question: str,
    assembly: Assembly,
    *,
    complete: Completer | None = None,
) -> AnswerPayload:
    """Answer `question` from `assembly`, or refuse. Never invents a fact.

    `complete` defaults to the real provider call. Tests pass a stub, which also
    bypasses the configuration check - a missing key must not make the code
    paths untestable.

    A weak `Assembly` is refused **without calling the model at all**
    (architecture §20: "refuse without calling the LLM when possible"). That
    saves a call and, more importantly, means a low-similarity question cannot
    reach a model that might talk its way to a plausible answer.
    """
    if assembly.weak:
        log.info(
            "refusing without generating (%s): %s",
            ",".join(assembly.weak_reasons),
            question[:60],
        )
        return refuse_not_in_corpus(
            assembly.detected_schemes, reason="weak_retrieval"
        )

    if complete is None and not llm_is_configured():
        # Re-raise the detailed message from the client rather than inventing
        # a second, thinner one.
        return call_openai_compatible(build_messages(question, assembly.context))

    raw = (complete or call_openai_compatible)(
        build_messages(question, assembly.context)
    )

    if looks_insufficient(raw):
        log.info("model declined: context did not answer %s", question[:60])
        return refuse_not_in_corpus(assembly.detected_schemes, reason="insufficient")

    text = _clean_prose(raw)
    if not text:
        log.info("model returned nothing usable for %s", question[:60])
        return refuse_not_in_corpus(assembly.detected_schemes, reason="insufficient")

    # A bare figure is not a sentence. The context is already retrieved, so the
    # only thing wrong with the reply is its shape - which is worth exactly one
    # more call to fix. Observed once in ten on a real run, so the retry is not
    # a meaningful share of the rate-limit budget.
    if is_prose_free(text):
        log.info("model returned a bare figure, asking again: %s", question[:60])
        retry = build_messages(question, assembly.context)
        retry.append({"role": "user", "content": RESHAPE_NUDGE})
        text = _clean_prose((complete or call_openai_compatible)(retry))
        if not text or is_prose_free(text):
            # Not worth a third call. Refuse, and say what actually went wrong
            # - `refuse_unusable_answer`, not `refuse_not_in_corpus`, which
            # would wrongly claim the corpus has no answer.
            log.info("still no sentence after the retry for %s", question[:60])
            return refuse_unusable_answer(assembly.detected_schemes)

    # Citation and date are metadata. Never the model's.
    text = _clamp_sentences(text, MAX_ANSWER_SENTENCES)

    ungrounded = ungrounded_numbers(text, assembly.context)
    if ungrounded:
        # Logged, never shown: restating the figure would repeat the claim
        # this check exists to withhold.
        log.warning(
            "discarding answer with %d ungrounded number(s): %s",
            len(ungrounded),
            ", ".join(ungrounded),
        )
        return refuse_ungrounded(ungrounded)

    return AnswerPayload(
        mode="fact",
        text=text,
        source_url=assembly.source_url,
        last_updated=assembly.last_updated,
        scheme_name=assembly.scheme_name,
        reason="",
    )
