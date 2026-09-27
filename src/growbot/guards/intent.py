"""Phase 5 - Intent guards.

Deterministic refusals that run **before** retrieval, so a "should I buy?"
question never reaches the vector store, the LLM, or the network
(architecture §8). Four outcomes, evaluated in a fixed order:

    PII  ->  advice / buy-sell / best fund  ->  returns compare-predict  ->  allow

Ordering is deliberate. PII is checked first because a message carrying a PAN
must be refused on privacy grounds even when it *also* asks for advice - the
privacy answer is the correct one, and it is the one that must not be
downgraded to a softer copy.

Two rules shape the implementation:

**Refusal links are never invented.** Every ``source_url`` comes from the
config allowlist - :data:`growbot.config.EDU_LINK` or the matching entry in
:data:`growbot.config.FACTSHEET_LINKS`. The model does not get to pick a URL,
and neither does this module.

**Raw messages are never logged or written.** On a PII hit the message is
redacted to a category list plus a character count. :func:`guard` performs no
I/O of any kind; the only thing that leaves this module is the payload object.

The module imports nothing from ``chromadb``, ``sentence_transformers`` or any
LLM client, so it stays fast and testable in isolation (architecture §14's
"ask should I buy? -> guard refusal" walkthrough step).
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field

from growbot.config import (
    DISCLAIMER,
    EDU_LINK,
    FACTSHEET_LINKS,
)
from growbot.config import detect_scheme as _detect_scheme

log = logging.getLogger("growbot.guards.intent")

__all__ = [
    "AnswerPayload",
    "classify",
    "detect_pii",
    "detect_scheme",
    "guard",
    "redact",
    "refuse_advice",
    "refuse_not_in_corpus",
    "refuse_pii",
    "refuse_returns",
    "refuse_ungrounded",
]


# ---------------------------------------------------------------------------
# Payload (architecture §7.3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnswerPayload:
    """What the UI receives. Mirrors architecture §7.3.

    The five documented keys are ``mode``, ``text``, ``source_url``,
    ``last_updated`` and ``scheme_name``. ``reason`` is an addition: the UI needs
    it to style a PII refusal differently from an advice refusal, and the
    guard's own tests assert on it. Consumers that only care about the
    documented contract can ignore it.

    ``last_updated`` defaults to empty because **a refusal has no source and so
    has no source date**. Making it required is what forced every refusal to
    stamp the current day, asserting a corpus freshness it had not earned. An
    answer that came from a retrieved chunk always sets it, from
    ``Assembly.last_updated``.
    """

    mode: str  # "fact" | "refuse"
    text: str
    source_url: str
    last_updated: str = ""
    scheme_name: str = ""
    reason: str = field(default="", compare=False)

    @property
    def refused(self) -> bool:
        return self.mode == "refuse"

    def to_dict(self) -> dict[str, str]:
        """The JSON shape shown in architecture §7.3."""
        return asdict(self)


# A note on `last_updated`, since it is easy to get wrong: it means "the source
# this answer came from was fetched on this date" (architecture §7.3). Answer
# payloads get it from the retrieved chunk metadata, via `Assembly.last_updated`.
# **No refusal sets it.** A refusal has no source, so there is no source date,
# and stamping today's date on one would claim the corpus was fresh when nothing
# had been read from it. The UI hides the field on refusal cards regardless;
# this is about the payload being honest, not about presentation.


# ---------------------------------------------------------------------------
# PII
# ---------------------------------------------------------------------------

#: (category, pattern). Each is checked independently and every hit is
#: reported, so a message carrying both a PAN and an email is logged as two
#: categories - never by value.
#:
#: Patterns are deliberately context-anchored where a bare number would be too
#: aggressive. ``\b\d{4,6}\b`` on its own would swallow every "1 year" and every
#: return figure in the corpus, so an OTP must be preceded by a word that means
#: OTP. The same reasoning keeps folio numbers to 7+ digits: HDFC scheme codes
#: are 6, so a scheme code is not mistaken for an account number.
_PII_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("pan", re.compile(r"\b[A-Za-z]{5}\d{4}[A-Za-z]\b")),
    (
        "aadhaar",
        re.compile(
            r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"
            r"|\b[Xx]{4}[\s-]?[Xx]{4}[\s-]?\d{4}\b"
        ),
    ),
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("phone", re.compile(r"(?:\+?91[\s-]?)?\b[6-9]\d{4}[\s-]?\d{5}\b")),
    (
        "otp",
        re.compile(
            r"\b(?:otp|one[\s-]?time\s+pass(?:word|code)|verification\s+code"
            r"|passcode|auth(?:entication)?\s?code|secure\s+code)\b"
            r"\s*(?:is|was|:|-)?\s*\d{4,6}\b",
            re.IGNORECASE,
        ),
    ),
    (
        "account_number",
        re.compile(
            r"\b(?:folio|account|acc|customer|cust|demat|dp)\s*"
            r"(?:no\.?|number|num|#)?\s*(?:is|was|[:\-])?\s*\d{7,}\b",
            re.IGNORECASE,
        ),
    ),
    ("long_number", re.compile(r"\b\d{7,}\b")),
)


def detect_pii(message: str) -> list[str]:
    """Return the PII *categories* present. Never returns the values."""
    if not message:
        return []
    found = [name for name, pattern in _PII_PATTERNS if pattern.search(message)]
    return sorted(set(found))


def redact(message: str) -> str:
    """Replace every PII span with ``[redacted]``, keeping the rest readable.

    Offered for logging and test output. The guard itself does not log messages
    at all, so this is never on the path to disk unless a caller asks for it.
    """
    if not message:
        return ""
    text = message
    for _name, pattern in _PII_PATTERNS:
        text = pattern.sub("[redacted]", text)
    return text


# ---------------------------------------------------------------------------
# Advice
# ---------------------------------------------------------------------------

#: Patterns that ask for a *recommendation about investing*, not a fact.
#:
#: The bar is high on purpose. "What does buy and hold mean?" is a legitimate
#: question the AMFI guides answer, so a bare "buy" or "invest" cannot trigger a
#: refusal - each pattern needs a recommendation shape (a modal, a superlative
#: next to "fund", or an explicit ask for a suggestion).
_ADVICE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("should_i", re.compile(r"\bshould\s+(?:i|we|my\b|one\b)", re.IGNORECASE)),
    ("recommend", re.compile(r"\b(?:recommend|suggest|advise|advise\s+me|tell\s+me\s+what\s+to)\b", re.IGNORECASE)),
    ("is_it_safe", re.compile(r"\b(?:is|are)\s+it\s+(?:safe|worth|risk[\s-]?free|good|a\s+good)\b", re.IGNORECASE)),
    ("good_time", re.compile(r"\b(?:good|right|best|wrong)\s+time\s+to\s+(?:buy|invest|enter|exit|start)\b", re.IGNORECASE)),
    ("worth_it", re.compile(r"\bworth\s+(?:investing|buying|it|purchasing)\b", re.IGNORECASE)),
    ("which_should", re.compile(r"\bwhich\s+(?:fund|scheme|one|mutual\s+fund)s?\s+(?:should|do|is\s+best|would\s+you)\b", re.IGNORECASE)),
    ("can_i_invest", re.compile(r"\bcan\s+i\s+(?:invest|buy|sell|purchase|start|allocate)\b", re.IGNORECASE)),
    ("what_should_i", re.compile(r"\bwhat\s+should\s+i\b", re.IGNORECASE)),
    ("how_much_should_i", re.compile(r"\bhow\s+much\s+should\s+i\b", re.IGNORECASE)),
    ("best_fund", re.compile(r"\bbest\s+(?:fund|scheme|mutual\s+fund)s?\b", re.IGNORECASE)),
    ("fund_verdict", re.compile(r"\b(?:good|safe|bad|profitable)\s+(?:investment|fund)\b", re.IGNORECASE)),
    ("allocation", re.compile(r"\b(?:portfolio\s+allocation|how\s+should\s+i\s+(?:allocate|split|divide))\b", re.IGNORECASE)),
    ("hold_or_sell", re.compile(r"\b(?:hold|keep)\s+(?:it\s+|this\s+)?or\s+(?:sell|exit|redeem|switch)\b", re.IGNORECASE)),
    ("switch", re.compile(r"\bshould\s+i\s+switch\b|\bswitch\s+from\b", re.IGNORECASE)),
    # "Which is better, X or Y?" is a recommendation request wearing a
    # question mark. The lookahead keeps the fact-comparison phrasing -
    # "which has the *better* expense ratio" - on the allow side, because that
    # one is answerable from the disclosed numbers.
    (
        "which_better",
        re.compile(
            r"\bwhich\b[^.?]*\b(?:better|worse|safer|riskier|profitable)\b"
            r"(?!\s*(?:expense|ratio|risk|load|return|yield|benchmark|nav"
            r"|aum|rating|value|performance|fee|charge))",
            re.IGNORECASE,
        ),
    ),
)


def detect_advice(message: str) -> list[str]:
    """Return the advice patterns that matched."""
    if not message:
        return []
    return [name for name, pattern in _ADVICE_PATTERNS if pattern.search(message)]


# ---------------------------------------------------------------------------
# Returns: compare or predict
# ---------------------------------------------------------------------------

#: Two things get refused: **comparing** funds on returns, and **predicting**
#: future returns. A single scheme's *historical* return is neither, and the
#: corpus contains return tables, so "what is the 1 year return of HDFC Large
#: Cap?" is allowed through. The line is tense and plurality: a superlative
#: spanning funds, or a "will ... return" in the future, gets refused.
_RETURNS_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "compare",
        re.compile(
            r"\b(?:compare|comparison|versus|vs\.?)\b[^.?]*\breturns?\b"
            r"|\breturns?\b[^.?]*\b(?:versus|vs\.?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "superlative",
        re.compile(
            r"\b(?:best|highest|lowest|maximum|top|more)\b[^.?]*\breturns?\b"
            r"|\breturns?\b[^.?]*\b(?:highest|lowest|maximum|most)\b",
            re.IGNORECASE,
        ),
    ),
    ("which_returns", re.compile(r"\bwhich\b[^.?]*\breturns?\b", re.IGNORECASE)),
    (
        "predict",
        re.compile(
            r"\b(?:predict|forecast|project)\b[^.?]*\breturns?\b"
            r"|\bexpect(?:ed)?\b[^.?]*\breturns?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "future",
        re.compile(
            r"\bwill\b[^.?]*\b(?:give|provide|deliver|earn|make|generate)\b[^.?]*\breturns?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "earn",
        re.compile(
            r"\b(?:will|would|could|shall)\s+(?:i|we|it|they)\s+"
            r"(?:get|earn|make|gain|receive)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "projection",
        re.compile(
            r"\bhow\s+(?:much|long|when)\s+(?:will|would|can|could)\s+(?:i|we)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "guarantee",
        re.compile(
            r"\b(?:guarantee\w*|assured?|double|tri+ple)\b[^.?]*"
            r"\b(?:return|profit|gain|money)\b",
            re.IGNORECASE,
        ),
    ),
)


def detect_returns(message: str) -> list[str]:
    """Return the returns patterns that matched."""
    if not message:
        return []
    return [name for name, pattern in _RETURNS_PATTERNS if pattern.search(message)]


# ---------------------------------------------------------------------------
# Scheme detection (chooses which allowlisted factsheet link to cite)
# ---------------------------------------------------------------------------


def detect_scheme(message: str) -> list[str]:  # noqa: F811 - re-export
    """Every in-scope scheme named in the message.

    Re-exported from :func:`growbot.config.detect_scheme`, which owns the
    implementation so the Phase 6 retriever can use the same alias matching
    without depending on the guard (architecture §6).
    """
    return _detect_scheme(message)


def display_name(scheme_name: str) -> str:
    """Human-facing form of a scheme name.

    The config key carries the plan suffix - "HDFC Small Cap Fund - Direct
    Growth" - because it has to match the corpus metadata exactly. That reads
    badly in a sentence, so prose uses the trimmed name while the
    ``scheme_name`` payload field keeps the canonical one for Phase 6's
    metadata filter.
    """
    for suffix in (" - Direct Plan Growth", " - Direct Growth"):
        if scheme_name.endswith(suffix):
            return scheme_name[: -len(suffix)]
    return scheme_name


def _factsheet_link(schemes: list[str]) -> tuple[str, str]:
    """Pick the one allowlisted URL to cite. Returns (url, scheme_name).

    With exactly one scheme named we can point at that fund's own factsheet,
    which is the most useful thing a refused question can hand back. With none
    or several we fall back to the education link, because guessing which
    factsheet was meant would be inventing an answer.
    """
    if len(schemes) == 1:
        name = schemes[0]
        return FACTSHEET_LINKS[name], name
    return EDU_LINK, ""


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def refuse_pii(categories: list[str] | None = None) -> AnswerPayload:
    """Refuse because the message carries personal data.

    No scheme link is possible or useful here, so the single citation is the
    investor-education glossary. The categories are not echoed into the text -
    telling a user exactly which pattern matched is a small but real leak.
    """
    return AnswerPayload(
        mode="refuse",
        text=(
            "I don't handle personal identifiers, and I don't store messages that "
            "contain them, so I've discarded this one. For what those terms mean, "
            f"AMFI's investor-education glossary is the right place to look. {DISCLAIMER}"
        ),
        source_url=EDU_LINK,
        reason="pii",
    )


def refuse_advice(schemes: list[str] | None = None) -> AnswerPayload:
    """Refuse a buy / sell / "best fund" request. Cites the education link."""
    schemes = schemes or []
    if len(schemes) == 1:
        subject = display_name(schemes[0])
    elif len(schemes) > 1:
        subject = "any of these five funds"
    else:
        subject = "a fund"
    return AnswerPayload(
        mode="refuse",
        text=(
            f"I can't say whether {subject} is worth buying - that's an investment "
            "decision, not a fact I can look up. What I can do is quote you the "
            "disclosed numbers, like expense ratio, exit load, benchmark and risk "
            f"level, straight from the fund's own pages. {DISCLAIMER}"
        ),
        source_url=EDU_LINK,
        scheme_name=schemes[0] if len(schemes) == 1 else "",
        reason="advice",
    )


def refuse_returns(schemes: list[str] | None = None) -> AnswerPayload:
    """Refuse a returns comparison or prediction. Cites a factsheet pointer.

    When one scheme is named the refusal links that fund's factsheet, which is
    where the official return figures live. No ranking or computed number is
    ever produced (PRD §12 Q9).
    """
    schemes = schemes or []
    url, scheme_name = _factsheet_link(schemes)
    if len(schemes) == 1:
        pointer = (
            f"the factsheet published by the AMC for {display_name(schemes[0])} "
            "is the official source for its own returns"
        )
    else:
        pointer = (
            "each AMC publishes its own fund factsheet, and comparing the official "
            "factsheets is the way to do this without anyone ranking them for you"
        )
    return AnswerPayload(
        mode="refuse",
        text=(
            "I can't rank the five funds by returns or tell you what they'll "
            f"return - I have no basis for either, and a made-up ranking would be "
            f"worse than none. For that, {pointer}. {DISCLAIMER}"
        ),
        source_url=url,
        scheme_name=scheme_name,
        reason="returns",
    )


# ---------------------------------------------------------------------------
# Retrieval and generation refusals (Phases 6-7)
# ---------------------------------------------------------------------------


def refuse_not_in_corpus(
    schemes: list[str] | None = None, *, reason: str
) -> AnswerPayload:
    """Refuse because retrieval did not find a strong enough match.

    Distinct from `refuse_returns`: nothing was asked about performance, the
    question simply is not answered by the corpus. Said plainly, because "I
    don't have that" is a more useful answer than a confident guess - and the
    corpus really is only twelve pages, so this is a common and honest outcome.

    `reason` is **required**, with no default, because this one function covers
    several genuinely different causes - weak retrieval, the model declining,
    and an unusable model reply - and the UI and the trace distinguish them.
    A default here was `reason="weak"`, which no call site ever used and no
    check anywhere recognised: a caller who forgot the keyword would have
    emitted a reason that silently meant nothing. Forcing it means the reason
    is always one the rest of the system has agreed on.
    """
    schemes = schemes or []
    url, scheme_name = _factsheet_link(schemes)
    if len(schemes) == 1:
        pointer = (
            f"the page I hold for {display_name(schemes[0])} is the place to check"
        )
    else:
        pointer = "the scheme pages listed in the sources file are what I hold"
    return AnswerPayload(
        mode="refuse",
        text=(
            "I don't have a passage in my source material that answers that, so "
            f"I won't guess at it. For this, {pointer}. {DISCLAIMER}"
        ),
        source_url=url,
        scheme_name=scheme_name,
        reason=reason,
    )


def refuse_ungrounded(numbers: list[str]) -> AnswerPayload:
    """Refuse because the model stated a figure that is not in the context.

    The single most important check in the system (architecture §13, "LLM
    ignores grounding"). A number the model produced from memory rather than
    from the retrieved text is indistinguishable, to the user, from one the
    page actually says - so the number is discarded along with the answer, not
    quietly corrected. Naming the figures here would restate the unverified
    claim, so they are logged by the caller instead of shown.
    """
    return AnswerPayload(
        mode="refuse",
        text=(
            "I can't state that one - the answer came back with figures that "
            "aren't in the source text I retrieved, so I'm discarding it rather "
            f"than passing on something I can't verify. {DISCLAIMER}"
        ),
        source_url=EDU_LINK,
        reason="ungrounded",
    )


def refuse_unusable_answer(schemes: list[str] | None = None) -> AnswerPayload:
    """Refuse because the model returned a fragment, not a statement.

    Distinct from `refuse_not_in_corpus`, and the distinction is the whole
    point. That copy says "I don't have a passage that answers that", which
    would be **false** here: the passage was retrieved, the model read it, and
    the model simply replied with a bare figure like `0.77%`. The corpus is
    fine. Saying otherwise would teach a user to distrust refusals that are
    usually accurate.

    So this says what actually happened - the source was there, the answer could
    not be phrased - and points at the page rather than inventing a cause.
    """
    schemes = schemes or []
    url, scheme_name = _factsheet_link(schemes)
    if len(schemes) == 1:
        pointer = (
            f"the page I hold for {display_name(schemes[0])} has the figure in it"
        )
    else:
        pointer = "the scheme pages listed in the sources file are what I hold"
    return AnswerPayload(
        mode="refuse",
        text=(
            "I have the source text for that, but I couldn't turn it into a "
            "clear statement, so I'm not going to guess at one. For this, "
            f"{pointer}. {DISCLAIMER}"
        ),
        source_url=url,
        scheme_name=scheme_name,
        reason="unusable",
    )


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def classify(message: str) -> str:
    """Return one of ``pii``, ``advice``, ``returns``, ``allow``.

    Fixed precedence: PII, then advice, then returns. The first match wins, so
    a message that is both PII and advice is refused as PII.
    """
    if not message or not message.strip():
        return "allow"
    if detect_pii(message):
        return "pii"
    if detect_advice(message):
        return "advice"
    if detect_returns(message):
        return "returns"
    return "allow"


def guard(message: str) -> AnswerPayload | None:
    """The single call the UI makes before retrieval.

    Returns a refusal :class:`AnswerPayload` when the message must not be
    answered, or ``None`` to proceed to embed -> retrieve -> generate.

    This function performs no I/O: no network, no disk, no vector store. The
    only side effect is a log line carrying the refusal *category* and the
    message length - never its content.
    """
    verdict = classify(message)
    if verdict == "allow":
        return None

    schemes = detect_scheme(message)
    payload = {
        "pii": lambda: refuse_pii(detect_pii(message)),
        "advice": lambda: refuse_advice(schemes),
        "returns": lambda: refuse_returns(schemes),
    }[verdict]()

    # Category and length only. Logging the message here would defeat the whole
    # point of the PII branch.
    log.info(
        "guard refused (%s, %d char(s), schemes=%d); message not logged",
        payload.reason,
        len(message),
        len(schemes),
    )
    return payload
