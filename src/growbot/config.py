"""Central configuration for Growbot.

Every tunable listed in `docs/architecture.md` §11 lives here so no stage has to
hard-code a path, a model name, or a refusal link. Environment variables (see
`.env.example`) override the defaults.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

#: Repository root - src/growbot/config.py -> parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# `.env` is loaded here, before any os.getenv call below, because those calls
# run at import time and would otherwise capture the pre-.env environment.
# The dotenv import is optional so that the loading stage (Phase 1) keeps
# working in a bare venv where only the core dependencies are installed.
try:  # pragma: no cover - exercised by whichever branch the venv provides
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:  # pragma: no cover
    pass


def _env_path(name: str, default: Path) -> Path:
    """Read a path from the environment, resolving it against PROJECT_ROOT."""
    raw = os.getenv(name)
    if not raw:
        return default
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


#: Auditable URL list that drives ingestion (Phase 1).
SOURCES_CSV = _env_path("SOURCES_CSV", PROJECT_ROOT / "data" / "sources.csv")

#: Local persist dir for the Chroma collection (gitignored, rebuildable).
CHROMA_PATH = _env_path("CHROMA_PATH", PROJECT_ROOT / "data" / "chroma")

# ---------------------------------------------------------------------------
# Embeddings and vector store (architecture §5.3-5.4)
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
#: all-MiniLM-L6-v2 output dimensionality.
EMBEDDING_DIM = 384
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "hdfc_mf_faq")

#: Which runtime executes the encoder. Both produce the same 384-d vector from
#: the same weights - measured cosine 1.0000000000, max elementwise difference
#: 1.0e-07, which is float32 rounding, so switching this cannot move a
#: similarity score across `SIMILARITY_FLOOR` and does not require re-ingesting.
#: `retrieve.checks` re-measures that equivalence on every run, so the two are
#: pinned rather than trusted.
#:
#:   onnx  - onnxruntime, no torch import. Measured peak RSS for a full
#:           `ask()` is 238.7 MB versus 579.7 MB for torch, because torch's
#:           runtime alone costs ~500 MB to evaluate a 22M-parameter model.
#:           This is the default because it is the difference between fitting a
#:           512 MB container and not fitting one.
#:   torch - sentence-transformers on torch. Kept as a fallback, not because it
#:           is needed for correctness but because it is the reference the ONNX
#:           path is checked against, and because it is the faster of the two
#:           for a large batch at ingest time.
#:
#: Chosen over `sentence_transformers(..., backend="onnx")`, which is NOT
#: equivalent: it requires `optimum` (not otherwise needed), downgraded
#: sentence-transformers 6.1.0 to 5.7.0, and still imported torch - measured at
#: 552.9 MB, i.e. *more* than plain torch. Going through onnxruntime directly
#: is the only version of this that actually avoids the 500 MB.
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "onnx").strip().lower()
if EMBEDDING_BACKEND not in {"onnx", "torch"}:
    raise ValueError(
        f"EMBEDDING_BACKEND must be 'onnx' or 'torch', got {EMBEDDING_BACKEND!r}"
    )

# ---------------------------------------------------------------------------
# Retrieval tuning (architecture §11)
# ---------------------------------------------------------------------------

#: Chunks handed to the generator. 3-5 is the documented band (architecture
#: §6.2); 5 is the measured choice, not the midpoint. At 4 the ELSS lock-in
#: fact ("3Y Lock-in") ranks #5 in its own scheme's pool and never reaches the
#: model - see the Phase 6 notes. Raise this to 8+ and the extra context mostly
#: adds rival numbers the model could quote by mistake.
TOP_K = int(os.getenv("TOP_K", "5"))
#: Max similarity below this -> weak retrieval -> refuse (architecture §6.3).
#: Measured, not guessed. Architecture §6.3 suggests starting at 0.35-0.45; the
#: Phase 6 sweep put three distinct groups in clearly separated bands:
#:
#:   0.13 - 0.33  off-topic junk ("is it good?", "give me market tips")
#:   0.41          on-topic but NOT in the corpus (the capital-gains question -
#:                 no chunk mentions capital gains, so the right answer is "I
#:                 don't have that")
#:   0.63 - 0.86  answerable PRD questions
#:
#: 0.45 sits inside the documented range and inside the measured 0.41-0.63 gap,
#: so it rejects junk and the corpus gap while keeping every real question. It
#: does NOT separate vague-but-on-topic ("tell me about HDFC funds", 0.80) from
#: specific questions - that is the mixed-scheme rule in retrieve/assemble.py,
#: and no similarity threshold can do it.
SIMILARITY_FLOOR = float(os.getenv("SIMILARITY_FLOOR", "0.45"))

#: Prior chat turns available to scheme resolution (see `growbot.memory`).
#: Not a prompt context window: history never reaches the generator, it only
#: fills in which fund an underspecified follow-up is about. Every entry is
#: screened for PII first, so the window is a window over safe entries.
#:
#: Capped rather than unbounded for predictability - an open-ended window makes
#: the answer depend on how long a tab has been open, which is a property worth
#: not having. 10 is comfortably more than the 1-2 a follow-up needs and small
#: enough that the scan is trivially cheap. Commented out of `.env.example` for
#: the same reason `TOP_K` is: `config.py` is the single source of truth.
MEMORY_TURNS = int(os.getenv("MEMORY_TURNS", "10"))

# ---------------------------------------------------------------------------
# Chunking (architecture §5.2)
# ---------------------------------------------------------------------------

#: Hard cap per chunk in characters (~80-120 MiniLM tokens).
CHUNK_SIZE = 512
#: Soft target; chunks below this get merged back into a neighbour.
CHUNK_MIN_SIZE = 400
#: Character overlap so a label/value pair is not cut in half.
CHUNK_OVERLAP = 80
#: A table block is never split mid-row. If one is larger than this (e.g. a
#: 300-row holdings list) it is divided at row boundaries instead, which still
#: honours the rule while avoiding a single giant chunk.
MAX_TABLE_CHARS = 1024

# ---------------------------------------------------------------------------
# HTTP fetching (Phase 1)
# ---------------------------------------------------------------------------

USER_AGENT = os.getenv(
    "GROWBOT_USER_AGENT", "growbot-class-demo/0.1 (educational RAG prototype)"
)
HTTP_TIMEOUT = 30.0
#: Guard against a single pathological page dominating the corpus.
MAX_DOC_CHARS = 200_000

# ---------------------------------------------------------------------------
# Generation (architecture §6.4, PRD §16 "Exact LLM (local vs API)")
# ---------------------------------------------------------------------------

#: OpenAI-compatible endpoints. Every free tier worth using speaks this dialect,
#: so one client shape covers them all and the provider only picks a default
#: base URL. Gemini's OpenAI-compatibility endpoint is included because it is
#: the most generous free tier for a demo.
#:
#: This is NOT part of the citation allowlist below. It is where prompts are
#: sent, which is the opposite of a URL the bot may show a user.
LLM_BASE_URLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
}

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()
LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "").strip()
#: An unknown provider with no explicit LLM_BASE_URL leaves this empty, which
#: surfaces as a clear configuration error rather than a request to nowhere.
LLM_BASE_URL = os.getenv(
    "LLM_BASE_URL", LLM_BASE_URLS.get(LLM_PROVIDER, "")
).strip()
#: 0 - this is a lookup, not a creative task, and a stray temperature above 0
#: is one more way for a number to drift from the retrieved text.
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "45"))
#: Three short sentences need ~60 tokens. This is deliberately far above that,
#: because a real provider call was measured returning finish_reason="length"
#: with the answer cut mid-sentence, and at low budgets the response carried no
#: content key at all. The headroom is cheap; a truncated answer shown to a user
#: is not recoverable. Raise this if a model reports truncation, lower it only
#: to cut cost on a paid tier.
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "800"))
#: Attempts per call, including the first. Not a nice-to-have: a measured ~50%
#: of calls to gemini-3.8-flash returned HTTP 503 "high demand", which is
#: transient by the provider's own wording, and Groq's free tier (the provider
#: `.env.example` ships) returns 429 once a minute's token allowance runs out.
#: Either way a single attempt makes the bot fail during a live demo.
LLM_MAX_ATTEMPTS = int(os.getenv("LLM_MAX_ATTEMPTS", "4"))
#: First backoff step in seconds; doubles per attempt, jittered, capped.
#: 4 attempts at 1.5s spans about 8s of waiting, which is long enough to cover
#: a demand spike and short enough not to look hung in a classroom.
LLM_RETRY_BACKOFF = float(os.getenv("LLM_RETRY_BACKOFF", "1.5"))
LLM_RETRY_BACKOFF_MAX = float(os.getenv("LLM_RETRY_BACKOFF_MAX", "20"))

#: Hard cap on answer length (architecture §6.4, "≤ 3 sentences").
MAX_ANSWER_SENTENCES = int(os.getenv("MAX_ANSWER_SENTENCES", "3"))


def llm_is_configured() -> bool:
    """True when there is enough in the environment to actually call a model."""
    return bool(LLM_API_KEY and LLM_MODEL and LLM_BASE_URL)


# ---------------------------------------------------------------------------
# The five schemes in scope (PRD §5.1)
# ---------------------------------------------------------------------------

#: Canonical `scheme_name` -> category, and the words a query might use.
#: Phase 6 resolves a user question to one of these names for metadata filtering.
SCHEMES: dict[str, dict[str, object]] = {
    "HDFC Large Cap Fund - Direct Growth": {
        "category": "large_cap",
        "aliases": ["hdfc large cap", "large cap fund", "hdfc large cap fund"],
    },
    "HDFC Flexi Cap Fund - Direct Growth": {
        "category": "flexi_cap",
        # Renamed from HDFC Equity Fund; both names are still in search traffic.
        "aliases": [
            "hdfc flexi cap",
            "hdfc equity fund",
            "flexi cap fund",
            "hdfc equity",
        ],
    },
    "HDFC ELSS Tax Saver Fund - Direct Plan Growth": {
        "category": "elss",
        "aliases": [
            "hdfc elss",
            "hdfc elss tax saver",
            "elss tax saver",
            "hdfc tax saver",
        ],
    },
    "HDFC Small Cap Fund - Direct Growth": {
        "category": "small_cap",
        "aliases": ["hdfc small cap", "small cap fund", "hdfc small cap fund"],
    },
    "HDFC Balanced Advantage Fund - Direct Growth": {
        "category": "hybrid",
        "aliases": [
            "hdfc balanced advantage",
            "balanced advantage fund",
            "hdfc balanced advantage fund",
        ],
    },
}

#: Longest alias first, so "hdfc large cap fund" wins over "hdfc large cap".
SCHEME_ALIASES: list[tuple[str, str]] = sorted(
    (
        (alias, name)
        for name, spec in SCHEMES.items()
        for alias in spec["aliases"]  # type: ignore[union-attr]
    ),
    key=lambda pair: len(pair[0]),
    reverse=True,
)

#: Value used by the AMFI guide chunks, which belong to no single scheme.
GENERAL_SCHEME = "general"


def detect_scheme(message: str) -> list[str]:
    """Return the in-scope schemes named in ``message``.

    Lives here rather than in ``guards/`` because both the guard and the Phase 6
    retriever need it, and architecture §6 keeps the retriever independent of
    the guard. It is pure alias matching over the tables above, so there is no
    guard policy hiding in it.

    Matching is substring-based against the longest-first alias list, so
    "hdfc flexi cap fund" is never truncated by "hdfc flexi cap". Aliases that
    map to the same scheme collapse to one entry, which is why "HDFC Equity
    Fund (Flexi Cap)" counts as a single scheme rather than an ambiguous two.
    """
    if not message:
        return []
    lowered = message.lower()
    found: list[str] = []
    for alias, name in SCHEME_ALIASES:
        if alias in lowered and name not in found:
            found.append(name)
    return found

# ---------------------------------------------------------------------------
# Refusal link allowlist (architecture §8)
# ---------------------------------------------------------------------------

#: Guards may only cite these. The model must never invent a URL.
#: AMFI's investor-education glossary ("Mutual Funds Sahi Hai").
EDU_LINK = os.getenv(
    "EDU_LINK", "https://www.mutualfundssahihai.com/en/glossary"
)

#: Where a "which fund performs best?" refusal points, per scheme.
FACTSHEET_LINKS: dict[str, str] = {
    "HDFC Large Cap Fund - Direct Growth": (
        "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth"
    ),
    "HDFC Flexi Cap Fund - Direct Growth": (
        "https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth"
    ),
    "HDFC ELSS Tax Saver Fund - Direct Plan Growth": (
        "https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth"
    ),
    "HDFC Small Cap Fund - Direct Growth": (
        "https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth"
    ),
    "HDFC Balanced Advantage Fund - Direct Growth": (
        "https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth"
    ),
}

# ---------------------------------------------------------------------------
# Product copy
# ---------------------------------------------------------------------------

DISCLAIMER = "Facts-only. No investment advice."
