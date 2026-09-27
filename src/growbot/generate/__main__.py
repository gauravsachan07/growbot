"""Phase 7 - generation demo.

    python -m growbot.generate                      # the PRD §12 fact questions
    python -m growbot.generate "your question"      # one question
    python -m growbot.generate --offline            # no key: show the pipeline
                                                     # up to the LLM call, then
                                                     # report what is missing

Retrieval is real, so the chunks and the citation are genuine. The only thing
that needs a key is the model call itself.

With no key configured, this does not invent an answer. It prints the context
it would have sent, then says exactly which environment variables are missing -
which is the honest demonstration for a room without API access.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Sequence

from growbot.config import (
    DISCLAIMER,
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_PROVIDER,
    MAX_ANSWER_SENTENCES,
    llm_is_configured,
)
from growbot.generate.answer import (
    LLMNotConfigured,
    LLMQuotaExceeded,
    LLMTruncated,
    generate,
)
from growbot.generate.prompt import build_messages
from growbot.guards.intent import display_name
from growbot.retrieve.assemble import assemble
from growbot.retrieve.query import search

#: PRD §12 fact questions 1-6. Q7 is excluded: the corpus has no capital-gains
#: content, so it is a refusal by design and belongs in the retrieval sweep.
EVAL_QUESTIONS = [
    "Expense ratio of HDFC Large Cap Fund Direct Growth?",
    "Exit load of HDFC Small Cap Fund Direct Growth?",
    "Minimum SIP for HDFC Balanced Advantage Fund Direct Growth?",
    "Lock-in period for HDFC ELSS Tax Saver Direct Growth?",
    "Riskometer level of HDFC Equity Fund (Flexi Cap) Direct Growth?",
    "Benchmark of HDFC Large Cap Fund Direct Growth?",
]

_NOISY_LOGGERS = (
    "httpx", "httpcore", "urllib3", "chromadb", "sentence_transformers",
    "transformers", "huggingface_hub", "filelock", "onnxruntime", "tokenizers",
)


def _banner() -> None:
    print()
    print("=" * 74)
    if llm_is_configured():
        print(f"  LLM: {LLM_PROVIDER} / {LLM_MODEL}")
        print(f"  endpoint: {LLM_BASE_URL}")
    else:
        print("  LLM: not configured")
        print("  Copy .env.example to .env and set LLM_PROVIDER, LLM_API_KEY,")
        print("  LLM_MODEL. Any free tier works - see docs/architecture.md §6.4.")
    print("=" * 74)


def _show(question: str, offline: bool, verbose: bool) -> bool:
    """Answer one question. Returns False if the provider failed."""
    result = search(question)
    payload = assemble(result)

    print()
    print(f"Q  {question}")
    print("-" * 74)
    if result.detected_schemes:
        print(f"   scheme: {display_name(result.detected_schemes[0])}"
              f"   filtered={result.filtered}")
    print(f"   {len(result.hits)} chunks, max_similarity={payload.max_similarity:.3f}")

    if payload.weak:
        for reason in payload.weak_reasons:
            print(f"   weak: {reason}")

    if verbose:
        print()
        print("   --- context sent to the model ---")
        for line in payload.context.splitlines():
            print(f"   | {line}")
        print("   --- end context ---")

    if payload.weak:
        print("   => refused without calling the model (no key needed)")

    if offline or not llm_is_configured():
        if not payload.weak:
            print("   => would call the model; no key configured, so stopping here")
            print("      the citation and date are already fixed from metadata:")
            print(f"      {payload.source_url}")
            print(f"      as of {payload.last_updated}")
        return True

    try:
        answer = generate(question, payload)
    except LLMNotConfigured as exc:
        print(f"   => cannot generate: {exc}")
        return False
    except LLMQuotaExceeded as exc:
        # Not a Growbot fault and not transient, so it gets its own line
        # rather than a generic provider error.
        print(f"   => provider quota exhausted, not retried: {exc}")
        return False
    except LLMTruncated as exc:
        # Not a crash: the provider ran out of budget mid-sentence. Reported on
        # its own line because the remedy is a config value, not a retry.
        print(f"   => incomplete answer discarded ({exc})")
        return False
    except Exception as exc:  # noqa: BLE001 - a demo should not traceback
        print(f"   => provider error: {exc.__class__.__name__}: {exc}")
        return False

    if answer.mode == "fact":
        print(f"   => {answer.text}")
        print(f"      source: {answer.source_url}")
        print(f"      as of:  {answer.last_updated}")
    else:
        print(f"   => refused ({answer.reason})")
        print(f"      {answer.text}")
        print(f"      link: {answer.source_url}")
    return True


def _split_questions(words: Sequence[str]) -> list[str]:
    """Turn the positional args into one or more questions.

    An unquoted question arrives as several args ("what is the exit load"),
    so the default has to be "join them". But a user who quotes two questions
    clearly means two questions, and joining those silently produces one
    nonsense question that then gets refused for naming three schemes.

    The rule: if every arg looks like a finished question, treat them as
    separate questions; otherwise join. Predictable in both directions, and it
    never needs the user to remember to quote.
    """
    if not words:
        return []
    if len(words) > 1 and all(w.strip().endswith("?") for w in words):
        return [w.strip() for w in words]
    return [" ".join(words)]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m growbot.generate",
        description="Phase 7 - generate a grounded answer. Needs an LLM key.",
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="a question (quoting optional); quote several to ask several",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="do not call the model; show retrieval and the citation it fixed",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="print the context")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="  %(message)s",
        stream=sys.stdout,
    )
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.ERROR)

    _banner()
    questions = _split_questions(args.question) or EVAL_QUESTIONS
    offline = args.offline or not llm_is_configured()
    ok = 0
    for question in questions:
        ok += bool(_show(question, offline, args.verbose))
    failed = len(questions) - ok

    print()
    if offline:
        print(f"  {DISCLAIMER}")
        print()
        return 0

    # Non-zero when the provider failed, so a demo run is honest about whether
    # it actually answered. A refusal is a successful answer; an error is not.
    print(f"  {ok}/{len(questions)} answered"
          + (f", {failed} failed (provider or truncation)" if failed else ""))
    print(f"  {DISCLAIMER}")
    print()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
