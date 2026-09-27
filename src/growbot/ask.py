"""Phase 8 - the orchestrator: `ask(question) -> AnswerPayload` (architecture §9-10).

This is the single function the UI will call, and the only place the three
halves of the system are wired together:

    guard -> search -> assemble -> generate

Two properties matter more than anything else here.

**The guard runs first, and a refusal short-circuits.** "Should I buy HDFC
Large Cap?" must never reach a model, not because the model would refuse - it
would probably answer helpfully - but because answering it at all is the
failure. Phase 7 proved this by counting model calls; `python -m growbot.ask`
counts them again per process so it is visible in a demo.

**Ingestion is never invoked.** `ask()` reads the index or it reports that
there is no index. It never fetches a page. A chat turn cannot reach the
network except through the LLM call, which is the one place it is allowed to.

Each stage can refuse. The order is deliberate: the cheapest check that can
catch a problem runs first, so a PII message never gets embedded, and a weak
retrieval never gets billed to the provider.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from typing import Sequence

from growbot import config
from growbot.config import DISCLAIMER, EDU_LINK
from growbot.generate.answer import (
    LLMError,
    LLMNotConfigured,
    LLMQuotaExceeded,
    LLMTruncated,
)
from growbot.generate.answer import generate as _generate
from growbot.guards.intent import AnswerPayload, display_name, guard
from growbot.memory import describe, memory_query, resolve_scheme
from growbot.retrieve.assemble import assemble
from growbot.retrieve.query import search

log = logging.getLogger("growbot.ask")

__all__ = [
    "Stage",
    "Trace",
    "UIStatus",
    "ask",
    "ask_with_trace",
    "main",
    "status",
]


@dataclass(frozen=True)
class Stage:
    """One step of the pipeline, and what it decided."""

    name: str
    outcome: str
    detail: str = ""


@dataclass
class Trace:
    """What happened for a single question.

    Exists so the demo can *show* that a refusal never reached the model, and
    so the checks can assert it rather than trusting the prompt. `model_called`
    is the number that matters: it must be 0 for every guard refusal.
    """

    stages: list[Stage] = field(default_factory=list)

    def add(self, name: str, outcome: str, detail: str = "") -> None:
        self.stages.append(Stage(name, outcome, detail))

    @property
    def model_called(self) -> bool:
        return any(s.name == "generate" and s.outcome == "called" for s in self.stages)

    @property
    def stopped_at(self) -> str:
        """The stage that actually decided the outcome.

        Not simply the last stage. On a weak-retrieval refusal the last stage is
        `generate`, but `generate` was *skipped* - the decision was made one
        stage earlier when assembly found the context too thin. Reporting
        "generate" there is worse than useless in a demo: it implies the model
        was asked and declined, which is the opposite of what happened.

        So the deciding stage is the last one that refused or errored, falling
        back to the final stage when nothing refused (a normal answer).
        """
        for stage in reversed(self.stages):
            if stage.outcome in ("refused", "error"):
                return stage.name
        return self.stages[-1].name if self.stages else "none"

    def render(self) -> str:
        lines = []
        for stage in self.stages:
            mark = {"called": "->", "refused": "X", "ok": "ok",
                    "error": "!", "skipped": "-"}.get(stage.outcome, "?")
            detail = f"  {stage.detail}" if stage.detail else ""
            lines.append(f"      [{mark}] {stage.name:<9} {stage.outcome}{detail}")
        return "\n".join(lines)


def _no_index_payload() -> AnswerPayload:
    """Friendly error when the vector store is missing or empty.

    An error payload rather than an exception, and never a generated answer.
    The distinction matters: a user who has not run the ingest CLI should be
    told what to do, not shown a plausible fact from a model that was given no
    source material.
    """
    return AnswerPayload(
        mode="refuse",
        text=(
            "I have no source material loaded yet, so I can't answer anything "
            "without guessing. Build the index first: python -m growbot.ingest "
            f"(it fetches the pages listed in data/sources.csv). {DISCLAIMER}"
        ),
        source_url=EDU_LINK,
        last_updated="",
        reason="no_index",
    )


def ask_with_trace(
    question: str,
    *,
    complete=None,
    check_index: bool = True,
    history: Sequence[str] | None = None,
) -> tuple[AnswerPayload, Trace]:
    """`ask()`, plus the trace of how the answer was reached.

    `complete` is the Phase 7 injection point for tests: pass a stub and the
    model is never contacted, so the guard and retrieval paths can be checked
    with no key at all.

    `history` is earlier turns of the conversation - prior **question** texts,
    most recent last. It exists for one purpose: to work out which fund an
    underspecified follow-up like "what about its exit load?" is about. See
    `growbot.memory` for the rules and, more importantly, for what memory is
    not allowed to do.

    Three properties are worth stating plainly, because each is a way this
    could quietly go wrong:

    - **The guard still sees only the current question.** History is never
      classified, so a PII message buried three turns back cannot trigger a
      refusal of an innocent follow-up.
    - **History never reaches the generator.** `_generate()` is called with
      `question`, not the memory-augmented retrieval text, so the prompt is
      unchanged and the grounding check still sees only chunks retrieved for
      the question actually asked.
    - **With no history, behaviour is unchanged.** The memory stage is not even
      recorded, so the trace for a single-turn call looks exactly as it did.
    """
    trace = Trace()

    question = (question or "").strip()
    if not question:
        trace.add("guard", "refused", "empty message")
        return refuse_empty(), trace

    # 1. Guard. No I/O, no vector store, no network.
    refusal = guard(question)
    if refusal is not None:
        trace.add("guard", "refused", refusal.reason)
        log.info("guard refused (%s); model not called", refusal.reason)
        return refusal, trace
    trace.add("guard", "ok", "allowed")

    # 2. The index must exist and hold something. Checked before retrieval so
    #    the failure reads as "you forgot to ingest", not as a broken search.
    #
    #    The path is passed explicitly rather than relying on index.py's
    #    default. A default argument binds CHROMA_PATH at import time, so a
    #    caller that reads config at call time and then hands over the path can
    #    be tested against a real empty directory - which is the only honest way
    #    to test the "you forgot to ingest" path.
    if check_index:
        from growbot.ingest.index import collection_count, is_indexed

        path = config.CHROMA_PATH
        try:
            ready = is_indexed(path)
            count = collection_count(path) if ready else 0
        except Exception as exc:  # noqa: BLE001 - any store failure is "no index"
            log.warning("index check failed: %s", exc.__class__.__name__)
            ready, count = False, 0
        if not ready or count == 0:
            found = "empty" if ready else "missing"
            trace.add("index", "error", f"{found} ({count} records)")
            return _no_index_payload(), trace
        trace.add("index", "ok", f"{count} records")

    # 3. Resolve which fund this is about, then retrieve and assemble.
    #    Read-only: opens the index, never writes.
    #
    #    Without history this is exactly what it always was - `detected=[]`
    #    because `detect_scheme` found nothing, which the assembler then treats
    #    as ambiguous. The `[]` is not the same as omitting the argument, and
    #    that difference is the whole point: omitting it would re-detect from
    #    the text, which for a question that named no fund gives the same
    #    answer, so the two agree here and only diverge when memory filled a
    #    gap. Passing it explicitly keeps one source of truth for the scheme.
    schemes, source = resolve_scheme(question, history)
    if history:
        trace.add("memory", "ok", describe(question, history))

    query_text = memory_query(question, schemes, source)
    result = search(query_text, detected=schemes)
    assembly = assemble(result)
    trace.add(
        "retrieve",
        "ok",
        f"{len(result.hits)} chunks, max_sim={assembly.max_similarity:.3f}"
        + (f", filtered={result.filtered}" if result.filtered else "")
        + (f", scheme from {source}" if source else ""),
    )

    # 4. Weak context refuses inside generate(), before the model is called.
    #    Recorded here so the trace shows where it stopped.
    if assembly.weak:
        trace.add("assemble", "refused", ", ".join(assembly.weak_reasons))
    else:
        trace.add("assemble", "ok", "context strong enough")

    # 5. Generate. The only stage allowed to leave the machine.
    #
    #    `generate()` raises on a provider failure; `ask()` must not. The two
    #    have different jobs - generate() is the tested unit and an exception
    #    is the honest result there, while ask() is what a UI renders, and a
    #    traceback in the chat pane is a worse outcome than an honest "I
    #    couldn't get an answer for that". So the exception is caught here and
    #    turned into a payload, and only here.
    #
    #    Note LLMNotConfigured does not inherit from LLMError, so both are
    #    caught. A missing key is a configuration state, not a provider fault.
    if assembly.weak:
        trace.add("generate", "skipped", "weak context, model not called")
        return _generate(question, assembly, complete=complete), trace

    trace.add("generate", "called")
    try:
        payload = _generate(question, assembly, complete=complete)
    except (LLMError, LLMNotConfigured) as exc:
        failure = _provider_failure_payload(exc)
        trace.stages[-1] = Stage("generate", "error", failure.reason)
        log.warning("provider failure, not raised to the UI: %s", failure.reason)
        return failure, trace
    if payload.mode == "refuse":
        trace.stages[-1] = Stage("generate", "called", f"refused: {payload.reason}")

    return payload, trace


def _provider_failure_payload(exc: Exception) -> AnswerPayload:
    """Turn a provider exception into a payload the UI can render.

    The message says what broke and offers the route that still works. What it
    deliberately does not do is retry more, or guess. A rate limit has already
    been retried inside the client by this point.
    """
    if isinstance(exc, LLMQuotaExceeded):
        reason = "provider_quota"
        text = (
            "I reached the model provider but it refused the call because this "
            "API key's allowance is used up, so there's no answer for me to "
            "give you. This is a quota limit on the provider's side, not "
            f"something wrong with your question. {DISCLAIMER}"
        )
    elif isinstance(exc, LLMTruncated):
        reason = "provider_truncated"
        text = (
            "The provider cut the response off mid-sentence, so I discarded "
            "the partial answer rather than show you half of it. Raising "
            "LLM_MAX_TOKENS in .env usually fixes this. "
            f"{DISCLAIMER}"
        )
    elif isinstance(exc, LLMNotConfigured):
        reason = "not_configured"
        text = (
            "I have no language-model key configured, so I can't write an "
            "answer from the source material I retrieved. Set LLM_API_KEY and "
            "LLM_MODEL in .env to enable answers. "
            f"{DISCLAIMER}"
        )
    else:
        reason = "provider_error"
        text = (
            f"I couldn't reach the model provider just now ({str(exc)[:110]}). "
            "I'd rather say that than answer from memory, so nothing was "
            f"guessed. Please try again shortly. {DISCLAIMER}"
        )
    return AnswerPayload(
        mode="refuse",
        text=text,
        source_url=EDU_LINK,
        last_updated="",
        reason=reason,
    )


def ask(question: str, **kwargs) -> AnswerPayload:
    """The one function the UI calls. See `ask_with_trace` for the details."""
    payload, _ = ask_with_trace(question, **kwargs)
    return payload


@dataclass(frozen=True)
class UIStatus:
    """Readiness of the two things `ask()` depends on. Phase 9 shows this.

    The UI needs to say "you haven't run the ingest CLI" and "you have no LLM
    key" *before* someone clicks a question, rather than discovering it from a
    refusal. But architecture §9 says the UI must not touch Chroma or the LLM
    directly, so the check lives here, behind the same module boundary, and
    performs no retrieval and no generation.

    Every field is a boolean or a count. No user text, no keys.
    """

    index_ready: bool
    record_count: int
    llm_ready: bool
    provider: str
    model: str

    @property
    def ready(self) -> bool:
        return self.index_ready and self.llm_ready


def status() -> UIStatus:
    """Report readiness. Cheap, read-only, and answers no questions."""
    from growbot.ingest.index import collection_count, is_indexed

    try:
        count = collection_count(config.CHROMA_PATH)
        ready = count > 0 and is_indexed(config.CHROMA_PATH)
    except Exception:  # noqa: BLE001 - any store problem means "not ready"
        count, ready = 0, False
    return UIStatus(
        index_ready=ready,
        record_count=count,
        llm_ready=config.llm_is_configured(),
        provider=config.LLM_PROVIDER,
        model=config.LLM_MODEL or "(unset)",
    )


def refuse_empty() -> AnswerPayload:
    """Nothing was asked, so there is nothing to refuse and nothing to say."""
    return AnswerPayload(
        mode="refuse",
        text=(
            "I didn't get a question. Ask me about expense ratio, exit load, "
            "minimum SIP, lock-in period, benchmark, or riskometer level for "
            f"the five HDFC Direct Growth schemes. {DISCLAIMER}"
        ),
        source_url=EDU_LINK,
        last_updated="",
        reason="empty",
    )


# ---------------------------------------------------------------------------
# Terminal front end
# ---------------------------------------------------------------------------
# Lives here rather than in a separate module because `python -m growbot.ask`
# works for a module as well as a package, and a package named `ask` cannot
# coexist with a module named `ask`. The REPL is the point of this phase - it
# is how the bot gets *used*, and how the trace above becomes visible.

import argparse  # noqa: E402 - kept beside the front end that needs it

_NOISY_LOGGERS = (
    "httpx", "httpcore", "urllib3", "chromadb", "sentence_transformers",
    "transformers", "huggingface_hub", "filelock", "onnxruntime", "tokenizers",
)

_HELP = """\
Ask about: expense ratio, exit load, minimum SIP, lock-in period, benchmark,
riskometer level, fund category, or the AMC - for the five HDFC Direct Growth
schemes. A recommendation, a return figure, or anything containing personal data
is refused. Type 'exit' to quit.
"""

_STUB_ANSWER = (
    "[stub] The model was not called. This stands in for a generated answer so "
    "you can see the citation and date that retrieval already fixed."
)


def _configure_logging(quiet: bool) -> None:
    logging.basicConfig(
        level=logging.WARNING if quiet else logging.INFO,
        format="  %(message)s",
        stream=sys.stdout,
    )
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.ERROR)


def _print_answer(payload: AnswerPayload, trace: Trace, show_trace: bool) -> None:
    marker = "FACT" if payload.mode == "fact" else "REFUSE"
    print()
    print(f"  [{marker}] {payload.text}")
    if payload.scheme_name:
        print(f"         {display_name(payload.scheme_name)}")
    if payload.source_url:
        print(f"         {'source' if payload.mode == 'fact' else 'link'}: "
              f"{payload.source_url}")
    if payload.last_updated:
        print(f"         as of:  {payload.last_updated}")
    if show_trace:
        print()
        print("  pipeline:")
        print(trace.render())
    print()


def _ask_one(
    question: str,
    show_trace: bool,
    stub: bool,
    history: Sequence[str] | None = None,
) -> bool:
    """Answer one question. Returns False when it could not be answered.

    No exception handling here on purpose: ask() has already converted every
    provider failure into a payload, so a traceback at this point would mean a
    bug in Growbot rather than a problem with the provider. The bare `except`
    is kept so an interactive session survives even that, but it is the last
    line, not the first.
    """
    complete = (lambda messages: _STUB_ANSWER) if stub else None
    try:
        payload, trace = ask_with_trace(
            question, complete=complete, history=history
        )
    except Exception as exc:  # noqa: BLE001 - a REPL should not traceback
        print()
        print(f"  [BUG] {exc.__class__.__name__}: {exc}")
        print("       This is a Growbot error, not a provider one.")
        print()
        return False
    _print_answer(payload, trace, show_trace)
    return payload.reason not in ("no_index", "empty")


def _repl(show_trace: bool, stub: bool) -> int:
    print()
    print("=" * 74)
    if stub:
        print("  Mode: STUB - no model is called; answers are placeholders.")
    elif config.llm_is_configured():
        print(f"  Mode: live  - {config.LLM_PROVIDER} / {config.LLM_MODEL}")
    else:
        print("  Mode: no LLM key - every refusal path still works, fact")
        print("         answers will not. Set LLM_API_KEY and LLM_MODEL in .env.")
    print("=" * 74)
    print(_HELP)

    asked = 0
    # Prior questions, most recent last. Held here so a follow-up can be
    # resolved without retyping the fund's name, and bounded by the same
    # config.MEMORY_TURNS the library uses - the window is the library's rule,
    # not a second one that could drift away from it.
    history: list[str] = []
    while True:
        try:
            question = input("\n  you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print()
            break
        _ask_one(question, show_trace, stub, history=history)
        history.append(question)
        # Same offset-from-length form as `memory.screen`, for the same reason:
        # `history[:-0]` would keep everything.
        del history[:len(history) - config.MEMORY_TURNS]
        asked += 1

    print(f"  {asked} question(s) asked. {config.DISCLAIMER}")
    print()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m growbot.ask",
        description="Phase 8 - ask Growbot a question.",
    )
    parser.add_argument("question", nargs="*", help="ask once, then exit")
    parser.add_argument(
        "-t", "--trace", action="store_true",
        help="show which pipeline stage answered or refused",
    )
    parser.add_argument(
        "--stub", action="store_true",
        help="never call the model; placeholders stand in for answers",
    )
    parser.add_argument("--quiet", action="store_true", help="less logging")
    args = parser.parse_args(argv)

    _configure_logging(args.quiet)

    if args.question:
        return 0 if _ask_one(" ".join(args.question), args.trace, args.stub) else 1
    return _repl(args.trace, args.stub)


if __name__ == "__main__":
    sys.exit(main())
