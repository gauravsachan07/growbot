"""Phase 8 - orchestrator checks.

    python -m growbot.ask_checks

Assertion-based, non-zero exit on regression. **No API key and no network call
is required**: a counting stub stands in for the model, which is what makes the
central claim of this phase testable at all.

That claim is not "the bot answers well" - that is Phase 7's job, and it
depends on which model is configured. It is narrower and more important:

    a refusal never reaches the model.

So every refusal test asserts on a call counter, not just on the returned text.
"Should I buy?" returning a refusal *and* having called the model would be a
pass on the first assertion and a failure in production, and only the counter
tells the two apart.

Also covered: a missing index produces a friendly error rather than a
generated fact, and the pipeline never mutates the index.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from growbot import config
from growbot.ask import ask, ask_with_trace
from growbot.config import DISCLAIMER, EDU_LINK, FACTSHEET_LINKS, SOURCES_CSV
from growbot.generate.answer import (
    LLMError,
    LLMNotConfigured,
    LLMQuotaExceeded,
    LLMTruncated,
)
from growbot.generate.answer import generate as m_generate
from growbot.guards.intent import AnswerPayload
from growbot.ingest import index as index_mod
from growbot.retrieve.assemble import assemble
from growbot.retrieve.query import search

FACT_Q = "Expense ratio of HDFC Large Cap Fund Direct Growth?"

#: Every refusal the guard can produce, with the reason it must carry. These
#: are the PRD §12 Q8/Q9 cases plus the PII and empty paths.
GUARD_CASES = [
    ("Should I buy HDFC Large Cap Fund?", "advice"),
    ("Which fund has the highest returns?", "returns"),
    ("My PAN is ABCDE1234F, what is the expense ratio?", "pii"),
    ("My OTP is 482913, is that valid?", "pii"),
    ("", "empty"),
    ("   ", "empty"),
]

#: Questions retrieval is expected to refuse on their own merits.
WEAK_CASES = [
    "tell me about HDFC funds",
    "How to download a capital-gains statement?",
]

#: Refused for *some* valid reason, where which layer caught it is not the
#: point. "is it good?" is here because the guard reaches it as advice while
#: retrieval would have scored it as off-topic junk - both are correct, and
#: pinning a specific layer would make this a test of the guard's wording.
ANY_REASON_CASES = [
    "is it good?",
    "give me market tips",
]


class Checks:
    def __init__(self) -> None:
        self.passed = 0
        self.failures: list[str] = []

    def check(self, condition: bool, label: str, detail: str = "") -> bool:
        if condition:
            self.passed += 1
            return True
        self.failures.append(f"{label}{f' - {detail}' if detail else ''}")
        return False

    def equal(self, actual, expected, label: str) -> bool:
        return self.check(
            actual == expected, label, f"expected {expected!r}, got {actual!r}"
        )


class CountingStub:
    """Stands in for the model and records every call."""

    def __init__(self, reply: str = "The expense ratio is 1.03%.") -> None:
        self.reply = reply
        self.calls = 0

    def __call__(self, messages) -> str:
        self.calls += 1
        return self.reply


def _allowlisted(url: str) -> bool:
    """A refusal may cite the education link or a scheme factsheet - nothing else."""
    return url in (EDU_LINK, *FACTSHEET_LINKS.values())


class ChecksRunner:
    def run(self) -> int:
        checks = Checks()
        print("orchestrator checks (Phase 8)")
        print("=" * 74)
        print("  A counting stub stands in for the model, so these checks")
        print("  need no key and make no network call.")

        # --- 1. the central claim: refusals never reach the model -----
        print("\n  a refusal never calls the model")
        stub = CountingStub()
        for question, reason in GUARD_CASES:
            before = stub.calls
            payload, trace = ask_with_trace(question, complete=stub)
            called = stub.calls > before
            label = question or "(empty)"
            checks.equal(payload.mode, "refuse", f"refused: {label[:38]!r}")
            checks.equal(payload.reason, reason, f"reason: {label[:38]!r}")
            checks.check(not called, f"model NOT called: {label[:38]!r}",
                         f"{stub.calls - before} call(s)")
            checks.equal(trace.stopped_at, "guard", f"stopped at guard: {label[:30]!r}")
            checks.check(not trace.model_called, f"trace agrees: {label[:30]!r}")
        print(f"    {len(GUARD_CASES)} guard refusals, {stub.calls} model calls total")

        # --- 2. weak retrieval never reaches the model -----------------
        print("\n  weak retrieval never calls the model either")
        for question in WEAK_CASES:
            before = stub.calls
            payload, trace = ask_with_trace(question, complete=stub)
            called = stub.calls > before
            checks.equal(payload.mode, "refuse", f"refused: {question[:38]!r}")
            checks.check(payload.reason == "weak_retrieval",
                         f"reason is weak_retrieval: {question[:30]!r}",
                         payload.reason)
            checks.check(not called, f"model NOT called: {question[:38]!r}",
                         f"{stub.calls - before} call(s)")
            # A weak refusal is decided in assemble; `generate` is only
            # skipped. Reporting "generate" as the stopping point used to imply
            # the model had been asked and declined, which is the opposite of
            # what happened and made the demo docs contradict themselves.
            checks.equal(trace.stopped_at, "assemble",
                         f"stopped_at names the deciding stage: {question[:30]!r}")
        print(f"    {len(WEAK_CASES)} weak refusals, {stub.calls} model calls total")

        # --- 2b. refused for any valid reason, model still untouched ---
        print("\n  off-topic junk is refused, whichever layer catches it")
        for question in ANY_REASON_CASES:
            before = stub.calls
            payload, trace = ask_with_trace(question, complete=stub)
            checks.equal(payload.mode, "refuse", f"refused: {question!r}")
            checks.check(
                payload.reason in ("advice", "weak_retrieval"),
                f"reason is a refusal kind: {question!r}", payload.reason,
            )
            checks.check(not trace.model_called, f"model NOT called: {question!r}")
        print(f"    {len(ANY_REASON_CASES)} junk refusals, {stub.calls} total calls")

        # --- 3. the happy path does call the model ---------------------
        print("\n  an answerable question does reach the model")
        before = stub.calls
        payload, trace = ask_with_trace(FACT_Q, complete=stub)
        checks.equal(stub.calls, before + 1, "model called exactly once")
        checks.equal(payload.mode, "fact", "mode is fact")
        checks.check(trace.model_called, "trace records the call")
        checks.equal(trace.stopped_at, "generate", "stopped at generate")
        stages = [s.name for s in trace.stages]
        checks.equal(
            stages, ["guard", "index", "retrieve", "assemble", "generate"],
            "full pipeline ran in order",
        )
        print(f"    pipeline: {' -> '.join(stages)}")

        # --- 4. every refusal is well formed ---------------------------
        print("\n  every refusal carries a disclaimer and one allowlisted link")
        corpus_urls = _corpus_urls()
        for question, _ in GUARD_CASES:
            payload = ask(question, complete=stub)
            checks.check(DISCLAIMER in payload.text,
                         f"disclaimer present: {(question or '(empty)')[:30]!r}")
            checks.equal(payload.text.count("https://"), 0,
                         f"no URL inside the prose: {(question or '(empty)')[:26]!r}")
            checks.check(payload.source_url.startswith("https://"),
                         f"has a citation: {(question or '(empty)')[:30]!r}")
            checks.check(
                _allowlisted(payload.source_url),
                f"link is from the refusal allowlist: {payload.source_url[:48]}",
            )
        for question in WEAK_CASES:
            payload = ask(question, complete=stub)
            checks.check(DISCLAIMER in payload.text, "disclaimer present (weak)")
            checks.check(_allowlisted(payload.source_url),
                         f"allowlisted (weak): {payload.source_url[:48]}")
        # A fact answer's citation must be a real corpus URL, from sources.csv.
        fact = ask(FACT_Q, complete=stub)
        checks.check(
            fact.source_url in corpus_urls,
            "fact citation is listed in data/sources.csv",
            fact.source_url,
        )
        print(f"    refusals checked against {len(FACTSHEET_LINKS)} factsheets + edu link")
        print(f"    fact citation found in the {len(corpus_urls)}-row corpus list")

        # --- 5. a missing index is a friendly error -------------------
        print("\n  a missing index is an error, never a generated fact")
        # ignore_cleanup_errors: checking a non-existent path makes Chroma
        # create it and hold an open handle on chroma.sqlite3, which Windows
        # refuses to unlink. Worth knowing in production too - is_indexed()
        # creates the directory it is asked about, so a mistyped CHROMA_PATH
        # produces an empty store rather than a crash. The refusal is the same
        # either way, which is why the check tolerates it.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            empty = Path(tmp) / "no-index-here"
            saved = config.CHROMA_PATH
            config.CHROMA_PATH = empty
            try:
                # The real collection_count, against a directory that does not
                # exist. No patching of the function under test.
                checks.equal(index_mod.collection_count(empty), 0,
                             "a non-existent path really reports 0 records")
                before = stub.calls
                payload, trace = ask_with_trace(FACT_Q, complete=stub)
                called = stub.calls > before
                checks.equal(payload.mode, "refuse", "refused")
                checks.equal(payload.reason, "no_index", "reason is no_index")
                checks.check(not called, "model NOT called with no index")
                checks.check("growbot.ingest" in payload.text,
                             "tells the user how to fix it", payload.text[:70])
                checks.check(DISCLAIMER in payload.text, "disclaimer present")
                checks.equal(trace.stopped_at, "index", "stopped at index")
                checks.check(
                    not any(s.name == "retrieve" for s in trace.stages),
                    "retrieval never ran without an index",
                )
                print("    no_index: refused, 0 model calls, fixable instructions")
            finally:
                config.CHROMA_PATH = saved
        checks.check(
            index_mod.collection_count() > 0, "the real index still works",
        )

        # --- 6. the pipeline never writes to the index -----------------
        print("\n  the pipeline does not ingest or mutate the index")
        before_count = index_mod.collection_count()
        for question, _ in GUARD_CASES:
            ask(question, complete=stub)
        ask(FACT_Q, complete=stub)
        after_count = index_mod.collection_count()
        checks.equal(after_count, before_count, "record count unchanged")
        checks.check(
            after_count > 0, "the real index is still there", str(after_count)
        )
        print(f"    {before_count} records before, {after_count} after")

        # --- 7. ask() is the thin wrapper it claims to be -------------
        print("\n  ask() and ask_with_trace() agree")
        one = ask(FACT_Q, complete=stub)
        two, _ = ask_with_trace(FACT_Q, complete=stub)
        checks.equal(one, two, "same payload from both entry points")
        checks.check(
            isinstance(one, AnswerPayload),
            "both entry points return an AnswerPayload",
            type(one).__name__,
        )
        # The payload is frozen, so the UI cannot mutate a citation after the
        # fact - worth pinning, because every citation guarantee rests on it.
        checks.check(
            hasattr(one, "__dataclass_fields__"),
            "payload is a dataclass with the §7.3 fields",
        )
        for field_name in ("mode", "text", "source_url", "last_updated",
                           "scheme_name"):
            checks.check(
                hasattr(one, field_name), f"payload has {field_name}"
            )
        print("    identical payloads, same frozen shape")

        # --- 7. a provider failure degrades, it does not crash ---------
        # A UI renders whatever ask() returns. An exception escaping here
        # would surface as a traceback in the chat pane, so every provider
        # exception must become a payload. This was a real defect: a rate
        # limit that outlived the retries killed the whole REPL loop.
        print("\n  a provider failure degrades to a payload, never a crash")
        for exc, expected in (
            (LLMError("HTTP 429 rate limit"), "provider_error"),
            (LLMQuotaExceeded("allowance used up"), "provider_quota"),
            (LLMTruncated("cut off mid-sentence"), "provider_truncated"),
            (LLMNotConfigured("no LLM_API_KEY set"), "not_configured"),
        ):
            def boom(messages, _e=exc):
                raise _e

            try:
                payload, trace = ask_with_trace(FACT_Q, complete=boom)
            except Exception as escaped:  # noqa: BLE001 - that is the failure
                checks.check(False, f"{expected}: did not raise",
                             f"escaped {type(escaped).__name__}")
                continue
            checks.equal(payload.mode, "refuse", f"{expected}: refused")
            checks.equal(payload.reason, expected, f"{expected}: reason")
            checks.check(DISCLAIMER in payload.text, f"{expected}: disclaimer")
            checks.check(payload.source_url == EDU_LINK,
                         f"{expected}: cites the allowlisted edu link")
            checks.equal(trace.stopped_at, "generate", f"{expected}: stopped at generate")
            checks.check(
                not any(s.name == "retrieve" and s.outcome == "error"
                        for s in trace.stages),
                f"{expected}: retrieval itself was not the failure",
            )
        # generate() must still raise - the two layers keep different contracts.
        try:
            m_generate(FACT_Q, assemble(search(FACT_Q)),
                       complete=lambda msgs: (_ for _ in ()).throw(
                           LLMError("boom")))
            checks.check(False, "generate() still raises LLMError", "it swallowed")
        except LLMError:
            checks.check(True, "generate() still raises LLMError")
        print("    4 provider failures -> 4 payloads; generate() still raises")

        # --- report ---------------------------------------------------
        print()
        print("=" * 74)
        if checks.failures:
            print(f"FAILED  {len(checks.failures)} of "
                  f"{checks.passed + len(checks.failures)}")
            for failure in checks.failures:
                print(f"  - {failure}")
            return 1
        print(f"OK      {checks.passed} checks passed")
        return 0


def _corpus_urls() -> set[str]:
    """URLs listed in data/sources.csv - where a citation is allowed to point."""
    urls: set[str] = set()
    with SOURCES_CSV.open(encoding="utf-8-sig", newline="") as handle:
        for row in __import__("csv").DictReader(handle):
            url = (row.get("url") or "").strip()
            if url:
                urls.add(url)
    return urls


def main() -> int:
    return ChecksRunner().run()


if __name__ == "__main__":
    sys.exit(main())
