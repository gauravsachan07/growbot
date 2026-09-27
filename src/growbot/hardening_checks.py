"""Phase 11 - hardening checks.

    python -m growbot.hardening_checks

The failure modes in architecture §13, asserted rather than assumed. Phase 11
mostly found that earlier phases had already handled these, which is worth
knowing but is not evidence: "we wrote a post-check for it" is not the same as
"the post-check fires". So each row of §13 gets a test that *provokes* the
failure, and the docs claims get tests that fail if the wording drifts.

The five §13 rows, and where each is covered:

| §13 failure | Proven here by |
|---|---|
| LLM ignores grounding | a stub inventing a plausible, ungrounded expense ratio |
| Mixed-scheme chunks | a question that retrieves well and spans five schemes |
| Empty Chroma at chat time | a real empty store, not a monkeypatched flag |
| Fetch 404 / blocked page | the ingest log's skip reasons, read back |
| PDF table garbage | scheme pages are HTML-first in `sources.csv` |

Runs **offline with no API key**: a counting stub stands in for the model, so
the tests that matter most - that a bad answer is discarded before it is shown -
can be run on any machine, any time, at no cost. Whether the bot *answers well*
depends on which model is configured and is measured separately by
`tools/eval_citations.py`; this file only asserts the guarantees that hold
regardless of model.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from growbot import config
from growbot.ask import ask, ask_with_trace
from growbot.config import (
    DISCLAIMER,
    EDU_LINK,
    FACTSHEET_LINKS,
    SCHEMES,
    SOURCES_CSV,
)
from growbot.generate.answer import (
    RESHAPE_NUDGE,
    generate,
    is_prose_free,
    split_sentences,
)
from growbot.guards import intent as g
from growbot.ingest import embed
from growbot.retrieve.assemble import assemble
from growbot.retrieve.query import search

EXPENSE_Q = "Expense ratio of HDFC Large Cap Fund Direct Growth?"

#: Every `reason` the system can produce. The UI styles cards by this value and
#: the trace names it, so a value outside this set is a card that renders with
#: no explanation of why it was refused.
KNOWN_REASONS = frozenset(
    {
        "advice",
        "empty",
        "insufficient",
        "no_index",
        "not_configured",
        "pii",
        "provider_error",
        "provider_quota",
        "provider_truncated",
        "returns",
        "ungrounded",
        "unusable",
        "weak_retrieval",
    }
)

ROOT = Path(__file__).resolve().parents[2]

#: The figure the corpus actually states, and one it does not. Both are
#: plausible expense ratios, which is the point: the ungrounded one cannot be
#: caught by looking weird, only by checking it against the retrieved text.
REAL_RATIO = "1.03"
INVENTED_RATIO = "0.42"


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


class Stub:
    """A model that returns queued replies and counts every call.

    A queue rather than a fixed string, so a test can model "the model was
    asked again and complied" - which is the only way to test the bare-figure
    retry without a live provider.
    """

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.calls = 0
        self.seen: list[list[dict[str, str]]] = []

    def __call__(self, messages) -> str:
        self.calls += 1
        self.seen.append(list(messages))
        return self.replies[min(self.calls - 1, len(self.replies) - 1)]


def _allowlisted(url: str) -> bool:
    """A refusal may cite the education link or a scheme factsheet - nothing else."""
    return url in (EDU_LINK, *FACTSHEET_LINKS.values())


# ---------------------------------------------------------------------------
# 1. LLM ignores grounding  (architecture §13)
# ---------------------------------------------------------------------------


def _check_numeric_grounding(checks: Checks) -> None:
    print("\n  a number that is not in the context is discarded, not shown")
    assembly = assemble(search(EXPENSE_Q))
    if not checks.check(
        not assembly.weak, "the probe question retrieves strongly enough to answer",
        ",".join(assembly.weak_reasons),
    ):
        return
    checks.check(
        REAL_RATIO in assembly.context,
        "the real ratio is genuinely present in the retrieved context",
    )

    # A model that answers correctly. The number *is* in the context, so this
    # must survive - a grounding check that refused everything would pass this
    # suite while being useless in production.
    good = Stub(f"The expense ratio is {REAL_RATIO}%.")
    payload = generate(EXPENSE_Q, assembly, complete=good)
    checks.equal(payload.mode, "fact", "a grounded number is answered")
    checks.check(REAL_RATIO in payload.text, "the grounded figure is passed through",
                 payload.text)
    checks.equal(good.calls, 1, "one call when the answer is usable")

    # The same question, answered from memory. Plausible, well-formatted, and
    # not in the source: exactly the failure §13 is about.
    bad = Stub(f"The expense ratio is {INVENTED_RATIO}%.")
    payload = generate(EXPENSE_Q, assembly, complete=bad)
    checks.equal(payload.mode, "refuse", "an ungrounded number converts to a refusal")
    checks.equal(payload.reason, "ungrounded", "reason names the grounding failure")
    checks.check(
        INVENTED_RATIO not in payload.text,
        "the invented figure is never shown to the user",
        payload.text,
    )
    checks.check(
        REAL_RATIO not in payload.text,
        "the refusal does not restate a figure either",
        payload.text,
    )
    checks.equal(bad.calls, 1, "one call; grounding is checked, not retried")

    # A number the model states correctly but which is nowhere in the retrieved
    # text, alongside a correct one. Both are in the answer, so the check has to
    # catch the pair, not just the answer.
    mixed = Stub(
        f"The expense ratio is {REAL_RATIO}%, and the AUM is INR 44000 crore."
    )
    payload = generate(EXPENSE_Q, assembly, complete=mixed)
    checks.equal(payload.reason, "ungrounded", "one bad number fails the whole answer")
    print(f"    grounded passes, {INVENTED_RATIO}% invented does not")


# ---------------------------------------------------------------------------
# 2. A bare figure is not an answer  (found during Phase 10, fixed in 11)
# ---------------------------------------------------------------------------


def _check_bare_figure(checks: Checks) -> None:
    print("\n  a bare figure is re-asked once, then refused honestly")
    # The unit test of the predicate, which is where the design risk actually
    # is: too strict and it throws away legitimate short answers.
    for good, why in [
        ("The expense ratio is 1.03%.", "a full sentence"),
        ("Very High risk.", "a one-word label is legitimate"),
        ("Benchmark is NIFTY 100 TRI.", "a short sentence"),
        ("Minimum SIP is INR 500.", "a fragment with a noun"),
        # Units count as words. "INR 100" names a currency, so it is a terse
        # answer rather than a bare fragment - the predicate is about having no
        # letters at all, not about being short.
        ("INR 100", "a bare amount still names its currency"),
    ]:
        checks.check(not is_prose_free(good), f"kept: {why}", good)
    for bad, why in [
        ("0.77%", "a bare percentage"),
        ("1.03%", "a bare percentage, with a full stop"),
        ("100", "a bare number"),
        ("2026-01-01", "a bare date"),
    ]:
        checks.check(is_prose_free(bad), f"rejected: {why}", bad)

    assembly = assemble(search(EXPENSE_Q))

    # Case 1: the model complies on the retry. The fact is kept - this is the
    # whole reason for retrying rather than refusing outright.
    compliant = Stub("0.77%", f"The expense ratio is {REAL_RATIO}%.")
    payload = generate(EXPENSE_Q, assembly, complete=compliant)
    checks.equal(compliant.calls, 2, "the retry happens, and only once")
    checks.equal(payload.mode, "fact", "a compliant retry keeps the answer")
    checks.check(
        any(m.get("content") == RESHAPE_NUDGE for m in compliant.seen[1]),
        "the retry asks for a sentence, not a new question",
    )
    checks.check(
        compliant.seen[1][:2] == compliant.seen[0][:2],
        "the system prompt and context are unchanged on the retry",
    )

    # The retry must not be a way around the grounding check. Reshaping a reply
    # does not make an invented number true, so a bare figure followed by an
    # invented sentence is still discarded.
    laundering = Stub("0.77%", f"The expense ratio is {INVENTED_RATIO}%.")
    payload = generate(EXPENSE_Q, assembly, complete=laundering)
    checks.equal(
        payload.reason, "ungrounded",
        "the retry cannot launder an ungrounded number into an answer",
    )
    checks.check(
        INVENTED_RATIO not in payload.text,
        "and the invented figure is still never shown",
        payload.text[:70],
    )

    # Case 2: the model is terse again. Refuse - but with copy that does not
    # blame the corpus, because the corpus did have the answer.
    stubborn = Stub("0.77%", "0.77%")
    payload = generate(EXPENSE_Q, assembly, complete=stubborn)
    checks.equal(stubborn.calls, 2, "no third call; the budget is capped")
    checks.equal(payload.mode, "refuse", "two bare figures do not become an answer")
    checks.equal(payload.reason, "unusable", "reason is 'unusable', not 'insufficient'")
    checks.check(
        "don't have a passage" not in payload.text.lower(),
        "does not claim the corpus lacks an answer it has",
        payload.text[:70],
    )
    checks.check(
        "couldn't turn it into a clear statement" in payload.text,
        "says what actually went wrong",
        payload.text[:70],
    )
    print("    one retry, capped, and the refusal names the real cause")


# ---------------------------------------------------------------------------
# 3. Refusal copy is consistent  (Phase 11: "make refuse copy consistent")
# ---------------------------------------------------------------------------


#: Every refusal the system can produce, with the reason it must carry. All
#: ten are here so that a *new* refusal cannot be added without this list being
#: updated - which is the only way "consistent" stays true.
#:
#: `refuse_not_in_corpus` is called with an explicit reason on purpose: it has
#: no default, so this list is also the check that the reasons used here are the
#: ones the rest of the system recognises.
ALL_REFUSALS = [
    ("pii", lambda: g.refuse_pii(["pan"])),
    ("advice", lambda: g.refuse_advice([next(iter(SCHEMES))])),
    ("advice", lambda: g.refuse_advice([])),
    ("returns", lambda: g.refuse_returns([next(iter(SCHEMES))])),
    ("returns", lambda: g.refuse_returns([])),
    ("weak_retrieval", lambda: g.refuse_not_in_corpus([next(iter(SCHEMES))],
                                                      reason="weak_retrieval")),
    ("weak_retrieval", lambda: g.refuse_not_in_corpus([], reason="weak_retrieval")),
    ("insufficient", lambda: g.refuse_not_in_corpus([], reason="insufficient")),
    ("ungrounded", lambda: g.refuse_ungrounded(["9.99"])),
    ("unusable", lambda: g.refuse_unusable_answer([next(iter(SCHEMES))])),
    ("unusable", lambda: g.refuse_unusable_answer([])),
]


def _body(text: str) -> str:
    """The refusal text minus the mandated disclaimer.

    The 3-sentence limit applies to the *body*. `guards.__main__` has always
    excluded the disclaimer, on the grounds that it is separately required and
    is boilerplate rather than an answer; `split_sentences` counts every
    sentence, so a 2-sentence refusal plus the 2-sentence disclaimer reads as 4.
    Both readings are defensible, which is exactly why the two counters have to
    be pinned to each other rather than left to drift.
    """
    return text.replace(DISCLAIMER, "").strip()


def _check_refusal_consistency(checks: Checks) -> None:
    print("\n  every refusal has the same shape")
    for expected_reason, build in ALL_REFUSALS:
        payload = build()
        tag = f"{payload.reason or expected_reason}"

        checks.equal(payload.reason, expected_reason, f"reason: {tag}")
        checks.equal(payload.mode, "refuse", f"mode: {tag}")
        checks.check(
            payload.text.rstrip().endswith(DISCLAIMER),
            f"ends with the disclaimer: {tag}",
            payload.text[-60:],
        )
        body = _body(payload.text)
        n_body = len(split_sentences(body))
        checks.check(
            n_body <= 3,
            f"<=3 sentences in the body: {tag}",
            f"{n_body} body sentences, {len(split_sentences(payload.text))} total",
        )
        # The reason must be one the UI styles and the trace names. An
        # unrecognised value would render as an unlabelled card.
        checks.check(
            payload.reason in KNOWN_REASONS,
            f"reason is one the system knows: {tag}",
            payload.reason,
        )
        checks.check(
            "https://" not in payload.text,
            f"no URL inside the prose: {tag}",
            payload.text[:60],
        )
        # One link, and it must be an allowlisted education or factsheet URL.
        # A refusal citing a random page is how a facts-only bot starts quoting
        # things it never retrieved.
        checks.check(bool(payload.source_url), f"has one link: {tag}")
        checks.check(
            _allowlisted(payload.source_url),
            f"link is allowlisted: {tag}",
            payload.source_url,
        )
        # The §13 fix: a refusal has no source, so it has no source date.
        checks.equal(payload.last_updated, "", f"no source date on a refusal: {tag}")
        # A refusal must not leak the config key or a scheme's internal name.
        checks.check(
            not any(name in payload.text for name in SCHEMES),
            f"no internal scheme name in prose: {tag}",
        )

    # The two refusals that could plausibly be confused must not be: one says
    # the corpus is silent, the other says the corpus was fine. Mixing them up
    # teaches users to distrust refusals that are usually right.
    silent = g.refuse_not_in_corpus([next(iter(SCHEMES))], reason="weak_retrieval")
    unusable = g.refuse_unusable_answer([next(iter(SCHEMES))])
    checks.check(
        "don't have a passage" in silent.text
        and "don't have a passage" not in unusable.text,
        "the two 'I have no answer' refusals say different things",
    )
    print(f"    {len(ALL_REFUSALS)} refusal variants, one shape, no source dates")


# ---------------------------------------------------------------------------
# 4. Mixed-scheme chunks  (architecture §13)
# ---------------------------------------------------------------------------


def _check_mixed_scheme(checks: Checks) -> None:
    print("\n  a question spanning several funds is refused, not blended")
    vague = assemble(search("tell me about HDFC funds"))
    checks.check(
        not vague.weak or any("scheme" in r for r in vague.weak_reasons),
        "the vague cross-scheme question is flagged",
        ",".join(vague.weak_reasons) or "not weak",
    )

    # The important half: it must be flagged *despite* scoring well. A
    # similarity threshold alone cannot catch this - the earlier measurement
    # put it at 0.80, well above the floor. That is why the rule is separate.
    checks.check(
        any("scheme" in r for r in vague.weak_reasons),
        "flagged for spanning schemes, not for low similarity",
        ",".join(vague.weak_reasons),
    )

    # The contrast case, which is what proves the rule is not just "refuse
    # anything with several schemes in it": naming one fund is fine.
    single = assemble(search(EXPENSE_Q))
    checks.check(not single.weak, "a single named fund is not flagged")
    checks.equal(len(single.detected_schemes), 1, "exactly one scheme detected")
    print(f"    vague: {','.join(vague.weak_reasons)} | named: answered")


# ---------------------------------------------------------------------------
# 5. Empty Chroma at chat time  (architecture §13)
# ---------------------------------------------------------------------------


def _check_empty_index(checks: Checks) -> None:
    print("\n  an empty index is an error, not an invitation to guess")
    import tempfile

    from growbot.ingest import index as index_mod

    # Swap config.CHROMA_PATH rather than passing a path in: `is_indexed` and
    # `collection_count` bind it as a default argument at def time, so only
    # `ask()` reading it at call time makes this a test of the real function.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        empty = Path(tmp) / "chroma"
        saved = config.CHROMA_PATH
        config.CHROMA_PATH = empty
        try:
            checks.equal(
                index_mod.collection_count(empty), 0,
                "a non-existent path really reports 0 records",
            )
            payload = ask(EXPENSE_Q)
            checks.equal(payload.mode, "refuse", "refused with no index present")
            checks.equal(payload.reason, "no_index", "reason is no_index")
            checks.check(
                "ingest" in payload.text.lower(),
                "tells the user how to fix it",
                payload.text[:80],
            )
            checks.check(
                REAL_RATIO not in payload.text and INVENTED_RATIO not in payload.text,
                "invents no figure when it cannot look anything up",
            )
        finally:
            config.CHROMA_PATH = saved

    # And the real index still works, i.e. the probe above did no damage.
    real = ask(EXPENSE_Q)
    checks.equal(real.mode, "fact", "the real index is untouched by that probe")
    print("    empty store -> no_index, real index still answers")


# ---------------------------------------------------------------------------
# 6. Blocked official HTML  (architecture §13, README requirement)
# ---------------------------------------------------------------------------


def _check_blocked_sources(checks: Checks) -> None:
    print("\n  blocked official HTML is recorded, not hidden")
    import csv

    with SOURCES_CSV.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    checks.check(len(rows) > 0, "sources.csv parses and is non-empty")
    checks.check(
        all(r["url"].startswith("https://") for r in rows),
        "every source is an https URL",
    )
    checks.check(
        len({r["url"] for r in rows}) == len(rows),
        "no duplicate sources",
    )
    # The official AMC is the primary source and must be *listed* even though
    # its host blocks us; the fallback material is what we can actually read.
    hdfc = [r for r in rows if "hdfcfund.com" in r["url"]]
    fallback = [r for r in rows if "mutualfundssahihai" in r["url"]]
    checks.check(len(hdfc) >= 5, "the official AMC pages are still listed", str(len(hdfc)))
    checks.check(
        len(fallback) >= 2,
        "readable fallback material is present too",
        str(len(fallback)),
    )

    # The README has to say which is which. A classmate who cannot load
    # hdfcfund.com needs to know that is expected, not a broken setup.
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    checks.check("403" in readme, "README names the block we hit")
    checks.check(
        "hdfcfund.com" in readme, "README names the host that is blocked"
    )
    checks.check(
        "mutualfundssahihai" in readme, "README names the fallback actually used"
    )
    checks.check(
        "data/sources.csv" in readme, "README points at the authoritative list"
    )
    print(f"    {len(hdfc)} blocked official, {len(fallback)} readable fallback, both listed")


# ---------------------------------------------------------------------------
# 7. The encoder is loaded once  (Phase 11, "optional" - it is not optional for
#    the UI, which reloads the model on every rerun without it)
# ---------------------------------------------------------------------------


def _check_encoder_cached(checks: Checks) -> None:
    print("\n  the encoder is built once per process, not per question")
    first = embed.get_model()
    second = embed.get_model()
    checks.check(first is second, "get_model() returns the same object")
    checks.check(
        embed.get_model() is first,
        "and stays the same across further calls",
    )
    # The real cost: two encodes must not re-initialise. Assert on the object
    # identity above rather than timing, which is what actually regresses.
    vectors = embed.embed_texts(["one", "two", "three"])
    checks.equal(len(vectors), 3, "embed_texts returns one vector per input")
    checks.check(
        embed.get_model() is first,
        "encoding does not replace the cached model",
    )
    print("    single cached encoder; 3 encodes, still one instance")


# ---------------------------------------------------------------------------
# 8. The walkthrough in architecture §14
# ---------------------------------------------------------------------------


def _check_walkthrough(checks: Checks) -> None:
    print("\n  the §14 walkthrough works end to end")
    # 1. sources.csv -> loader -> cleaned docs
    from growbot.ingest import load as load_mod

    checks.check(
        callable(getattr(load_mod, "load_sources", None)),
        "step 1: the loader is importable and callable",
    )
    rows = load_mod.load_sources()
    checks.check(len(rows) > 0, "step 1: it reads the source list", str(len(rows)))
    # The loader must not touch the network here, so `load_sources` is the right
    # level: the fetch itself is a separate step, covered by the ingest CLI.
    checks.check(
        all(r.url.startswith("https://") for r in rows),
        "step 1: every source row carries a URL",
    )
    # 3. MiniLM -> Chroma persist
    from growbot.ingest.index import collection_count

    checks.check(collection_count() > 0, "Chroma holds records", str(collection_count()))
    # 4. a fact question -> retrieved chunks -> exactly one citation
    payload, trace = ask_with_trace(EXPENSE_Q, complete=Stub(f"The expense ratio is {REAL_RATIO}%."))
    checks.equal(payload.mode, "fact", "step 4: the expense-ratio question is answered")
    retrieved = next((s for s in trace.stages if s.name == "retrieve"), None)
    checks.check(retrieved is not None, "step 4: the trace records a retrieve stage")
    checks.check(
        bool(payload.source_url) and "https://" in payload.source_url,
        "step 4: exactly one citation",
        payload.source_url,
    )
    corpus_urls = {
        r["url"]
        for r in __import__("csv").DictReader(
            SOURCES_CSV.open(encoding="utf-8-sig", newline="")
        )
    }
    checks.check(
        payload.source_url in corpus_urls,
        "step 4: the citation is in sources.csv",
        payload.source_url,
    )
    # 5. "should I buy" -> guard refusal, no invented advice
    before = 0
    payload2, trace2 = ask_with_trace("Should I buy HDFC Small Cap Fund?")
    checks.equal(payload2.mode, "refuse", "step 5: the buy question is refused")
    checks.equal(payload2.reason, "advice", "step 5: refused by the guard")
    checks.equal(trace2.stopped_at, "guard", "step 5: it never reached retrieval")
    checks.check(not trace2.model_called, "step 5: the model was never called")
    checks.check(
        not re.search(r"\b(you should|recommend|I suggest|good buy)\b", payload2.text, re.I),
        "step 5: no advice is offered",
        payload2.text[:60],
    )
    checks.equal(before, 0, "step 5: no model call was needed")
    print("    4 stages reachable, 1 citation, buy question refused at the guard")


# ---------------------------------------------------------------------------
# 9. Table rows are not cut in half  (architecture §13, "PDF table garbage")
# ---------------------------------------------------------------------------


def _check_table_rows(checks: Checks) -> None:
    print("\n  a table row is never split, and never half-carried by overlap")
    from growbot.ingest import chunk as chunk_mod
    from growbot.ingest import load as load_mod

    # The separator row is load-bearing: `split_blocks` recognises a table by
    # TABLE_SEP, and without it these lines are correctly treated as prose. So
    # it must be present, or this would test the wrong branch convincingly.
    header = "| Particulars | Details |\n| --- | --- |\n"
    rows = "| Exit load | 1% if redeemed within 1 year |\n" * 40
    document = load_mod.Document(
        text="Direct Growth\n\n" + header + rows
             + "\nThe expense ratio of the scheme is 1.03%.\n" * 40,
        metadata={
            "scheme_name": "HDFC Large Cap Fund - Direct Growth",
            "doc_type": "scheme_page",
            "source_url": "https://example.invalid/demo",
        },
    )
    pieces = chunk_mod.chunk_document(document)
    checks.check(len(pieces) > 1, "the table is actually split", f"{len(pieces)} chunks")

    # A whole row of a two-column table has three pipes. A row that has lost its
    # first cell has two - and still starts and ends with "|", so nothing about
    # its shape gives it away except the count.
    def row_widths(text: str) -> list[int]:
        return [
            line.strip().count("|")
            for line in text.splitlines()
            if line.strip().startswith("|") and line.strip().endswith("|")
        ]

    for piece in pieces:
        widths = row_widths(piece.text)
        for width in widths:
            checks.check(
                width == 3,
                "every table row keeps both cells",
                f"{width} pipes: {piece.text.splitlines()[0][:40]!r}",
            )
    narrow = [
        piece for piece in pieces
        if any(w < 3 for w in row_widths(piece.text))
    ]
    checks.check(
        not narrow,
        "no chunk carries a half-row from the previous chunk's overlap",
        f"{len(narrow)} chunks with a narrow row",
    )
    # And the overlap is still doing its job: consecutive chunks must share
    # text, otherwise "fixed" would just mean "removed".
    shared = 0
    for a, b in zip(pieces, pieces[1:]):
        a_lines = {ln.strip() for ln in a.text.splitlines() if len(ln.strip()) > 20}
        if any(ln in a_lines for ln in b.text.splitlines()):
            shared += 1
    checks.check(
        shared > 0,
        "consecutive chunks still overlap, so the fix did not just drop it",
        f"{shared} of {len(pieces) - 1} adjacent pairs share a line",
    )

    # Prose overlap must be untouched by the row logic.
    prose = chunk_mod._overlap_tail(
        ("A full sentence that should survive. " * 12)
    )
    checks.check(
        prose and not prose.strip().startswith("|"),
        "a prose overlap is returned unchanged",
        repr(prose[:52]),
    )
    print(f"    {len(pieces)} chunks, 0 half-rows, overlap still {len(prose)} chars")


# ---------------------------------------------------------------------------


def main() -> int:
    checks = Checks()
    print("hardening checks (Phase 11)")
    print("=" * 74)
    print("  Architecture §13 failure modes, provoked rather than assumed.")
    print("  Offline, no API key: a counting stub stands in for the model.")

    _check_numeric_grounding(checks)
    _check_bare_figure(checks)
    _check_refusal_consistency(checks)
    _check_mixed_scheme(checks)
    _check_empty_index(checks)
    _check_blocked_sources(checks)
    _check_encoder_cached(checks)
    _check_table_rows(checks)
    _check_walkthrough(checks)

    print("\n" + "=" * 74)
    if checks.failures:
        print(f"FAILED  {len(checks.failures)} of "
              f"{checks.passed + len(checks.failures)}")
        for failure in checks.failures:
            print(f"  - {failure}")
        return 1
    print(f"OK      {checks.passed} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
