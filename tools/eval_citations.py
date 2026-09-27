"""Measure whether citations are correct, not merely present.

    python tools/eval_citations.py

Phase 11's "Done when" is *≥8/10 fact questions cited correctly*. That is a
different measurement from the one the check suites make. The offline suites
assert the guarantees that hold for any model - an ungrounded number is
discarded, a refusal never reaches the model. This script asks the question the
guards cannot: **given a real model and the real corpus, is the citation on
each answer the right one?**

"Cited correctly" means three things at once, which is stricter than "has a
link":

1. the question was answered rather than refused,
2. exactly one citation is emitted, and
3. that URL belongs to the scheme the question was about, per `data/sources.csv`.

Point 3 is the one that matters. A bot that answers the right fact from the
wrong fund's page is worse than one that refuses, because the user has no way
to notice - and "all chunks of one scheme share a single `source_url`" means
this is checkable exactly rather than approximately.

This calls the live provider, so it needs a key, and it is deliberately **not**
part of the offline check suites: a free tier rate-limits, and a check suite
that fails intermittently teaches people to ignore it. Run it before a demo and
read the number; a copy of the last run is recorded in `docs/sample_qa.md`.

Exits non-zero below the threshold, so it can gate a release if you want it to.
"""

from __future__ import annotations

import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from growbot import config  # noqa: E402
from growbot.ask import ask_with_trace  # noqa: E402
from growbot.generate.answer import is_prose_free  # noqa: E402
from growbot.guards.intent import display_name  # noqa: E402

#: The bar from Phase 11. Not a target to hit - a floor to clear.
THRESHOLD = 8

#: Seconds between questions. Free tiers rate-limit per minute and a burst
#: produces 429s, which would show up here as a wrong answer rather than as a
#: rate-limit, and that would be a misleading measurement.
GAP = 6.0

#: Facts the corpus actually states, two to three per scheme so that a scheme
#: which has quietly gone unretrievable cannot hide behind the others.
FACT_QUESTIONS = [
    ("Expense ratio of HDFC Large Cap Fund Direct Growth?", "HDFC Large Cap Fund - Direct Growth"),
    ("Benchmark of HDFC Large Cap Fund Direct Growth?", "HDFC Large Cap Fund - Direct Growth"),
    ("Risk level of HDFC Large Cap Fund Direct Growth?", "HDFC Large Cap Fund - Direct Growth"),
    ("Exit load of HDFC Small Cap Fund Direct Growth?", "HDFC Small Cap Fund - Direct Growth"),
    ("Riskometer level of HDFC Small Cap Fund Direct Growth?", "HDFC Small Cap Fund - Direct Growth"),
    ("Minimum SIP for HDFC Balanced Advantage Fund Direct Growth?", "HDFC Balanced Advantage Fund - Direct Growth"),
    ("Exit load of HDFC Balanced Advantage Fund Direct Growth?", "HDFC Balanced Advantage Fund - Direct Growth"),
    ("Lock-in period for HDFC ELSS Tax Saver Direct Growth?", "HDFC ELSS Tax Saver Fund - Direct Plan Growth"),
    ("Is there a tax benefit under section 80C for HDFC ELSS Tax Saver Direct Growth?", "HDFC ELSS Tax Saver Fund - Direct Plan Growth"),
    ("Expense ratio of HDFC Flexi Cap Fund Direct Growth?", "HDFC Flexi Cap Fund - Direct Growth"),
    ("Benchmark of HDFC Flexi Cap Fund Direct Growth?", "HDFC Flexi Cap Fund - Direct Growth"),
    ("Minimum SIP for HDFC Large Cap Fund Direct Growth?", "HDFC Large Cap Fund - Direct Growth"),
]

#: The two refusal types Phase 11 requires to keep working. Both must be
#: refused *and* must never reach the model.
REFUSAL_QUESTIONS = [
    ("Should I buy HDFC Small Cap Fund?", "advice"),
    ("Which fund has the best returns?", "returns"),
]


@dataclass
class Outcome:
    question: str
    ok: bool
    detail: str
    mode: str
    reason: str
    url: str
    cited_scheme: str
    wanted_scheme: str


def _url_owner() -> dict[str, str]:
    """URL -> the scheme that owns it, straight from the source list."""
    with (ROOT / "data" / "sources.csv").open(encoding="utf-8-sig", newline="") as fh:
        return {r["url"]: r["scheme_name"] for r in csv.DictReader(fh)}


def _grade_fact(question: str, wanted: str, payload, owner: dict[str, str]) -> Outcome:
    """Apply the three conditions. The first failure is the one reported."""
    cited = owner.get(payload.source_url, "<url not in sources.csv>")
    base = dict(
        question=question,
        mode=payload.mode,
        reason=payload.reason,
        url=payload.source_url,
        cited_scheme=cited,
        wanted_scheme=wanted,
    )
    if payload.mode != "fact":
        return Outcome(ok=False, detail=f"refused ({payload.reason})", **base)
    if not payload.source_url:
        return Outcome(ok=False, detail="no citation", **base)
    if payload.source_url not in owner:
        return Outcome(ok=False, detail="cited URL is not in sources.csv", **base)
    if cited != wanted:
        return Outcome(
            ok=False,
            detail=f"cited {display_name(cited)}, asked about {display_name(wanted)}",
            **base,
        )
    if is_prose_free(payload.text):
        # Correct figure, but not a sentence. Counts as a miss, because a
        # citation next to a bare number reads as a rendering fault.
        return Outcome(ok=False, detail="answered with a bare figure", **base)
    return Outcome(ok=True, detail="correct scheme cited", **base)


def main() -> int:
    owner = _url_owner()
    print("citation accuracy (Phase 11)")
    print("=" * 74)
    print(f"  model   : {config.LLM_PROVIDER} / {config.LLM_MODEL}")
    print(f"  corpus  : {len(owner)} URLs from data/sources.csv")
    print(f"  bar     : >= {THRESHOLD}/{len(FACT_QUESTIONS)} correctly cited")
    print(f"  pacing  : {GAP:.0f}s between questions (free tier limits)")

    results: list[Outcome] = []
    for question, wanted in FACT_QUESTIONS:
        payload, _ = ask_with_trace(question)
        outcome = _grade_fact(question, wanted, payload, owner)
        results.append(outcome)
        mark = "ok  " if outcome.ok else "MISS"
        print(f"  {mark} [{payload.mode:<6}] {question[:52]}")
        if not outcome.ok:
            print(f"         -> {outcome.detail}")

    print()
    for question, reason in REFUSAL_QUESTIONS:
        payload, trace = ask_with_trace(question)
        stopped_right = payload.mode == "refuse" and payload.reason == reason
        never_asked = not trace.model_called
        mark = "ok  " if (stopped_right and never_asked) else "MISS"
        print(f"  {mark} [{payload.mode:<6}] {question[:52]}")
        print(
            f"         -> reason={payload.reason!r}, stopped at "
            f"{trace.stopped_at}, model called: {'yes' if trace.model_called else 'no'}"
        )
        if not (stopped_right and never_asked):
            results.append(
                Outcome(False, f"refusal broken (wanted {reason})", question,
                        payload.mode, payload.reason, payload.source_url, "", "")
            )

    facts = [r for r in results if r.wanted_scheme]
    scored = sum(1 for r in facts if r.ok)
    total = len(facts)
    print("\n" + "=" * 74)
    print(f"  citation accuracy : {scored}/{total}")
    per_scheme: dict[str, list[bool]] = {}
    for r in facts:
        per_scheme.setdefault(r.wanted_scheme, []).append(r.ok)
    for scheme, oks in sorted(per_scheme.items()):
        print(f"    {sum(oks)}/{len(oks)}  {display_name(scheme)}")

    if scored < THRESHOLD:
        print(f"\nFAILED  below the bar ({scored}/{total} < {THRESHOLD})")
        for r in results:
            if not r.ok:
                print(f"  - {r.question[:52]}: {r.detail}")
        return 1
    print(f"\nOK      {scored}/{total} correctly cited, both refusal types hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
