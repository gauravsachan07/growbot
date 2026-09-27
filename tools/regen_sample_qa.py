"""Regenerate docs/sample_qa.md from live answers.

    python tools/regen_sample_qa.py

Phase 10 requires `docs/sample_qa.md` to hold *actual* assistant answers, not
ones written by hand to look good. This script produces that file by calling
`growbot.ask.ask()` and pasting the payloads verbatim.

Run it again whenever the model, the corpus, or the prompt changes - the
captured date and model id in the generated file are meant to go stale, so
that a reader can tell how old the evidence is.

Note the sleep between calls. Free LLM tiers rate-limit per minute, and a burst
produces `429`s rather than answers, which would silently leave gaps in the
sample. That is the reason this is a script and not a shell loop.
"""

from __future__ import annotations

import csv
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))

from growbot import config  # noqa: E402
from growbot.ask import ask_with_trace  # noqa: E402
from growbot.generate.answer import is_prose_free  # noqa: E402
from growbot.guards.intent import display_name  # noqa: E402

#: The nine from PRD section 12 in order, then one extra to show a second
#: scheme, then a follow-up. `expect` is what PRD section 12 says should
#: happen. A third element, where present, is the prior turns the question
#: depends on - only the follow-up has one.
QUESTIONS = [
    ("Expense ratio of HDFC Large Cap Fund Direct Growth?", "fact"),
    ("Exit load of HDFC Small Cap Fund Direct Growth?", "fact"),
    ("Minimum SIP for HDFC Balanced Advantage Fund Direct Growth?", "fact"),
    ("Lock-in period for HDFC ELSS Tax Saver Direct Growth?", "fact"),
    ("Riskometer / risk level of HDFC Equity Fund (Flexi Cap) Direct Growth?", "fact"),
    ("Benchmark of HDFC Large Cap Fund Direct Growth?", "fact"),
    ("How to download a capital-gains statement?", "refuse"),
    ("Should I buy HDFC Small Cap Fund?", "refuse"),
    ("Which fund has the best returns?", "refuse"),
    ("Expense ratio of HDFC Flexi Cap Fund (Direct Growth)?", "fact"),
    # A follow-up, deliberately last. It is the one entry that cannot be read
    # as a standalone question, so the generated block says what it follows -
    # a bare "What about its exit load?" in a list of self-contained questions
    # looks like a bug rather than the demonstration it is.
    (
        "What about its exit load?",
        "fact",
        ["Expense ratio of HDFC Small Cap Fund Direct Growth?"],
    ),
]

OUT = ROOT / "docs" / "sample_qa.md"
SOURCES = ROOT / "data" / "sources.csv"
GAP = 6.0

#: The last full citation-accuracy run, as measured by `tools/eval_citations.py`.
#: Recorded here so the document carries a number rather than a promise. Update
#: it when you re-run the eval - a stale figure is worse than none, which is why
#: the date is next to it.
LAST_EVAL = "2026-09-27"
LAST_EVAL_SCORE = "11/12"
LAST_EVAL_BAR = "8/12"


def _corpus_size() -> int:
    with SOURCES.open(encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def main() -> int:
    rows = []
    for entry in QUESTIONS:
        question, expect = entry[0], entry[1]
        history = entry[2] if len(entry) > 2 else None
        payload, trace = ask_with_trace(question, history=history)
        ok = (payload.mode == "fact") if expect == "fact" else (payload.mode == "refuse")
        note = f"  (after: {history[0][:34]!r})" if history else ""
        print(f"  {'ok  ' if ok else 'MISS'} [{payload.mode:<6}] {question[:52]}{note}")
        rows.append((question, expect, payload, trace, ok, history))
        time.sleep(GAP)

    now = date.today().isoformat()
    # Indexed, not `*_, ok`. The rows grew a sixth field (the follow-up's
    # history) and `*_, ok` kept working syntactically while rebinding `ok` to
    # that history - truthy for the one follow-up, an empty list for the rest -
    # so the tally read 1/11 while all eleven had actually matched.
    passed = sum(1 for row in rows if row[4])
    lines: list[str] = []
    w = lines.append

    w("# Sample Q&A")
    w("")
    w("Every answer below is a **real** response from `growbot.ask.ask()`, pasted")
    w("verbatim - not written by hand, and not edited to look better than it was.")
    w("Regenerate with `python tools/regen_sample_qa.py`.")
    w("")
    w("| | |")
    w("|---|---|")
    w(f"| Captured | {now} |")
    w(f"| Model | `{config.LLM_PROVIDER}` / `{config.LLM_MODEL}` |")
    w(f"| Corpus | `data/sources.csv` ({_corpus_size()} URLs) |")
    w(f"| Outcome | {passed}/{len(rows)} matched the expected mode |")
    w("")
    w("The citations and the `Last updated from sources` date come from retrieval")
    w("metadata, never from the model - the model is only ever asked for prose.")
    w("")
    w(f"> **{config.DISCLAIMER}**")
    w("")
    w("---")
    w("")

    for n, (question, expect, payload, trace, ok, history) in enumerate(rows, 1):
        tag = "FACT" if payload.mode == "fact" else "REFUSAL"
        w(f"## Q{n}. {question}")
        w("")
        if history:
            w(f"_Follow-up. Asked after: \"{history[0]}\"_")
            w("")
        w(f"**{tag}**" + ("" if ok else f" - _expected {expect}, got {payload.mode}_"))
        w("")
        w(f"> {payload.text}")
        w("")
        if payload.scheme_name:
            w(f"- **Scheme:** {display_name(payload.scheme_name)}")
        if payload.source_url:
            label = "Source" if payload.mode == "fact" else "Educational link"
            w(f"- **{label}:** {payload.source_url}")
        if payload.last_updated:
            w(f"- **Last updated from sources:** {payload.last_updated}")
        w(f"- **Refusal reason:** `{payload.reason or 'n/a (answered)'}`")
        w(
            f"- **Stopped at:** `{trace.stopped_at}` | "
            f"**model called:** {'yes' if trace.model_called else '**no**'}"
        )
        w("")

    w("---")
    w("")
    w("## What the trace lines show")
    w("")
    w("`Stopped at` is the pipeline stage that **decided** the answer, and")
    w("`model called` is whether the provider was contacted at all. Read together")
    w("they separate the two kinds of refusal, which look identical in the UI:")
    w("")
    w("- **Never reached the model because a rule blocked it.** Q8 and Q9 stop at")
    w("  `guard`. The question was refused before retrieval, so no prompt change")
    w("  can make the bot answer it.")
    w("- **Never reached the model because the evidence was too weak.** Q7 stops at")
    w("  `assemble`, where the retrieved context scored below the similarity floor")
    w("  and spanned two schemes. The bot declined on its own evidence.")
    w("")
    w("In all three cases `model called` is **no**, which is the claim worth")
    w("making: the refusals are structural, not the model choosing to be cautious.")
    w("")
    w("Q11 adds a `memory` stage, which appears only when the question depends on")
    w("earlier turns. It records *which fund the follow-up was resolved to* and")
    w("where that came from. It is the one stage that reads the conversation")
    w("rather than the question, and it deliberately does no more than that: it")
    w("picks the fund for retrieval and nothing else, so the retrieval, the")
    w("grounding check and the prompt are all unchanged by it.")
    w("")
    w("## Citation accuracy")
    w("")
    w("The eleven questions above check *whether* the bot answers. A separate")
    w("measurement checks whether the citation is the **right** one, which is")
    w("stricter: answered, exactly one link, and that link belonging to the")
    w("scheme the question was about. Run `python tools/eval_citations.py` over")
    w("twelve fact questions spanning all five schemes.")
    w("")
    w(f"Last run **{LAST_EVAL}**: **{LAST_EVAL_SCORE}** correctly cited, against a bar")
    w(f"of {LAST_EVAL_BAR}.")
    w("")
    w("The one miss was the section 80C question, refused as `insufficient`")
    w("because the retrieved text does not state it - a genuine gap in the")
    w("corpus, not a wrong citation. **No answer was ever cited to the wrong")
    w("fund**, which is the failure that matters: it is the one a reader has no")
    w("way to notice.")
    w("")
    w("## Known limits")
    w("")
    w("- Answers are only as current as the fetch that built the index. The")
    w("  `Last updated from sources` line is the fetch time, not a live AMC feed.")
    w("- Only the five HDFC Direct Growth schemes in `data/sources.csv` are")
    w("  covered. Anything else is out of scope, not merely unknown.")
    w("- No NAV, no tax computation, no portfolio questions, no recommendations.")
    w("- One scheme per answer. A question spanning two funds is refused rather")
    w("  than answered from a blend of both.")
    w("")
    # Data-driven, because the earlier version of this file hardcoded a claim
    # about one answer being too terse. That answer is fine now, and a
    # hardcoded complaint about a specific question would outlive the bug.
    terse = [
        n for n, (_q, _e, p, _t, _ok, _h) in enumerate(rows, 1)
        if is_prose_free(p.text)
    ]
    if terse:
        w(f"- Q{', Q'.join(map(str, terse))} came back as a bare figure with no words")
        w("  around it, twice in a row, so it was refused rather than shown. The")
        w("  refusal says the model could not phrase the answer - it does not")
        w("  claim the corpus is missing, because the corpus was not.")
    else:
        w("- No answer in this run was a bare figure. An earlier capture had one")
        w("  (`0.77%`, with no sentence around it): correct and grounded, but not")
        w("  usable. `generate()` now re-asks once when a reply contains no words")
        w("  at all, and if that fails too it refuses with copy that names the real")
        w("  cause instead of blaming the corpus. See")
        w("  `python -m growbot.hardening_checks`.")
    w("")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n  wrote {OUT.relative_to(ROOT)} ({len(lines)} lines), "
          f"{passed}/{len(rows)} as expected")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
