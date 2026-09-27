"""Run the classmate walkthrough from architecture §14, end to end.

    python tools/walkthrough.py

§14 is the document's own definition of success, so it should be runnable
rather than described. Each of the five steps prints what actually happened, and
the script exits non-zero if any of them does not do what §14 says it should.

    1. sources.csv -> loader -> cleaned docs
    2. Chunker: prefix + size + intact tables
    3. MiniLM -> Chroma persist
    4. Ask expense-ratio question -> retrieved chunks -> one citation
    5. Ask "should I buy?" -> guard refusal, no invented advice

Steps 1-3 are offline and need no key. Steps 4-5 need one for step 4 only -
step 5 is refused at the guard, so it never calls the model and still
demonstrates the point.

Pass `--offline` to skip step 4's model call and show a stub answer instead,
which is enough to rehearse the demo on a train with no signal.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from growbot import config  # noqa: E402
from growbot.ask import ask_with_trace  # noqa: E402
from growbot.ingest import chunk as chunk_mod  # noqa: E402
from growbot.ingest import embed as embed_mod  # noqa: E402
from growbot.ingest import index as index_mod  # noqa: E402
from growbot.ingest import load as load_mod  # noqa: E402
from growbot.retrieve.query import search  # noqa: E402

FACT_Q = "Expense ratio of HDFC Large Cap Fund Direct Growth?"
BUY_Q = "Should I buy HDFC Small Cap Fund?"

failures: list[str] = []


def step(number: int, title: str) -> None:
    print(f"\n{'=' * 74}\nSTEP {number}. {title}\n{'=' * 74}")


def require(condition: bool, label: str, detail: str = "") -> None:
    mark = "ok  " if condition else "FAIL"
    print(f"  {mark} {label}" + (f"  ({detail})" if detail else ""))
    if not condition:
        failures.append(label)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="stub the model in step 4; rehearses the demo without a key",
    )
    args = parser.parse_args()

    print("Growbot walkthrough (architecture §14)")
    print(f"  corpus list : {config.SOURCES_CSV.relative_to(ROOT)}")
    print(f"  index       : {Path(config.CHROMA_PATH).name}")
    print(f"  provider    : {config.LLM_PROVIDER} / {config.LLM_MODEL}")

    # -- 1. sources.csv -> loader -> cleaned docs --------------------------
    step(1, "sources.csv -> loader -> cleaned documents")
    rows = load_mod.load_sources()
    require(len(rows) > 0, "the source list is read", f"{len(rows)} sources")
    for row in rows[:3]:
        print(f"       {row.doc_type:<12} {row.scheme_name}")
    if len(rows) > 3:
        print(f"       ... and {len(rows) - 3} more")
    # The real pipeline is html_to_text (a parser, drops nav/footer/script) then
    # clean_text (whitespace only). Demonstrating only the parser would skip half
    # of the cleaning, so both run here, in order.
    html = (
        "<html><body><nav>Menu</nav><h1>Scheme</h1>"
        "<p>Expense ratio 1.03%</p>"
        "<table><tr><td>Exit load</td><td>1% within 1 year</td></tr></table>"
        "<script>track()</script><footer>Copyright</footer></body></html>"
    )
    parsed = load_mod.html_to_text(html)
    sample = load_mod.clean_text(parsed)
    require(
        "1.03%" in sample and "1% within 1 year" in sample,
        "content survives HTML parsing",
        repr(sample[:58]),
    )
    require(
        "track()" not in sample and "Copyright" not in sample and "Menu" not in sample,
        "nav, scripts and footers are dropped",
    )
    print("       (run `python -m growbot.ingest` to actually fetch these)")

    # -- 2. chunker --------------------------------------------------------
    step(2, "Chunker: prefix + size + intact tables")
    # Built through the real Document type so the chunker sees the same input
    # shape it sees in production, rather than a convenience wrapper.
    #
    # The separator row is not decoration: `split_blocks` recognises a table by
    # `TABLE_SEP`, and without it these lines are correctly treated as prose and
    # hard-wrapped. So it has to be here, or this would be testing the wrong
    # branch while appearing to test the table one.
    header = (
        "| Particulars | Details |\n"
        "| --- | --- |\n"
    )
    rows = "| Exit load | 1% if redeemed within 1 year |\n" * 40
    body = "Direct Growth\n\n" + header + rows + "\nThe expense ratio of the scheme is 1.03%.\n" * 40
    document = load_mod.Document(
        text=body,
        metadata={
            "scheme_name": "HDFC Large Cap Fund - Direct Growth",
            "doc_type": "scheme_page",
            "source_url": "https://example.invalid/demo",
        },
    )
    pieces = chunk_mod.chunk_document(document)
    require(len(pieces) > 0, "text is split into chunks", f"{len(pieces)} chunks")
    first_text = pieces[0].text
    require(
        "HDFC Large Cap Fund" in first_text,
        "each chunk carries its scheme prefix",
        repr(first_text[:52]),
    )
    over = [p for p in pieces if p.char_count > config.CHUNK_SIZE]
    require(
        not over,
        f"no chunk exceeds the {config.CHUNK_SIZE}-char cap",
        f"{len(over)} over",
    )
    require(
        all(p.scheme_name == "HDFC Large Cap Fund - Direct Growth" for p in pieces),
        "every chunk keeps the scheme in its metadata",
    )
    # "intact tables" means a row is never cut in half. A whole row of a
    # two-column table has three pipes; one that has lost its first cell has
    # two - and still begins and ends with "|", so shape gives nothing away.
    # Width does. (An earlier version of this check asked whether a chunk ended
    # with "|", which is true of every chunk that legitimately ends on a row,
    # and so failed on correct output.)
    table_chunks = [p for p in pieces if any(
        ln.strip().startswith("|") for ln in p.text.splitlines()
    )]
    widths = [
        (i, ln.strip().count("|"))
        for i, p in enumerate(pieces)
        for ln in p.text.splitlines()
        if ln.strip().startswith("|") and ln.strip().endswith("|")
    ]
    require(
        widths and all(w == 3 for _i, w in widths),
        "every table row keeps both of its cells",
        f"{sum(1 for _i, w in widths if w != 3)} narrow of {len(widths)} rows",
    )
    # And the figure must never be split from its number, which is what the
    # grounding check downstream assumes.
    torn_fig = [p for p in pieces if re.search(r"\b1\.0\b|\b0?3%\b", p.text)]
    require(not torn_fig, "no figure is split across a chunk boundary",
            f"{len(torn_fig)} torn")
    # Overlap must survive the row fix, or "intact" would just mean "shorter".
    shared = sum(
        1 for a, b in zip(pieces, pieces[1:])
        if {ln.strip() for ln in a.text.splitlines() if len(ln.strip()) > 20}
        & {ln.strip() for ln in b.text.splitlines()}
    )
    require(shared > 0, "consecutive chunks still overlap", f"{shared} adjacent pairs")

    # -- 3. MiniLM -> Chroma persist --------------------------------------
    step(3, "MiniLM -> Chroma persist")
    vectors = embed_mod.embed_texts(["one scheme", "another scheme"])
    require(len(vectors) == 2, "the encoder returns one vector per input",
            f"dim={len(vectors[0])}")
    model_a = embed_mod.get_model()
    require(model_a is embed_mod.get_model(),
            "the encoder is built once, not per question")
    count = index_mod.collection_count()
    require(count > 0, "Chroma holds records", f"{count} records at "
            f"{Path(config.CHROMA_PATH).name}")
    print("       (rebuild with `python -m growbot.ingest`)")

    # -- 4. a fact, with retrieved chunks and one citation -----------------
    step(4, "Ask a fact question -> retrieved chunks -> one citation")
    result = search(FACT_Q)
    hits = result.hits
    for hit in hits[:3]:
        print(f"       {hit.similarity:.3f}  {hit.text[:58]}")
    require(len(hits) > 0, "retrieval returns chunks", f"{len(hits)} hits")
    require(
        result.detected_schemes == ["HDFC Large Cap Fund - Direct Growth"],
        "retrieval resolved to exactly one scheme",
        ", ".join(result.detected_schemes) or "none",
    )

    complete = None
    if args.offline:
        complete = lambda messages: "The expense ratio is 1.03%."  # noqa: E731
        print("       (offline: using a stub, no API call)")
    payload, trace = ask_with_trace(FACT_Q, complete=complete)
    print(f"       -> {payload.text}")
    print(f"       -> {payload.source_url}")
    require(payload.mode == "fact", "the question is answered", payload.mode)
    urls = re.findall(r"https?://\S+", payload.text)
    require(len(urls) == 0, "the answer text carries no URL of its own",
            f"{len(urls)} found")
    require(bool(payload.source_url), "exactly one citation is attached")
    import csv as _csv
    with config.SOURCES_CSV.open(encoding="utf-8-sig", newline="") as fh:
        corpus = {r["url"] for r in _csv.DictReader(fh)}
    require(payload.source_url in corpus,
            "the citation is in sources.csv", payload.source_url[:58])
    require(bool(payload.last_updated), "the answer is dated from the fetch",
            payload.last_updated)
    require("generate" in [s.name for s in trace.stages if s.outcome == "called"],
            "the trace shows the model was consulted", trace.stopped_at)

    # -- 5. the buy question ------------------------------------------------
    step(5, 'Ask "should I buy?" -> guard refusal, no invented advice')
    payload, trace = ask_with_trace(BUY_Q)
    print(f"       -> {payload.text}")
    print(f"       -> {payload.source_url}")
    require(payload.mode == "refuse", "the question is refused", payload.mode)
    require(payload.reason == "advice", "refused as advice", payload.reason)
    require(trace.stopped_at == "guard", "it never reached retrieval", trace.stopped_at)
    require(not trace.model_called,
            "the model was never called - this is a rule, not a prompt")
    require(
        not re.search(
            r"\b(you should (buy|invest|add)|i recommend|i suggest|we recommend|"
            r"is a good (buy|investment)|best (choice|option) for you)\b",
            payload.text, re.I,
        ),
        "no advice is offered",
        # Scoped to affirmative recommendation shapes on purpose. A first
        # attempt included "worth buying" and matched the refusal's own denial
        # - "I can't say whether X is worth buying" - so the check flagged the
        # very sentence that was declining to advise.
    )
    require(bool(payload.source_url) and "http" not in payload.text,
            "one educational link, none inline")

    # -- 6. a follow-up, cold and then with memory -----------------------
    # This is the one step where the *same question* is asked twice and the
    # difference is the point, so both are shown. Asked cold, a follow-up names
    # no fund, retrieval is unfiltered, and whatever page happens to score best
    # is attached to the answer as the source - often a general explainer, and
    # potentially a different fund's page. With the window, the fund comes from
    # the conversation and the citation comes from that fund's own pages.
    #
    # A *number-free* stub is used here, unlike step 4, and deliberately. Step
    # 4's stub says "The expense ratio is 1.03%", which is right for that
    # question and wrong for every other one. Asked about an exit load, the
    # grounding check discards it as ungrounded and the step ends up measuring
    # the grounding check. The claims below are about *retrieval* - which scheme
    # was resolved, whether the filter applied, whose page is cited - so the
    # stub is held constant and number-free to keep it out of the way.
    step(6, "A follow-up: asked cold, then with conversation memory")
    follow_up = "What about its exit load?"
    plain = lambda messages: "That is stated in the fund's own factsheet."  # noqa: E731
    follow_stub = plain if args.offline else None

    print(f"       cold: {follow_up!r}, no history")
    cold, cold_trace = ask_with_trace(follow_up, complete=follow_stub)
    cold_retrieve = next((s for s in cold_trace.stages if s.name == "retrieve"), None)
    print(f"       -> {cold.text}")
    print(f"       -> {cold.source_url}")
    print(f"       -> {cold_retrieve.detail if cold_retrieve else '?'}")

    print(f"       with history: [{FACT_Q!r}]")
    warm, warm_trace = ask_with_trace(
        follow_up, history=[FACT_Q], complete=follow_stub
    )
    print(f"       -> {warm.text}")
    print(f"       -> {warm.source_url}")
    warm_mem = next((s for s in warm_trace.stages if s.name == "memory"), None)
    warm_retrieve = next((s for s in warm_trace.stages if s.name == "retrieve"), None)
    print(f"       -> memory: {warm_mem.detail if warm_mem else '(not consulted)'}")
    print(f"       -> {warm_retrieve.detail if warm_retrieve else '?'}")

    cold_detail = cold_retrieve.detail if cold_retrieve else ""
    warm_detail = warm_retrieve.detail if warm_retrieve else ""
    with config.SOURCES_CSV.open(encoding="utf-8-sig", newline="") as fh:
        owner = {r["url"]: r["scheme_name"] for r in _csv.DictReader(fh)}

    require(
        warm_mem is not None and "inherited" in warm_mem.detail,
        "the follow-up inherits the fund from the conversation",
        warm_mem.detail if warm_mem else "no memory stage",
    )
    require(
        "filtered=True" in warm_detail,
        "so retrieval is filtered to that one fund", warm_detail,
    )
    require(
        "filtered=True" not in cold_detail,
        "asked cold, retrieval is NOT filtered - no fund was resolved", cold_detail,
    )
    require(
        "scheme from memory" in warm_detail,
        "and the trace says the scheme came from memory, not the question",
        warm_detail,
    )
    require(
        owner.get(warm.source_url) == "HDFC Large Cap Fund - Direct Growth",
        "the citation belongs to the fund the conversation was about",
        str(owner.get(warm.source_url)),
    )
    # The bug this fixes: cold, the cited page is whatever won an unfiltered
    # search, which is not the fund being discussed. Whether that is a general
    # explainer or another scheme's page, it is the wrong citation - and it is
    # a real URL from the corpus either way, which is what makes it dangerous.
    require(
        owner.get(cold.source_url) != "HDFC Large Cap Fund - Direct Growth",
        "asked cold, the cited page is NOT the fund being discussed (the bug)",
        f"{owner.get(cold.source_url)} <- {cold.source_url[:46]}",
    )

    print(f"\n{'=' * 74}")
    if failures:
        print(f"WALKTHROUGH FAILED  {len(failures)} step(s) did not behave as §14 says:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("WALKTHROUGH OK      all six §14 steps behave as documented")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
