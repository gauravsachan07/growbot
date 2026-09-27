"""Phase 6 - retrieval checks.

    python -m growbot.retrieve.checks

Assertion-based, non-zero exit on regression. Requires a built index
(``python -m growbot.ingest``). Read-only: no network, no LLM.

What is asserted:

1. The two implementation.md "Done when" cases.
2. The scheme ``where``-filter actually confines results, for every alias of
   every scheme - including the HDFC Equity Fund / Flexi Cap rename, where two
   different aliases must collapse to one canonical scheme.
3. The citation contract: exactly one URL, taken from the best chunk, and it
   must be one the guard would have been willing to cite.
4. Weak retrieval fires for vague questions and for multi-scheme questions.
5. The ``append_scheme`` option stays off by default - the measured comparison
   showed it lowers accuracy.
"""

from __future__ import annotations

import re
import sys

from growbot.config import (
    EDU_LINK,
    EMBEDDING_BACKEND,
    EMBEDDING_DIM,
    FACTSHEET_LINKS,
    GENERAL_SCHEME,
    SCHEMES,
    SIMILARITY_FLOOR,
    SOURCES_CSV,
    detect_scheme,
)
from growbot.ingest.load import load_sources
from growbot.retrieve.assemble import assemble
from growbot.retrieve.query import Hit, SearchResult, search

from growbot.retrieve.__main__ import EVAL_QUESTIONS

ALLOWED_URLS = {EDU_LINK, *FACTSHEET_LINKS.values()}


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


def main() -> int:
    checks = Checks()
    print("retrieval checks")
    print("=" * 74)

    # --- 1. "Done when" case 1: expense ratio -----------------------------
    print("\n  'expense ratio HDFC Large Cap Direct Growth'")
    result = search("expense ratio HDFC Large Cap Direct Growth")
    assembly = assemble(result)
    checks.check(bool(result.hits), "returns chunks")
    checks.check(
        len(assembly.context) > 0, "assembles a context block"
    )
    checks.check(
        assembly.context.count("source: https://") >= 1,
        "context carries source lines",
    )
    checks.equal(
        assembly.source_url,
        FACTSHEET_LINKS["HDFC Large Cap Fund - Direct Growth"],
        "cites the one expected URL",
    )
    checks.equal(
        assembly.source_url.count("https://"), 1, "exactly one URL in the citation"
    )
    checks.check(
        not assembly.weak, "not weak", str(assembly.weak_reasons)
    )
    print(f"    {len(result.hits)} chunks, max_sim={assembly.max_similarity:.3f}")
    print(f"    cite {assembly.source_url}")

    # --- 2. "Done when" case 2: ELSS preferred -----------------------------
    print("\n  ELSS question prefers ELSS chunks")
    elss = search("Lock-in period for HDFC ELSS Tax Saver Direct Growth?")
    elss_hits = [h for h in elss.hits if not h.is_general]
    checks.check(bool(elss_hits), "returns ELSS chunks")
    schemes_found = {h.scheme_name for h in elss_hits}
    checks.equal(
        schemes_found,
        {"HDFC ELSS Tax Saver Fund - Direct Plan Growth"},
        "every non-guide chunk is ELSS",
    )
    checks.check(elss.filtered, "scheme filter was applied")
    print(f"    schemes={schemes_found}, filtered={elss.filtered}")

    # The lock-in fact must actually be inside the context, not just nearby.
    elss_context = assemble(elss).context.lower()
    checks.check(
        "lock" in elss_context,
        "lock-in text is inside the context handed to the generator",
    )

    # --- 3. the filter holds for every alias -------------------------------
    print("\n  scheme filter for every alias of every scheme")
    for name, spec in SCHEMES.items():
        for alias in spec["aliases"]:  # type: ignore[union-attr]
            found = detect_scheme(alias)
            checks.equal(found, [name], f"alias resolves: {alias!r}")
            probe = search(f"{alias} expense ratio", top_k=5)
            wrong = [
                h.scheme_name
                for h in probe.hits
                if not h.is_general and h.scheme_name != name
            ]
            checks.check(
                not wrong, f"filter confines {alias!r}", str(set(wrong))
            )
    # The rename must collapse to one scheme, not look ambiguous.
    checks.equal(
        detect_scheme("HDFC Equity Fund (Flexi Cap)"),
        ["HDFC Flexi Cap Fund - Direct Growth"],
        "Equity/Flexi aliases collapse to one scheme",
    )
    print(f"    {sum(len(s['aliases']) for s in SCHEMES.values())} aliases checked")

    # --- 4. citation contract ---------------------------------------------
    # Two different URL sets are in play and conflating them hides bugs:
    #   * the *refusal* allowlist (EDU_LINK + FACTSHEET_LINKS) is what the
    #     Phase 5 guard is allowed to offer in place of an answer;
    #   * the *corpus* source list is where a citation may point, because a
    #     citation is by definition the page the fact was read from.
    # A cited URL must be in the corpus list, not the refusal allowlist - an
    # AMFI guide chunk legitimately cites an AMFI article.
    corpus_urls = {row.url for row in load_sources(SOURCES_CSV)}
    print("\n  citation contract")
    print(f"    corpus urls: {len(corpus_urls)}  refusal allowlist: {len(ALLOWED_URLS)}")
    for question in EVAL_QUESTIONS:
        assembly = assemble(search(question))
        checks.check(
            assembly.source_url in corpus_urls,
            f"cited url is in sources.csv: {question[:30]}",
            assembly.source_url,
        )
        checks.check(
            assembly.source_url == assembly.hits[0].source_url,
            f"url comes from rank 1: {question[:30]}",
        )
        checks.check(
            assembly.source_url.count("https://") == 1,
            f"exactly one URL: {question[:30]}",
        )
        checks.check(
            len(assembly.last_updated) == 10,
            f"ISO last_updated: {question[:30]}",
            assembly.last_updated,
        )
        # Grounded: the cited URL appears in the context that was assembled.
        checks.check(
            assembly.source_url in assembly.context,
            f"cited url appears in context: {question[:30]}",
        )
    # Every corpus URL we could cite must be a real, listed source.
    checks.check(
        corpus_urls.issuperset({"https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth"}),
        "sources.csv still contains the Large Cap page",
    )
    print(f"    {len(EVAL_QUESTIONS)} questions, one corpus url each")

    # --- 5. weak retrieval ------------------------------------------------
    print("\n  weak retrieval")
    vague = [
        "is it good?",
        "what should I do with my money",
        "give me market tips",
    ]
    for question in vague:
        assembly = assemble(search(question))
        checks.check(
            assembly.weak,
            f"weak for vague: {question[:32]}",
            f"sim={assembly.max_similarity:.3f}",
        )
    # A question naming two funds cannot be answered from one citation.
    multi = assemble(
        search("Expense ratio of HDFC Large Cap Fund and HDFC Small Cap Fund?")
    )
    checks.check(
        multi.weak, "multi-scheme question is weak", str(multi.weak_reasons)
    )
    checks.check(
        any("2 schemes" in r for r in multi.weak_reasons),
        "multi-scheme reason is explicit",
        str(multi.weak_reasons),
    )
    # The mixed-scheme rule is tested on synthetic results rather than through
    # a live query. The corpus is re-fetched from live pages, so which chunk
    # ranks first for any given question drifts between runs; a rule test that
    # depends on that is a flaky test, not a strict one.
    guide_result = SearchResult(
        question="synthetic: guide chunk cited, other schemes below",
        hits=[
            Hit(rank=1, text="A guide chunk.", similarity=0.62,
                scheme_name=GENERAL_SCHEME, source_url="https://example.test/guide",
                fetched_at="2026-09-27"),
            Hit(rank=2, text="Large Cap chunk.", similarity=0.58,
                scheme_name="HDFC Large Cap Fund - Direct Growth",
                source_url="https://example.test/lc", fetched_at="2026-09-27"),
            Hit(rank=3, text="Small Cap chunk.", similarity=0.54,
                scheme_name="HDFC Small Cap Fund - Direct Growth",
                source_url="https://example.test/sc", fetched_at="2026-09-27"),
        ],
    )
    guide = assemble(guide_result)
    checks.check(
        not any("spans" in r for r in guide.weak_reasons),
        "a guide-cited answer is not flagged as mixed",
        str(guide.weak_reasons),
    )
    checks.check(
        not guide.weak, "a strong guide-cited answer is answerable", str(guide.weak_reasons)
    )
    # Same hits, but now a single fund's page is the one being cited. Now the
    # one-URL citation could misattribute, so the rule must fire.
    scheme_first = SearchResult(
        question="synthetic: scheme chunk cited, several funds in context",
        hits=[
            Hit(rank=1, text="Large Cap chunk.", similarity=0.62,
                scheme_name="HDFC Large Cap Fund - Direct Growth",
                source_url="https://example.test/lc", fetched_at="2026-09-27"),
            Hit(rank=2, text="Small Cap chunk.", similarity=0.58,
                scheme_name="HDFC Small Cap Fund - Direct Growth",
                source_url="https://example.test/sc", fetched_at="2026-09-27"),
        ],
    )
    mixed = assemble(scheme_first)
    checks.check(
        any("spans" in r for r in mixed.weak_reasons),
        "a scheme-cited answer spanning funds is flagged",
        str(mixed.weak_reasons),
    )
    # A single-fund context is clean even though nothing was named.
    single = assemble(SearchResult(
        question="synthetic: one fund only",
        hits=[
            Hit(rank=1, text="Large Cap chunk.", similarity=0.62,
                scheme_name="HDFC Large Cap Fund - Direct Growth",
                source_url="https://example.test/lc", fetched_at="2026-09-27"),
            Hit(rank=2, text="More Large Cap.", similarity=0.58,
                scheme_name="HDFC Large Cap Fund - Direct Growth",
                source_url="https://example.test/lc", fetched_at="2026-09-27"),
        ],
    ))
    checks.check(
        not any("spans" in r for r in single.weak_reasons),
        "a single-fund context is not flagged mixed",
        str(single.weak_reasons),
    )
    # A known corpus gap must be refused, whatever the reason. Checked as
    # "weak" rather than as a specific cause, because which of the two weak
    # rules fires depends on how the live pages happen to rank today. If this
    # ever fails, the corpus gained capital-gains content and the floor and
    # KNOWN_GAPS in retrieve/__main__.py should be revisited.
    from growbot.retrieve.__main__ import KNOWN_GAPS

    for question in KNOWN_GAPS:
        checks.check(
            assemble(search(question)).weak,
            f"known gap is refused: {question[:30]}",
            "it became answerable - corpus changed",
        )
    print(f"    {len(vague)} vague + 1 multi-scheme + {len(KNOWN_GAPS)} known gap weak")
    print("    mixed-scheme rule tested synthetically (corpus drifts between runs)")

    # --- 6. append_scheme stays off --------------------------------------
    print("\n  append_scheme default")
    off = search("Riskometer level of HDFC Equity Fund (Flexi Cap) Direct Growth?")
    on = search(
        "Riskometer level of HDFC Equity Fund (Flexi Cap) Direct Growth?",
        append_scheme=True,
    )
    checks.check(not off.appended_scheme, "default does not append the scheme")
    checks.check(on.appended_scheme, "flag appends when asked")
    checks.check(
        on.max_similarity >= off.max_similarity,
        "appending inflates similarity (so the option is not free)",
        f"off={off.max_similarity:.3f} on={on.max_similarity:.3f}",
    )
    print(f"    off={off.max_similarity:.3f}  on={on.max_similarity:.3f} (inflated)")

    # --- 7. context format ------------------------------------------------
    print("\n  context block format")
    assembly = assemble(search("Expense ratio of HDFC Large Cap Fund?"))
    hits = assembly.hits
    checks.check(
        assembly.context.startswith("[1] "), "starts with entry 1"
    )
    # Count entry markers with a strict pattern. A bare startswith("[") also
    # matches chunk body lines such as "[email protected]".
    entries = re.findall(r"^\[(\d+)\] ", assembly.context, re.MULTILINE)
    checks.equal(len(entries), len(hits), "one numbered entry per chunk")
    checks.equal(
        [int(n) for n in entries],
        list(range(1, len(hits) + 1)),
        "entries are numbered 1..k",
    )
    for hit in hits:
        checks.check(
            f"source: {hit.source_url}" in assembly.context,
            f"entry {hit.rank} carries its own source",
        )
        checks.check(
            f"fetched_at: {hit.fetched_at}" in assembly.context,
            f"entry {hit.rank} carries its own fetch date",
        )
    print(f"    {len(hits)} entries, {len(assembly.context)} chars")

    # --- 8. the two embedding backends agree ------------------------------
    # `EMBEDDING_BACKEND` picks the runtime that encodes every chunk and every
    # question, and the default exists purely to save ~341 MB. The only way
    # that trade can go wrong is if the two runtimes disagree - and then the
    # symptom is not a crash, it is a similarity score that quietly stops
    # matching the index it was measured against, which shows up much later as
    # a wrong citation. So it is asserted on every run rather than left to the
    # measurement quoted in embed.py's docstring.
    print("\n  embedding backend equivalence")
    import numpy as np

    from growbot.ingest.embed import embed_texts, get_torch_model

    # The real serving function, not a re-implementation of it, so the check
    # covers whatever `EMBEDDING_BACKEND` actually selects.
    probes = [
        *EVAL_QUESTIONS,
        "HDFC Small Cap Fund - Direct Growth. 3Y Lock-in. Benchmark: Nifty Smallcap 250.",
    ]
    produced = embed_texts(probes)
    reference = get_torch_model().encode(
        list(probes),
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    )

    worst_cosine = 1.0
    worst_gap = 0.0
    for position, probe in enumerate(probes):
        actual = np.asarray(produced[position], dtype="float64")
        expected = np.asarray(reference[position], dtype="float64")
        checks.equal(
            actual.shape[0], EMBEDDING_DIM, f"vector is {EMBEDDING_DIM}-d: {probe[:26]}"
        )
        cosine = float(
            actual @ expected / (np.linalg.norm(actual) * np.linalg.norm(expected))
        )
        worst_cosine = min(worst_cosine, cosine)
        worst_gap = max(worst_gap, float(np.max(np.abs(actual - expected))))

    checks.check(
        worst_cosine >= 0.9999,
        f"every probe matches the torch reference (worst cosine {worst_cosine:.10f})",
        f"only {worst_cosine:.10f} - the backends produce different vectors",
    )
    checks.check(
        worst_gap < 1e-4,
        f"elementwise agreement within 1e-4 (worst {worst_gap:.2e})",
        f"worst elementwise gap {worst_gap:.3e}",
    )
    # Guards the specific silent-drift case: chroma's onnx encoder implements
    # one model only, so a custom EMBEDDING_MODEL with the onnx backend falls
    # back to torch in _load_onnx. If that guard ever regressed, the vectors
    # above would still agree - but only because the fallback fired. This
    # asserts the guard's own precondition so the reason stays visible.
    checks.check(
        EMBEDDING_BACKEND in {"onnx", "torch"},
        f"EMBEDDING_BACKEND is a known backend ({EMBEDDING_BACKEND})",
    )
    print(
        f"    backend={EMBEDDING_BACKEND}, {len(probes)} probes, "
        f"worst cosine {worst_cosine:.10f}, worst elementwise {worst_gap:.2e}"
    )

    # --- report -----------------------------------------------------------
    print()
    print("=" * 74)
    print(f"  SIMILARITY_FLOOR = {SIMILARITY_FLOOR}")
    if checks.failures:
        print(f"FAILED  {len(checks.failures)} of {checks.passed + len(checks.failures)}")
        for failure in checks.failures:
            print(f"  - {failure}")
        return 1
    print(f"OK      {checks.passed} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
