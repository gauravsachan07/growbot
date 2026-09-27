"""Phase 6 - retrieval demo and similarity-floor calibration.

    python -m growbot.retrieve                 # the PRD evaluation questions
    python -m growbot.retrieve "your question" # one question, full detail
    python -m growbot.retrieve --sweep         # calibrate SIMILARITY_FLOOR

Read-only: opens the index built by ``python -m growbot.ingest`` and never
touches the network. No LLM is called (architecture §6, phase 6).
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Sequence

from growbot.config import SIMILARITY_FLOOR, TOP_K
from growbot.retrieve.assemble import assemble
from growbot.retrieve.query import search

#: Loading the model fires dozens of HuggingFace HTTP HEAD requests and an
#: unauthenticated-access warning. All of it is noise in a demo walkthrough;
#: real failures still surface as exceptions. Set to ERROR, not WARNING -
#: huggingface_hub reports its token notice at WARNING, so WARNING would
#: silence the request spam but still print the notice.
_NOISY_LOGGERS = (
    "httpx", "httpcore", "urllib3", "chromadb", "sentence_transformers",
    "transformers", "huggingface_hub", "filelock", "onnxruntime", "tokenizers",
)

#: PRD §12 evaluation questions 1-7. Q8/Q9 are guard refusals and never reach
#: retrieval, which is the point of Phase 5.
EVAL_QUESTIONS = [
    "Expense ratio of HDFC Large Cap Fund Direct Growth?",
    "Exit load of HDFC Small Cap Fund Direct Growth?",
    "Minimum SIP for HDFC Balanced Advantage Fund Direct Growth?",
    "Lock-in period for HDFC ELSS Tax Saver Direct Growth?",
    "Riskometer level of HDFC Equity Fund (Flexi Cap) Direct Growth?",
    "Benchmark of HDFC Large Cap Fund Direct Growth?",
    "How to download a capital-gains statement?",
]

#: Questions that are on-topic but too vague to answer. These are the ones the
#: similarity floor can never catch - they match the corpus well. The
#: mixed-scheme rule in `assemble` is what refuses them.
VAGUE_ON_TOPIC = [
    "tell me about HDFC funds",
    "hdfc",
    "how do mutual funds work in general",
]

#: Questions with no relationship to the corpus. These are the ones the
#: similarity floor *is* for.
JUNK = [
    "is it good?",
    "what should I do with my money",
    "give me market tips",
]

#: On-topic, but the corpus cannot answer them. Recorded so a future corpus
#: rebuild can be checked against a known gap rather than rediscovering it.
KNOWN_GAPS = {
    "How to download a capital-gains statement?":
        "no chunk in the corpus mentions capital gains; the CAS article is "
        "present but the PRD's wording does not reach it",
}


def _show(question: str, top_k: int | None, append_scheme: bool, verbose: bool) -> None:
    result = search(question, top_k=top_k, append_scheme=append_scheme)
    assembly = assemble(result)

    print()
    print(f"Q  {question}")
    print("-" * 74)
    flags = []
    if result.filtered:
        flags.append("filtered")
    if result.fell_back:
        flags.append("FELL BACK")
    if result.appended_scheme:
        flags.append("scheme-appended")
    print(
        f"   detected={result.detected_schemes or ['-']}  "
        f"k={len(result.hits)}/{result.top_k}  "
        f"[{','.join(flags) or 'unfiltered'}]"
    )
    for hit in result.hits:
        flag = "  <- cited" if hit.rank == 1 else ""
        print(f"   [{hit.rank}] {hit.similarity:.3f}  {hit.label[:58]}{flag}")
        if verbose:
            body = hit.text.replace("\n", " / ")
            print(f"        {body[:150]}...")
    print(f"   {assembly.summary}")
    if assembly.weak:
        for reason in assembly.weak_reasons:
            print(f"        weak: {reason}")
    else:
        print(f"   cite   {assembly.source_url}")
        print(f"   as of  {assembly.last_updated}")


def _band(label: str, questions: list[str], top_k: int | None) -> list[float]:
    print(f"\n  {label}")
    print(f"  {'-' * 66}")
    scores: list[float] = []
    for question in questions:
        assembly = assemble(search(question, top_k=top_k))
        scores.append(assembly.max_similarity)
        mark = "refuse" if assembly.weak else "answer"
        note = ""
        if assembly.weak:
            note = "  <- " + assembly.weak_reasons[0]
        print(f"  {assembly.max_similarity:>6.3f} {mark:<7} {question[:44]}{note}")
    return scores


def _sweep(top_k: int | None) -> int:
    """Show where each class of question actually lands, and what refuses it.

    The point of this table is that there are two different jobs and two
    different mechanisms. A threshold can separate junk from answerable text.
    It cannot separate a vague-but-on-topic question from a specific one,
    because "tell me about HDFC funds" is a *good* vector match for a corpus
    made of HDFC funds.
    """
    print("similarity sweep")
    print("=" * 74)
    print(f"  SIMILARITY_FLOOR = {SIMILARITY_FLOOR}   TOP_K = {top_k or TOP_K}")

    answerable_qs = [q for q in EVAL_QUESTIONS if q not in KNOWN_GAPS]
    answerable = _band("answerable (PRD §12 fact questions)", answerable_qs, top_k)
    vague = _band("on-topic but too vague to answer", VAGUE_ON_TOPIC, top_k)
    junk = _band("off-topic junk", JUNK, top_k)

    weak_answerable = min(answerable)
    junk_ceiling = max(junk)
    vague_peak = max(vague)

    print()
    print("=" * 74)
    print(f"  weakest answerable question   {weak_answerable:.3f}")
    print(f"  strongest junk question       {junk_ceiling:.3f}")
    print(f"  strongest vague question      {vague_peak:.3f}")
    print()
    if weak_answerable > junk_ceiling:
        print(f"  The floor's job: a threshold in {junk_ceiling:.3f}-{weak_answerable:.3f}")
        print(f"  rejects every junk question and keeps every answerable one.")
        inside = junk_ceiling < SIMILARITY_FLOOR < weak_answerable
        print(f"  {SIMILARITY_FLOOR} {'sits inside that band.' if inside else 'is OUTSIDE that band.'}")
    else:
        print("  The floor cannot separate these two classes by similarity alone.")
    print()
    print(f"  Vague questions peak at {vague_peak:.3f}, above most answerable ones, so no")
    print("  similarity threshold can refuse them. They are refused by the")
    print("  mixed-scheme / ambiguity rule in retrieve/assemble.py instead:")
    for question in VAGUE_ON_TOPIC:
        assembly = assemble(search(question, top_k=top_k))
        if assembly.weak:
            print(f"    {question[:40]:<42} {assembly.weak_reasons[0]}")

    if KNOWN_GAPS:
        print()
        print("  known corpus gaps (excluded from the band above):")
        for question, why in KNOWN_GAPS.items():
            assembly = assemble(search(question, top_k=top_k))
            mark = "refuse" if assembly.weak else "ANSWER (recheck this)"
            print(f"    {mark:<22} {question[:40]:<42} {assembly.max_similarity:.3f}")
            print(f"      {why}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m growbot.retrieve",
        description="Phase 6 - retrieve and assemble context. No LLM.",
    )
    parser.add_argument("question", nargs="*", help="one question; omit for the eval set")
    parser.add_argument("--top-k", type=int, default=None, help=f"default {TOP_K}")
    parser.add_argument(
        "--append-scheme",
        action="store_true",
        help="architecture 6.1 option: append the detected scheme to the query",
    )
    parser.add_argument("--sweep", action="store_true", help="calibrate the floor")
    parser.add_argument("-v", "--verbose", action="store_true", help="show chunk text")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="  %(message)s",
        stream=sys.stdout,
    )
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.ERROR)

    if args.sweep:
        return _sweep(args.top_k)

    questions = [" ".join(args.question)] if args.question else EVAL_QUESTIONS
    for question in questions:
        _show(question, args.top_k, args.append_scheme, args.verbose)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
