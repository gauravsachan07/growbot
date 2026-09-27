"""Phase 4 - the ingestion CLI.

Wires the offline pipeline into one command:

    python -m growbot.ingest

load -> chunk -> embed -> store

This is the *only* supported way to populate the vector store. Chat never
ingests (architecture §3, §10): a question must never trigger a web fetch or a
rebuild, or the demo stops being reproducible and every answer costs a network
round trip.

Re-running replaces the collection rather than appending to it, so the demo is
rebuildable from `data/sources.csv` at any time.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import Counter
from typing import Callable, Sequence

from growbot.config import (
    CHROMA_PATH,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    PROJECT_ROOT,
    SOURCES_CSV,
)
from growbot.ingest.chunk import chunk_documents
from growbot.ingest.embed import embed_texts
from growbot.ingest.index import (
    chroma_version,
    collection_count,
    describe_space,
    get_collection,
    write_index,
)
from growbot.ingest.load import load_documents, load_sources

log = logging.getLogger("growbot.ingest")

#: Third-party loggers that are informative but drown the demo walkthrough.
#: Set to ERROR, not WARNING: huggingface_hub emits its unauthenticated-access
#: notice at WARNING, so a WARNING threshold would only hide the request spam.
_NOISY_LOGGERS = (
    "httpx", "httpcore", "urllib3", "chromadb", "sentence_transformers",
    "transformers", "huggingface_hub", "filelock", "onnxruntime", "tokenizers",
)


def configure_logging(quiet: bool = False) -> None:
    """Show our own stage logs clearly and quiet the libraries underneath."""
    logging.basicConfig(
        level=logging.WARNING if quiet else logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.ERROR)


def _short(path: str) -> str:
    """Trim the absolute project path so logs stay readable."""
    text = str(path)
    root = str(PROJECT_ROOT)
    return text[len(root) + 1:] if text.startswith(root) else text


def _timed(fn: Callable, *args, **kwargs):
    """Run a stage, returning (result, seconds)."""
    started = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - started


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m growbot.ingest",
        description="Growbot ingestion: load -> chunk -> embed -> store.",
    )
    parser.add_argument("--sources", default=str(SOURCES_CSV), help="path to sources.csv")
    parser.add_argument(
        "--no-rebuild",
        action="store_true",
        help="append to the existing collection instead of replacing it",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--quiet", action="store_true", help="summary line only")
    args = parser.parse_args(argv)

    configure_logging(args.quiet)
    overall = time.perf_counter()
    say = (lambda *a: None) if args.quiet else print

    say()
    say("growbot ingest")
    say("=" * 72)

    # --- stage 1: load -----------------------------------------------------
    try:
        rows = load_sources(args.sources)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    say(f"  {'sources':<9} {len(rows)} row(s) from {_short(args.sources)}")

    report, seconds = _timed(load_documents, args.sources, None, rows)
    say(
        f"  {'load':<9} {len(report.documents)} document(s), "
        f"{report.total_chars:,} chars, {len(report.failures)} skipped "
        f"({seconds:.1f}s)"
    )

    if not report.documents:
        say()
        print("ERROR: no documents loaded - nothing to index.", file=sys.stderr)
        if report.failures:
            print(
                "Every source failed. Check network access, and whether "
                "hdfcfund.com is blocking this host (HTTP 403).",
                file=sys.stderr,
            )
            for failure in report.failures[:5]:
                print(f"  - {failure.reason}: {failure.url}", file=sys.stderr)
        return 1

    # --- stage 2: chunk ----------------------------------------------------
    chunks, seconds = _timed(chunk_documents, report.documents)
    if not chunks:
        print("ERROR: documents loaded but produced no chunks.", file=sys.stderr)
        return 1

    sizes = sorted(c.char_count for c in chunks)
    say(
        f"  {'chunk':<9} {len(chunks)} chunk(s), {sizes[0]}-{sizes[-1]} chars, "
        f"median {sizes[len(sizes) // 2]}, cap {CHUNK_SIZE} ({seconds:.1f}s)"
    )

    # --- stage 3: embed ----------------------------------------------------
    previous = collection_count()
    try:
        vectors, seconds = _timed(embed_texts, [c.text for c in chunks])
    except ImportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    say(
        f"  {'embed':<9} {len(vectors)} vector(s) via {EMBEDDING_MODEL.split('/')[-1]} "
        f"({seconds:.1f}s)"
    )

    # --- stage 4: store ----------------------------------------------------
    stored, seconds = _timed(
        write_index, chunks, not args.no_rebuild, args.batch_size, CHROMA_PATH, vectors
    )
    say(
        f"  {'store':<9} {stored} record(s) -> {COLLECTION_NAME} at "
        f"{_short(str(CHROMA_PATH))} ({seconds:.1f}s)"
    )

    if stored == 0:
        print("ERROR: indexed zero records.", file=sys.stderr)
        return 1
    if stored != len(chunks):
        print(
            f"ERROR: collection holds {stored} record(s) but {len(chunks)} "
            "chunk(s) were produced.",
            file=sys.stderr,
        )
        return 1

    # --- summary -----------------------------------------------------------
    total = time.perf_counter() - overall
    say()
    say("-" * 72)
    for name, count in Counter(c.scheme_name for c in chunks).most_common():
        say(f"    {count:>4}  {name}")
    say()
    say(f"  index space  {describe_space(get_collection())}")
    say(f"  chromadb     {chroma_version()}")
    say(f"  replaced     {previous} previous record(s)" if previous
        else "  replaced     nothing (first build)")
    say(f"  elapsed      {total:.1f}s")
    say("-" * 72)
    say(f"Ingest complete: {stored} records. Retrieval never re-ingests.")
    say()

    return 0


if __name__ == "__main__":
    sys.exit(main())
