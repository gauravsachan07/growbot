"""Phase 3 - Chroma store (ingestion step 4).

Writes chunk text, vectors and metadata to a persistent Chroma collection at
`CHROMA_PATH` named `hdfc_mf_faq` (architecture §5.4).

`source_url` is mandatory on every record: it is the citation the UI shows, and
architecture §13's failure table treats a missing citation as a defect rather
than a cosmetic gap. `_require_citation` refuses to write a record without one.

Rebuild is a delete-then-write, never an append, so running ingestion twice
cannot inflate the collection or leave stale chunks from a changed source page.
Record ids are derived from `source_url` + `chunk_index`, which makes a rebuild
deterministic.

Run directly to build the index from the source list:

    python -m growbot.ingest.index
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

from growbot.config import CHROMA_PATH, COLLECTION_NAME, SOURCES_CSV
from growbot.ingest.chunk import Chunk, chunk_documents
from growbot.ingest.embed import embed_texts
from growbot.ingest.load import load_documents

log = logging.getLogger("growbot.ingest.index")

#: Architecture §6.2 specifies cosine. With cosine distance Chroma reports
#: `1 - cosine_similarity`, so Phase 6 recovers similarity as `1 - distance`.
DISTANCE_SPACE = "cosine"

#: Chroma only accepts str / int / float / bool metadata values.
_SCALARS = (str, int, float, bool)

#: How many records to send per `add` call.
DEFAULT_BATCH_SIZE = 64


# ---------------------------------------------------------------------------
# Client and collection
# ---------------------------------------------------------------------------


def get_client(path: Path | str = CHROMA_PATH):
    """Open (or create) the persistent Chroma client."""
    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError as exc:  # pragma: no cover - setup guidance
        raise ImportError(
            'chromadb is not installed. Run: pip install -e ".[rag]"'
        ) from exc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(path),
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )


def _create_collection(client, name: str):
    """Create the collection with a cosine index.

    ChromaDB moved the index-space setting from `metadata={"hnsw:space": ...}`
    to `configuration={"hnsw": {"space": ...}}`. Try the modern form and fall
    back so the code works across the versions a classmate is likely to
    install. `describe_space` reports what was actually applied, because a
    silently-ignored keyword argument would otherwise leave L2 in place while
    Phase 6 reads the result as a cosine distance.
    """
    attempts: list[dict[str, Any]] = [
        {"configuration": {"hnsw": {"space": DISTANCE_SPACE}}},
        {"metadata": {"hnsw:space": DISTANCE_SPACE}},
        {},
    ]
    last: Exception | None = None
    for kwargs in attempts:
        try:
            collection = client.create_collection(name=name, **kwargs)
            if not kwargs:
                log.warning(
                    "could not request a %s index; falling back to the client "
                    "default. Phase 6 must not assume 1 - distance.",
                    DISTANCE_SPACE,
                )
            return collection
        except Exception as exc:  # noqa: BLE001 - version differences
            last = exc
    raise RuntimeError(f"could not create collection {name!r}: {last}")


def chroma_version() -> str:
    """Installed chromadb version, for the demo README."""
    try:
        import chromadb

        return str(chromadb.__version__)
    except Exception:  # noqa: BLE001
        return "unknown"


def describe_space(collection) -> str:
    """Best-effort report of the index space actually in use."""
    for attribute in ("configuration_json", "configuration"):
        value = getattr(collection, attribute, None)
        if isinstance(value, dict) and value.get("hnsw"):
            return str(value["hnsw"].get("space", "unknown"))
        if value is not None:
            return str(value)[:80]
    metadata = getattr(collection, "metadata", None) or {}
    if isinstance(metadata, dict) and "hnsw:space" in metadata:
        return str(metadata["hnsw:space"])
    return "client default (unverified)"


def delete_collection(client) -> None:
    """Drop the collection if it exists. Never raises."""
    try:
        client.delete_collection(COLLECTION_NAME)
        log.info("deleted existing collection %s", COLLECTION_NAME)
    except Exception:  # noqa: BLE001 - absent collection is fine
        log.debug("no existing collection %s", COLLECTION_NAME)


def get_collection(path: Path | str = CHROMA_PATH, create: bool = False):
    """Return the collection, optionally creating it if absent.

    Phase 6 uses `create=False` so a missing index surfaces as a clear
    "run ingest first" error instead of an empty-but-silent collection.
    """
    client = get_client(path)
    if create:
        try:
            return client.get_collection(COLLECTION_NAME)
        except Exception:  # noqa: BLE001
            return _create_collection(client, COLLECTION_NAME)
    return client.get_collection(COLLECTION_NAME)


# ---------------------------------------------------------------------------
# Record helpers
# ---------------------------------------------------------------------------


def chunk_id(chunk: Chunk) -> str:
    """Deterministic id from source_url + chunk_index.

    Stable across rebuilds (so nothing duplicates) and unique per document
    (guides all use scheme_name="general", so the index alone would collide).
    """
    url = chunk.metadata.get("source_url", "")
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
    index = chunk.metadata.get("chunk_index", "0")
    return f"{digest}-{index}"


def _require_citation(metadata: dict[str, Any]) -> str:
    url = metadata.get("source_url")
    if not url or not str(url).strip():
        raise ValueError(
            f"chunk {metadata.get('chunk_index')!r} has no source_url; "
            "every record must be citable"
        )
    return str(url).strip()


def sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Coerce metadata to the scalar types Chroma accepts."""
    clean: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, _SCALARS):
            clean[str(key)] = value
        else:
            clean[str(key)] = str(value)
    return clean


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------


def write_index(
    chunks: Sequence[Chunk],
    rebuild: bool = True,
    batch_size: int = DEFAULT_BATCH_SIZE,
    path: Path | str = CHROMA_PATH,
    embeddings: Sequence[Sequence[float]] | None = None,
) -> int:
    """Embed chunks and persist them to Chroma. Returns the record count.

    With `rebuild=True` the collection is deleted first, so a second run
    replaces the data rather than adding to it (architecture §5.4).

    `embeddings` lets a caller embed as its own visible stage - the Phase 4 CLI
    reports "N vectors" separately from "N records stored". When omitted the
    chunks are embedded here.
    """
    if not chunks:
        raise ValueError("refusing to write an empty index")

    client = get_client(path)
    if rebuild:
        delete_collection(client)
    collection = _create_collection(client, COLLECTION_NAME)

    ids = [chunk_id(chunk) for chunk in chunks]
    duplicates = len(ids) - len(set(ids))
    if duplicates:
        raise ValueError(
            f"{duplicates} duplicate chunk id(s); source_url + chunk_index "
            "must be unique per document"
        )

    # Fail before embedding anything if a record would be uncitable.
    for chunk in chunks:
        _require_citation(chunk.metadata)

    if embeddings is None:
        log.info("embedding %d chunk(s) with the shared MiniLM model", len(chunks))
        embeddings = embed_texts([chunk.text for chunk in chunks])

    if len(embeddings) != len(chunks):
        raise RuntimeError(
            f"got {len(embeddings)} vectors for {len(chunks)} chunks"
        )

    documents = [chunk.text for chunk in chunks]
    metadatas = [sanitize_metadata(chunk.metadata) for chunk in chunks]

    for start in range(0, len(chunks), batch_size):
        stop = start + batch_size
        collection.add(
            ids=ids[start:stop],
            embeddings=embeddings[start:stop],
            documents=documents[start:stop],
            metadatas=metadatas[start:stop],
        )

    count = collection.count()
    log.info("wrote %d record(s) to %s (space=%s)", count, COLLECTION_NAME, describe_space(collection))
    return count


def collection_count(path: Path | str = CHROMA_PATH) -> int:
    """Number of records in the collection, or 0 when it does not exist."""
    try:
        return get_collection(path).count()
    except Exception:  # noqa: BLE001 - missing collection is not an error here
        return 0


def is_indexed(path: Path | str = CHROMA_PATH) -> bool:
    """True when a non-empty collection exists. Phase 8 uses this to refuse."""
    return collection_count(path) > 0


def peek(limit: int = 1, path: Path | str = CHROMA_PATH) -> dict[str, Any]:
    """Return a few raw records, for the manual verification in Phase 3."""
    return get_collection(path).peek(limit=limit)


# ---------------------------------------------------------------------------
# CLI: `python -m growbot.ingest.index`
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m growbot.ingest.index",
        description="Phase 3 - load, chunk, embed and persist to Chroma.",
    )
    parser.add_argument("--sources", default=str(SOURCES_CSV))
    parser.add_argument("--no-rebuild", action="store_true", help="append instead of replace")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)-7s %(message)s",
    )

    report = load_documents(args.sources)
    if not report.documents:
        print("no documents loaded - nothing to index", file=sys.stderr)
        return 1

    chunks = chunk_documents(report.documents)
    if not chunks:
        print("no chunks produced - nothing to index", file=sys.stderr)
        return 1

    count = write_index(chunks, rebuild=not args.no_rebuild, batch_size=args.batch_size)

    print()
    print("Index report")
    print("-" * 78)
    print(f"  documents      : {len(report.documents)}")
    print(f"  chunks         : {len(chunks)}")
    print(f"  vectors stored : {count}")
    print(f"  collection     : {COLLECTION_NAME}")
    print(f"  index space    : {describe_space(get_collection())}")
    print(f"  chromadb       : {chroma_version()}")
    print(f"  persist path   : {CHROMA_PATH}")

    if count != len(chunks):
        print(f"  WARNING: collection count {count} != chunk count {len(chunks)}")
        return 1

    sample = peek(limit=1)
    ids = sample.get("ids") or []
    if ids:
        record = sample["documents"][0]
        meta = sample["metadatas"][0]
        print()
        print(f"  peek() id            : {ids[0]}")
        print(f"  peek() source_url    : {meta.get('source_url')}")
        print(f"  peek() scheme_name   : {meta.get('scheme_name')}")
        print(f"  peek() section       : {meta.get('section')}")
        print(f"  peek() fetched_at    : {meta.get('fetched_at')}")
        print(f"  peek() document[{len(record)} chars]: {record[:100]}...")

    return 0


if __name__ == "__main__":
    sys.exit(main())
