"""Phase 6 - Vector search (retrieval step 1).

Embeds the question with the same MiniLM model used at ingestion, takes top-k
chunks from `hdfc_mf_faq`, and filters on `scheme_name` when the question names
a scheme (architecture §6.1-6.2).

Two deliberate properties:

**Read-only.** Nothing here fetches, writes or re-ingests. The index was built
once by ``python -m growbot.ingest``; a question only reads it. Chroma is opened
lazily so importing this module stays cheap.

**One encoder, in one process.** The vector comes from
:func:`growbot.ingest.embed.embed_query`, the same function ingestion used. A
second model instance here would silently produce vectors in a different space
and every similarity score would be meaningless.

The scheme ``where``-filter is the main defence against cross-scheme bleed: five
schemes share almost identical wording, and "expense ratio" is not distinctive
on its own. When a filter returns nothing the search falls back to unfiltered
top-k and records that it did, so the assembler can mark the result weak rather
than pretend the filter worked (architecture §6.2).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Sequence

from growbot.config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    GENERAL_SCHEME,
    TOP_K,
    detect_scheme,
)
from growbot.ingest.embed import embed_query

log = logging.getLogger("growbot.retrieve.query")

__all__ = ["Hit", "SearchResult", "search", "get_collection"]


@dataclass(frozen=True)
class Hit:
    """One retrieved chunk, with the metadata the UI and generator need."""

    rank: int
    text: str
    similarity: float
    scheme_name: str
    scheme_category: str = ""
    section: str = ""
    doc_type: str = ""
    source_url: str = ""
    fetched_at: str = ""

    @property
    def is_general(self) -> bool:
        """True for the AMFI guide chunks, which belong to no single scheme."""
        return self.scheme_name in ("", GENERAL_SCHEME)

    @property
    def label(self) -> str:
        return f"{self.scheme_name} | {self.section}".strip(" |")


@dataclass(frozen=True)
class SearchResult:
    """The outcome of one search, including how it was narrowed."""

    question: str
    hits: list[Hit]
    detected_schemes: list[str] = field(default_factory=list)
    filtered: bool = False
    fell_back: bool = False
    appended_scheme: bool = False
    top_k: int = 0

    @property
    def max_similarity(self) -> float:
        return self.hits[0].similarity if self.hits else 0.0

    @property
    def schemes_in_hits(self) -> list[str]:
        """Distinct real schemes present, ignoring the neutral guide chunks."""
        seen: list[str] = []
        for hit in self.hits:
            if not hit.is_general and hit.scheme_name and hit.scheme_name not in seen:
                seen.append(hit.scheme_name)
        return seen


@lru_cache(maxsize=1)
def _collection():
    """Open the collection once per process. Read-only, so no rebuild path."""
    from growbot.ingest.index import get_client

    client = get_client(CHROMA_PATH)
    return client.get_collection(COLLECTION_NAME)


def get_collection():
    """The indexed collection. Raises if ingest has never been run."""
    return _collection()


def _build_hits(raw: dict[str, Any]) -> list[Hit]:
    """Turn a Chroma query response into `Hit` objects, best first.

    Chroma returns parallel lists. Similarity is `1 - distance` because the
    collection was created with cosine space - see `describe_space` in
    `ingest/index.py`, which reports what was actually applied rather than
    assuming it.
    """
    ids = raw.get("ids") or [[]]
    documents = raw.get("documents") or [[]]
    metadatas = raw.get("metadatas") or [[]]
    distances = raw.get("distances") or [[]]

    docs = documents[0] if documents else []
    metas = metadatas[0] if metadatas else []
    dists = distances[0] if distances else []
    _ids = ids[0] if ids else []

    hits: list[Hit] = []
    for index, document in enumerate(docs):
        meta = dict(metas[index]) if index < len(metas) and metas[index] else {}
        distance = float(dists[index]) if index < len(dists) else 1.0
        hits.append(
            Hit(
                rank=index + 1,
                text=document or "",
                similarity=round(1.0 - distance, 4),
                scheme_name=str(meta.get("scheme_name", "")),
                scheme_category=str(meta.get("scheme_category", "")),
                section=str(meta.get("section", "")),
                doc_type=str(meta.get("doc_type", "")),
                source_url=str(meta.get("source_url", "")),
                fetched_at=str(meta.get("fetched_at", "")),
            )
        )
    hits.sort(key=lambda hit: hit.similarity, reverse=True)
    for position, hit in enumerate(hits, start=1):
        hits[position - 1] = Hit(**{**hit.__dict__, "rank": position})
    return hits


def search(
    question: str,
    top_k: int | None = None,
    append_scheme: bool = False,
    detected: Sequence[str] | None = None,
) -> SearchResult:
    """Embed `question` and return the top-k chunks, best match first.

    `append_scheme` implements architecture §6.1's optional "append the detected
    scheme name to the query for better alignment". It is off by default -
    every chunk already begins with the scheme name, so appending it tends to
    match the chunk *prefix* rather than its content, and the metadata filter
    already guarantees the right scheme. See the note in `__main__` for the
    measured comparison.

    `detected` overrides scheme detection. It exists for one caller: a
    follow-up that names no fund, where `growbot.memory` has resolved the scheme
    from the conversation instead. Passing `[]` is meaningful and different from
    omitting it - it says "this question is about no single fund", which the
    assembler then treats as ambiguous. Omitting it detects from the text as
    before, which is the path every non-conversational caller takes.
    """
    if not question or not question.strip():
        raise ValueError("cannot search an empty question")

    k = top_k or TOP_K
    if detected is None:
        detected = detect_scheme(question)
    else:
        detected = list(detected)
    text = question
    if append_scheme and len(detected) == 1:
        text = f"{question} {detected[0]}"

    vector = embed_query(text)
    collection = get_collection()

    filtered = False
    fell_back = False
    raw: dict[str, Any] = {}

    # A question naming exactly one scheme gets a metadata filter. Two or more
    # is ambiguous, so no filter - the assembler flags that as weak instead.
    if len(detected) == 1:
        raw = collection.query(
            query_embeddings=[vector],
            n_results=k,
            where={"scheme_name": detected[0]},
            include=["documents", "metadatas", "distances"],
        )
        filtered = True
        if not (raw.get("ids") or [[]])[0]:
            log.info(
                "scheme filter for %s returned nothing; falling back to unfiltered",
                detected[0],
            )
            filtered = False
            fell_back = True

    if not raw or not (raw.get("ids") or [[]])[0]:
        raw = collection.query(
            query_embeddings=[vector],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

    hits = _build_hits(raw)
    if filtered and len(hits) < k:
        # Kept rather than topped up from other schemes: mixing in a different
        # fund's chunks would poison the context, and a thin context is the
        # honest outcome when a scheme simply has less to say.
        log.info(
            "scheme filter returned %d of %d requested chunk(s) for %s",
            len(hits), k, detected[0],
        )

    return SearchResult(
        question=question,
        hits=hits,
        detected_schemes=detected,
        filtered=filtered,
        fell_back=fell_back,
        appended_scheme=bool(append_scheme and len(detected) == 1),
        top_k=k,
    )
