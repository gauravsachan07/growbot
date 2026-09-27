"""Phase 6 - Context assembly (retrieval step 2).

Turns a :class:`~growbot.retrieve.query.SearchResult` into the numbered context
block the generator will read, and decides whether the result is strong enough
to answer from (architecture §6.3).

Three rules are enforced here rather than in the prompt, because a prompt is a
suggestion and this is a gate:

**Exactly one citation URL.** The answer cites the ``source_url`` of the single
highest-scoring chunk. The other k-1 URLs stay in the context for the model to
reason over but are never promoted into the payload, because listing five links
is how a citation turns into an unverified suggestion.

**Last-updated comes from metadata, not the model.** It is read off the cited
chunk's ``fetched_at``. Nothing downstream can talk the date into being newer
than the fetch.

**Weak retrieval is refused, not answered thinly.** ``weak`` goes true when the
best similarity is under the floor, when the retrieved chunks contradict the
scheme the user asked about, or when the question named several funds and the
context spans several. The generator turns that into ``mode=refuse`` instead of
a confident answer built on the wrong fund.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from growbot.config import DISCLAIMER, SIMILARITY_FLOOR
from growbot.retrieve.query import Hit, SearchResult

log = logging.getLogger("growbot.retrieve.assemble")

__all__ = ["Assembly", "assemble", "format_context"]


@dataclass(frozen=True)
class Assembly:
    """Context plus the single citation and the weak-retrieval verdict."""

    context: str
    source_url: str
    last_updated: str
    scheme_name: str
    max_similarity: float
    hits: list[Hit] = field(default_factory=list)
    weak: bool = False
    weak_reasons: list[str] = field(default_factory=list)
    detected_schemes: list[str] = field(default_factory=list)

    @property
    def supporting(self) -> Hit | None:
        """The chunk the citation comes from: the best-scoring one."""
        return self.hits[0] if self.hits else None

    @property
    def summary(self) -> str:
        bits = [f"max_similarity={self.max_similarity:.3f}"]
        if self.weak:
            bits.append("weak=" + ",".join(self.weak_reasons))
        else:
            bits.append("strong")
        return " ".join(bits)


def format_context(hits: list[Hit]) -> str:
    """The architecture §6.3 block: numbered, one chunk per entry.

    Every entry carries its own source and fetch date so the model can attribute
    a fact to a page, while the *user-facing* answer still cites only one.
    """
    blocks: list[str] = []
    for hit in hits:
        blocks.append(
            f"[{hit.rank}] {hit.label}\n"
            f"{hit.text}\n"
            f"source: {hit.source_url}\n"
            f"fetched_at: {hit.fetched_at}"
        )
    return "\n\n".join(blocks)


def _weak_reasons(
    result: SearchResult,
    hits: list[Hit],
    floor: float,
) -> list[str]:
    """Every reason this context must not be answered from, in plain words."""
    reasons: list[str] = []

    if not hits:
        return ["no chunks retrieved"]

    best = hits[0].similarity
    if best < floor:
        reasons.append(f"similarity {best:.3f} below floor {floor:.2f}")

    detected = result.detected_schemes
    present = [
        hit.scheme_name for hit in hits
        if not hit.is_general and hit.scheme_name
    ]
    distinct = sorted(set(present))

    if len(detected) > 1:
        # The user asked about more than one fund. Whatever we retrieve is only
        # ever about some of them, so an answer would silently drop the rest.
        reasons.append(f"question names {len(detected)} schemes")
    elif len(detected) == 1:
        # Filtered and still clean? Then this cannot fire. It fires when the
        # filter had to fall back, or when a guide chunk outranks the scheme.
        if detected[0] not in distinct and distinct:
            reasons.append(
                f"asked about {detected[0]} but top chunks are "
                f"{', '.join(distinct)}"
            )
    elif hits[0].is_general:
        # No scheme named and the best chunk is an AMFI guide: the citation is
        # unambiguous, because the guide is not one fund's page. Extra scheme
        # chunks further down are background, not the thing being cited.
        pass
    elif len(distinct) > 1:
        # No scheme named, and the *cited* chunk is one specific fund. With
        # several funds in the context, that single citation could easily
        # misattribute the answer.
        reasons.append(f"unfiltered context spans {len(distinct)} schemes")

    if result.fell_back:
        reasons.append("scheme filter returned nothing; used unfiltered top-k")

    return reasons


def assemble(
    result: SearchResult,
    floor: float | None = None,
) -> Assembly:
    """Build the context block and decide whether it is strong enough.

    Returns an `Assembly` even when weak - the generator decides what to do
    with it, and it needs the context to explain *why* it is refusing.
    """
    threshold = SIMILARITY_FLOOR if floor is None else floor
    hits = result.hits
    best = hits[0] if hits else None

    reasons = _weak_reasons(result, hits, threshold)
    weak = bool(reasons)

    if weak:
        log.info("weak retrieval (%s): %s", ",".join(reasons), result.question[:60])

    return Assembly(
        context=format_context(hits),
        # Citation rule: one URL, from the best-scoring chunk.
        source_url=best.source_url if best else "",
        # Last-updated is metadata, never model output.
        last_updated=best.fetched_at if best else "",
        scheme_name=best.scheme_name if best else "",
        max_similarity=best.similarity if best else 0.0,
        hits=hits,
        weak=weak,
        weak_reasons=reasons,
        detected_schemes=list(result.detected_schemes),
    )
