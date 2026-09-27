"""Phase 2 - Chunking (ingestion step 2).

Structure-aware recursive split of loaded documents into embeddable chunks
(docs/architecture.md §5.2):

  1. split on headings, then blank lines, then sentences
  2. target 400-512 characters per chunk
  3. 80 characters of overlap
  4. never split a table mid-row
  5. prefix every chunk with ``{scheme_name} | {section}``

The prefix is not decoration: five HDFC schemes share nearly identical wording
("Expense ratio", "Exit load"), so the scheme name has to be part of the text
that gets embedded or two funds' numbers become indistinguishable. Chunks are
also built per document, so one chunk can never hold two schemes' figures.

A heading is a *preferred* boundary, not a forced one. Scheme pages have many
short sections ("Return calculator", "Fund management"); cutting at every
heading produced 41% scraps under the 400-character floor. A chunk therefore
only breaks at a heading once it has already reached CHUNK_MIN_SIZE, and any
heading a chunk crosses is re-emitted inline so the structure is not lost.

Run directly to inspect the chunked corpus:

    python -m growbot.ingest.chunk
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

from growbot.config import (
    CHUNK_MIN_SIZE,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    SOURCES_CSV,
)
from growbot.ingest.load import Document, load_documents

log = logging.getLogger("growbot.ingest.chunk")

#: A line containing this separator came from a <table> row in html_to_text.
TABLE_SEP = " | "

_HEADING_RE = re.compile(r"^(#{1,6})\s+(\S.*?)\s*$")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.?!])\s+(?=[A-Z0-9(\"'\u201c])")
_DEFAULT_SECTION = "overview"


# ---------------------------------------------------------------------------
# Data contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Chunk:
    """One embeddable unit. Phase 3 embeds `text` and stores `metadata`."""

    text: str
    metadata: dict[str, str]

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def section(self) -> str:
        return self.metadata.get("section", "")

    @property
    def scheme_name(self) -> str:
        return self.metadata.get("scheme_name", "")

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"[{self.char_count:>4} chars] {self.text[:70]}..."


@dataclass(frozen=True)
class _Section:
    """A heading and the body beneath it."""

    heading: str
    body: str


@dataclass
class _Open:
    """A chunk being filled. `items` are (section_heading, text) pairs.

    `length` mirrors exactly what _build_body will emit, including the
    `## heading` markers re-inserted for every section a chunk crosses. If the
    two disagree, chunks silently overshoot CHUNK_SIZE.
    """

    items: list[tuple[str, str]] = None  # type: ignore[assignment]
    length: int = 0

    def __post_init__(self) -> None:
        if self.items is None:
            self.items = []

    @property
    def section(self) -> str:
        return self.items[0][0] if self.items else _DEFAULT_SECTION

    def projected_add(self, heading: str, text: str) -> int:
        """Body length if (heading, text) were appended right now."""
        if not self.items:
            return len(text)
        extra = 2  # blank-line separator
        if heading != self.items[-1][0]:
            extra += len(f"## {heading}") + 2
        return self.length + extra + len(text)

    def add(self, heading: str, text: str) -> None:
        self.length = self.projected_add(heading, text)
        self.items.append((heading, text))

    def recompute(self) -> None:
        """Recalculate `length` after items were added or removed directly."""
        self.length = 0
        items, self.items = self.items, []
        for heading, text in items:
            self.add(heading, text)


# ---------------------------------------------------------------------------
# Step 1: headings
# ---------------------------------------------------------------------------


def split_sections(text: str) -> list[_Section]:
    """Split on headings (architecture §5.2 step 1).

    The heading line is not repeated in the body - it becomes the chunk's
    `section` and the second half of the ``{scheme_name} | {section}`` prefix.
    A heading with no body of its own (a page title) just relabels the next one.
    """
    sections: list[_Section] = []
    heading = _DEFAULT_SECTION
    body: list[str] = []

    for line in text.split("\n"):
        match = _HEADING_RE.match(line)
        if match:
            joined = "\n".join(body).strip()
            if joined:
                sections.append(_Section(heading=heading, body=joined))
            heading = match.group(2).strip()
            body = []
        else:
            body.append(line)

    joined = "\n".join(body).strip()
    if joined:
        sections.append(_Section(heading=heading, body=joined))

    return sections


# ---------------------------------------------------------------------------
# Step 2: blocks (blank lines, then sentences) - tables stay whole
# ---------------------------------------------------------------------------


def split_blocks(body: str) -> list[str]:
    """Split a section body on blank lines, keeping table rows together."""
    blocks: list[str] = []

    for paragraph in body.split("\n\n"):
        table_run: list[str] = []
        prose_run: list[str] = []

        for line in paragraph.split("\n"):
            if not line.strip():
                continue
            if TABLE_SEP in line:
                if prose_run:
                    blocks.append("\n".join(prose_run))
                    prose_run.clear()
                table_run.append(line)
            else:
                if table_run:
                    blocks.append("\n".join(table_run))
                    table_run.clear()
                prose_run.append(line)

        if table_run:
            blocks.append("\n".join(table_run))
        if prose_run:
            blocks.append("\n".join(prose_run))

    return blocks


def _split_table(block: str, cap: int) -> list[str]:
    """Split an oversized table at row boundaries - never mid-row.

    Architecture §5.2.4 keeps a table that overruns the cap in one piece, which
    is right for an exit-load slab. A 300-row holdings list would become one
    enormous chunk, so past the cap it is divided between whole rows.
    """
    if len(block) <= cap:
        return [block]

    rows = [row for row in block.split("\n") if row.strip()]
    pieces: list[str] = []
    current: list[str] = []
    length = 0

    for row in rows:
        addition = len(row) + (1 if current else 0)
        if current and length + addition > cap:
            pieces.append("\n".join(current))
            current, length = [row], len(row)
        else:
            current.append(row)
            length += addition

    if current:
        pieces.append("\n".join(current))
    return pieces


def _hard_wrap(text: str, cap: int) -> list[str]:
    """Split a pathologically long line on word boundaries."""
    words = text.split()
    pieces: list[str] = []
    current: list[str] = []
    length = 0

    for word in words:
        addition = len(word) + (1 if current else 0)
        if current and length + addition > cap:
            pieces.append(" ".join(current))
            current, length = [word], len(word)
        else:
            current.append(word)
            length += addition

    if current:
        pieces.append(" ".join(current))
    return pieces or [text]


def _split_sentences(block: str, cap: int) -> list[str]:
    """Greedily pack sentences into pieces of at most `cap` characters."""
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(block) if s.strip()]
    if not sentences:
        return []

    pieces: list[str] = []
    current: list[str] = []
    length = 0

    for sentence in sentences:
        if len(sentence) > cap:
            if current:
                pieces.append(" ".join(current))
                current, length = [], 0
            pieces.extend(_hard_wrap(sentence, cap))
            continue
        addition = len(sentence) + (1 if current else 0)
        if current and length + addition > cap:
            pieces.append(" ".join(current))
            current, length = [sentence], len(sentence)
        else:
            current.append(sentence)
            length += addition

    if current:
        pieces.append(" ".join(current))
    return pieces


def split_oversized(block: str, cap: int) -> list[str]:
    """Break a block larger than the cap into cap-sized pieces."""
    if len(block) <= cap:
        return [block]
    if TABLE_SEP in block:
        return _split_table(block, cap)
    return _split_sentences(block, cap)


# ---------------------------------------------------------------------------
# Step 3: pack into chunks with overlap
# ---------------------------------------------------------------------------


def _trim_partial_row(candidate: str) -> str:
    """Drop a half-carried table row from the front of an overlap.

    Sentence and word boundaries are the wrong axis inside a table, because a
    row is a *line*. Snapping to a word can therefore leave

        | 1% if redeemed within 1 year |

    at the top of a chunk while the `Exit load` that gave the number its
    meaning stayed in the previous one. That is precisely the "PDF table
    garbage" mode architecture §13 warns about: a figure sitting in the context
    with nothing saying what it refers to. A grounding check cannot catch it,
    because the number itself is perfectly present - it is the label that went
    missing, and a label is not a number.

    Note the fragment still begins and ends with `|`, so "does it look like a
    row" is not a usable test - it looks exactly like one. The discriminator is
    *width*: a whole row of a two-column table has three pipes, and this one has
    two. The width is taken as the commonest among the rows in the window, which
    is why at least two are required; with fewer, nothing is inferred and the
    overlap is left exactly as it was.

    Prose overlaps are untouched: the first line has to start with `|` to be
    considered at all, and if it is already full width it is kept.
    """
    lines = candidate.split("\n")
    if len(lines) < 2:
        return candidate
    rows = [line.strip() for line in lines
            if line.strip().startswith("|") and line.strip().endswith("|")]
    if len(rows) < 2:
        return candidate
    widths = sorted(row.count("|") for row in rows)
    full_width = widths[len(widths) // 2]
    first = lines[0].strip()
    if not first.startswith("|") or first.count("|") >= full_width:
        return candidate
    rest = "\n".join(lines[1:]).strip()
    return rest if rest else ""


def _overlap_tail(text: str) -> str:
    """Last ~CHUNK_OVERLAP characters, snapped to a sentence or word start.

    Snapping stops the overlap beginning mid-word, which would otherwise embed a
    meaningless fragment. `_trim_partial_row` then handles the one case word
    boundaries get wrong on their own: a fragment of a table row.
    """
    if len(text) <= CHUNK_OVERLAP:
        return ""
    window = text[-CHUNK_OVERLAP:]

    for match in reversed(list(_SENTENCE_SPLIT_RE.finditer(window))):
        candidate = window[match.end():].strip()
        if candidate:
            return _trim_partial_row(candidate)

    space = re.search(r"\s", window)
    if space:
        return _trim_partial_row(window[space.end():].strip())
    return _trim_partial_row(window.strip())


def _build_body(items: Sequence[tuple[str, str]], first_section: str) -> str:
    """Join (section, block) pairs, re-emitting any heading the chunk crossed."""
    parts: list[str] = []
    previous = first_section

    for heading, block in items:
        if heading != previous:
            parts.append(f"## {heading}")
            previous = heading
        parts.append(block)

    return "\n\n".join(parts)


def _prefix(scheme: str, section: str) -> str:
    return f"{scheme} | {section}"


def _cap_for(scheme: str, section: str) -> int:
    """Body budget left once the `{scheme} | {section}` prefix is accounted for.

    The prefix is stored in the chunk text, so counting it here is what keeps
    the final chunk inside the 400-512 character target instead of overshooting
    by the length of the heading.
    """
    return max(120, CHUNK_SIZE - len(_prefix(scheme, section)) - 1)


def _iter_units(sections: Iterable[_Section], scheme: str) -> Iterator[tuple[str, str]]:
    """Yield (section_heading, block) with every block already within its cap.

    The cap is computed from the real `{scheme} | {section}` prefix, so a block
    can never be sized for one scheme and then packed into another.
    """
    for section in sections:
        cap = _cap_for(scheme, section.heading)
        for block in split_blocks(section.body):
            for piece in split_oversized(block, cap):
                if piece.strip():
                    yield section.heading, piece.strip()


# ---------------------------------------------------------------------------
# Step 4: document -> chunks
# ---------------------------------------------------------------------------


def _merge_undersized(groups: list[_Open], scheme: str) -> list[_Open]:
    """Fuse a short chunk into its neighbour when the result still fits.

    Scheme pages are full of genuinely short sections (one fund-manager card,
    one "Fund house" row). Cutting at every heading leaves those stranded well
    under the 400-character floor, so a short chunk pulls in as much of the
    *following* chunk as will fit - not necessarily all of it. `section` stays
    the first group's, and crossed headings are re-emitted inline by
    _build_body, so no structure is lost.
    """
    changed = True
    while changed:
        changed = False
        for index in range(len(groups) - 1):
            current, following = groups[index], groups[index + 1]
            if current.length >= CHUNK_MIN_SIZE:
                continue

            cap = _cap_for(scheme, current.section)
            taken = 0
            for heading, text in list(following.items):
                if current.projected_add(heading, text) > cap:
                    break
                current.add(heading, text)
                taken += 1

            if taken:
                del following.items[:taken]
                following.recompute()
                if not following.items:
                    del groups[index + 1]
                changed = True
                break

    return groups


def chunk_document(document: Document) -> list[Chunk]:
    """Turn one loaded document into prefixed, metadata-carrying chunks."""
    scheme = document.metadata.get("scheme_name") or "general"
    groups: list[_Open] = []
    open_chunk = _Open()
    cap = _cap_for(scheme, _DEFAULT_SECTION)

    def close() -> None:
        nonlocal open_chunk
        if open_chunk.items:
            groups.append(open_chunk)
            open_chunk = _Open()

    for heading, block in _iter_units(split_sections(document.text), scheme):
        if open_chunk.items:
            projected = open_chunk.projected_add(heading, block)
            starts_new_section = heading != open_chunk.section

            # Hard rule: never exceed the cap, because the prefix is already
            # counted against it. Soft rule: when a chunk is big enough, break
            # cleanly at a heading so it stays topically coherent.
            must_split = projected > cap
            prefer_split = starts_new_section and open_chunk.length >= CHUNK_MIN_SIZE

            if must_split or prefer_split:
                # Overlap comes from the tail of the previous rendered chunk.
                previous_body = (
                    _build_body(groups[-1].items, groups[-1].section) if groups else ""
                )
                close()
                # The next chunk starts in this block's section, so budget
                # against that prefix rather than the emptied one.
                cap = _cap_for(scheme, heading)
                tail = _overlap_tail(previous_body)
                if tail and len(tail) + len(block) + 2 <= cap:
                    open_chunk.add(heading, tail)

        open_chunk.add(heading, block)

    close()

    chunks: list[Chunk] = []
    for group in _merge_undersized(groups, scheme):
        first = group.section
        text = f"{_prefix(scheme, first)}\n{_build_body(group.items, first)}".strip()
        metadata = dict(document.metadata)
        metadata["scheme_name"] = scheme
        metadata["section"] = first
        metadata["chunk_index"] = str(len(chunks))
        chunks.append(Chunk(text=text, metadata=metadata))

    total = len(chunks)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = str(index)
        chunk.metadata["chunk_count"] = str(total)

    return chunks


def chunk_documents(documents: Sequence[Document]) -> list[Chunk]:
    """Chunk every document, preserving document order."""
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document))
    log.info("chunked %d document(s) into %d chunk(s)", len(documents), len(chunks))
    return chunks


# ---------------------------------------------------------------------------
# CLI: `python -m growbot.ingest.chunk`
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m growbot.ingest.chunk",
        description="Phase 2 - load sources.csv, then split into chunks.",
    )
    parser.add_argument("--sources", default=str(SOURCES_CSV))
    parser.add_argument("--samples", type=int, default=2, help="chunks to print in full")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)-7s %(message)s",
    )

    report = load_documents(args.sources)
    chunks = chunk_documents(report.documents)
    if not chunks:
        print("no chunks produced - run the loading stage first")
        return 1

    sizes = [c.char_count for c in chunks]
    short = [c for c in chunks if c.char_count < CHUNK_MIN_SIZE]
    oversized = [c for c in chunks if c.char_count > CHUNK_SIZE]

    print()
    print(f"Chunk report - {len(report.documents)} document(s) -> {len(chunks)} chunk(s)")
    print("-" * 78)
    print(f"  chars: min {min(sizes)}  median {sorted(sizes)[len(sizes) // 2]}  max {max(sizes)}")
    print(f"  under CHUNK_MIN_SIZE ({CHUNK_MIN_SIZE}): {len(short)}")
    print(f"  over CHUNK_SIZE ({CHUNK_SIZE}): {len(oversized)}")
    print()

    for scheme, count in Counter(c.scheme_name for c in chunks).most_common():
        print(f"  {count:>4}  {scheme}")

    for index, chunk in enumerate(chunks[: args.samples], start=1):
        print()
        print("=" * 78)
        print(f"Sample chunk {index}  ({chunk.char_count} chars)")
        print(
            f"metadata: scheme_name={chunk.metadata['scheme_name']!r} "
            f"section={chunk.metadata['section']!r} doc_type={chunk.metadata['doc_type']!r}"
        )
        print("-" * 78)
        print(chunk.text)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
