"""Phase 1 - Loading (ingestion step 1).

Reads the auditable URL list in `data/sources.csv`, fetches each public page,
strips navigation/boilerplate, and returns `Document { text, metadata }` objects
for the chunker (Phase 2) to consume.

Design rules from docs/implementation.md (Phase 1):
  * Official AMC / AMFI / SEBI pages are the corpus. Groww URLs identify schemes
    and are only used as a fallback when the official page is unreachable.
  * Failures are skipped and logged - one dead URL never aborts ingestion.
  * Nothing is fetched at chat time.

Run directly to inspect the corpus:

    python -m growbot.ingest.load
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup, NavigableString, Tag

from growbot.config import (
    HTTP_TIMEOUT,
    MAX_DOC_CHARS,
    SOURCES_CSV,
    USER_AGENT,
)

log = logging.getLogger("growbot.ingest.load")

#: Columns required in sources.csv (architecture §7.1).
REQUIRED_COLUMNS = ("url", "scheme_name", "scheme_category", "doc_type")

#: Tags whose text is never scheme fact. `title` is included because its text
#: is captured separately as page metadata and would otherwise be duplicated as
#: a stray first line of the body.
DROP_TAGS = (
    "script", "style", "noscript", "svg", "canvas", "iframe",
    "nav", "footer", "aside", "form", "button", "input", "select",
    "textarea", "link", "meta", "template", "title",
)

#: Substrings that mark a wrapper as chrome rather than content.
#: Matched per class/id *token*, never against the whole attribute string, so a
#: Tailwind utility class such as "overflow-hidden" cannot accidentally match.
_BOILERPLATE_HINTS = (
    "nav", "navbar", "menu", "footer", "sidebar", "breadcrumb",
    "cookie", "consent", "social", "share", "newsletter", "subscribe",
    "related", "recommend", "promo", "banner", "advert", "popup",
    "modal", "skip-link", "skiplink", "copyright",
)

#: Only public web URLs. Guards against file://, ftp://, data: and friends.
ALLOWED_SCHEMES = ("http", "https")

#: Visible lines that are interface chrome, not scheme fact. Matched as whole
#: lines so a real value ("1.03%") can never be dropped.
_BOILERPLATE_LINES = frozenset(
    {
        "skip to main content", "skip to content", "main content",
        "compare", "education", "view details", "read more", "read more less",
        "load more", "more details are loading", "calculators", "watch",
        "learn", "search", "home", "next", "previous", "back", "menu",
        "all", "apply", "reset", "clear", "invest now", "know more",
        "view all", "show more", "i am ready to invest", "get started",
        "mr.", "ms.", "mrs.", "dr.", "disclaimer", "terms of use",
        "privacy notice", "contact us", "about amfi", "important updates",
    }
)

_WS_RUN = re.compile(r"[ \t\u00a0]+")
_BLANK_RUN = re.compile(r"\n{3,}")


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceRow:
    """One auditable row of sources.csv."""

    url: str
    scheme_name: str
    scheme_category: str
    doc_type: str


@dataclass(frozen=True)
class Document:
    """A loaded page. Phase 2 turns this into prefixed chunks."""

    text: str
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return len(self.text)

    def __str__(self) -> str:  # pragma: no cover - display only
        scheme = self.metadata.get("scheme_name", "?")
        return f"{scheme} ({self.char_count} chars)"


@dataclass(frozen=True)
class LoadFailure:
    """A source that was skipped, with the reason kept for the log."""

    url: str
    scheme_name: str
    reason: str


@dataclass
class LoadReport:
    """Outcome of one ingestion run's loading stage."""

    documents: list[Document] = field(default_factory=list)
    failures: list[LoadFailure] = field(default_factory=list)
    fetched_at: str = ""

    @property
    def total_chars(self) -> int:
        return sum(doc.char_count for doc in self.documents)

    def __len__(self) -> int:
        return len(self.documents)


class FetchError(RuntimeError):
    """A source could not be turned into a document."""


# ---------------------------------------------------------------------------
# Step 0: read the source list
# ---------------------------------------------------------------------------


def load_sources(path: Path | str = SOURCES_CSV) -> list[SourceRow]:
    """Parse sources.csv into SourceRow objects.

    Malformed rows are logged and skipped so one bad line cannot stop a build.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"source list not found: {path}")

    rows: list[SourceRow] = []
    # utf-8-sig strips a byte-order mark if present and is a no-op otherwise.
    # Excel and Notepad write BOMs on Windows, and a BOM turns the first column
    # name into a BOM-prefixed "url" - so a file that visibly *has* a url
    # column gets reported as missing one.
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        found = reader.fieldnames or []
        missing = [c for c in REQUIRED_COLUMNS if c not in found]
        if missing:
            raise ValueError(
                f"{path} is missing required column(s): {', '.join(missing)} "
                f"(header found: {', '.join(found) or 'none'})"
            )

        for line_no, raw in enumerate(reader, start=2):
            url = (raw.get("url") or "").strip()
            if not url:
                continue
            if urlparse(url).scheme.lower() not in ALLOWED_SCHEMES:
                log.warning("%s:%d skipped - not a public http(s) URL: %s", path.name, line_no, url)
                continue
            rows.append(
                SourceRow(
                    url=url,
                    scheme_name=(raw.get("scheme_name") or "").strip(),
                    scheme_category=(raw.get("scheme_category") or "").strip(),
                    doc_type=(raw.get("doc_type") or "").strip(),
                )
            )

    log.info("read %d source row(s) from %s", len(rows), path.name)
    return rows


# ---------------------------------------------------------------------------
# Step 1: fetch
# ---------------------------------------------------------------------------


def fetch(url: str, client: httpx.Client) -> tuple[bytes, str]:
    """GET a URL and return (body, content_type).

    Raises FetchError on a non-2xx response or a transport error, so the caller
    can log and skip instead of crashing the whole ingest.
    """
    try:
        response = client.get(url)
    except httpx.HTTPError as exc:
        raise FetchError(f"transport error: {type(exc).__name__}: {exc}") from exc

    if response.status_code >= 400:
        raise FetchError(f"HTTP {response.status_code}")
    if not response.is_success:
        raise FetchError(f"HTTP {response.status_code}")

    return response.content, response.headers.get("content-type", "").lower()


# ---------------------------------------------------------------------------
# Step 2: turn bytes into readable text
# ---------------------------------------------------------------------------


def _attr_text(tag: Tag, name: str) -> str:
    """Flatten an attribute to text.

    BeautifulSoup returns a list (AttributeValueList) for multi-valued attributes
    such as `class` and `rel`, so `tag.get("class") or ""` is not always a str.
    """
    value = tag.get(name)
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(str(part) for part in value)
    return str(value)


def _is_boilerplate(tag: Tag) -> bool:
    """True when a tag is page chrome rather than scheme fact.

    Note: `aria-hidden` is deliberately NOT treated as boilerplate. It is an
    accessibility flag, not a visibility guarantee - AMFI's article body sits in
    an aria-hidden collapsible, and dropping it deletes the whole page.
    """
    if tag.name in DROP_TAGS:
        return True
    role = _attr_text(tag, "role").lower()
    if role in ("navigation", "banner", "contentinfo", "search", "menu", "dialog"):
        return True
    for attr in ("class", "id"):
        for token in _attr_text(tag, attr).lower().split():
            if any(hint in token for hint in _BOILERPLATE_HINTS):
                return True
    return False


def _flatten_tables(soup: BeautifulSoup) -> None:
    """Rewrite each table as `cell | cell | cell` rows.

    Keeping one row per line is what lets Phase 2 hold an exit-load slab
    together instead of splitting a row's label away from its number.
    """
    for table in soup.find_all("table"):
        lines: list[str] = []
        for row in table.find_all("tr"):
            cells = [
                _WS_RUN.sub(" ", cell.get_text(" ", strip=True))
                for cell in row.find_all(["th", "td"])
            ]
            cells = [c for c in cells if c]
            if cells:
                lines.append(" | ".join(cells))
        table.replace_with(NavigableString("\n" + "\n".join(lines) + "\n"))


def _mark_headings(soup: BeautifulSoup) -> None:
    """Prefix h1-h6 with markdown hashes so Phase 2 can split on sections."""
    for level in range(1, 7):
        for heading in soup.find_all(f"h{level}"):
            text = _WS_RUN.sub(" ", heading.get_text(" ", strip=True))
            if text:
                heading.replace_with(NavigableString(f"\n{'#' * level} {text}\n"))


def clean_text(text: str) -> str:
    """Normalise whitespace without destroying paragraph or row structure."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS_RUN.sub(" ", text)
    kept = [
        line
        for line in (raw.strip() for raw in text.split("\n"))
        if line and line.lower() not in _BOILERPLATE_LINES
    ]
    text = _BLANK_RUN.sub("\n\n", "\n".join(kept))
    return text.strip()


def html_to_text(html: str) -> str:
    """Extract visible text from HTML, dropping nav/footer/scripts."""
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup.find_all(True):
        # A parent may already have been decomposed, taking its children with it.
        if getattr(tag, "decomposed", False):
            continue
        if _is_boilerplate(tag):
            tag.decompose()

    _flatten_tables(soup)
    _mark_headings(soup)

    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    body = soup.get_text(separator="\n")
    text = clean_text(body)

    # Keep the page title for chunker section fallback; it is not body prose.
    if title:
        text = f"# {clean_text(title)}\n\n{text}"
    return text


def pdf_to_text(blob: bytes) -> str:
    """Extract text from a PDF factsheet/KIM/SID.

    PDF table extraction is unreliable; architecture §13 says to prefer the
    HTML scheme page when both exist. This is the fallback path.
    """
    from pypdf import PdfReader  # imported lazily: only needed for PDF sources

    reader = PdfReader(io.BytesIO(blob))
    pages = [page.extract_text() or "" for page in reader.pages]
    if not any(pages):
        raise FetchError("PDF contains no extractable text (likely a scan)")
    return clean_text("\n\n".join(pages))


def to_text(body: bytes, content_type: str, url: str) -> str:
    """Dispatch on content type to the right text extractor."""
    if "pdf" in content_type or url.lower().endswith(".pdf"):
        return pdf_to_text(body)
    if "html" in content_type:
        return html_to_text(body.decode("utf-8", errors="replace"))
    if content_type.startswith("text/"):
        return clean_text(body.decode("utf-8", errors="replace"))
    raise FetchError(f"unsupported content type: {content_type or 'unknown'}")


# ---------------------------------------------------------------------------
# Step 3: one row -> one Document
# ---------------------------------------------------------------------------


def today_iso() -> str:
    """UTC date the corpus was fetched, ISO-8601 (architecture §7.2)."""
    return datetime.now(timezone.utc).date().isoformat()


def build_metadata(row: SourceRow, fetched_at: str) -> dict[str, str]:
    """Document-level metadata attached to every loaded page."""
    return {
        "scheme_name": row.scheme_name,
        "scheme_category": row.scheme_category,
        "source_url": row.url,
        "doc_type": row.doc_type,
        "fetched_at": fetched_at,
    }


def load_document(
    row: SourceRow,
    client: httpx.Client,
    fetched_at: str | None = None,
) -> Document:
    """Fetch and clean one source into a Document.

    Raises FetchError for anything that should be skipped rather than crash.
    """
    body, content_type = fetch(row.url, client)
    text = to_text(body, content_type, row.url)

    if not text.strip():
        raise FetchError("no visible text after cleaning")

    if len(text) > MAX_DOC_CHARS:
        log.warning(
            "truncated %s from %d to %d chars", row.url, len(text), MAX_DOC_CHARS
        )
        text = text[:MAX_DOC_CHARS]

    metadata = build_metadata(row, fetched_at or today_iso())
    return Document(text=text, metadata=metadata)


# ---------------------------------------------------------------------------
# Step 4: the whole loading stage
# ---------------------------------------------------------------------------


def load_documents(
    path: Path | str = SOURCES_CSV,
    client: httpx.Client | None = None,
    rows: list[SourceRow] | None = None,
) -> LoadReport:
    """Run the loading stage over every row in the source list.

    `rows` lets a caller that already validated the CSV (the Phase 4 CLI reads
    it to report the row count) hand the parsed rows in, so the file is read
    once rather than twice.
    """
    rows = load_sources(path) if rows is None else rows
    fetched_at = today_iso()
    report = LoadReport(fetched_at=fetched_at)

    owns_client = client is None
    client = client or httpx.Client(
        headers={"User-Agent": USER_AGENT, "Accept-Language": "en-IN,en;q=0.9"},
        timeout=HTTP_TIMEOUT,
        follow_redirects=True,
    )
    try:
        for row in rows:
            try:
                report.documents.append(load_document(row, client, fetched_at))
                log.info("loaded %s", row.scheme_name or row.url)
            except FetchError as exc:
                report.failures.append(
                    LoadFailure(row.url, row.scheme_name, str(exc))
                )
                log.warning("skipped %s - %s", row.url, exc)
            except Exception as exc:  # noqa: BLE001 - never abort the stage
                report.failures.append(
                    LoadFailure(row.url, row.scheme_name, f"unexpected {type(exc).__name__}: {exc}")
                )
                log.warning("skipped %s - unexpected %s", row.url, type(exc).__name__)
    finally:
        if owns_client:
            client.close()

    return report


# ---------------------------------------------------------------------------
# CLI: `python -m growbot.ingest.load`
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m growbot.ingest.load",
        description="Phase 1 - load sources.csv into cleaned Documents.",
    )
    parser.add_argument("--sources", default=str(SOURCES_CSV), help="path to sources.csv")
    parser.add_argument("--quiet", action="store_true", help="only print the summary")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)-7s %(message)s",
    )

    report = load_documents(args.sources)

    print()
    print(f"Loading report - fetched_at {report.fetched_at}")
    print("-" * 78)
    for doc in report.documents:
        meta = doc.metadata
        print(
            f"  {doc.char_count:>7,} chars  "
            f"{meta['scheme_name']:<48} "
            f"[{meta['doc_type']}]"
        )
        print(f"           {meta['source_url']}")

    if report.failures:
        print()
        print(f"Skipped {len(report.failures)} source(s):")
        for failure in report.failures:
            print(f"  - {failure.reason:<22} {failure.url}")

    print("-" * 78)
    print(
        f"{len(report.documents)} document(s), {report.total_chars:,} chars total, "
        f"{len(report.failures)} skipped"
    )
    return 0 if report.documents else 1


if __name__ == "__main__":
    sys.exit(main())
