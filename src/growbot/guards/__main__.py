"""Phase 5 - guard checks.

    python -m growbot.guards

Self-contained and assertion-based: a non-zero exit means a guard regressed.
Nothing here imports chromadb, sentence-transformers or an LLM client, so the
whole file runs in well under a second.

Four things are checked, in the order they would bite in a demo:

1. The four cases from implementation.md's "Done when".
2. Every fact question in PRD §12 must still be allowed. This is the
   over-refusal regression guard - a guard that blocks "expense ratio of HDFC
   Large Cap?" is worse than no guard at all.
3. PII handling, including that a PAN never appears in the payload and that
   nothing reaches disk.
4. Payload contract: one allowlisted URL, at most three sentences, disclaimer
   present.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from growbot.config import DISCLAIMER, EDU_LINK, FACTSHEET_LINKS, PROJECT_ROOT
from growbot.guards.intent import (
    classify,
    detect_pii,
    detect_scheme,
    display_name,
    guard,
    redact,
)

#: Every URL the guard is ever allowed to emit.
ALLOWED_URLS = {EDU_LINK, *FACTSHEET_LINKS.values()}

#: The seven fact questions from PRD §12, which must never be refused.
PRD_FACT_QUESTIONS = [
    "Expense ratio of HDFC Large Cap Fund Direct Growth?",
    "Exit load of HDFC Small Cap Fund Direct Growth?",
    "Minimum SIP for HDFC Balanced Advantage Fund Direct Growth?",
    "Lock-in period for HDFC ELSS Tax Saver Direct Growth?",
    "Riskometer / risk level of HDFC Equity Fund (Flexi Cap) Direct Growth?",
    "Benchmark of HDFC Large Cap Fund Direct Growth?",
    "How to download a capital-gains statement?",
]

#: Messages chosen to attack the guards from both sides: numbers that look
#: like identifiers, opinions dressed as questions, and predictions phrased in
#: the present tense. Each entry is (message, expected verdict).
ADVERSARIAL: list[tuple[str, str]] = [
    # must stay allowed - numbers that are not identifiers
    ("What is the expense ratio, 1.03%?", "allow"),
    ("NAV is 123.4567 today", "allow"),
    ("Benchmark of HDFC Large Cap is NIFTY 100", "allow"),
    ("Which fund has the lowest expense ratio?", "allow"),
    ("Which scheme has the better expense ratio?", "allow"),
    ("Is HDFC Small Cap Fund risky?", "allow"),
    ("Riskometer level of HDFC Equity Fund?", "allow"),
    ("What is the exit load after 1 year?", "allow"),
    ("What does buy and hold mean?", "allow"),
    ("As on 2026-09-27 what was the NAV?", "allow"),
    # must be refused as advice
    ("Should I buy HDFC Small Cap Fund?", "advice"),
    ("Which is better, Large Cap or Small Cap?", "advice"),
    ("Recommend a fund for my portfolio", "advice"),
    ("Is HDFC Small Cap a good investment?", "advice"),
    ("How much should I invest monthly?", "advice"),
    # must be refused as returns
    ("Which fund has the best returns?", "returns"),
    ("Compare HDFC Large Cap and Small Cap returns", "returns"),
    ("Will HDFC Small Cap give good returns?", "returns"),
    ("How much will I get in 5 years?", "returns"),
    ("What is the 1 year return of HDFC Large Cap?", "allow"),
]

#: Synthetic identifiers. Not real people's data.
PII_SAMPLES = [
    ("pan", "My PAN is ABCDE1234F, can you check the fund?"),
    ("aadhaar", "Aadhaar number 1234 5678 9012"),
    ("email", "mail me at priya.sharma@example.in"),
    ("phone", "call me on 9876543210"),
    ("otp", "the otp is 448213"),
    ("account_number", "my folio number is 6012345678"),
]


class Checks:
    """Minimal assertion collector so every failure is reported, not just the first."""

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


def sentence_count(text: str) -> int:
    """Sentences in the refusal body, ignoring the trailing disclaimer line."""
    body = text.replace(DISCLAIMER, "").strip()
    if not body:
        return 0
    parts = re.split(r"(?<=[.!?])\s+", body)
    return len([p for p in parts if p.strip()])


def project_files() -> set[Path]:
    """Project files a stray disk write would show up in."""
    skip = {".venv", "venv", "__pycache__", "chroma", ".git", ".ruff_cache"}
    found: set[Path] = set()
    for path in PROJECT_ROOT.rglob("*"):
        if any(part in skip for part in path.parts):
            continue
        if path.is_file():
            found.add(path)
    return found


def main() -> int:
    checks = Checks()
    before = project_files()
    print("guard checks")
    print("=" * 74)

    # --- 1. the four "Done when" cases -------------------------------------
    print("\n  implementation.md 'Done when' cases")

    verdict = classify("Should I buy HDFC Small Cap Fund?")
    checks.equal(verdict, "advice", "buy question is advice")
    payload = guard("Should I buy HDFC Small Cap Fund?")
    checks.check(
        payload is not None and payload.mode == "refuse", "buy question refuses"
    )
    checks.check(
        payload is not None and payload.source_url == EDU_LINK,
        "buy refusal cites the education link",
        payload.source_url if payload else "no payload",
    )
    checks.equal(
        detect_scheme("Should I buy HDFC Small Cap Fund?"),
        ["HDFC Small Cap Fund - Direct Growth"],
        "buy question resolves the scheme",
    )
    print(f"    {verdict:<8} Should I buy HDFC Small Cap Fund?")

    verdict = classify("Which fund has the best returns?")
    checks.equal(verdict, "returns", "best-returns question is returns")
    payload = guard("Which fund has the best returns?")
    checks.check(
        payload is not None and payload.mode == "refuse",
        "best-returns question refuses",
    )
    checks.check(
        payload is not None and payload.source_url in ALLOWED_URLS,
        "best-returns refusal cites an allowlisted URL",
    )
    checks.check(
        payload is not None
        and not re.search(r"\b\d+(?:\.\d+)?\s?%", payload.text),
        "best-returns refusal states no percentage",
        payload.text if payload else "",
    )
    print(f"    {verdict:<8} Which fund has the best returns?")

    pan_message = "Is this right? PAN ABCDE1234F"
    verdict = classify(pan_message)
    checks.equal(verdict, "pii", "PAN-like string is PII")
    payload = guard(pan_message)
    checks.check(
        payload is not None and "ABCDE1234F" not in payload.text,
        "PAN value never reaches the payload",
    )
    print(f"    {verdict:<8} PAN-like string")

    verdict = classify("Expense ratio of HDFC Large Cap Fund Direct Growth?")
    checks.equal(verdict, "allow", "fact question is allowed")
    checks.check(guard("Expense ratio of HDFC Large Cap Fund Direct Growth?") is None,
                 "allowed question returns None so retrieval proceeds")
    print(f"    {verdict:<8} Expense ratio of HDFC Large Cap Fund Direct Growth?")

    # --- 2. PRD fact questions must not be over-refused --------------------
    print("\n  PRD §12 fact questions (over-refusal regression)")
    for question in PRD_FACT_QUESTIONS:
        verdict = classify(question)
        checks.equal(verdict, "allow", f"allowed: {question[:44]}")
        print(f"    {verdict:<8} {question}")

    # --- 3. adversarial ---------------------------------------------------
    print("\n  adversarial cases (numbers vs identifiers, opinion vs fact)")
    for message, expected in ADVERSARIAL:
        verdict = classify(message)
        checks.equal(verdict, expected, f"verdict for: {message[:44]}")
        print(f"    {verdict:<8} {message}")

    # --- 4. PII coverage ---------------------------------------------------
    print("\n  PII detection")
    for category, message in PII_SAMPLES:
        categories = detect_pii(message)
        checks.check(category in categories, f"detects {category}", str(categories))
        checks.equal(classify(message), "pii", f"{category} message is PII")
        print(f"    {category:<16} {message}")

    # A PII message that also asks for advice must still refuse as PII.
    both = "Should I buy HDFC Small Cap? My PAN is ABCDE1234F"
    checks.equal(classify(both), "pii", "PII outranks advice")

    redacted = redact("PAN ABCDE1234F and priya@example.in")
    checks.check("ABCDE1234F" not in redacted, "redact removes the PAN")
    checks.check("priya@example.in" not in redacted, "redact removes the email")
    print(f"    {'redacted':<16} {redacted}")

    # Values must not leak through the category list either.
    checks.check(
        not any("ABCDE1234F" in c for c in detect_pii(both)),
        "detect_pii returns categories, not values",
    )

    # --- 5. payload contract ----------------------------------------------
    print("\n  payload contract")
    messages = [
        "Should I buy HDFC Small Cap Fund?",
        "Which fund has the best returns?",
        "Is this right? PAN ABCDE1234F",
        "Should I invest in HDFC Large Cap or HDFC Flexi Cap?",
    ]
    for message in messages:
        payload = guard(message)
        if not checks.check(payload is not None, f"refuses: {message[:40]}"):
            continue
        assert payload is not None
        url = payload.source_url
        checks.check(url in ALLOWED_URLS, f"allowlisted url for: {message[:36]}", url)
        checks.check(url.startswith("https://"), f"https url for: {message[:36]}", url)
        checks.equal(url.count("https://"), 1, f"exactly one url for: {message[:36]}")
        checks.check(
            sentence_count(payload.text) <= 3,
            f"<=3 sentences for: {message[:36]}",
            f"{sentence_count(payload.text)} sentences",
        )
        checks.check(
            DISCLAIMER in payload.text, f"disclaimer present for: {message[:36]}"
        )
        keys = set(payload.to_dict())
        checks.check(
            {"mode", "text", "source_url", "last_updated", "scheme_name"} <= keys,
            f"§7.3 keys present for: {message[:36]}",
        )
        # A refusal has no source, so it must carry no source date. Asserting
        # the inverse - that every payload has an ISO date - used to pass only
        # because every refusal was stamping the current day, which asserted a
        # freshness the refusal had not earned.
        if payload.mode == "fact":
            checks.check(
                bool(re.match(r"\d{4}-\d{2}-\d{2}$", payload.last_updated)),
                f"ISO last_updated on a fact: {message[:36]}",
                payload.last_updated,
            )
        else:
            checks.check(
                payload.last_updated == "",
                f"refusal carries no source date: {message[:36]}",
                repr(payload.last_updated),
            )
        print(f"    {payload.reason:<8} {url[:58]}")

    # The canonical config key must never appear in user-facing prose.
    print("\n  prose does not leak the config key")
    for name in FACTSHEET_LINKS:
        payload = guard(f"Should I buy {name}?")
        if payload is None:
            continue
        checks.check(
            name not in payload.text,
            f"no canonical key in prose: {name[:34]}",
            payload.text[:80],
        )
        checks.check(
            display_name(name) in payload.text,
            f"uses the trimmed name: {name[:34]}",
        )
        # the payload field itself stays canonical, for Phase 6's filter
        checks.equal(
            payload.scheme_name, name, f"chip stays canonical: {name[:34]}"
        )
        print(f"    {payload.scheme_name}")

    # Multi-scheme question must not arbitrarily pick one factsheet.
    multi = guard("Should I invest in HDFC Large Cap or HDFC Flexi Cap?")
    checks.check(
        multi is not None and multi.source_url == EDU_LINK,
        "multi-scheme refusal falls back to the education link",
    )
    checks.check(
        multi is not None and multi.scheme_name == "",
        "multi-scheme refusal names no single scheme",
    )

    # --- 6. nothing written to disk ---------------------------------------
    print("\n  side effects")
    after = project_files()
    new_files = {p for p in after - before if "guards" not in p.name}
    checks.check(not new_files, "no new files written", str(new_files)[:200])
    print(f"    {len(after)} project file(s) before and after, none new")

    # --- report ------------------------------------------------------------
    print()
    print("=" * 74)
    if checks.failures:
        print(f"FAILED  {len(checks.failures)} of {checks.passed + len(checks.failures)}")
        for failure in checks.failures:
            print(f"  - {failure}")
        return 1
    print(f"OK      {checks.passed} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

