"""Phase 9 - UI checks.

    python -m growbot.ui.checks

Streamlit's own `AppTest` runs the real script headlessly, so these are not
mocks of the UI - the actual `app.py` executes and the actual widget tree is
inspected. What is substituted is the *model*, and only where a model call
would otherwise happen.

That split is deliberate:

- **Refusal questions run the real pipeline.** "Should I buy?" and the PAN case
  are refused by the guard, which does no I/O, so they cost nothing and prove
  the guard is genuinely wired into the page rather than a stub faking a
  refusal.
- **Fact questions stub `ask_with_trace`.** Rendering a fact card is a
  formatting concern, and Phase 8 already proves what the model returns. Paying
  for an LLM call in a test of markdown layout would add rate-limit flakiness
  and cover nothing new. To watch the fact path run for real, click an example
  in the app: `streamlit run src/growbot/ui/app.py`.

The invariant worth stating: the UI must not be able to answer a question
except by calling `ask()`. `_no_direct_retrieval` checks that statically,
because nothing at runtime would catch a stray `search()` call in a page.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

from growbot import ask as ask_mod
from growbot.guards.intent import AnswerPayload

APP_PATH = Path(__file__).with_name("app.py")

DISCLAIMER = "Facts-only. No investment advice."

#: Spec'd at implementation.md:320-322, including the third that must refuse.
EXPECTED_EXAMPLES = [
    "Expense ratio of HDFC Large Cap Fund (Direct Growth)?",
    "Lock-in for HDFC ELSS Tax Saver?",
    "How to download a capital-gains statement?",
]

#: A well-formed fact payload, standing in for a real model answer. The citation
#: and the date are the two fields the UI must show for a fact and must never
#: show for a refusal, so they are the interesting part here.
STUB_FACT = AnswerPayload(
    mode="fact",
    text=f"Expense ratio: 1.03%. {DISCLAIMER}",
    source_url="https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
    last_updated="2026-09-27",
    scheme_name="HDFC Large Cap Fund - Direct Growth",
    reason="",
)


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


def _app() -> AppTest:
    at = AppTest.from_file(str(APP_PATH), default_timeout=240)
    at.run()
    return at


def _body_text(at: AppTest) -> str:
    """Every piece of visible text on the page, flattened.

    Note: AppTest also exposes chat-bubble content through `at.markdown`, so
    walking `at.chat_message` as well would count every answer twice. Only the
    flattened view is used here; `_turn_text` reaches into a single turn when
    that is what the assertion is about.
    """
    parts: list[str] = []
    for element in ("title", "markdown", "caption", "info", "warning", "error"):
        for item in getattr(at, element, []):
            parts.append(str(getattr(item, "value", "")))
    return "\n".join(parts)


def _turn_text(at: AppTest, index: int) -> str:
    """Text of one chat turn. Even indices are questions, odd are answers."""
    if index >= len(at.chat_message):
        return ""
    chat = at.chat_message[index]
    parts = [str(m.value) for m in chat.markdown]
    parts += [str(c.value) for c in chat.caption]
    return "\n".join(parts)


def _links(at: AppTest) -> list[str]:
    """Every markdown link target on the page."""
    text = _body_text(at)
    return re.findall(r"\]\((https?://[^)]+)\)", text)


def _no_direct_retrieval(checks: Checks) -> None:
    """The UI may not reach the vector store or the LLM client itself.

    architecture.md §9: "UI talks only to a ask(question) -> AnswerPayload
    function. It does not access Chroma or the LLM directly." A stray
    `search(...)` in a page would be invisible at runtime, so it is checked in
    the source.
    """
    source = APP_PATH.read_text(encoding="utf-8")
    forbidden = {
        "chromadb": "vector store",
        "torch": "torch",
        "sentence_transformers": "embedding model",
        "gatherbot": "",
        "retrieve.query": "retrieval",
        "retrieve.assemble": "assembly",
        "generate.answer": "LLM client",
        "ingest.index": "index internals",
        "search(": "retrieval call",
        "assemble(": "assembly call",
    }
    for needle, what in forbidden.items():
        if not what:
            continue
        checks.check(
            f"import {needle}" not in source and f"{needle}." not in source,
            f"UI does not touch the {what}",
            f"found {needle!r} in app.py",
        )
    # The only Growbot imports allowed in a page.
    imports = re.findall(r"^from growbot[.\w]*", source, re.M)
    allowed_prefixes = (
        "from growbot.ask",
        "from growbot.config",
        "from growbot.guards.intent",
    )
    for line in imports:
        checks.check(
            line.startswith(allowed_prefixes),
            f"import stays within the presentation layer: {line}",
            "ask/config/guards only",
        )
    print(f"    {len(imports)} Growbot import(s), all in the allowed set")


def _check_load(checks: Checks) -> AppTest:
    print("\n  the page loads")
    at = _app()
    checks.equal([e.type for e in at.exception], [], "no exception on load")
    checks.equal([t.value for t in at.title], ["Growbot"], "title")
    return at


def _check_examples(checks: Checks, at: AppTest) -> None:
    print("\n  three example questions, clickable")
    labels = [b.label for b in at.button]
    checks.equal(labels, EXPECTED_EXAMPLES, "examples match the spec verbatim")
    keys = [b.key for b in at.button]
    checks.equal(keys, ["example_0", "example_1", "example_2"], "unique button keys")
    print(f"    {len(labels)} examples, keys {keys}")


def _check_disclaimer(checks: Checks, at: AppTest) -> None:
    print("\n  the disclaimer is visible before anything is asked")
    body = _body_text(at)
    checks.check(DISCLAIMER in body, "disclaimer on the empty page")
    checks.check(
        any(DISCLAIMER in str(i.value) for i in at.info),
        "disclaimer is a persistent banner, not a footnote",
    )
    body_low = body.lower()
    for phrase in ("expense ratio", "exit load", "lock-in", "five hdfc"):
        checks.check(phrase in body_low, f"welcome line mentions {phrase!r}")


def _check_fact_card(checks: Checks) -> None:
    print("\n  clicking an example renders a fact card")
    original = ask_mod.ask_with_trace
    ask_mod.ask_with_trace = lambda question, **kw: (
        STUB_FACT,
        ask_mod.Trace(stages=[ask_mod.Stage("generate", "called")]),
    )
    try:
        at = _app()
        at.button[0].click().run()
        checks.equal([e.type for e in at.exception], [], "no exception after click")
        body = _body_text(at)
        checks.check("1.03%" in body, "the answer text is shown")
        checks.check("2026-09-27" in body, "last-updated date is shown")
        checks.check("HDFC Large Cap Fund" in body, "scheme chip is shown")
        links = _links(at)
        checks.equal(len(links), 1, "exactly one hyperlink on the card")
        checks.check(
            links and links[0].startswith("https://groww.in/"),
            "the link is the source URL", str(links),
        )
        checks.check(DISCLAIMER in body, "disclaimer still present")
        print(f"    text + date + chip + 1 link ({links[0][:44] if links else 'none'}...)")
    finally:
        ask_mod.ask_with_trace = original


def _check_refusals(checks: Checks) -> None:
    print("\n  refusal questions run the real pipeline and show a refusal card")
    # No stub here on purpose: the guard does no I/O, so this is free and it
    # proves the page reaches the genuine guard.
    cases = [
        ("Should I buy HDFC Small Cap Fund?", "example button 3 / typed"),
        ("Which fund has the best returns?", "typed"),
    ]
    for question, how in cases:
        at = _app()
        at.chat_input[0].set_value(question).run()
        checks.equal(
            [e.type for e in at.exception], [], f"no exception: {question[:30]!r}"
        )
        body = _body_text(at)
        checks.check(DISCLAIMER in body, f"disclaimer present: {question[:26]!r}")
        checks.check(
            "worth buying" in body or "rank the five funds" in body,
            f"a refusal message is shown: {question[:26]!r}",
            body[-160:],
        )
        links = _links(at)
        checks.equal(len(links), 1, f"exactly one link on refusal: {question[:24]!r}")
        checks.check(
            bool(links) and "groww.in" not in links[0],
            f"refusal links to education, not a scheme page: {question[:20]!r}",
            str(links),
        )
        checks.check(
            "2026-09-27" not in body and "Last updated" not in body,
            f"no source date is claimed on a refusal: {question[:20]!r}",
        )
        print(f"    {how:<28} -> refusal card, 1 edu link, no date")


def _check_gap_example(checks: Checks) -> None:
    print("\n  example 3 is the known corpus gap and declines honestly")
    at = _app()
    at.button[2].click().run()          # "How to download a capital-gains statement?"
    checks.equal([e.type for e in at.exception], [], "no exception")
    answer = _turn_text(at, 1)
    checks.check(
        "capital-gains statement" in _turn_text(at, 0), "the question is echoed"
    )
    checks.check(
        "don't have a passage" in answer or "won't guess" in answer,
        "it declines instead of inventing a download guide",
        answer[:180],
    )
    # The specific thing to rule out: a plausible-looking download guide with a
    # scheme URL attached, which is exactly what a hallucinating bot produces.
    links = re.findall(r"\]\((https?://[^)]+)\)", answer)
    checks.equal(len(links), 1, "one educational link on the card")
    checks.check(
        bool(links) and "groww.in" not in links[0],
        "no scheme page is cited for a question the corpus cannot answer",
        str(links),
    )
    # Scan the prose only. Link markup is stripped first because the allowed
    # educational link necessarily contains "www.", which would trip a naive
    # pattern.
    prose = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", answer)
    checks.check(
        not re.search(
            r"\bstep \d|\bclick (here|on)\b|\blog ?in\b|\bdashboard\b", prose, re.I
        ),
        "no invented download instructions",
        prose[:180],
    )
    checks.check(DISCLAIMER in answer, "disclaimer present")
    print("    declined, no invented guide, no scheme URL, disclaimer present")


def _check_thread(checks: Checks) -> None:
    print("\n  the thread accumulates and re-renders without duplicating")
    at = _app()
    at.chat_input[0].set_value("Should I buy HDFC Small Cap Fund?").run()
    at.chat_input[0].set_value("My PAN is ABCDE1234F, what is the exit load?").run()
    checks.equal(len(at.chat_message), 4, "two questions, two answers")

    first_answer = _turn_text(at, 1)
    second_answer = _turn_text(at, 3)
    checks.equal(
        first_answer.count("worth buying"), 1, "the first answer appears once"
    )
    checks.equal(
        second_answer.count("personal identifiers"), 1,
        "the second answer appears once",
    )
    # Echoing the user's own words is correct - that is their message. What
    # must never happen is the identifier coming back inside an answer.
    checks.check(
        "ABCDE1234F" not in second_answer,
        "the PAN is not echoed back in the refusal",
        second_answer[:120],
    )
    checks.check(
        "ABCDE1234F" in _turn_text(at, 2),
        "the user still sees their own question",
    )
    checks.check(
        "redact" in second_answer.lower() or "discard" in second_answer.lower(),
        "the refusal says the identifier was discarded",
    )
    print("    2 turns, no duplicated cards, PAN never returns in an answer")


def _check_demo_pack(checks: Checks) -> None:
    """Phase 10's "Done when", enforced rather than asserted in prose.

    Two of the three conditions are about the docs agreeing with the code, so
    they are checkable and are checked here: the disclaimer file must contain
    the *actual* UI strings, and the sample Q&A must contain real citations.
    Documentation drifts silently; this fails loudly.
    """
    from growbot.config import DISCLAIMER
    from growbot.ui.app import LONG_DISCLAIMER

    root = Path(__file__).resolve().parents[3]

    # --- docs/disclaimer.md matches the UI --------------------------
    print("\n  docs/disclaimer.md matches the strings the UI shows")
    dpath = root / "docs" / "disclaimer.md"
    if not checks.check(dpath.exists(), "docs/disclaimer.md exists"):
        return
    text = dpath.read_text(encoding="utf-8")
    checks.check(
        DISCLAIMER in text,
        "the short banner string is present verbatim",
        repr(DISCLAIMER),
    )
    checks.check(
        LONG_DISCLAIMER in " ".join(text.split()),
        "the long PRD section 9 snippet is present verbatim",
        LONG_DISCLAIMER[:70],
    )
    checks.check(
        "mutual fund investments are subject to market risks" in text.lower(),
        "the market-risk line is present",
    )
    print(f"    both UI strings found verbatim ({len(text)} bytes)")

    # --- docs/sample_qa.md has citations ----------------------------
    print("\n  docs/sample_qa.md carries real answers and citations")
    qpath = root / "docs" / "sample_qa.md"
    if not checks.check(qpath.exists(), "docs/sample_qa.md exists"):
        return
    qa = qpath.read_text(encoding="utf-8")
    questions = re.findall(r"^## Q\d+\.", qa, re.M)
    checks.check(
        5 <= len(questions) <= 12, "holds a usable set of Q&A entries",
        f"{len(questions)} entries",
    )
    blocks = re.split(r"^## Q\d+\.", qa, flags=re.M)[1:]
    facts = [b for b in blocks if "**FACT**" in b]
    refusals = [b for b in blocks if "**REFUSAL**" in b]
    checks.check(len(facts) >= 4, "has several answered questions", str(len(facts)))
    checks.check(len(refusals) >= 2, "has at least two refusals", str(len(refusals)))

    # Every fact must carry exactly one source URL, and it must be a real one.
    sources_path = root / "data" / "sources.csv"
    corpus = {
        row["url"]
        for row in __import__("csv").DictReader(
            sources_path.open(encoding="utf-8-sig")
        )
    }
    for block in facts:
        urls = re.findall(r"^- \*\*Source:\*\* (\S+)", block, re.M)
        head = block.strip().splitlines()[0][:44]
        checks.equal(len(urls), 1, f"exactly one source cited: {head}")
        if urls:
            checks.check(
                urls[0] in corpus,
                f"cited URL is in data/sources.csv: {head}",
                urls[0],
            )
        checks.check(
            "**Last updated from sources:**" in block,
            f"fact carries a source date: {head}",
        )

    # The two refusals the spec explicitly asks for.
    joined = qa.lower()
    checks.check("should i buy" in joined, "includes a buy refusal")
    checks.check("best returns" in joined, "includes a best-returns refusal")
    for block in refusals:
        head = block.strip().splitlines()[0][:44]
        checks.check(
            "**Educational link:**" in block, f"refusal carries one link: {head}"
        )
        checks.check(
            "**Source:**" not in block,
            f"refusal cites education, not a scheme page: {head}",
        )
    # And the structural claim the whole project rests on.
    checks.check(
        joined.count("**model called:** **no**") >= 2,
        "at least two refusals show the model was never called",
        f"counted {joined.count('**model called:** **no**')}",
    )
    print(f"    {len(facts)} facts (all cited, all in sources.csv), "
          f"{len(refusals)} refusals")

    # --- README is a usable cold start ------------------------------
    print("\n  README is a cold-start guide")
    rpath = root / "README.md"
    if not checks.check(rpath.exists(), "README.md exists"):
        return
    readme = rpath.read_text(encoding="utf-8")
    required = {
        "venv creation": "python -m venv .venv",
        "install command": "pip install -e",
        "ingest command": "python -m growbot.ingest",
        "UI run command": "streamlit run src/growbot/ui/app.py",
        "API key setup": ".env",
        "python version warning": "3.11",
        "AMC named": "HDFC Asset Management Company",
        "source list linked": "data/sources.csv",
        "architecture linked": "docs/architecture.md",
        "sample Q&A linked": "docs/sample_qa.md",
        "disclaimer linked": "docs/disclaimer.md",
        "known limits section": "## Known limits",
        "disclaimer string": DISCLAIMER,
    }
    for label, needle in required.items():
        checks.check(needle in readme, f"README mentions {label}", needle[:50])
    for name in (
        "HDFC Large Cap Fund",
        "HDFC Flexi Cap Fund",
        "HDFC ELSS Tax Saver Fund",
        "HDFC Small Cap Fund",
        "HDFC Balanced Advantage Fund",
    ):
        checks.check(name in readme, f"README lists {name}")
    for stage in ("load.py", "chunk.py", "embed.py", "index.py",
                  "query.py", "assemble.py", "answer.py", "intent.py"):
        checks.check(stage in readme, f"README names the {stage} stage")

    # The counts in the README go stale the moment a check is added, and a
    # wrong count in a submission looks like sloppiness. The ui suite knows its
    # own size, so it can check the others against the README.
    counts = {
        "python -m growbot.guards": "growbot.guards",
        "python -m growbot.retrieve.checks": "growbot.retrieve.checks",
        "python -m growbot.generate.checks": "growbot.generate.checks",
        "python -m growbot.ask_checks": "growbot.ask_checks",
        "python -m growbot.ui.checks": "growbot.ui.checks",
        "python -m growbot.memory_checks": "growbot.memory_checks",
        "python -m growbot.hardening_checks": "growbot.hardening_checks",
    }
    for command in counts:
        line = next(
            (ln for ln in readme.splitlines() if ln.strip().startswith(command)),
            "",
        )
        match = re.search(r"#\s*(\d+)\s+checks", line)
        checks.check(
            bool(match), f"README states a count for {command}", line
        )
    # Note: the ui suite's own count is deliberately not asserted exactly. A
    # suite cannot know its own size mid-run - the assertion would count itself
    # and the next check would move the number again. The sum check below is
    # the enforceable one.
    total = sum(
        int(m.group(1))
        for m in (
            re.search(r"#\s*(\d+)\s+checks",
                      next((ln for ln in readme.splitlines()
                            if ln.strip().startswith(cmd)), ""))
            for cmd in counts
        )
        if m
    )
    checks.equal(
        len(counts), 7, "all seven suites are listed with a count"
    )
    checks.check(
        f"{total} checks" in readme,
        "README's stated total matches the sum of its per-suite counts",
        f"sum={total}",
    )
    for tool in ("tools/regen_sample_qa.py", "tools/regen_disclaimer.py",
                 "tools/eval_citations.py", "tools/walkthrough.py"):
        checks.check(tool in readme, f"README documents {tool}")
        checks.check(
            (root / tool).exists(), f"{tool} exists where the README says"
        )
    print(f"    {len(required)} required mentions, 5 schemes, 8 pipeline stages, "
          f"counts sum to {total} across {len(counts)} suites")


# --- the UI passes conversation history ---------------------------
def _check_memory_wiring(checks) -> None:
    """The thread on screen is the memory, and only the user half of it.

    `memory.py` is covered on its own in `memory_checks`. What is checked here
    is the wiring that only the UI has: that a second question carries the
    first one, that the first carries nothing, and that assistant text is
    never fed back - an answer's own wording resolving a fund is how a
    conversation starts citing the previous scheme.

    `ask_with_trace` is replaced rather than called, so this asserts on the
    *arguments the UI passes* and costs no model call. The real script still
    runs; only the model is substituted.
    """
    from growbot import ask as ask_mod
    from growbot.guards.intent import AnswerPayload

    small = "HDFC Small Cap Fund - Direct Growth"
    first, follow_up = f"Expense ratio of {small}?", "What about its exit load?"

    seen: list[tuple[str, list[str]]] = []
    real = ask_mod.ask_with_trace

    def recording(question, **kwargs):
        seen.append((question, list(kwargs.get("history") or [])))
        return (
            AnswerPayload(
                mode="fact",
                text="The expense ratio is stated in the factsheet.",
                source_url="https://example.invalid/factsheet",
                last_updated="2026-01-01",
            ),
            ask_mod.Trace(),
        )

    try:
        ask_mod.ask_with_trace = recording
        at = AppTest.from_file(str(APP_PATH), default_timeout=240)
        at.run()
        at.chat_input[0].set_value(first).run()
        at.chat_input[0].set_value(follow_up).run()
    finally:
        ask_mod.ask_with_trace = real

    checks.equal(len(seen), 2, "the app asked twice")
    if len(seen) == 2:
        checks.equal(seen[0][0], first, "turn 1 is the first question")
        checks.equal(seen[0][1], [], "turn 1 has nothing before it")
        checks.equal(seen[1][0], follow_up, "turn 2 is the follow-up")
        checks.equal(
            seen[1][1], [first],
            "turn 2 carries the earlier user question, and only that",
        )

    # `_user_turns` is the pure half of the wiring, and it is checked directly.
    # `_history` itself cannot be: it reads `st.session_state`, which does not
    # exist between script runs, so calling it from a check raises rather than
    # proving anything. The two-turn run above already covers the part that
    # needs a live session.
    from growbot.config import MEMORY_TURNS
    from growbot.ui.app import _user_turns

    thread = [
        {"role": "user", "text": "q1"},
        {"role": "assistant", "payload": AnswerPayload(
            mode="fact", text="a1", source_url="https://example.invalid/x",
        )},
        {"role": "user", "text": "q2"},
    ]
    got = _user_turns(thread)
    checks.equal(got, ["q1", "q2"], "only user turns become history")
    checks.check(
        "a1" not in got, "assistant text is never fed back as context"
    )
    long_thread = [{"role": "user", "text": f"q{i}"} for i in range(MEMORY_TURNS + 7)]
    capped = _user_turns(long_thread)
    checks.equal(
        len(capped), MEMORY_TURNS,
        f"the window is capped at MEMORY_TURNS={MEMORY_TURNS}",
    )
    checks.check(
        capped[-1] == f"q{MEMORY_TURNS + 6}", "and it keeps the most recent",
    )
    checks.equal(
        _user_turns(thread, 0), [],
        "a window of 0 remembers nothing (not everything)",
    )


def main() -> int:
    checks = Checks()
    print("UI checks (Phase 9)")
    print("=" * 74)
    print("  Streamlit's AppTest runs the real app.py headlessly. Refusal")
    print("  paths use the real pipeline; fact rendering uses a stub answer")
    print("  so layout is tested without an LLM call.")

    at = _check_load(checks)
    _check_examples(checks, at)
    _check_disclaimer(checks, at)
    _check_fact_card(checks)
    _check_refusals(checks)
    _check_gap_example(checks)
    _check_thread(checks)
    print("\n  the page cannot bypass ask()")
    _no_direct_retrieval(checks)
    _check_demo_pack(checks)
    print("\n  the thread on screen is the conversation memory")
    _check_memory_wiring(checks)

    print()
    print("=" * 74)
    if checks.failures:
        print(f"FAILED  {len(checks.failures)} of "
              f"{checks.passed + len(checks.failures)}")
        for failure in checks.failures:
            print(f"  - {failure}")
        return 1
    print(f"OK      {checks.passed} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
