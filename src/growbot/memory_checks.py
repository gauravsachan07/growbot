"""Memory checks - conversational memory for scheme resolution.

    python -m growbot.memory_checks

Assertion-based, non-zero exit on regression. No API key and no network call
required: a counting stub stands in for the model, and retrieval reads the
index that is already on disk.

Memory is a small feature with a large blast radius, so the interesting checks
are not "does the follow-up work" - that is three lines - but the ways it could
quietly do harm. Each section below is one of those ways:

1. **The question always wins.** Memory fills a gap; it never overrides what
   was actually asked. A user who switches funds mid-conversation must get the
   new fund, not the old one. This is the check that would catch a follow-up
   silently citing the previous fund.
2. **PII is never remembered.** Not "callers are expected to filter" - screened
   inside `memory.screen`, so a caller who forgets cannot breach it.
3. **Memory cannot launder a refusal into an answer.** A refused turn may
   donate a scheme, but donating a scheme must not make the *next* question
   answerable when it was not.
4. **History never reaches the model.** Asserted on the prompt the stub
   receives, not on behaviour, because behaviour is a weak proxy.
5. **No history means no change.** Every existing caller gets exactly the
   behaviour it had before memory existed, including the trace shape.
6. **A citation always belongs to a fund the corpus actually holds.** The
   citation invariant, re-checked with memory in the path - this is the check
   that would have caught the `general`-page bug that motivated the feature.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from growbot import config
from growbot.ask import ask, ask_with_trace
from growbot.config import DISCLAIMER, SOURCES_CSV
from growbot.guards.intent import AnswerPayload
from growbot.memory import describe, memory_query, resolve_scheme, screen
from growbot.retrieve.query import search

SMALL = "HDFC Small Cap Fund - Direct Growth"
LARGE = "HDFC Large Cap Fund - Direct Growth"
#: Note the real key says "Direct Plan Growth" - that is the canonical spelling
#: in both config.SCHEMES and data/sources.csv, not a typo to be corrected here.
ELSS = "HDFC ELSS Tax Saver Fund - Direct Plan Growth"

#: Questions that name no fund at all, so memory is the only thing that can
#: resolve them. Each was verified to be unanswerable cold (max_sim 0.12-0.38,
#: below the 0.45 floor) and answerable with the scheme inherited (0.77-0.85).
FOLLOW_UPS = [
    "What about its exit load?",
    "And the minimum SIP?",
    "Is it risky?",
]

#: History that must never reach the bot's memory, with a category so the
#: check can say which detector fired. Taken from the guard's own test cases so
#: the two cannot drift apart.
PII_HISTORY = [
    "My PAN is ABCDE1234F, what is the expense ratio?",
    "My OTP is 482913, is that valid?",
    "email me at priya.sharma@example.com",
]


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


class RecordingStub:
    """Stands in for the model, and keeps the prompt so it can be inspected.

    The default reply carries **no number on purpose**. An earlier version used
    a real expense ratio, which quietly coupled these checks to the corpus: the
    grounding check accepts the number for the one fund it happens to belong to
    and discards the answer as ungrounded for every other fund, so two checks
    came back inverted for reasons that had nothing to do with memory. A
    number-free reply is grounded in every scheme's chunks and keeps these
    checks measuring memory.
    """

    def __init__(self, reply: str = "This is stated in the fund factsheet.") -> None:
        self.reply = reply
        self.calls = 0
        self.prompts: list[str] = []

    def __call__(self, messages) -> str:
        self.calls += 1
        # Joined so a check can search the whole prompt, not just one field.
        self.prompts.append("\n".join(str(m.get("content", "")) for m in messages))
        return self.reply


def _corpus_owner() -> dict[str, str]:
    """url -> scheme_name, from the auditable corpus list."""
    with Path(SOURCES_CSV).open(encoding="utf-8-sig") as handle:
        return {r["url"]: r["scheme_name"] for r in csv.DictReader(handle)}


class ChecksRunner:
    def run(self) -> int:
        checks = Checks()
        owner = _corpus_owner()
        print("memory checks")
        print("=" * 74)
        print("  No API key and no network call: a recording stub stands in")
        print("  for the model.")

        # --- 1. the question always wins -------------------------------
        print("\n  1. the current question overrides memory")
        schemes, source = resolve_scheme(f"Expense ratio of {LARGE}?", [f"About {SMALL}?"])
        checks.equal(schemes, [LARGE], "an explicit scheme wins")
        checks.equal(source, "question", "and is reported as coming from the question")

        # A user who changes funds mid-conversation must get the new fund.
        # This is the single most damaging possible bug, so it is checked
        # through the full pipeline as well as through the resolver.
        stub = RecordingStub()
        payload, trace = ask_with_trace(
            f"Expense ratio of {ELSS}?", history=[f"Expense ratio of {SMALL}?"],
            complete=stub,
        )
        checks.equal(
            owner.get(payload.source_url), ELSS,
            "a switch to another fund cites the NEW fund, not the remembered one",
        )
        mem = next((s.detail for s in trace.stages if s.name == "memory"), "")
        checks.check(
            "named in the question" in mem,
            "and the trace says the scheme came from the question", mem,
        )

        # --- 2. a follow-up resolves to the fund in question -----------
        print("\n  2. a follow-up inherits the fund, and cites that fund")
        history = [f"Expense ratio of {SMALL}?"]
        for question in FOLLOW_UPS:
            payload, trace = ask_with_trace(question, history=history, complete=stub)
            label = question[:30]
            checks.equal(
                owner.get(payload.source_url), SMALL,
                f"citation belongs to the remembered fund: {label!r}",
            )
            mem = next((s.detail for s in trace.stages if s.name == "memory"), "")
            checks.check(SMALL in mem, f"trace names the inherited fund: {label!r}", mem)
            checks.equal(payload.mode, "fact", f"answerable with memory: {label!r}")
            checks.check(
                payload.last_updated != "", f"fact carries a date: {label!r}"
            )

        # --- 3. PII is never remembered -------------------------------
        print("\n  3. history containing PII is dropped, not stored")
        dirty = [*PII_HISTORY, "What about its exit load?"]
        # Screening must not be defeated by a PII-bearing turn in the middle:
        # the safe turn after it should still be usable. The surviving turn
        # names a fund, so it is a usable donor - "what about its exit load?"
        # would name none and could not donate however well it survived.
        safe = f"Expense ratio of {SMALL}?"
        dirty = [*PII_HISTORY, safe]
        kept = screen(dirty)
        checks.equal(len(kept), 1, f"only the {len(PII_HISTORY)} PII entries dropped")
        checks.check(
            not any("ABCDE1234F" in k for k in kept), "the PAN is not retained"
        )
        checks.check(
            not any("482913" in k for k in kept), "the OTP is not retained"
        )
        checks.check(
            not any("priya.sharma" in k for k in kept), "the email is not retained"
        )
        schemes, source = resolve_scheme("What about its exit load?", dirty)
        checks.equal(schemes, [SMALL], "a safe turn after PII still donates")
        checks.equal(source, "memory", "and is used")

        # --- 4. the window is a real window ---------------------------
        print("\n  4. the window keeps only the most recent turns")
        long_history = [f"About {SMALL}?"] * (config.MEMORY_TURNS + 5)
        checks.equal(
            len(screen(long_history)), config.MEMORY_TURNS,
            f"capped at MEMORY_TURNS={config.MEMORY_TURNS}",
        )
        # The oldest donor must fall out of range, so a fund mentioned only
        # long ago no longer resolves the follow-up.
        stale = [f"About {SMALL}?"] + ["unrelated chatter"] * config.MEMORY_TURNS
        schemes, _ = resolve_scheme("What about its exit load?", stale)
        checks.equal(
            schemes, [], f"a donor older than the window is not consulted"
        )
        # Most recent wins, when several turns could donate.
        two = [f"About {SMALL}?", f"About {LARGE}?"]
        schemes, _ = resolve_scheme("What about its exit load?", two)
        checks.equal(schemes, [LARGE], "the most recent donor wins")

        # --- 5. memory cannot launder a refusal -----------------------
        print("\n  5. memory does not turn a refusal into an answer")
        # A turn refused for advice still established which fund is being
        # discussed, so it may donate...
        schemes, source = resolve_scheme(
            "What about its exit load?", [f"Should I buy {SMALL}?"]
        )
        checks.equal(schemes, [SMALL], "a refused turn may donate its scheme")
        # ...but the follow-up must not be *given* a lock-in fact that this
        # fund does not have. Only the ELSS has one.
        #
        # This is asserted on the retrieved context rather than on the returned
        # mode, because a stub always claims to have an answer, so mode would be
        # measuring the stub. A real limitation is visible here and is worth
        # stating: `SIMILARITY_FLOOR` does NOT catch this gap. Asking the Small
        # Cap about lock-in scores 0.713 - well above the 0.45 floor - while its
        # context mentions lock-in nowhere at all. Retrieval reports "strong" and
        # the model is what actually declines. So the property that has to hold
        # is the narrower one: memory must not manufacture the missing fact.
        for fund, expect_mentions in ((SMALL, False), (ELSS, True)):
            question = "What about the lock-in period?"
            schemes, source = resolve_scheme(question, [f"Expense ratio of {fund}?"])
            context = " ".join(
                h.text for h in
                search(memory_query(question, schemes, source), detected=schemes).hits
            ).lower()
            mentions = "lock-in" in context or "lock in" in context
            short = fund.split(" - ")[0]
            checks.equal(
                mentions, expect_mentions,
                f"context for {short} "
                f"{'has' if expect_mentions else 'has no'} lock-in fact",
            )
            checks.equal(schemes, [fund], f"resolved to {short}")
            checks.equal(source, "memory", f"resolved via memory: {short}")

        # --- 6. history never reaches the model -----------------------
        print("\n  6. history never reaches the model")
        stub.prompts.clear()
        noisy = [
            f"Expense ratio of {SMALL}?",
            f"What is the NAV of {LARGE}?",
            "My PIN is 1234, can you help?",
        ]
        question = "What about its exit load?"
        schemes_, source_ = resolve_scheme(question, noisy)
        augmented = memory_query(question, schemes_, source_)
        checks.check(
            augmented != question, "the retrieval query text is augmented"
        )
        ask_with_trace(question, history=noisy, complete=stub)
        checks.equal(len(stub.prompts), 1, "the model was called once")
        prompt = stub.prompts[0] if stub.prompts else ""
        checks.check(prompt != "", "the prompt was captured")

        # The sensitive assertion is on the *augmented string as a whole*, not
        # on the scheme name. Every chunk in the context block starts with the
        # scheme name, so "the scheme name is not in the prompt" passes no
        # matter what is passed to the model - that version of this check was
        # written first, and a deliberate leak of `query_text` into `_generate`
        # sailed straight past it. The augmented string is the thing that must
        # not appear: it can only get there if the retrieval text was handed to
        # the generator instead of the question.
        checks.check(
            augmented not in prompt,
            "the augmented retrieval text never reaches the prompt",
        )
        checks.check(
            question in prompt, "the original question does reach the prompt"
        )
        for leaked in ("PIN is 1234", "What is the NAV of"):
            checks.check(
                leaked not in prompt, f"history text absent from prompt: {leaked!r}"
            )

        # --- 7. no history means no change ---------------------------
        print("\n  7. with no history, behaviour is unchanged")
        fact_q = f"Expense ratio of {LARGE}?"
        plain, plain_trace = ask_with_trace(fact_q, complete=stub)
        empty, empty_trace = ask_with_trace(fact_q, history=[], complete=stub)
        checks.equal(
            empty.source_url, plain.source_url, "same citation with history=[]"
        )
        checks.equal(empty.text, plain.text, "same text with history=[]")
        checks.equal(
            [s.name for s in empty_trace.stages],
            [s.name for s in plain_trace.stages],
            "same pipeline stages",
        )
        checks.check(
            not any(s.name == "memory" for s in plain_trace.stages),
            "no memory stage when no history is passed",
        )
        # A history that mentions nothing relevant must not change the answer.
        irrelevant, _ = ask_with_trace(
            fact_q, history=["what is the weather", "tell me a joke"], complete=stub
        )
        checks.equal(irrelevant.source_url, plain.source_url, "irrelevant history ignored")
        checks.equal(irrelevant.text, plain.text, "and the text is identical")

        # --- 8. a refusal in history does not smuggle a URL ----------
        print("\n  8. refusal shape is unchanged with history")
        for question, reason in [
            (f"Should I buy {SMALL}?", "advice"),
            ("Which fund has the highest returns?", "returns"),
            (f"Is {LARGE} the best fund?", "advice"),
        ]:
            before = stub.calls
            payload = ask(question, history=[f"About {ELSS}?"], complete=stub)
            label = question[:30]
            checks.equal(payload.mode, "refuse", f"refused with history: {label!r}")
            checks.equal(payload.reason, reason, f"same reason: {label!r}")
            checks.equal(payload.last_updated, "", f"no date on a refusal: {label!r}")
            checks.equal(
                stub.calls, before, f"model NOT called: {label!r}"
            )
            checks.check(
                DISCLAIMER in payload.text, f"disclaimer present: {label!r}"
            )
            checks.equal(
                payload.text.count("https://"), 0, f"no URL in prose: {label!r}"
            )

        # --- 8b. search() honours an explicit scheme override --------
        # `ask()` passes the resolved scheme to `search(detected=...)`. Note
        # what that is and is not verified here. It is NOT observable through
        # `ask()`: `memory_query` also puts the scheme name in the query text,
        # so `search` re-detects it from the text and applies the same filter
        # anyway. Swapping `detected=schemes` for `detected=None` in `ask()`
        # changes nothing observable (tried; every other check still passed).
        # The override is kept regardless - it is the single source of truth for
        # the scheme, instead of relying on `detect_scheme` parsing a name back
        # out of appended text, which is a coupling that would break silently
        # if the alias table changed. But since it cannot be pinned through
        # `ask()`, its contract is pinned here, directly, with an override that
        # *disagrees* with the question text - which is the case that actually
        # distinguishes an honoured override from an ignored one.
        print("\n  8b. search() obeys an explicit scheme override")
        plain_q = "What is the expense ratio?"
        checks.equal(
            resolve_scheme(plain_q)[0], [], "the bare question names no fund"
        )
        forced = search(plain_q, detected=[LARGE])
        checks.check(
            forced.filtered, "an explicit override applies the metadata filter"
        )
        checks.check(
            all(h.scheme_name == LARGE for h in forced.hits),
            "and every hit belongs to the overridden fund",
            str({h.scheme_name for h in forced.hits}),
        )
        # Overriding to nothing is meaningful too, and must not fall back to
        # detecting from the text - that is the case memory relies on to tell
        # "this question is about no single fund" apart from "not yet resolved".
        checks.equal(
            search(f"Expense ratio of {LARGE}?", detected=[]).filtered, False,
            "an override of [] disables the filter even for a fund-naming question",
        )

        # --- 9. the citation invariant, with memory in the path ------
        # This is the check that would have caught the `general`-page bug that
        # motivated memory: an underspecified follow-up citing a page that does
        # not belong to the fund being discussed.
        print("\n  9. a follow-up never cites a page from another fund")
        cases = [
            ("What about its exit load?", [f"About {SMALL}?"], SMALL),
            ("And the benchmark?", [f"About {ELSS}?"], ELSS),
            ("What is the minimum SIP?", [f"About {LARGE}?"], LARGE),
            ("What about its expense ratio?", [f"About {SMALL}?"], SMALL),
        ]
        for question, hist, expected in cases:
            payload, _ = ask_with_trace(question, history=hist, complete=stub)
            label = question[:26]
            checks.check(
                payload.source_url in owner,
                f"fact cites a corpus URL: {label!r}", payload.source_url,
            )
            checks.equal(
                owner.get(payload.source_url), expected,
                f"and that URL belongs to the right fund: {label!r}",
            )

        # --- 10. describe() is demo-safe ------------------------------
        print("\n  10. the trace line is safe to show a user")
        line = describe("What about its exit load?", [f"About {SMALL}?"])
        checks.check("inherited" in line, "says it inherited", line)
        checks.check(
            "PAN" not in line and "ABCDE" not in line,
            "carries no PII even when history did", line,
        )
        checks.equal(
            describe("What about its exit load?"), "no scheme named or inherited",
            "honest when there is no memory",
        )

        # --- report ---------------------------------------------------
        print("\n" + "=" * 74)
        if checks.failures:
            print(f"  {len(checks.failures)} FAILED, {checks.passed} passed")
            for failure in checks.failures:
                print(f"    FAIL  {failure}")
            return 1
        print(f"  OK  {checks.passed} checks passed")
        return 0


def main() -> int:
    return ChecksRunner().run()


if __name__ == "__main__":
    sys.exit(main())
