"""Phase 7 - generation checks.

    python -m growbot.generate.checks

Assertion-based, non-zero exit on regression. **No API key and no network
call is required**: every model reply is a stub.

That is the point of the suite. The guarantees Phase 7 makes are not the
model's promises, so they are tested against a model that is trying to break
them. Each hostile stub returns a URL the bot must not show, a date it must not
use, a ratio that appears nowhere in the corpus, or five sentences when three
are allowed - and the assertions are that none of it survives.

Retrieval is real (the built index is read), so the grounding checks run
against genuine retrieved text rather than a fixture that might be kinder than
the real corpus.

Read-only: opens the Phase 3 index, never rebuilds it, never fetches a page.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import re
import sys
import threading

from growbot.config import (
    EDU_LINK,
    LLM_API_KEY,
    LLM_BASE_URL,
    MAX_ANSWER_SENTENCES,
    PROJECT_ROOT,
    llm_is_configured,
)
from growbot.generate import answer as answer_mod
from growbot.generate.answer import (
    LLMError,
    LLMNotConfigured,
    LLMQuotaExceeded,
    LLMTruncated,
    generate,
    split_sentences,
    ungrounded_numbers,
)
from growbot.generate.prompt import INSUFFICIENT, SYSTEM_PROMPT, build_messages
from growbot.guards.intent import refuse_not_in_corpus  # noqa: F401 - re-export guard
from growbot.retrieve.assemble import Assembly, assemble
from growbot.retrieve.query import search

EXPENSE_Q = "Expense ratio of HDFC Large Cap Fund Direct Growth?"
LARGE_CAP = "HDFC Large Cap Fund - Direct Growth"
#: Mirrors growbot.config.DISCLAIMER, asserted here so the copy the model is
#: told to append is the same string the UI shows.
DISCLAIMER_TEXT = "Facts-only. No investment advice."


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


class Stub:
    """A model that returns a fixed string and counts how often it was called."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = 0
        self.seen: list = []

    def __call__(self, messages) -> str:
        self.calls += 1
        self.seen.append(messages)
        return self.reply


def real_assembly(question: str = EXPENSE_Q) -> Assembly:
    return assemble(search(question))


# --- secret scan -----------------------------------------------------------
#: Provider key formats, matched by shape. A real key pasted into a template is
#: the failure this guards: `.env` is gitignored, `.env.example` is not, and the
#: template is the file that gets shared with classmates. It happened once
#: during development of this very phase, which is why the check exists.
_SECRET_PATTERNS = (
    r"AIza[0-9A-Za-z_\-]{20,}",          # Google / Gemini
    r"sk-[A-Za-z0-9_\-]{20,}",           # OpenAI
    r"sk-ant-[A-Za-z0-9_\-]{20,}",       # Anthropic
    r"gsk_[A-Za-z0-9]{20,}",             # Groq
    r"sk-or-v1-[A-Za-z0-9]{20,}",        # OpenRouter
    r"hf_[A-Za-z0-9]{20,}",              # HuggingFace
)
_SECRET_RE = re.compile("|".join(_SECRET_PATTERNS))

#: Suffixes worth scanning. Deliberately excludes .env, which is where a key is
#: supposed to be, and the venv/cache trees, which are not ours to police.
_SCAN_GLOBS = (".env.example", "*.md", "*.py", "*.toml", "*.txt", "*.cfg", "*.csv")
_SKIP_DIRS = {".venv", "venv", "__pycache__", "chroma", ".git", "node_modules"}


def _scan_for_secrets() -> list[tuple[str, int]]:
    """Committed-file paths holding a key-shaped string, with occurrence counts."""
    found: list[tuple[str, int]] = []
    for pattern in _SCAN_GLOBS:
        for path in PROJECT_ROOT.glob(pattern):
            if path.name == ".env" or not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            count = len(_SECRET_RE.findall(text))
            if count:
                found.append((path.name, count))
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file() or path.name == ".env":
            continue
        if _SKIP_DIRS & set(path.parts):
            continue
        if path.suffix not in {".md", ".py", ".toml", ".txt", ".cfg", ".csv", ".example"}:
            continue
        if any(path.name == p for p, _ in found):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        count = len(_SECRET_RE.findall(text))
        if count:
            found.append((str(path.relative_to(PROJECT_ROOT)), count))
    return found


def _gitignore_covers_env() -> bool:
    """True when .gitignore lists .env. The repo is not a git repo during early
    phases, so this reads the file rather than shelling out to git."""
    ignore = PROJECT_ROOT / ".gitignore"
    if not ignore.exists():
        return False
    for line in ignore.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.strip() in (".env", ".env*", "*.env"):
            return True
    return False


# --- fake provider ---------------------------------------------------------
# A real provider on a free tier fails often enough that the retry path is not
# something to test by waiting for a 503. This stands up a throwaway HTTP server
# on localhost that replays a scripted list of statuses, so the retry, the
# no-retry-on-401 and the truncation branches are all driven deterministically
# with no key and no external call.
@contextlib.contextmanager
def _fake_provider():
    server = _FakeServer()
    thread = threading.Thread(target=server.serve, daemon=True)
    thread.start()
    server.started.wait(timeout=5)
    try:
        yield server
    finally:
        server.stop()


class _FakeServer:
    """Replays `script` (a list of HTTP statuses) one per request."""

    def __init__(self) -> None:
        from http.server import BaseHTTPRequestHandler, HTTPServer

        outer = self
        self.calls = 0
        self.started = threading.Event()
        self._script: list[int] = []
        self._content: str | None = ""
        self._finish = "stop"
        self._omit_content = False

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # silence stderr access log
                pass

            def do_POST(self):  # noqa: N802 - name fixed by BaseHTTPRequestHandler
                length = int(self.headers.get("Content-Length", 0))
                self.rfile.read(length)
                outer.calls += 1
                index = min(outer.calls - 1, len(outer._script) - 1)
                entry = outer._script[index]
                # A script entry may be a plain status, or a marker object for
                # the rate-limit variant of 429.
                reason = getattr(entry, "body", None)
                status = 429 if reason else entry

                if status >= 400:
                    body = json.dumps(
                        [{"error": {"code": status,
                                    "message": reason or _FAKE_REASON[status]}}]
                    ).encode()
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                message: dict = {"role": "assistant"}
                if not outer._omit_content and outer._content is not None:
                    message["content"] = outer._content
                body = json.dumps(
                    {"choices": [{"index": 0, "message": message,
                                  "finish_reason": outer._finish}]}
                ).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}/v1"

    def serve(self) -> None:
        self.started.set()
        self._server.serve_forever(poll_interval=0.05)

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def reset(self) -> None:
        self.calls = 0

    def script(
        self,
        statuses: list[int],
        content: str | None = "",
        finish_reason: str = "stop",
        omit_content: bool = False,
    ) -> None:
        self.reset()
        self._script = statuses
        self._content = content
        self._finish = finish_reason
        self._omit_content = omit_content


#: The exact wording the real providers use, so the unwrapping check is
#: asserting on realistic text rather than whatever the test invented.
_FAKE_REASON = {
    401: "API key not valid. Please pass a valid API key.",
    429: "You exceeded your current quota, please check your plan and billing "
         "details. For more information on this error, head to: "
         "https://ai.google.dev/gemini-api/docs/",
    503: "This model is currently experiencing high demand. Spikes in demand "
         "are usually temporary. Please try again later.",
}

#: A 429 that means "slow down" rather than "allowance spent". Uses the same
#: status code as _FAKE_REASON[429] on purpose: the two are only distinguishable
#: by wording, which is exactly what _is_quota_exhausted has to work with.
_RATE_LIMIT_BODY = "Rate limit reached for requests. Please try again in a moment."


class _RateLimit429:
    """Marker so the fake server can emit a rate-limit 429 instead of a quota one."""

    body = _RATE_LIMIT_BODY


_RATE_LIMIT_SCRIPT = [_RateLimit429]


def _call_fake(
    server: _FakeServer,
    statuses: list[int],
    content: str | None = "",
    finish_reason: str = "stop",
    omit_content: bool = False,
):
    """Run the real client against the fake server, with no sleeping.

    Backoff is set to zero so the retry branch is exercised at full speed; the
    attempt count is left real so "used all 4 attempts" stays meaningful.
    """
    import contextlib as _contextlib

    server.script(statuses, content, finish_reason, omit_content)
    saved = (
        answer_mod.LLM_BASE_URL, answer_mod.LLM_API_KEY, answer_mod.LLM_MODEL,
        answer_mod.LLM_RETRY_BACKOFF, answer_mod.LLM_RETRY_BACKOFF_MAX,
        answer_mod.llm_is_configured,
    )
    answer_mod.LLM_BASE_URL = server.base_url
    answer_mod.LLM_API_KEY = "fake-key-for-localhost"
    answer_mod.LLM_MODEL = "fake-model"
    answer_mod.LLM_RETRY_BACKOFF = 0.0
    answer_mod.LLM_RETRY_BACKOFF_MAX = 0.0
    answer_mod.llm_is_configured = lambda: True
    try:
        with _contextlib.redirect_stdout(io.StringIO()):
            return answer_mod.call_openai_compatible(
                [{"role": "user", "content": "hi"}]
            )
    except answer_mod.LLMError as exc:
        return exc
    finally:
        (
            answer_mod.LLM_BASE_URL, answer_mod.LLM_API_KEY, answer_mod.LLM_MODEL,
            answer_mod.LLM_RETRY_BACKOFF, answer_mod.LLM_RETRY_BACKOFF_MAX,
            answer_mod.llm_is_configured,
        ) = saved


class ChecksRunner:
    def run(self) -> int:
        checks = Checks()
        print("generation checks")
        print("=" * 74)
        key_present = llm_is_configured()
        print(f"  LLM configured in this environment: {key_present}")
        print("  (every model reply below is a stub; no key is needed)")

        assembly = real_assembly()
        context = assembly.context
        print(f"\n  real retrieved context: {len(context)} chars, "
              f"max_sim={assembly.max_similarity:.3f}")

        # --- 1. the happy path, grounded -------------------------------
        print("\n  grounded answer is accepted")
        grounded = (
            f"The expense ratio of {LARGE_CAP} is 1.03% as published on the "
            f"scheme page. {DISCLAIMER_TEXT}"
        )
        stub = Stub(grounded)
        payload = generate(EXPENSE_Q, assembly, complete=stub)
        checks.equal(payload.mode, "fact", "mode is fact")
        checks.equal(stub.calls, 1, "model called once")
        checks.equal(
            payload.source_url,
            assembly.source_url,
            "citation comes from the assembly, not the model",
        )
        checks.equal(
            payload.last_updated,
            assembly.last_updated,
            "last_updated comes from fetched_at metadata",
        )
        checks.equal(
            payload.source_url.count("https://"), 1, "exactly one URL in the payload"
        )
        checks.check(
            payload.scheme_name == LARGE_CAP, "scheme chip is the canonical name",
            payload.scheme_name,
        )
        checks.check(
            len(split_sentences(payload.text)) <= MAX_ANSWER_SENTENCES,
            f"at most {MAX_ANSWER_SENTENCES} sentences",
            payload.text,
        )
        print(f"    {payload.text[:64]}...")
        print(f"    cite {payload.source_url}")

        # --- 2. a hostile model cannot introduce a URL or a date -------
        print("\n  hostile model: invented URL and date are discarded")
        hostile = (
            "The expense ratio is 1.03%. See "
            "https://totally-made-up-site.example/fake-fund for details, "
            "last updated 1999-01-01."
        )
        payload = generate(EXPENSE_Q, assembly, complete=Stub(hostile))
        checks.check(
            "https://" not in payload.text and "www." not in payload.text,
            "no URL survives in the prose",
            payload.text,
        )
        checks.check(
            "1999" not in payload.text, "invented date is not repeated", payload.text
        )
        # Stripping the URL must not leave the connective behind it stranded.
        for debris in ["for details", "see for", "for reference", "for source"]:
            checks.check(
                debris not in payload.text.lower(),
                f"removing a URL leaves no stranded {debris!r}",
                payload.text,
            )
        checks.equal(
            payload.text,
            "The expense ratio is 1.03%.",
            "hostile reply cleans down to just the verified fact",
        )
        checks.check(
            payload.text.rstrip().endswith("."), "prose still ends cleanly", payload.text
        )
        # A bare URL with no connective must go too.
        bare = generate(
            EXPENSE_Q,
            assembly,
            complete=Stub("It is 1.03% (https://x.example/a, https://x.example/b)."),
        )
        checks.check(
            "x.example" not in bare.text, "multiple bare URLs all removed", bare.text
        )
        checks.check(
            "(" not in bare.text or ")" not in bare.text,
            "no empty bracket left behind",
            bare.text,
        )
        checks.equal(
            payload.source_url,
            assembly.source_url,
            "citation still the assembly's",
        )
        checks.equal(
            payload.last_updated,
            assembly.last_updated,
            "last_updated still the chunk's",
        )
        print(f"    prose: {payload.text[:70]}")

        # --- 3. a fabricated ratio is refused -------------------------
        print("\n  fabricated ratio is refused, not corrected")
        for claim in [
            "The expense ratio of this fund is 4.2%.",
            "The expense ratio works out to 2.75 percent.",
            "The exit load is 12.5% if you redeem early.",
        ]:
            payload = generate(EXPENSE_Q, assembly, complete=Stub(claim))
            checks.equal(payload.mode, "refuse", f"refused: {claim[:34]}")
            checks.equal(payload.reason, "ungrounded", f"reason: {claim[:34]}")
            checks.check(
                "4.2" not in payload.text and "2.75" not in payload.text
                and "12.5" not in payload.text,
                "the unverified figure is not restated",
                payload.text,
            )
        print("    3 fabricated ratios refused, none restated")

        # --- 4. the model declining is a refusal ----------------------
        print("\n  model declines -> refusal, not an invention")
        for reply in [INSUFFICIENT, f"  {INSUFFICIENT.lower()}  ", "I don't know."]:
            payload = generate(EXPENSE_Q, assembly, complete=Stub(reply))
            checks.equal(payload.mode, "refuse", f"refused: {reply[:24]!r}")
            checks.equal(
                payload.reason, "insufficient", f"reason: {reply[:24]!r}"
            )
        print("    sentinel, lowercase sentinel and 'I don't know' all refused")

        # --- 5. weak retrieval never reaches the model ----------------
        print("\n  weak assembly: the model is not called at all")
        vague = assemble(search("tell me about HDFC funds"))
        checks.check(vague.weak, "vague question is weak", str(vague.weak_reasons))
        stub = Stub("Some plausible answer.")
        payload = generate("tell me about HDFC funds", vague, complete=stub)
        checks.equal(stub.calls, 0, "model was never called")
        checks.equal(payload.mode, "refuse", "mode is refuse")
        checks.equal(payload.reason, "weak_retrieval", "reason is weak_retrieval")
        checks.check(
            payload.source_url in (EDU_LINK,) or payload.source_url.startswith("https://"),
            "refusal still carries one allowlisted link",
            payload.source_url,
        )
        gap = assemble(search("How to download a capital-gains statement?"))
        checks.check(gap.weak, "known corpus gap is weak", str(gap.weak_reasons))
        print("    0 model calls for a weak assembly")

        # --- 6. sentence clamp ----------------------------------------
        print("\n  verbosity is clamped at a sentence boundary")
        five = (
            "The expense ratio is 1.03%. It is charged daily. It varies with "
            "the fund. It is disclosed monthly. That is all."
        )
        payload = generate(EXPENSE_Q, assembly, complete=Stub(five))
        got = split_sentences(payload.text)
        checks.equal(len(got), MAX_ANSWER_SENTENCES, f"clamped to {MAX_ANSWER_SENTENCES}")
        checks.check(
            got[-1].endswith("."), "last kept sentence is complete", got[-1]
        )
        checks.check("1.03" in payload.text, "the fact survives clamping")
        checks.check("That is all." not in payload.text, "excess removed")
        print(f"    {len(split_sentences(five))} -> {len(got)} sentences")

        # --- 7. the splitter does not break decimals -----------------
        print("\n  sentence splitting survives decimals and abbreviations")
        cases = [
            ("The expense ratio is 1.03%.", 1),
            ("The ratio is 1.03% and the exit load is 1%.", 1),
            ("It is 3.5 times the benchmark.", 1),
            ("This costs Rs. 100 per unit. That is all.", 2),
            ("The lock-in is 3Y. It ends then.", 2),
        ]
        for text, expected in cases:
            got = split_sentences(text)
            checks.equal(len(got), expected, f"split {text[:36]!r}")
        print(f"    {len(cases)} splitter cases")

        # --- 8. the numeric check's coverage --------------------------
        print("\n  numeric check: what it enforces and what it does not")
        ctx = "Expense ratio 1.03% and exit load 1% after 1 year. Benchmark NIFTY 100."
        enforced = [
            ("4.2%", True), ("1.03%", False), ("2.75 percent", True),
            ("NIFTY 999", True), ("NIFTY 100", False), ("1.03", False),
        ]
        for claim, should_flag in enforced:
            found = ungrounded_numbers(claim, ctx)
            checks.equal(
                bool(found), should_flag, f"ungrounded_numbers({claim!r})"
            )
        # Bare small integers are deliberately not enforced.
        for claim in ["one of the five funds", "3 years", "top 10 holdings"]:
            checks.equal(
                ungrounded_numbers(claim, ctx), [], f"small int ignored: {claim!r}"
            )
        print("    decimals, units and 3+ digit ints enforced; 1-2 digit ints not")

        # --- 9. missing configuration fails clearly -------------------
        print("\n  missing key fails clearly and never leaks the key")
        saved = (
            answer_mod.LLM_API_KEY, answer_mod.LLM_MODEL,
            answer_mod.LLM_BASE_URL, answer_mod.llm_is_configured,
        )
        answer_mod.LLM_API_KEY = ""
        answer_mod.LLM_MODEL = ""
        answer_mod.LLM_BASE_URL = ""
        answer_mod.llm_is_configured = lambda: False
        try:
            raised = None
            try:
                generate(EXPENSE_Q, assembly)
            except LLMNotConfigured as exc:
                raised = exc
            checks.check(raised is not None, "raises LLMNotConfigured")
            if raised is not None:
                message = str(raised)
                checks.check(
                    "LLM_API_KEY" in message and "LLM_MODEL" in message,
                    "names the missing settings", message,
                )
                checks.check(
                    ".env" in message, "says how to fix it", message
                )
                checks.check(
                    "sk-" not in message
                    and (not LLM_API_KEY or LLM_API_KEY not in message),
                    "no key material in the error",
                )
        finally:
            (
                answer_mod.LLM_API_KEY, answer_mod.LLM_MODEL,
                answer_mod.LLM_BASE_URL, answer_mod.llm_is_configured,
            ) = saved
        # And a configured provider is still required to have a base URL.
        checks.check(
            isinstance(LLM_BASE_URL, str), "base url is a string"
        )
        print("    error names the missing vars and how to set them")

        # --- 10. prompt shape -----------------------------------------
        print("\n  prompt contract")
        messages = build_messages(EXPENSE_Q, context)
        checks.equal(messages[0]["role"], "system", "first turn is the system policy")
        checks.equal(messages[1]["role"], "user", "second turn carries the question")
        system = messages[0]["content"]
        for phrase, label in [
            ("ONLY from the CONTEXT", "context-only rule present"),
            (INSUFFICIENT, "refusal sentinel named"),
            ("Never state a number", "no-invented-numbers rule present"),
            ("at most 3 sentences", "sentence cap present"),
            ("no investment advice", "advice prohibition present"),
            ("Do not output URLs", "no-URL instruction present"),
        ]:
            checks.check(phrase in system, label, system[:60])
        checks.check(
            EXPENSE_Q in messages[1]["content"], "question is in the user turn"
        )
        checks.check(context in messages[1]["content"], "context is in the user turn")
        checks.check(
            DISCLAIMER_TEXT in SYSTEM_PROMPT, "system prompt repeats the disclaimer"
        )
        print(f"    6 prompt rules asserted, context {len(context)} chars delivered")

        # --- 10. the client survives a flaky provider -----------------
        print("\n  the client retries a flaky provider and refuses partial text")
        # The retry log is expected output here, but five lines of it would
        # bury the check results.
        answer_log = logging.getLogger("growbot.generate.answer")
        previous_level = answer_log.level
        answer_log.setLevel(logging.CRITICAL)
        try:
            with _fake_provider() as server:
                # 503 twice, then a good answer: retry must get through.
                text = _call_fake(server, [503, 503, 200], "Recovered answer.")
                checks.equal(text, "Recovered answer.", "recovers after two 503s")
                checks.check(server.calls >= 3, "made more than one attempt",
                             str(server.calls))

                # 503 on every attempt: a clear error naming the real reason.
                err = _call_fake(server, [503, 503, 503, 503], "")
                checks.check("high demand" in str(err),
                             "503 error shows the provider's own reason", str(err))
                checks.check("[{" not in str(err) and '"error"' not in str(err),
                             "error message is unwrapped, not raw JSON", str(err))
                checks.equal(server.calls, 4, "used all 4 attempts then gave up")

                # 429 quota is not transient: fail fast, do not retry. Measured
                # on a real free key that had exhausted its daily allowance.
                server.reset()
                err = _call_fake(server, [429, 429, 429, 429], "")
                checks.check(isinstance(err, LLMQuotaExceeded),
                             "spent quota raises LLMQuotaExceeded", type(err).__name__)
                checks.equal(server.calls, 1, "spent quota is not retried")
                checks.check("quota" in str(err).lower(), "names the cause", str(err))

                # A per-minute rate limit IS transient and must still retry.
                server.reset()
                err = _call_fake(server, _RATE_LIMIT_SCRIPT, "")
                checks.check(isinstance(err, LLMError)
                             and not isinstance(err, LLMQuotaExceeded),
                             "rate limit is not mislabelled as quota",
                             type(err).__name__)
                checks.equal(server.calls, 4, "rate limit is retried")

                # 401 must NOT be retried - it is a credentials mistake.
                server.reset()
                err = _call_fake(server, [401], "")
                checks.check("401" in str(err), "401 reported", str(err))
                checks.equal(server.calls, 1, "401 is not retried")

                # finish_reason=length means a half sentence. Never returned.
                server.reset()
                err = _call_fake(server, [200], "The expense ratio of HDFC Large Cap",
                                 finish_reason="length")
                checks.check(isinstance(err, LLMTruncated), "length raises LLMTruncated",
                             type(err).__name__)
                checks.check("LLM_MAX_TOKENS" in str(err), "names the remedy", str(err))

                # gemini's exact truncation shape: finish_reason=length *and* no
                # content key at all. Truncation, not a malformed reply.
                server.reset()
                err = _call_fake(server, [200], None, finish_reason="length",
                                 omit_content=True)
                checks.check(isinstance(err, LLMTruncated),
                             "length with no content key is still truncation",
                             type(err).__name__)

                # A clean stop with no content is a different fault: a
                # well-formed reply with nothing in it. Must not be mislabelled
                # as truncation, and must say what it actually saw.
                server.reset()
                err = _call_fake(server, [200], None, omit_content=True)
                checks.check(
                    isinstance(err, LLMError) and not isinstance(err, LLMTruncated),
                    "stop with no content is LLMError, not truncation",
                    type(err).__name__,
                )
                checks.check("message keys" in str(err),
                             "malformed reply reports what it saw", str(err))

                # A clean 'stop' still works.
                server.reset()
                text = _call_fake(server, [200], "The expense ratio is 1.03%.")
                checks.equal(text, "The expense ratio is 1.03%.", "normal reply passes")
        finally:
            answer_log.setLevel(previous_level)
        print("    retry, no-retry on 401, truncation, clean reply")

        # --- 11. no secret in a committed file -----------------------
        print("\n  no API key in any file that gets committed")
        leaks = _scan_for_secrets()
        for path, count in leaks:
            checks.check(
                False, f"secret-shaped string in {path}", f"{count} occurrence(s)"
            )
        checks.check(
            not leaks,
            "no committed file holds a key-shaped string",
            ", ".join(p for p, _ in leaks),
        )
        checks.check(
            _gitignore_covers_env(),
            ".gitignore covers .env",
        )
        print(f"    scanned {len(_SCAN_GLOBS)} patterns over the repo, "
              f"{len(leaks)} leak(s)")

        # --- report ---------------------------------------------------
        print()
        print("=" * 74)
        if checks.failures:
            print(
                f"FAILED  {len(checks.failures)} of "
                f"{checks.passed + len(checks.failures)}"
            )
            for failure in checks.failures:
                print(f"  - {failure}")
            return 1
        print(f"OK      {checks.passed} checks passed")
        return 0


def main() -> int:
    return ChecksRunner().run()


if __name__ == "__main__":
    sys.exit(main())
