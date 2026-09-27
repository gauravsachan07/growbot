"""Regenerate docs/disclaimer.md from the strings the UI actually shows.

    python tools/regen_disclaimer.py

The Phase 10 "Done when" is "Disclaimer matches the UI string". A disclaimer
document is exactly the kind of file that drifts - someone edits the UI copy,
or fixes a typo in the markdown, and now the published document is quietly
wrong. So the document is *derived* from the code rather than retyped, and
`python -m growbot.ui.checks` asserts the two still agree.

The assertions at the top are the point: this script refuses to write a file
whose content has drifted from PRD section 9 or from `config.DISCLAIMER`.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from growbot import config  # noqa: E402
from growbot.ui.app import DISCLAIMER as UI_BANNER  # noqa: E402
from growbot.ui.app import LONG_DISCLAIMER  # noqa: E402

OUT = ROOT / "docs" / "disclaimer.md"

#: PRD section 9, transcribed. Kept separate from the imported value on
#: purpose: if the UI copy is edited, the comparison below fails instead of
#: silently rewriting the PRD reference out of existence.
PRD_9 = (
    "This assistant answers **facts only** from public scheme documents "
    "(expense ratio, exit load, SIP minimum, lock-in, riskometer, benchmark, "
    "and document download guides). It is **not** investment advice, a "
    "recommendation to buy or sell, or a substitute for the SID/KIM/factsheet. "
    "Mutual fund investments are subject to market risks. Read all "
    "scheme-related documents carefully."
)


def main() -> int:
    drift = []
    if LONG_DISCLAIMER != PRD_9:
        drift.append("LONG_DISCLAIMER in ui/app.py no longer matches PRD section 9")
    if UI_BANNER != config.DISCLAIMER:
        drift.append("the UI banner no longer matches config.DISCLAIMER")
    if drift:
        for line in drift:
            print(f"  DRIFT: {line}")
        print("  refusing to write docs/disclaimer.md until that is resolved.")
        return 1

    lines = [
        "# Disclaimer",
        "",
        "The exact strings Growbot displays. Generated from the source by",
        "`tools/regen_disclaimer.py`, so it cannot drift from what the app shows:",
        "",
        "## 1. Persistent banner",
        "",
        "Shown at the top of the page on load, before any question is asked, and",
        "repeated as a caption at the bottom. This is the short form, held in",
        "`src/growbot/config.py` as `DISCLAIMER`:",
        "",
        f"> **{config.DISCLAIMER}**",
        "",
        "Every refusal payload also ends with this sentence, so it appears again",
        "in the chat thread wherever the bot declines something.",
        "",
        "## 2. Full disclaimer",
        "",
        "From PRD section 9, shown in the UI under \"Disclaimer and what I will not",
        "do\" and carried in `src/growbot/ui/app.py` as `LONG_DISCLAIMER`:",
        "",
        "> " + " ".join(LONG_DISCLAIMER.split()),
        "",
        "## Why two forms",
        "",
        "The long form is complete but too heavy to sit above every answer. The",
        "short form is the persistent one, and it is deliberately the first thing",
        "on the page - a disclaimer that only appears after a user has already read",
        "an answer is not doing much work.",
        "",
        "## Where these come from",
        "",
        "| String | Lives in | Shown |",
        "|---|---|---|",
        "| `DISCLAIMER` | `src/growbot/config.py` | banner, footer, every refusal |",
        "| `LONG_DISCLAIMER` | `src/growbot/ui/app.py` | expander in the chat page |",
        "",
        "`python -m growbot.ui.checks` asserts that the UI strings,",
        "`config.DISCLAIMER` and this document all still agree, so an edit to any",
        "one of them fails the checks rather than shipping a stale document.",
        "",
    ]

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  wrote {OUT.relative_to(ROOT)}")
    print(f"  banner == config.DISCLAIMER == UI banner : {UI_BANNER == config.DISCLAIMER}")
    print(f"  long   == PRD section 9 snippet          : {LONG_DISCLAIMER == PRD_9}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
