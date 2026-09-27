# Disclaimer

The exact strings Growbot displays. Generated from the source by
`tools/regen_disclaimer.py`, so it cannot drift from what the app shows:

## 1. Persistent banner

Shown at the top of the page on load, before any question is asked, and
repeated as a caption at the bottom. This is the short form, held in
`src/growbot/config.py` as `DISCLAIMER`:

> **Facts-only. No investment advice.**

Every refusal payload also ends with this sentence, so it appears again
in the chat thread wherever the bot declines something.

## 2. Full disclaimer

From PRD section 9, shown in the UI under "Disclaimer and what I will not
do" and carried in `src/growbot/ui/app.py` as `LONG_DISCLAIMER`:

> This assistant answers **facts only** from public scheme documents (expense ratio, exit load, SIP minimum, lock-in, riskometer, benchmark, and document download guides). It is **not** investment advice, a recommendation to buy or sell, or a substitute for the SID/KIM/factsheet. Mutual fund investments are subject to market risks. Read all scheme-related documents carefully.

## Why two forms

The long form is complete but too heavy to sit above every answer. The
short form is the persistent one, and it is deliberately the first thing
on the page - a disclaimer that only appears after a user has already read
an answer is not doing much work.

## Where these come from

| String | Lives in | Shown |
|---|---|---|
| `DISCLAIMER` | `src/growbot/config.py` | banner, footer, every refusal |
| `LONG_DISCLAIMER` | `src/growbot/ui/app.py` | expander in the chat page |

`python -m growbot.ui.checks` asserts that the UI strings,
`config.DISCLAIMER` and this document all still agree, so an edit to any
one of them fails the checks rather than shipping a stale document.

