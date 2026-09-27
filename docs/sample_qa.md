# Sample Q&A

Every answer below is a **real** response from `growbot.ask.ask()`, pasted
verbatim - not written by hand, and not edited to look better than it was.
Regenerate with `python tools/regen_sample_qa.py`.

| | |
|---|---|
| Captured | 2026-09-27 |
| Model | `groq` / `openai/gpt-oss-120b` |
| Corpus | `data/sources.csv` (17 URLs) |
| Outcome | 11/11 matched the expected mode |

The citations and the `Last updated from sources` date come from retrieval
metadata, never from the model - the model is only ever asked for prose.

> **Facts-only. No investment advice.**

---

## Q1. Expense ratio of HDFC Large Cap Fund Direct Growth?

**FACT**

> The expense ratio is 1.03%.

- **Scheme:** HDFC Large Cap Fund
- **Source:** https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth
- **Last updated from sources:** 2026-09-27
- **Refusal reason:** `n/a (answered)`
- **Stopped at:** `generate` | **model called:** yes

## Q2. Exit load of HDFC Small Cap Fund Direct Growth?

**FACT**

> Exit load of 1% if redeemed within 1 year.

- **Scheme:** HDFC Small Cap Fund
- **Source:** https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth
- **Last updated from sources:** 2026-09-27
- **Refusal reason:** `n/a (answered)`
- **Stopped at:** `generate` | **model called:** yes

## Q3. Minimum SIP for HDFC Balanced Advantage Fund Direct Growth?

**FACT**

> Minimum SIP Investment is ₹100.

- **Scheme:** HDFC Balanced Advantage Fund
- **Source:** https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth
- **Last updated from sources:** 2026-09-27
- **Refusal reason:** `n/a (answered)`
- **Stopped at:** `generate` | **model called:** yes

## Q4. Lock-in period for HDFC ELSS Tax Saver Direct Growth?

**FACT**

> The lock‑in period is 3 years.

- **Scheme:** HDFC ELSS Tax Saver Fund
- **Source:** https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth
- **Last updated from sources:** 2026-09-27
- **Refusal reason:** `n/a (answered)`
- **Stopped at:** `generate` | **model called:** yes

## Q5. Riskometer / risk level of HDFC Equity Fund (Flexi Cap) Direct Growth?

**FACT**

> Very High risk.

- **Scheme:** HDFC Flexi Cap Fund
- **Source:** https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth
- **Last updated from sources:** 2026-09-27
- **Refusal reason:** `n/a (answered)`
- **Stopped at:** `generate` | **model called:** yes

## Q6. Benchmark of HDFC Large Cap Fund Direct Growth?

**FACT**

> NIFTY 100 Total Return Index.

- **Scheme:** HDFC Large Cap Fund
- **Source:** https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth
- **Last updated from sources:** 2026-09-27
- **Refusal reason:** `n/a (answered)`
- **Stopped at:** `generate` | **model called:** yes

## Q7. How to download a capital-gains statement?

**REFUSAL**

> I don't have a passage in my source material that answers that, so I won't guess at it. For this, the scheme pages listed in the sources file are what I hold. Facts-only. No investment advice.

- **Educational link:** https://www.mutualfundssahihai.com/en/glossary
- **Refusal reason:** `weak_retrieval`
- **Stopped at:** `assemble` | **model called:** **no**

## Q8. Should I buy HDFC Small Cap Fund?

**REFUSAL**

> I can't say whether HDFC Small Cap Fund is worth buying - that's an investment decision, not a fact I can look up. What I can do is quote you the disclosed numbers, like expense ratio, exit load, benchmark and risk level, straight from the fund's own pages. Facts-only. No investment advice.

- **Scheme:** HDFC Small Cap Fund
- **Educational link:** https://www.mutualfundssahihai.com/en/glossary
- **Refusal reason:** `advice`
- **Stopped at:** `guard` | **model called:** **no**

## Q9. Which fund has the best returns?

**REFUSAL**

> I can't rank the five funds by returns or tell you what they'll return - I have no basis for either, and a made-up ranking would be worse than none. For that, each AMC publishes its own fund factsheet, and comparing the official factsheets is the way to do this without anyone ranking them for you. Facts-only. No investment advice.

- **Educational link:** https://www.mutualfundssahihai.com/en/glossary
- **Refusal reason:** `returns`
- **Stopped at:** `guard` | **model called:** **no**

## Q10. Expense ratio of HDFC Flexi Cap Fund (Direct Growth)?

**FACT**

> The expense ratio of HDFC Flexi Cap Fund (Direct Growth) is 0.77%.

- **Scheme:** HDFC Flexi Cap Fund
- **Source:** https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth
- **Last updated from sources:** 2026-09-27
- **Refusal reason:** `n/a (answered)`
- **Stopped at:** `generate` | **model called:** yes

## Q11. What about its exit load?

_Follow-up. Asked after: "Expense ratio of HDFC Small Cap Fund Direct Growth?"_

**FACT**

> Exit load of 1% if redeemed within 1 year.

- **Scheme:** HDFC Small Cap Fund
- **Source:** https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth
- **Last updated from sources:** 2026-09-27
- **Refusal reason:** `n/a (answered)`
- **Stopped at:** `generate` | **model called:** yes

---

## What the trace lines show

`Stopped at` is the pipeline stage that **decided** the answer, and
`model called` is whether the provider was contacted at all. Read together
they separate the two kinds of refusal, which look identical in the UI:

- **Never reached the model because a rule blocked it.** Q8 and Q9 stop at
  `guard`. The question was refused before retrieval, so no prompt change
  can make the bot answer it.
- **Never reached the model because the evidence was too weak.** Q7 stops at
  `assemble`, where the retrieved context scored below the similarity floor
  and spanned two schemes. The bot declined on its own evidence.

In all three cases `model called` is **no**, which is the claim worth
making: the refusals are structural, not the model choosing to be cautious.

Q11 adds a `memory` stage, which appears only when the question depends on
earlier turns. It records *which fund the follow-up was resolved to* and
where that came from. It is the one stage that reads the conversation
rather than the question, and it deliberately does no more than that: it
picks the fund for retrieval and nothing else, so the retrieval, the
grounding check and the prompt are all unchanged by it.

## Citation accuracy

The eleven questions above check *whether* the bot answers. A separate
measurement checks whether the citation is the **right** one, which is
stricter: answered, exactly one link, and that link belonging to the
scheme the question was about. Run `python tools/eval_citations.py` over
twelve fact questions spanning all five schemes.

Last run **2026-09-27**: **11/12** correctly cited, against a bar
of 8/12.

The one miss was the section 80C question, refused as `insufficient`
because the retrieved text does not state it - a genuine gap in the
corpus, not a wrong citation. **No answer was ever cited to the wrong
fund**, which is the failure that matters: it is the one a reader has no
way to notice.

## Known limits

- Answers are only as current as the fetch that built the index. The
  `Last updated from sources` line is the fetch time, not a live AMC feed.
- Only the five HDFC Direct Growth schemes in `data/sources.csv` are
  covered. Anything else is out of scope, not merely unknown.
- No NAV, no tax computation, no portfolio questions, no recommendations.
- One scheme per answer. A question spanning two funds is refused rather
  than answered from a blend of both.

- No answer in this run was a bare figure. An earlier capture had one
  (`0.77%`, with no sentence around it): correct and grounded, but not
  usable. `generate()` now re-asks once when a reply contains no words
  at all, and if that fails too it refuses with copy that names the real
  cause instead of blaming the corpus. See
  `python -m growbot.hardening_checks`.

