# Growbot

A **facts-only** retrieval-augmented chatbot for five HDFC Mutual Fund Direct
Growth schemes. It answers disclosed scheme facts - expense ratio, exit load,
minimum SIP, lock-in, benchmark, riskometer level - and refuses everything
else.

> **Facts-only. No investment advice.** See [`docs/disclaimer.md`](docs/disclaimer.md).

It will not tell you whether to buy a fund, rank funds by returns, or compute
anything for your portfolio. Those are refused by code before any model is
contacted, which is the point of the project.

- **Design and rationale:** [`docs/architecture.md`](docs/architecture.md)
- **Requirements:** [`docs/PRD.md`](docs/PRD.md)
- **Real answers with citations:** [`docs/sample_qa.md`](docs/sample_qa.md)
- **Build phases:** [`docs/implementation.md`](docs/implementation.md)

---

## Quick start

Requires **Python 3.11** (see the note below). From a fresh clone:

```bash
# 1. venv
python -m venv .venv
.\.venv\Scripts\activate          # Windows
# source .venv/bin/activate       # macOS / Linux

# 2. dependencies
pip install -e ".[all]"

# 3. API key
copy .env.example .env           # Windows
# cp .env.example .env           # macOS / Linux
# then set LLM_API_KEY and LLM_MODEL in .env

# 4. build the index (fetches the pages, ~1 min)
python -m growbot.ingest

# 5. run it
python -m streamlit run src/growbot/ui/app.py
```

Then open the URL Streamlit prints, usually <http://localhost:8501>.

Prefer the terminal? No server needed:

```bash
python -m growbot.ask "Expense ratio of HDFC Large Cap Fund Direct Growth?"
python -m growbot.ask -t                # interactive, showing each pipeline stage
python -m growbot.ask --stub            # no API call; exercises retrieval only
```

### Why Python 3.11 and not the newest

`sentence-transformers` pulls in `torch`, and **torch has no wheels for Python
3.13/3.14**. Building the venv on 3.14 fails at install time with a long
unhelpful resolution error. 3.11 has wheels for everything here. This bit cost
real time to diagnose, so it is stated up front rather than discovered.

### If you have no API key

Everything except answer *generation* still works: the guard refuses advice,
PII, and return questions; retrieval refuses vague and out-of-corpus questions.
`python -m growbot.ask --stub` exercises the rest without a key. Only facts
need one.

---

## What it covers

**AMC: HDFC Asset Management Company Limited.** Five schemes, all Direct Growth:

| Scheme | Category | Aliases accepted |
|---|---|---|
| HDFC Large Cap Fund | `large_cap` | HDFC Large Cap |
| HDFC Flexi Cap Fund | `flexi_cap` | HDFC Equity Fund (its former name) |
| HDFC ELSS Tax Saver Fund | `elss` | HDFC Tax Saver |
| HDFC Small Cap Fund | `small_cap` | HDFC Small Cap |
| HDFC Balanced Advantage Fund | `hybrid` | HDFC Balanced Advantage |

### The corpus

[`data/sources.csv`](data/sources.csv) is the authoritative list: 17 URLs across
3 hosts, and every citation the bot emits must appear in it.

| Host | URLs | Role |
|---|---|---|
| `www.hdfcfund.com` | 5 | AMC scheme pages - the primary source |
| `groww.in` | 5 | scheme identity pages (used to *identify* schemes, not as the source of truth) |
| `www.mutualfundssahihai.com` | 7 | AMFI's investor-education guides and glossary |

**A caveat worth stating plainly:** the two most authoritative hosts do not
work from this machine. `hdfcfund.com` returns HTTP 403 behind an Akamai bot
protection, and SEBI's host refuses connections outright. The list therefore
contains both the official URLs (fetched where possible, skipped with a logged
reason where not) and the AMFI material that *is* reachable. No scraper was
written to work around either block - the PRD's fallback clause was used
instead, and which URLs actually yielded text is visible in the ingest log.
This is why `docs/sample_qa.md` cites Groww pages for some facts: the facts are
cross-checked against AMFI guidance, not invented.

---

## How it works

Ingestion and retrieval are **separate programs on purpose**. Nothing is ever
fetched at question time - a chat turn cannot reach the network except through
the model call.

**Offline, run once** (`python -m growbot.ingest`):

| Stage | File | What it does |
|---|---|---|
| Load | `ingest/load.py` | fetch + parse each URL in `data/sources.csv` |
| Chunk | `ingest/chunk.py` | split into passage-sized chunks, keep metadata |
| Embed | `ingest/embed.py` | `all-MiniLM-L6-v2`, one shared cached encoder, onnxruntime by default |
| Store | `ingest/index.py` | write to Chroma with a cosine index |

**Online, per question** (`python -m growbot.ask`):

| Stage | File | What it does |
|---|---|---|
| Guard | `guards/intent.py` | refuse PII / advice / returns. No I/O. |
| Retrieve | `retrieve/query.py` | embed the question, top-k, filter to one scheme |
| Assemble | `retrieve/assemble.py` | build context, apply the similarity floor |
| Generate | `generate/answer.py` | ask the model for prose only |
| Orchestrate | `ask.py` | `ask(question) -> AnswerPayload` |

Four properties that are enforced rather than hoped for:

- **A refusal never reaches the model.** Not because the prompt asks nicely -
  because `ask()` returns before `generate()` is called. Asserted with a
  counting stub in `python -m growbot.ask_checks`.
- **The model never supplies a URL or a date.** It is asked for prose; the
  citation and `Last updated from sources` come from chunk metadata, and any
  URL the model invents is stripped from its answer.
- **Every answer is grounded.** A number in the answer must appear in the
  retrieved context or the answer is discarded.
- **A missing index is an error, not an invitation to guess.** You get told to
  run the ingest CLI.

### Conversation memory

A follow-up like *"and its exit load?"* names no fund, and without help it goes
wrong in a way that is easy to miss: retrieval has nothing fund-specific to
match, a general AMFI explainer page wins, and **that page is then attached to
the answer as the source** - a citation that is real, plausible, and about a
different question than the one you asked. Three of four ordinary follow-ups
simply refused instead.

So `ask()` takes an optional `history` of prior question texts and uses it for
one job: working out which fund an underspecified question is about
(`memory.py`). The rules are deliberately narrow, and each is enforced by
`python -m growbot.memory_checks`:

- **The current question always wins.** A question naming its own fund never
  consults history, so switching funds mid-conversation gives the new fund.
- **It only affects retrieval.** History never enters the prompt, so the model
  cannot be led onto the previous fund, and the grounding check still sees only
  chunks retrieved for the question actually asked.
- **PII is never remembered.** Every history entry is screened through the
  guard's own `detect_pii` and dropped, inside the layer that stores it - a
  caller who forgets to filter cannot breach it. Nothing is written to disk;
  the window lives in the browser session and dies with the tab.
- **A refusal can still donate a fund.** *"Should I buy HDFC Small Cap?"* is
  refused, but it established what the conversation is about, so the next
  question inherits the fund while remaining subject to every other rule.
- **The window is the last `MEMORY_TURNS` (10) question turns** -
  `MEMORY_TURNS` in `config.py`. With no history, behaviour and the trace are
  unchanged.

It is not a prompt context window. Folding ten turns of text into the query
would drag it away from the question actually asked, so "remembers more" would
mean "retrieves worse"; the effect is confined to resolving a gap that
`detect_scheme` left empty.

### Tuning

All tunables live in `src/growbot/config.py`. The ones worth knowing:

| Setting | Value | Why |
|---|---|---|
| `SIMILARITY_FLOOR` | `0.45` | Measured, not guessed. Below this the context is too thin to answer from. |
| `TOP_K` | `5` | Measured. Larger pulls in more schemes and trips the mixed-scheme refusal. |
| `MEMORY_TURNS` | `10` | How many prior question turns can resolve an underspecified follow-up. `0` disables memory. |
| `EMBEDDING_BACKEND` | `onnx` | Which runtime executes the encoder. `onnx` needs 239 MB of RAM, `torch` needs 580 MB, for identical vectors. `torch` stays available as the reference. |

All of them are commented out of `.env.example` on purpose, so `config.py`
stays the single source of truth.

### Memory

Measured peak RSS for one `ask()`, torch included and excluded:

| | peak | fits a 512 MB container |
|---|---|---|
| `EMBEDDING_BACKEND=torch` | 579.7 MB | no |
| `EMBEDDING_BACKEND=onnx` (default) | **238.7 MB** | yes, 273 MB spare |

Torch costs ~500 MB of runtime before it evaluates a 22M-parameter model, so
the default runs the same weights through onnxruntime — already installed as a
chromadb dependency, so it costs no extra package. The two agree to
**cosine 1.0000000000**, max elementwise difference 1.6e-07, in the raw form
ingestion uses, which is float32 rounding and far below the precision
`SIMILARITY_FLOOR` is specified to. So the switch needs no re-ingest, and
`retrieve.checks` re-measures that agreement on every run rather than trusting
it.

---

## Verification

```bash
python -m growbot.guards           # 105 checks
python -m growbot.retrieve.checks  # 118 checks
python -m growbot.generate.checks  #  94 checks
python -m growbot.ask_checks       # 122 checks
python -m growbot.ui.checks        # 153 checks
python -m growbot.memory_checks    #  78 checks
python -m growbot.hardening_checks # 227 checks
```

897 checks, each suite exiting non-zero on regression. All run offline and need
no API key.

The last suite covers the failure modes in [`docs/architecture.md`](docs/architecture.md)
§13, and it **provokes** each one rather than assuming it: it hands the pipeline
a model that invents a plausible, ungrounded expense ratio and asserts the
answer is discarded; it points the pipeline at an empty Chroma directory and
asserts a friendly refusal instead of a guess; it feeds the chunker a table
long enough to force a split and asserts no row is cut in half.

The ui suite also covers this file: it asserts the five schemes, the pipeline
stage list, the setup commands, and the disclaimer strings are all present and
accurate, so the documentation cannot quietly go stale.

### Two things that need a key, kept separate on purpose

```bash
python tools/eval_citations.py     # citation accuracy against the live model
python tools/regen_sample_qa.py    # re-asks all 11 questions, rewrites sample_qa.md
python tools/regen_disclaimer.py   # rewrites disclaimer.md from the UI strings
```

`eval_citations.py` is the one measurement the offline suites cannot make. They
assert the guarantees that hold for *any* model; it asks whether the citation
on a real answer is the *right* one — answered, exactly one link, and that link
belonging to the scheme the question was about. It is not part of the check
suites because a free tier rate-limits, and a suite that fails intermittently
teaches people to ignore it.

### The classmate walkthrough

```bash
python tools/walkthrough.py            # needs a key for step 4
python tools/walkthrough.py --offline  # stubs step 4; no key needed
```

This runs the five steps of architecture §14 in order and exits non-zero if any
of them stops behaving as documented, so it doubles as a demo script and as a
regression test. It prints the retrieved chunks and their similarity scores at
step 4, which is the part worth showing: the citation is chosen from the
retrieved metadata before the model is ever called.

---

## Known limits

From PRD section 13, plus what this build actually ran into:

- **Five schemes, not all mutual funds.** Out of scope, not merely unknown.
- **Facts go stale.** `Last updated from sources` is the fetch time, not a live
  AMC feed. Re-run `python -m growbot.ingest` to refresh.
- **Groww pages may differ from the SID.** Where they conflict, the
  SID/KIM/factsheet wins.
- **MiniLM with small chunks can miss poorly worded queries.** Name the scheme
  in your question; "which fund" is refused as too vague to answer safely.
- **No live NAV, no tax computation, no portfolio questions.**
- **Corpus drift is real.** The chunk count moved between 195 and 223 across
  fetches of the same pages, because the source pages are not versioned. The
  index is not snapshotted, so two people ingesting a week apart get slightly
  different corpora.
- **Vague cross-scheme questions are refused.** "Tell me about HDFC funds"
  retrieves well (0.80 similarity) but spans five schemes, so the
  mixed-scheme rule declines it. A high similarity score does not imply a
  question is answerable.
- **The model can be too terse.** An earlier capture had an answer that was the
  bare figure `0.77%` — correct and grounded, but not a usable sentence.
  `generate()` now re-asks once when a reply contains no words at all, and if
  that fails too it refuses with copy that names the real cause rather than
  blaming a corpus that answered fine.
- **Some AMFI guide URLs have started 404ing.** Of the 17 rows in
  `data/sources.csv`, 10 currently fail to fetch (5 official AMC pages at HTTP
  403, 5 guide URLs at HTTP 404) and are skipped with a logged reason. The
  five scheme pages on the fallback host still load, so the corpus is thinner
  than the list suggests. A linked page can rot; the ingest log says which.
- **Free LLM tiers rate-limit.** Expect `429`s on a burst of questions; the
  client retries with backoff, and if the allowance is genuinely spent the bot
  says so instead of failing silently.

## Security notes

- `.env` is gitignored and holds the API key. `.env.example` must never contain
  a real key - `python -m growbot.generate.checks` scans the repo for one and
  fails if it finds it.
- A message containing a PAN, Aadhaar, OTP or similar is refused, discarded, and
  **never logged**. The guard logs a category and a character count only.
- No database, no user accounts, no message history on disk. The chat thread
  lives in the browser session and dies with the tab.
- Conversation memory is the last 10 question turns, in memory only, and is
  screened for PII by the same detector the guard uses - a turn carrying a PAN
  or an OTP is dropped before it can be consulted, even if a caller forgets to
  filter. Nothing is persisted and none of it is sent anywhere except, as
  always, to the model for the single question being answered.
