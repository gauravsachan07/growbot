# Implementation guide (phased)

**Product:** Growbot  
**Source of truth:** `docs/architecture.md` (behavior: `docs/PRD.md`)  
**How to use with Cursor:** Complete **one phase per chat** (or one prompt). Paste the **Cursor prompt** for that phase. Do not skip ahead. After each phase, run the **Done when** checks before starting the next.

**Rules for every phase**

- Keep **ingestion** and **retrieval** in separate modules. Do not fetch URLs at chat time.
- Do not invent investment advice, returns, or extra AMCs.
- No secrets in git. `.env` is gitignored.
- Prefer the file layout in architecture §4. Stage files must stay visible (`load.py`, `chunk.py`, `embed.py`, `index.py`, `query.py`, `assemble.py`, `prompt.py`, `answer.py`, `intent.py`).
- Official AMC / AMFI / SEBI pages are the corpus. Groww URLs identify schemes only unless an official page is unreachable.

**Suggested Cursor prefix (add to every prompt):**

> Follow `docs/architecture.md` and this phase only. Do not implement later phases. Match existing code style. After changes, tell me how to verify.

---

## Phase map

| Phase | RAG / product stage | You should have |
|-------|---------------------|-----------------|
| 0 | Repo + Python package | App imports; no crawl yet |
| 1 | Loading | `sources.csv` + cleaned documents |
| 2 | Chunking | Prefixed chunks, tables intact |
| 3 | Embedding + Chroma store | Persist collection `hdfc_mf_faq` |
| 4 | Ingest CLI | One command rebuilds the index |
| 5 | Guards | PII / advice / returns refuse **without** LLM |
| 6 | Retrieve + assemble | top-k + one citation candidate |
| 7 | Generate | Grounded ≤3 sentences + date |
| 8 | `ask()` orchestrator | Full online path, no ingest |
| 9 | UI | Welcome, 3 examples, disclaimer, chat |
| 10 | Demo pack | README, sample Q&A, disclaimer snippet |
| 11 | Hardening | Similarity floor, numeric post-check, empty index message |

---

## Phase 0 — Bootstrap the repo

**Goal:** Runnable Python package and ignored data dirs. No scraping, no UI.

**Create**

- `pyproject.toml` or `requirements.txt`: `sentence-transformers`, `chromadb`, `beautifulsoup4`, `httpx` or `requests`, `pypdf` (optional), Streamlit **or** Gradio (UI comes in phase 9; dependency can wait until then)
- `src/growbot/__init__.py` and empty stage modules (stubs OK)
- `.gitignore`: `data/chroma/`, `.env`, `__pycache__/`, `.venv/`
- `.env.example`: `LLM_*` placeholders, `CHROMA_PATH=data/chroma`, `EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2`
- `src/growbot/config.py`: constants from architecture §11 (`TOP_K=4`, chunk 512 / overlap 80, `SIMILARITY_FLOOR=0.4`, collection name `hdfc_mf_faq`, allowlist link placeholders)

**Done when**

- `python -c "import growbot"` works from the project (src layout or editable install)
- Chroma path and model name live in config, not scattered strings

**Cursor prompt**

```
Implement Phase 0 only from docs/implementation.md and docs/architecture.md.

Create a Python package under src/growbot with the module layout in architecture §4 (stub files allowed). Add pyproject.toml or requirements.txt, .gitignore, .env.example, and src/growbot/config.py with EMBEDDING_MODEL sentence-transformers/all-MiniLM-L6-v2, CHROMA_PATH data/chroma, collection hdfc_mf_faq, TOP_K 4, chunk size 512, overlap 80, SIMILARITY_FLOOR 0.4.

Do not fetch URLs, do not implement chunking/retrieval/UI. Make `import growbot` work. Summarize files created and how to install.
```

---

## Phase 1 — Loading (ingestion step 1)

**Depends on:** Phase 0  
**Goal:** Fetch public pages listed in `data/sources.csv` and return `Document { text, metadata }`.

**Create**

- `data/sources.csv` columns: `url,scheme_name,scheme_category,doc_type`
- Five HDFC Direct Growth schemes from the PRD (large cap, flexi cap / HDFC Equity Fund, ELSS tax saver, small cap, balanced advantage)
- Prefer **official HDFC AMC / AMFI / SEBI** URLs. Groww links are identity only.
- Optional extra row(s) with `doc_type=guide` for capital-gains / statement download how-to
- `src/growbot/ingest/load.py`: HTTP GET, HTML → visible text (strip nav/footer/scripts), PDF text if a source is PDF; skip + log on 404; `fetched_at` ISO date
- Metadata on every doc: `scheme_name`, `scheme_category`, `source_url`, `doc_type`, `fetched_at`

**Done when**

- Loading all CSV rows prints scheme names and character counts
- Failed URLs are skipped, not crashed
- No chat code

**Cursor prompt**

```
Implement Phase 1 only (Loading) per docs/implementation.md and architecture §5.1 and §7.1.

Add data/sources.csv with official public HDFC AMC / AMFI / SEBI pages for the five schemes in docs/PRD.md (Groww URLs identify schemes; do not use blogs). Implement src/growbot/ingest/load.py: fetch, strip boilerplate, attach metadata (scheme_name, scheme_category, source_url, doc_type, fetched_at). Skip failures with a log. Add a small `python -m growbot.ingest.load` or `__main__` that prints doc counts.

Do not chunk, embed, or build UI. If a listed official URL fails in your environment, pick another official page for that scheme and record it in the CSV.
```

---

## Phase 2 — Chunking (ingestion step 2)

**Depends on:** Phase 1  
**Goal:** Structure-aware chunks; label and value stay together.

**Create**

- `src/growbot/ingest/chunk.py`
- Split order: headings (`\n## `, `\n### `), then `\n\n`, then sentences
- Size **400–512 characters**, overlap **80**
- Do **not** split table blocks mid-row (exit load slabs stay one chunk)
- Prefix: `{scheme_name} | {section}`
- Copy parent metadata; set `section` when known
- Guides may use `scheme_name=general`

**Done when**

- Running load → chunk on the corpus prints ~dozens of chunks, not 5 giant blobs and not thousands of 20-char scraps
- Spot-check: an expense-ratio chunk contains both the words and the number
- Two schemes’ ratios are not in the same chunk

**Cursor prompt**

```
Implement Phase 2 only (Chunking) per docs/implementation.md and architecture §5.2.

Add src/growbot/ingest/chunk.py: structure-aware recursive split (headings, then blank lines, then sentences), 400–512 chars, 80 overlap, keep table blocks intact, prefix each chunk with "{scheme_name} | {section}". Preserve metadata from loaded documents.

Add a tiny CLI or script that loads then chunks and prints chunk count plus 2 sample chunks. Do not embed or write Chroma.
```

---

## Phase 3 — Embedding + Chroma store (ingestion steps 3–4)

**Depends on:** Phase 2  
**Goal:** Embed chunks with MiniLM; persist Chroma. Retrieval still later.

**Create**

- `src/growbot/ingest/embed.py`: `sentence-transformers/all-MiniLM-L6-v2`, encode chunk texts only, 384-d
- `src/growbot/ingest/index.py`: persistent Chroma at `CHROMA_PATH`, collection `hdfc_mf_faq`; each record `id`, embedding, document text, metadata including **mandatory** `source_url`
- Rebuild = delete collection/dir then write fresh
- Single embedding helper used later by query path (put shared encode in `embed.py` or `src/growbot/embeddings.py` so ingest and retrieve do not load two different models)

**Done when**

- After index, Chroma count equals chunk count
- A manual `collection.peek()` shows text + `source_url`
- `data/chroma/` is gitignored

**Cursor prompt**

```
Implement Phase 3 only (Embedding + Chroma store) per docs/implementation.md and architecture §5.3–5.4.

Use sentence-transformers/all-MiniLM-L6-v2 on chunks only. Persist ChromaDB at CHROMA_PATH, collection hdfc_mf_faq. Metadata must include source_url. Share one embed function for later query-time use. Rebuild replaces the collection.

Do not implement chat retrieval or LLM. Provide a way to run embed+index after load+chunk and print the collection count.
```

---

## Phase 4 — Ingest CLI (wire the offline pipeline)

**Depends on:** Phase 3  
**Goal:** One command runs load → chunk → embed → store. Chat still does not ingest.

**Create**

- `src/growbot/ingest/__main__.py` (or `python -m growbot.ingest`)
- Clear logs per stage (N docs, N chunks, N vectors)
- Exit non-zero if zero documents indexed

**Done when**

- Fresh clone path: install → `python -m growbot.ingest` → chroma folder populated
- Second run replaces data; no duplicate explosion

**Cursor prompt**

```
Implement Phase 4 only: wire ingestion CLI per architecture §5 and §4.

`python -m growbot.ingest` must load sources.csv, chunk, embed, and write Chroma, replacing the previous collection. Log counts per stage. Do not add retrieval, guards, or UI.
```

---

## Phase 5 — Guards

**Depends on:** Phase 0 (can parallel after 0; must exist before `ask()`)  
**Goal:** Deterministic refuse **before** retrieval. Allowlisted URLs only.

**Create**

- `src/growbot/guards/intent.py`
- Order: PII → advice/buy-sell/best fund → returns compare/predict → else `allow`
- PII: PAN, Aadhaar, account/folio-like numbers, OTP, email, phone — **do not log** the raw message
- Advice and returns refusals: polite facts-only copy + **one** URL from config allowlist (`EDU_LINK`, factsheet map)
- Return `AnswerPayload` shape from architecture §7.3 (`mode=refuse`)

**Done when**

- Unit tests or a tiny script: “Should I buy HDFC Small Cap?” → refuse + edu link  
- “Which has best returns?” → refuse + factsheet pointer, no ranking  
- Fake PAN-like string → refuse, nothing written to disk  
- “Expense ratio of HDFC Large Cap?” → `allow`

**Cursor prompt**

```
Implement Phase 5 only (Guards) per docs/implementation.md and architecture §8 and §7.3.

Add src/growbot/guards/intent.py: PII then advice then returns-comparison. Refusals use allowlisted EDU_LINK / factsheet URLs from config, never invented links. Do not store PII. Return AnswerPayload with mode refuse. Add a few tests or a __main__ demo of the four cases (buy, best returns, PII, allow fact question).

Do not call Chroma or the LLM.
```

---

## Phase 6 — Retrieve + assemble

**Depends on:** Phases 4 and 5 (index must exist; guards not required inside retriever)  
**Goal:** Online read path: embed query → Chroma top-k → context string + citation pick. No LLM.

**Create**

- `src/growbot/retrieve/query.py`: same MiniLM; detect scheme name from query; metadata `where` filter; if empty, unfiltered top-k; `TOP_K` 3–5
- `src/growbot/retrieve/assemble.py`: numbered context block as architecture §6.3; pick **one** `source_url` from highest-scoring supporting chunk; expose max similarity; if below `SIMILARITY_FLOOR` or schemes mixed/ambiguous, flag `weak=True`

**Done when**

- After ingest, script: query “expense ratio HDFC Large Cap Direct Growth” prints top chunks and one URL
- Query naming ELSS prefers ELSS chunks when filter works

**Cursor prompt**

```
Implement Phase 6 only (Retrieve + assemble) per architecture §6.1–6.3.

query.py: embed with the shared MiniLM, top-k 3–5, scheme_name metadata filter with fallback. assemble.py: context block with scheme, section, text, source_url, fetched_at; select exactly one citation URL from the best supporting chunk; set weak=True if similarity < SIMILARITY_FLOOR or mixed schemes.

CLI or script to retrieve one sample question. Do not call an LLM. Do not fetch web pages.
```

---

## Phase 7 — Generate

**Depends on:** Phase 6  
**Goal:** LLM answers from **context only**.

**Create**

- `src/growbot/generate/prompt.py`: facts-only; ≤3 sentences; no buy/sell; no return math; if context insufficient, refuse
- `src/growbot/generate/answer.py`: call LLM from env (`LLM_*`); parse `text`, `source_url`, `last_updated`, `mode`
- `source_url` and date **must** come from assembled metadata, not the model
- Optional post-check: if the answer contains a number not present in context, convert to refuse (can wait for phase 11 if you keep a TODO)

**Done when**

- With real retrieved context, expense-ratio question returns ≤3 sentences + that chunk’s URL + `Last updated from sources`
- With empty/weak context, `mode=refuse`, no fabricated ratio

**Cursor prompt**

```
Implement Phase 7 only (Generate) per architecture §6.4.

prompt.py: grounded facts-only system prompt. answer.py: LLM from env vars, context from assembler; user-facing source_url and last_updated from chunk metadata only; ≤3 sentences. Document the model in a comment or README stub.

Do not build Streamlit/Gradio. Provide a function generate(question, assembled) -> AnswerPayload. If LLM key missing, fail clearly.
```

---

## Phase 8 — `ask()` orchestrator

**Depends on:** Phases 5–7  
**Goal:** Single function the UI will call. Ingestion is never invoked here.

**Create**

- `src/growbot/ask.py` (or `retrieve/pipeline.py`):  
  `ask(question) -> AnswerPayload`  
  1) guards  
  2) if refuse, return  
  3) query + assemble  
  4) if weak, refuse (edu or “not in corpus”) with allowlisted link  
  5) generate  
- If Chroma empty / missing, return a clear error payload (not a hallucinated fact)

**Done when**

- REPL: fact question works; “should I buy?” never hits LLM (log or mock to prove)
- Empty chroma: friendly error

**Cursor prompt**

```
Implement Phase 8 only: ask(question) -> AnswerPayload per architecture §9–10.

Wire guards → retrieve/assemble → generate. Never run ingest. If guards refuse, skip LLM. If retrieval weak or Chroma missing, refuse/error without inventing facts. No UI yet. Add a `python -m growbot.ask` that reads one question from CLI.
```

---

## Phase 9 — Tiny UI

**Depends on:** Phase 8  
**Goal:** Class-demo chat page. PRD §8 / architecture §9.

**Create**

- `src/growbot/ui/app.py` — Streamlit **or** Gradio (pick one; Streamlit is fine)
- No auth, no PAN/email fields
- On load: welcome one-liner (five HDFC schemes), **three clickable example questions**, persistent **Facts-only. No investment advice.**
- Examples:
  1. Expense ratio of HDFC Large Cap Fund (Direct Growth)?
  2. Lock-in for HDFC ELSS Tax Saver?
  3. How to download a capital-gains statement?
- Fact card: text, one hyperlink, last updated, optional scheme chip  
- Refusal card: polite + one link + same disclaimer  
- Call only `ask()`

**Done when**

- App runs locally; examples clickable; disclaimer always visible
- Buy/sell typed in chat shows refusal card

**Cursor prompt**

```
Implement Phase 9 only: tiny chat UI per architecture §9 and PRD UX.

Streamlit or Gradio single page, no auth. Welcome line, 3 example questions, persistent "Facts-only. No investment advice." Chat calls growbot.ask.ask only. Fact vs refuse cards: one source link, last updated on facts.

Do not change RAG logic except to hook the UI. Document the run command.
```

---

## Phase 10 — Demo pack (milestone files)

**Depends on:** Phase 9 (or 8 if UI video later)  
**Goal:** What the class asked to submit besides the app.

**Create**

- `README.md`: setup (venv, ingest, UI), AMC + five schemes, known limits (PRD §13), RAG stage list
- `data/sources.csv` already exists — mention it as the source list
- `docs/sample_qa.md`: 5–10 queries with **actual** assistant answers + links (run the app or `ask()` and paste; include one buy refusal and one best-returns refusal)
- `docs/disclaimer.md`: PRD §9 snippet as used in the UI
- Architecture already exists; README should link `docs/architecture.md`

**Done when**

- A classmate can follow README cold-start
- Sample Q&A has citations
- Disclaimer matches the UI string

**Cursor prompt**

```
Implement Phase 10 only: demo deliverables.

Write README.md (setup, HDFC + five schemes, known limits, ingest vs retrieval). Add docs/sample_qa.md by running ask() or documenting expected answers with source links for the PRD evaluation questions. Add docs/disclaimer.md with the PRD disclaimer used in the UI. Do not refactor the pipeline.
```

---

## Phase 11 — Hardening (demo quality)

**Depends on:** Phase 9  
**Goal:** Failure modes from architecture §13.

**Do**

- Numeric claim must appear in retrieved context or convert to refuse
- Mixed-scheme / low similarity already flagged — make refuse copy consistent
- Empty index message in UI
- README: if official HTML is blocked, which source was used
- Optional: cache MiniLM in memory so the UI does not reload the model every question

**Done when**

- Walkthrough in architecture §14 works end to end  
- Sample set: aim ≥8/10 fact questions cited correctly; both refusal types work

**Cursor prompt**

```
Implement Phase 11 only: hardening per architecture §13–14.

Add numeric grounding post-check, consistent weak-retrieval refusal, UI message when Chroma is empty, and avoid reloading MiniLM per request if not already. Do not add new product features or extra AMCs. Then list how to run the classmate walkthrough.
```

---

## Verification cheat sheet (after phase 11)

```text
python -m growbot.ingest
python -m growbot.ask "What is the expense ratio of HDFC Large Cap Fund Direct Growth?"
python -m growbot.ask "Should I buy HDFC Small Cap Fund?"
# then: streamlit run src/growbot/ui/app.py   (or gradio command from README)
```

Walkthrough: CSV → logs of load/chunk/index → fact + one link → buy-question refusal.

---

## What not to do in any phase

- Crawl the whole web or index third-party blogs
- Compute or rank returns
- Store chat logs that include PII
- Call the LLM during ingest
- Merge ingest into the chat request path
- Skip citations “until later”
