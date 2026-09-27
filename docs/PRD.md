# Product Requirements Document

**Product:** Growbot — Mutual Fund Facts-Only FAQ Assistant  
**Type:** Class-demo RAG chatbot  
**Version:** 1.0  
**Date:** 27 September 2026  
**Source brief:** `docs/ProblemStatement.txt`

---

## 1. Summary

Growbot is a small retrieval-augmented generation (RAG) chatbot that answers **factual** questions about a fixed set of HDFC mutual fund schemes. Answers come only from an indexed corpus of **official public pages**. Every answer includes **one source link**. The product does **not** give investment advice, compare returns, or handle personal data.

This PRD is scoped for a **classroom demo**: a working prototype that makes the RAG stages visible (load → chunk → embed → store → retrieve → generate), not a production support tool.

---

## 2. Problem

Retail investors and support/content teams repeatedly ask the same scheme facts: expense ratio, exit load, minimum SIP, ELSS lock-in, riskometer, benchmark, and how to download statements. Those facts live on public AMC / AMFI / SEBI pages, but they are hard to find quickly and easy to mix with opinion.

**Without Growbot:** users hunt factsheets and get mixed “should I buy?” answers.  
**With Growbot:** a facts-only Q&A over a small, cited corpus.

---

## 3. Goals and non-goals

### 3.1 Goals

| ID | Goal | Success for the demo |
|----|------|----------------------|
| G1 | Answer factual MF scheme questions from retrieved sources | ≥8/10 sample queries answered with a correct fact + citation |
| G2 | Always cite | Every answer shows exactly one source URL |
| G3 | Stay facts-only | Advice / buy-sell / “best fund” questions are refused with a polite message + educational link |
| G4 | Demonstrate a full RAG pipeline | Architecture and README show ingestion and retrieval as separate stages |
| G5 | Tiny, honest UI | Welcome line, 3 example questions, “Facts-only. No investment advice.” |

### 3.2 Non-goals

- Portfolio advice, suitability, tax optimization, or “which fund should I pick?”
- Computing or ranking returns, CAGR, or performance vs peers
- Login, KYC, transactions, or live AMC account actions
- Multi-AMC coverage, daily auto-refresh of the full AMFI universe
- Production-grade auth, observability, or SLA
- Accepting or storing PII (PAN, Aadhaar, account numbers, OTP, email, phone)

---

## 4. Users

| User | Need in this demo |
|------|-------------------|
| Retail user comparing schemes | Fast facts (expense ratio, SIP min, exit load, lock-in) with a link to check |
| Support / content teammate | Repeatable answers without inventing guidance |
| Instructor / classmates | See RAG stages, corpus scope, refusals, and citations |

---

## 5. Scope

### 5.1 AMC and schemes

**AMC:** HDFC Mutual Fund (one AMC).

| Category | Scheme (as named on seed URLs) | Seed URL |
|----------|--------------------------------|----------|
| Large Cap | HDFC Large Cap Fund — Direct Growth | https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth |
| Flexi Cap | HDFC Equity Fund — Direct Growth | https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth |
| ELSS | HDFC ELSS Tax Saver Fund — Direct Plan Growth | https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth |
| Small Cap | HDFC Small Cap Fund — Direct Growth | https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth |
| Balanced Advantage (Hybrid) | HDFC Balanced Advantage Fund — Direct Growth | https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth |

### 5.2 Corpus policy

**Public sources only.** Prefer official pages:

- HDFC AMC scheme pages, factsheets, KIM / SID, fee & charges, scheme FAQs
- AMFI / SEBI pages for riskometer, benchmark notes, investor education
- Statement / capital-gains / tax-document **guides** (how-to), not user account data

**Seed URLs** in the brief are Groww scheme pages. Use them to **identify** the five schemes. Index **official AMC / AMFI / SEBI** documents as the answer sources whenever those pages exist. Do not treat blogs, YouTube, or unofficial “best funds” articles as sources.

**Source list deliverable:** CSV or Markdown of the **five primary URLs actually used** in the index (plus extras only if needed for statement/tax how-tos). Keep the list small and auditable.

### 5.3 Question types in scope

- Expense ratio  
- Exit load  
- Minimum SIP / min lumpsum (if present in corpus)  
- ELSS lock-in  
- Riskometer  
- Benchmark  
- How to download statements / capital-gains documents (process from public guides)

### 5.4 Out of scope questions (refuse)

- “Should I buy / sell / switch?”  
- “Which of these is best?”  
- “Will this beat the market?”  
- Return comparisons or predicted performance  
- Anything requiring PAN, folio, or personal holdings  

---

## 6. RAG architecture (required for the demo)

The system must follow **both** RAG stages: **data ingestion** and **data retrieval**. Each stage is distinct in code and in the architecture write-up.

```
Ingestion (offline / one-shot)          Retrieval (per query)
─────────────────────────────          ─────────────────────
1. Loading                             5. Query embed
2. Chunking                            6. Vector search (ChromaDB)
3. Embedding                           7. Context assembly
4. Store in Vector DB                  8. LLM generate + cite + refuse if needed
```

### 6.1 Loading

- Fetch allowed public pages (HTML and/or PDF factsheet/KIM where used).
- Strip navigation/boilerplate; keep scheme facts and how-to text.
- Attach metadata: `scheme_name`, `scheme_category`, `source_url`, `doc_type` (factsheet | kim | sid | faq | amfi | guide), `fetched_at`.

### 6.2 Chunking strategy (chosen for this corpus)

This corpus is **short, sectioned, fact-dense** (tables and headings: expense ratio, loads, SIP, lock-in, risk, benchmark). Chunks must keep a **complete fact + its label** together so retrieval does not split “1.05%” from “expense ratio”.

**Decision: structure-aware recursive split, then size cap.**

1. Split on headings and table-ish boundaries first (`\n## `, `\n### `, `\n\n`, then sentences).
2. Target chunk size **400–512 characters** (~80–120 tokens for MiniLM), **overlap 80 characters**.
3. If a table cell group (e.g. exit-load slabs) is still larger than the cap, keep the **whole table block** in one chunk rather than splitting mid-row.
4. Prefix each chunk with a one-line header: `{scheme_name} | {section}` so similar numbers across five funds stay distinguishable.

**Why not naive 1000-token chunks:** five schemes share similar templates; large chunks mix funds and raise citation errors.  
**Why not tiny 100-token chunks:** MiniLM embeddings lose the “what this number is” context.

### 6.3 Embedding

- **Model:** `sentence-transformers/all-MiniLM-L6-v2` (as specified).
- Embed chunks only (not raw full pages).
- Same model for query embeddings at retrieval time.

### 6.4 Vector store

- **Vector DB:** ChromaDB (local, demo-friendly).
- Persist on disk so the demo does not re-ingest every run (optional one-command reindex).
- Store document text + metadata (`source_url` required for citations).

### 6.5 Retrieval and generation

- Embed the user question; retrieve **top-k = 3–5** chunks.
- Filter or boost by scheme name if the query names a fund.
- Generate from retrieved context only (no unaided “knowledge” of NAVs or ratios).
- **Answer rules:**
  - ≤ 3 sentences
  - Exactly **one** citation URL (the best matching source)
  - Include `Last updated from sources: <date>` (from `fetched_at` or document date if present)
  - If retrieval is weak / question is opinion: refuse; do not guess

---

## 7. Functional requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| F1 | Chat UI accepts a question and returns an answer | P0 |
| F2 | Answers are grounded in retrieved chunks | P0 |
| F3 | Every factual answer includes one source link | P0 |
| F4 | Advice / performance-comparison questions are refused with a facts-only message and an educational (AMFI/SEBI/AMC investor-ed) link | P0 |
| F5 | UI shows welcome text, **3 example questions**, and **“Facts-only. No investment advice.”** | P0 |
| F6 | No form fields or logs for PAN, Aadhaar, account numbers, OTP, email, or phone; if pasted, refuse and do not store | P0 |
| F7 | Do not compute or compare returns; if asked, point to the official factsheet link | P0 |
| F8 | Ingestion pipeline can rebuild Chroma from the source list | P1 |
| F9 | Sample Q&A file (5–10 queries) with answers + links | P1 |
| F10 | README: setup, AMC + schemes, known limits | P1 |

---

## 8. UX requirements

**Layout:** single chat page (no auth).

**On load:**

- Welcome one-liner (e.g. HDFC scheme facts assistant for the five listed funds).
- Three click-to-ask examples, e.g.:
  1. What is the expense ratio of HDFC Large Cap Fund (Direct Growth)?
  2. What is the lock-in for HDFC ELSS Tax Saver?
  3. How do I download a capital-gains statement?
- Persistent disclaimer: **Facts-only. No investment advice.**

**Answer card:**

- Short answer (≤ 3 sentences)
- Source: hyperlink
- `Last updated from sources: YYYY-MM-DD`
- Optional: scheme name chip if detected

**Refusal card:** polite, no recommendation, one educational link, same disclaimer.

---

## 9. Disclaimer (required snippet)

Use this (or equivalent) in the UI and README:

> This assistant answers **facts only** from public scheme documents (expense ratio, exit load, SIP minimum, lock-in, riskometer, benchmark, and document download guides). It is **not** investment advice, a recommendation to buy or sell, or a substitute for the SID/KIM/factsheet. Mutual fund investments are subject to market risks. Read all scheme-related documents carefully.

---

## 10. Technical constraints (class demo)

| Area | Choice |
|------|--------|
| RAG | Full pipeline: load → chunk → embed → Chroma → retrieve → generate |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector DB | ChromaDB |
| LLM | Any small hosted or local chat model suitable for a demo (document the choice in README). Generation must stay grounded. |
| App | Lightweight web app or notebook with a chat-like UI |
| Data | Public URLs only; no app-backend screenshots |

---

## 11. Deliverables (milestone)

1. **Working prototype** (app URL) **or** ≤ 3-minute demo video if hosting is not possible  
2. **Source list** (CSV or Markdown) of the five URLs used  
3. **README** — setup, scope (AMC + schemes), known limits  
4. **Sample Q&A** — 5–10 queries with assistant answers + links  
5. **Disclaimer** snippet as used in the UI  
6. **Architecture note** (can live in README) showing ingestion vs retrieval stages  

---

## 12. Sample evaluation set (minimum)

Use these (plus 3–5 more) in `docs` sample Q&A:

1. Expense ratio of HDFC Large Cap Fund Direct Growth?  
2. Exit load of HDFC Small Cap Fund Direct Growth?  
3. Minimum SIP for HDFC Balanced Advantage Fund Direct Growth?  
4. Lock-in period for HDFC ELSS Tax Saver Direct Growth?  
5. Riskometer / risk level of HDFC Equity Fund (Flexi Cap) Direct Growth?  
6. Benchmark of HDFC Large Cap Fund Direct Growth?  
7. How to download a capital-gains statement?  
8. Should I buy HDFC Small Cap Fund? *(expect refusal)*  
9. Which fund has the best returns? *(expect refusal + factsheet pointer, no computed ranking)*  

---

## 13. Known limits (to document in README)

- Corpus is **five HDFC Direct Growth schemes**, not all mutual funds.  
- Facts can go stale; `Last updated from sources` is fetch/index time, not a live AMC feed.  
- Groww pages may differ from the SID; **SID/KIM/factsheet win** if they conflict.  
- MiniLM + small chunks can miss poorly worded queries; demo queries should name the scheme.  
- No live NAV, no personalised tax computation.

---

## 14. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Third-party seed pages vs “official sources only” | Index AMC/AMFI/SEBI documents; Groww URLs identify schemes |
| Model hallucinates a ratio | Strict grounding prompt; refuse if no chunk supports the fact |
| Mixing two schemes’ numbers | Chunk prefix with scheme name; metadata filter |
| User pastes PAN | Pattern refuse; no persistence |
| PDF factsheet table extraction quality | Prefer HTML scheme/fee pages; keep table blocks intact when chunking |

---

## 15. Demo success checklist

- [ ] Ingestion runs: pages → chunks → MiniLM → Chroma  
- [ ] Chat retrieves and answers with one link  
- [ ] Advice question is refused  
- [ ] UI disclaimer + 3 examples visible  
- [ ] Source list, README, sample Q&A, disclaimer file/snippet ready  

---

## 16. Open decisions (implementation, not product)

- Exact LLM (local vs API) for generation  
- Hosting (Streamlit / Gradio / FastAPI + simple HTML)  
- Whether Groww HTML is ingested at all vs used only as scheme identifiers  

Default for the demo: **official AMC/AMFI pages as corpus**; Groww links in the brief as **scheme identity only**, unless official HTML is blocked and a listed public page is the only option.
