Mutual Fund FAQs (Facts-Only Q&A)
Milestone brief
Build a small FAQ assistant that answers facts about mutual fund schemes—e.g., expense ratio, exit load, minimum SIP, lock-in (ELSS), riskometer, benchmark, and how to download statements—using only official public pages. Every answer must include one source link. No advice.
Who this helps
Retail users comparing schemes; support/content teams answering repetitive MF questions.
What you must build
Scope your corpus: Pick one AMC and 3–5 schemes under it (e.g., one large-cap, one flexi-cap, one ELSS).
Collect the public pages as mentioned below public pages from AMC/SEBI/AMFI (factsheets, KIM/SID, scheme FAQs, fee/charges pages, riskometer/benchmark notes, statement/tax-doc guides).
Large Cap: https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth
Flexi Cap: https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth
ELSS: https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth
Small Cap: https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth
Balanced Advantage (Hybrid): https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth
FAQ assistant (working prototype):
Answers factual queries only (e.g., “Expense ratio of ?”, “ELSS lock-in?”, “Minimum SIP?”, “Exit load?”, “Riskometer/benchmark?”, “How to download capital-gains statement?”).
Shows one clear citation link in every answer.
Refuses opinionated/portfolio questions (e.g., “Should I buy/sell?”) with a polite, facts-only message and a relevant educational link.
Tiny UI: welcome line + 3 example questions and a note: “Facts-only. No investment advice.”
Key constraints
**Public sources only.** No screenshots of the app back-end; no third-party blogs as sources.
**No PII.** Do not accept/store PAN, Aadhaar, account numbers, OTPs, emails, or phone numbers.
**No performance claims.** Don’t compute/compare returns; link to the official factsheet if asked.
**Clarity & transparency.** Keep answers ≤3 sentences; add “Last updated from sources: ”.
What to submit (deliverables)
Working prototype link (app/notebook) or a ≤3-min demo video if hosting isn’t possible.
Source list (CSV/MD) of the 5 URLs you used.
README with setup steps, scope (AMC + schemes), and known limits.
Sample Q&A file (5–10 queries with the assistant’s answers + links).
Disclaimer snippet used in your UI (facts-only, no advice).
End output - RAG Chatbot 
Each stage is different and when we create architecture we wwant to follow all the stages on RAG ( Data ingestion + Data retrieval) 
Loading → Chunking → Embedding → store Vector Data
Embedding Models – sentence-transformers/all-MiniLM-L6-v2
Chunking Strategy → Ask cursor to decide the chunking strategy based on the data 
VectorDB - ChromaDB 
We are building a RAG ChatBot
