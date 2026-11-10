# Just Allah backend

## What it is

Just Allah is a source-finder over the Quran and the hadith collections Bukhari, Muslim, and Abu Dawud. The LLM summarizes retrieved passages but does not write source text; `/ask` citations are emitted only when their references resolve to a database record. This is a study tool, not a fatwa tool.

## Features

- `POST /ask` streams an answer and cited passage cards as server-sent events.
- `POST /verify` in hadith mode checks similarity against a threshold and applies a lexical check; it returns stored grades and reports unmatched text as “This hadith cannot be verified in the sources.”
- `POST /verify` in claim mode uses a per-passage support judge, then returns a summary and its source cards.
- Both endpoints have in-memory, per-IP rate limits. Defaults are 5 requests per minute and 50 per day for `/ask`, and 10 per minute and 100 per day for `/verify`.

## Architecture

**`/ask`**

1. An LLM rewrites the question into search phrases; if rewriting fails, retrieval uses the original question.
2. Each phrase is searched, results are merged by reference, and the top five passages per kind are selected.
3. The answer LLM streams text. Each citation reference is looked up in the database before the citation and card are sent.

**`/verify`**

1. Hadith mode retrieves hadith candidates, then requires similarity of at least `0.85` and lexical evidence for a match.
2. Claim mode retrieves Quran and hadith passages, filters them with a passage-support judge, and summarizes the supported passages.

**Data layer:** Supabase Postgres stores the passages and 384-dimensional vectors. `intfloat/multilingual-e5-small` creates embeddings. The Alembic migration installs `pgvector` and defines the hybrid search function, which combines vector and English full-text ranks using reciprocal-rank fusion.

## Demo

This project is not hosted. The embedding model runs inside the backend and needs more memory than free hosting tiers provide, so a live demo is not available. The recording below shows the app running locally.

<img width="1471" height="955" alt="Screenshot 2026-11-10 110800" src="https://github.com/user-attachments/assets/30c61531-231a-4ba0-8d42-dbf121213f65" />
<img width="1917" height="800" alt="Screenshot 2026-11-10 110847" src="https://github.com/user-attachments/assets/c7e6d64a-70bc-4ce4-a755-c42f9e115eae" />
<img width="1917" height="1026" alt="Screenshot 2026-11-10 110928" src="https://github.com/user-attachments/assets/a9d17329-1e73-4d25-94fb-7c17af6312ae" />
<img width="1917" height="1018" alt="Screenshot 2026-11-10 110936" src="https://github.com/user-attachments/assets/1fb67ad4-c848-4fa6-84f5-16b47dc222c5" />
<img width="1915" height="1018" alt="Screenshot 2026-11-10 111219" src="https://github.com/user-attachments/assets/db14f422-19f2-4393-81cf-5e468d19097f" />
<img width="1097" height="773" alt="Screenshot 2026-11-10 111229" src="https://github.com/user-attachments/assets/9e550320-f381-4128-896a-6a3d6478c951" />
<img width="1117" height="665" alt="Screenshot 2026-11-10 111708" src="https://github.com/user-attachments/assets/09616b7f-369c-41b1-a84a-325339ea326a" />

To run it yourself, see Setup.

## Setup

Prerequisites: Python 3.13 or later, `uv`, a Postgres database with `pgvector`, the local source files described below, and provider credentials for LLM answers. From this backend directory:

```sh
uv sync
```

The current `.env.example` lists the first three names below. The application also reads the remaining names; values are intentionally omitted here. `REWRITE_MODEL` is optional: when absent, rewriting falls back to the original question.

```dotenv
ALLOWED_ORIGINS=
RATE_LIMIT_ASK=
RATE_LIMIT_VERIFY=
DATABASE_URL=
OPENROUTER_API_KEY=
OPENROUTER_MODEL=
OPENROUTER_BASE_URL=
REWRITE_MODEL=
```

Apply migrations, then ingest Quran and hadith data in this order:

```sh
uv run alembic upgrade head
uv run python scripts/ingest_quran.py
uv run python scripts/ingest_hadith.py bukhari
uv run python scripts/ingest_hadith.py muslim
uv run python scripts/ingest_hadith.py abudawud
```

The Quran ingestion script reads `app/quran.db/quran.db`. Hadith ingestion reads the matching Arabic and English JSON files in `data/hadith/`. Start the API from this directory with:

```sh
uv run python -m uvicorn app.main:app --reload
```

## Evaluation

From this directory, run selected cases with `uv run python eval/run.py --ids 1,2,3`. `uv run python eval/run.py --skip-llm` selects only `verify_hadith` cases. The cases cover hadith matching and non-matching, expected references and grades, claim support, and cited `/ask` answers; known failures are marked in `eval/questions.json`.

The latest results recorded in `PLAN.md` are partial: hadith eval cases 1–8 had 0 failures and recall@5 of 11/11; cases 22, 24–26 passed while case 23 remains a known failure; cases 28–30 and 31–33 were recorded as passing. The handoff records 15 pytest tests passed. These are recorded results, not a claim that the full eval was rerun.

## Known limitations

- Loose paraphrases can miss the intended passage (cases 9, 12, and 23).
- Hadith verification is English-only. Arabic questions in `/ask` are unsupported (case 34).
- Sahaba reports are not included. Bukhari and Muslim have no grades in the source data.
- English translation may differ from Arabic; the plan records a discrepancy for Abu Dawud 407.
- Provider free-tier rate limits can make requests slow or unavailable. Disputed topics may appear one-sided when a relevant passage is missing.
- The evaluation set covers known failure modes; it does not prove correctness for every question.

## Data sources and attribution

- **Quran:** the `gaitco` `quran.db` repository. The ingestion selects the `en.sahih` edition (Sahih International).
- **Hadith:** `fawazahmed0/hadith-api`, using separate Arabic and English editions for Bukhari, Muslim, and Abu Dawud; grades are stored as supplied by the source.

The source files are not included in the repository. `data/` and local database files are gitignored; the ingestion scripts load the local source files and rebuild the passages in Postgres. Reuse terms for the Arabic and English hadith editions and for the Quran translation have not been confirmed and must be confirmed before public deployment. No license is asserted here.
