# RAG Vector-DB Assistant

A retrieval-augmented-generation app built with **LangChain**, **LangGraph**,
**ChromaDB**, and a **React** frontend. It ingests PDF / Word / text documents,
creates embeddings, and answers questions grounded in the retrieved context —
using the **parent-child (small-to-big) retrieval algorithm** and **input /
output guardrails**.

```
┌──────────────┐   /api/upload   ┌────────────────────────────────────────┐
│ React (Vite) │ ──────────────► │ FastAPI                                │
│  chat + docs │   /api/query    │  ├─ ingest: PDF/Word/txt → parent+child │
└──────────────┘ ◄────────────── │  │            chunks → local embeddings │
                                 │  │            → Chroma (children)         │
                                 │  │            → file docstore (parents)   │
                                 │  └─ LangGraph pipeline:                   │
                                 │       input guardrail → retrieve →        │
                                 │       generate (Claude) → output guardrail│
                                 └────────────────────────────────────────┘
```

## Key pieces

| Concern            | Implementation                                                        |
| ------------------ | -------------------------------------------------------------------- |
| Vector DB          | ChromaDB (persistent, cosine)                                        |
| Embeddings         | `sentence-transformers/all-MiniLM-L6-v2` (local, offline, no cost)   |
| Parent-child       | LangChain `ParentDocumentRetriever` — embed small chunks, return big |
| Orchestration      | LangGraph `StateGraph` with conditional guardrail edges             |
| Generation         | Claude (`claude-sonnet-5`) via `langchain-anthropic`                 |
| Guardrails         | input (injection / size), grounding (no-context refusal), output (secret redaction + grounding check) |
| Doc parsing        | `pypdf` (PDF), `docx2txt` (Word), `TextLoader` (txt/md)             |

### The parent-child algorithm

Documents are split into **large parent chunks** and **small child chunks**.
Only the children are embedded (small chunks retrieve more precisely). At query
time we search the children, then hand the LLM their **parent** chunks so it has
the surrounding context. Parents live in a file-backed docstore; children live
in Chroma.

### The LangGraph pipeline

```
START → input_guardrail ─(blocked)────────────────► END
              │
           (ok) → retrieve ─(no docs)─► no_context → END
                       │
                    (docs) → generate → output_guardrail → END
```

## Prerequisites

- Python 3.11 or 3.12, and Node 18+
- An Anthropic API key

## Backend setup

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then edit .env and set ANTHROPIC_API_KEY
uvicorn app.main:app --reload --port 8000
```

First run downloads the embedding model (~90 MB) once. API docs at
`http://localhost:8000/docs`.

## Frontend setup

```bash
cd frontend
npm ci
npm run dev                    # http://localhost:5173 (proxies /api → :8000)
```

## Using it

1. Open `http://localhost:5173`.
2. Upload a PDF / Word / text file (left panel) — it is parsed, chunked,
   embedded, and indexed.
3. Ask a question. The answer is grounded in retrieved chunks, shows its
   sources, and displays which guardrails fired.

## API

| Method | Path             | Body                     | Purpose                        |
| ------ | ---------------- | ------------------------ | ------------------------------ |
| GET    | `/api/health`    | –                        | Service + model info           |
| GET    | `/api/documents` | –                        | List ingested documents        |
| POST   | `/api/upload`    | multipart `file`         | Ingest a document              |
| POST   | `/api/query`     | `{"question": "..."}`    | Answer + sources + guardrails  |

## Configuration

All settings are environment variables (see `backend/.env.example`): model,
chunk sizes, retrieval `k`, persistence directories, CORS origins.

## Choosing the generation model

Set `LLM_PROVIDER` in `backend/.env`:

| `LLM_PROVIDER` | Cost      | Setup                                                            |
| -------------- | --------- | --------------------------------------------------------------- |
| `anthropic`    | paid API  | Set `ANTHROPIC_API_KEY`; pick `LLM_MODEL` (`claude-haiku-4-5` = cheapest, `claude-sonnet-5` = balanced, `claude-opus-5` = best). |
| `openai`       | paid API  | Set `OPENAI_API_KEY`; pick `OPENAI_MODEL` (default `gpt-4o-mini`). |
| `groq`         | **free**  | Free key at [console.groq.com](https://console.groq.com); set `GROQ_API_KEY`. Fast hosted Llama — ideal for a $0 always-on deploy. |
| `ollama`       | **free**  | Install [Ollama](https://ollama.com), run `ollama pull llama3.1:8b`, set `OLLAMA_MODEL`. No API key; local only (won't fit free cloud tiers). |

## Deploying (free)

GitHub Pages hosts the **frontend** (static, no secrets). The **backend** runs
on any Python host; secrets live there as server-side env vars, never in the
bundle.

1. **Backend → Render (free):** In Render, *New → Blueprint*, connect this repo
   (it reads [`render.yaml`](render.yaml)). Set `GROQ_API_KEY` in the dashboard
   (hidden). Render gives you a URL like `https://rag-backend.onrender.com`.
2. **Frontend → point it at the backend:** add an Actions repository variable
   named `VITE_API_BASE` with the full Render URL (repo *Settings → Secrets and
   variables → Actions → Variables*).
3. **GitHub Pages:** under *Settings → Pages → Build and deployment*, choose
   **GitHub Actions** as the source. Pushes that change `frontend/` or the Pages
   workflow then rebuild and publish the site automatically.
4. **CORS:** already set to `https://akshayjaitly.github.io` in `render.yaml`;
   change it if your Pages origin differs.

> Free-tier notes: Render's free web service sleeps after inactivity (~50 s cold
> start) and has no persistent disk, so the Chroma index resets on restart —
> fine for a demo. For always-on + persistence, use a paid disk or a host with
> more RAM (e.g. Hugging Face Spaces gives 16 GB free).

Embeddings are always local (free); only generation differs. To go fully
offline at $0:

```bash
# .env
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.1:8b
```

## Working Site Preview(local)
<img width="1144" height="622" alt="Screenshot 2026-07-29 at 1 18 37 PM" src="https://github.com/user-attachments/assets/bb4149fc-cd67-40a4-9b64-d4f944fbdc3f" />

## Notes

- Embeddings run locally, so re-indexing and retrieval cost nothing.
- Guardrails here are intentionally lightweight/heuristic — for production,
  consider a dedicated moderation model and a stricter faithfulness grader.
```
