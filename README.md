# RAG Vector-DB Assistant

A retrieval-augmented-generation app built with **LangChain**, **LangGraph**,
**ChromaDB**, and a **React** frontend. It ingests PDF / Word / text documents,
creates embeddings, and answers questions grounded in the retrieved context —
using the **parent-child (small-to-big) retrieval algorithm** and **input /
output guardrails**.

**Live app:** [akshayjaitly.github.io/langchain-rag-assistant](https://akshayjaitly.github.io/langchain-rag-assistant/)

**Backend health:** [langchain-rag-assistant-tw27.onrender.com/api/health](https://langchain-rag-assistant-tw27.onrender.com/api/health)

```
┌──────────────┐   /api/upload   ┌────────────────────────────────────────┐
│ React (Vite) │ ──────────────► │ FastAPI                                │
│ chat + docs  │   /api/query    │  ├─ ingest: PDF/Word/txt/md             │
│ local profile│                 │  │          → parent + child chunks     │
└──────────────┘ ◄────────────── │  │            chunks → local embeddings │
                                 │  │            → Chroma (children)         │
                                 │  │            → file docstore (parents)   │
                                 │  └─ LangGraph pipeline:                   │
                                 │       input guardrail → retrieve →        │
                                 │       generate → output guardrail         │
                                 └────────────────────────────────────────┘
```

## Key pieces

| Concern | Implementation |
| --- | --- |
| Frontend | React 18 + Vite 6, deployed to GitHub Pages |
| Backend | FastAPI, deployed to Render |
| Vector DB | ChromaDB with a file-backed parent document store |
| Hosted embeddings | FastEmbed `BAAI/bge-small-en-v1.5` (local to the backend, no embedding API) |
| Local embeddings | Hugging Face `sentence-transformers/all-MiniLM-L6-v2` by default |
| Parent-child retrieval | LangChain `ParentDocumentRetriever` — embed small chunks, return larger parent chunks |
| Orchestration | LangGraph `StateGraph` with conditional guardrail edges |
| Hosted generation | Groq `llama-3.3-70b-versatile` |
| Other providers | Anthropic, OpenAI, and local Ollama are configurable |
| Observability | LangSmith traces in project `pr-puzzled-robot-90` |
| Guardrails | Input injection/size checks, no-context refusal, secret redaction, and grounding checks |
| Parsing | `pypdf` (PDF), `docx2txt` (Word), `TextLoader` (txt/md) |
| UI persistence | Display name, avatar, theme, and the latest 100 messages in browser `localStorage` |

### The parent-child algorithm

Documents are split into **large parent chunks** and **small child chunks**.
Only the children are embedded (small chunks retrieve more precisely). At query
time we search the children, then hand the LLM their **parent** chunks so it has
the surrounding context. Parents live in a file-backed docstore; children live
in Chroma.

### The LangGraph pipeline

The hosted deployment currently uses `PIPELINE=simple`:

```
START → input_guardrail ─(blocked)────────────────► END
              │
           (ok) → retrieve ─(no docs)─► no_context → END
                       │
                    (docs) → generate → output_guardrail → END
```

An optional corrective pipeline is available with `PIPELINE=multi_agent`:

```text
START → input_guardrail → retrieve → grade_documents → generate → verify
                                      │                         │
                                      └─ no relevant docs       └─ revise once
                                             ↓                         ↓
                                         no_context             output_guardrail
```

The multi-agent pipeline adds relevance grading and answer verification, but it
also adds model calls and latency. Keep the simple pipeline as the production
default until both versions have been compared with a LangSmith evaluation
dataset.

## Prerequisites

- Python 3.11 or 3.12
- Node.js 20+
- One generation provider: Anthropic, OpenAI, Groq, or local Ollama

## Backend setup

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # choose LLM_PROVIDER and set its API key
uvicorn app.main:app --reload --port 8000
```

The first run downloads the selected embedding model and caches it locally.
Interactive API documentation is available at `http://localhost:8000/docs`.

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
4. Use the profile control in the top-right corner to select an included avatar
   or upload a custom image.

Profile settings and chat history currently persist only in the browser. They
are not authenticated, shared between devices, or supplied to LangGraph as
conversation memory.

## API

| Method | Path | Body | Purpose |
| --- | --- | --- | --- |
| GET | `/api/health` | – | Provider, model, embeddings, pipeline, and tracing status |
| GET | `/api/documents` | – | List documents in the current backend index |
| POST | `/api/upload` | multipart `file` | Parse, embed, and index a document |
| POST | `/api/query` | `{"question": "..."}` | Answer, sources, guardrails, blocked status, and LangSmith `trace_id` |

## Configuration

All settings are environment variables (see `backend/.env.example`): model,
chunk sizes, retrieval `k`, persistence directories, CORS origins.

## LangSmith tracing

The backend emits one LangSmith trace per `/api/query` request, with child runs
for the LangGraph nodes, retriever, and model calls. Traces are tagged with the
environment, pipeline, and model provider and include retrieval configuration
as metadata.

1. Create a LangSmith API key at
   [smith.langchain.com](https://smith.langchain.com).
2. Set `LANGSMITH_API_KEY` as a secret in the Render service.
3. Keep `LANGSMITH_TRACING=true`. The Render Blueprint already sets the
   production project to `pr-puzzled-robot-90` and enables tracing.
4. If the key can access multiple LangSmith workspaces, also set
   `LANGSMITH_WORKSPACE_ID`.

`LANGSMITH_HIDE_INPUTS` and `LANGSMITH_HIDE_OUTPUTS` default to `true`, so
uploaded document content, questions, and answers are not sent in trace
payloads. For a non-sensitive demo, set them to `false` to inspect prompts,
retrieved context, and generated answers in LangSmith.

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
   (it reads [`render.yaml`](render.yaml)). Set `GROQ_API_KEY` and
   `LANGSMITH_API_KEY` in the dashboard as secret environment variables. The
   current backend is
   `https://langchain-rag-assistant-tw27.onrender.com`.
2. **Frontend → point it at the backend:** add an Actions repository variable
   named `VITE_API_BASE` with the full Render URL (repo *Settings → Secrets and
   variables → Actions → Variables*).
3. **GitHub Pages:** under *Settings → Pages → Build and deployment*, choose
   **GitHub Actions** as the source. Pushes that change `frontend/` or the Pages
   workflow then rebuild and publish the site automatically.
4. **CORS:** already set to `https://akshayjaitly.github.io` in `render.yaml`;
   change it if your Pages origin differs.

> Free-tier note: the Render service can spin down while idle and does not have
> a persistent disk. Uploaded files, Chroma vectors, the parent docstore, and
> the document manifest can therefore disappear after a restart or redeploy.
> Production persistence requires a persistent disk or a managed vector store.

Embeddings are always local (free); only generation differs. To go fully
offline at $0:

```bash
# .env
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.1:8b
```

## Working Site Preview(local)
<img width="1144" height="622" alt="Screenshot 2026-07-29 at 1 18 37 PM" src="https://github.com/user-attachments/assets/bb4149fc-cd67-40a4-9b64-d4f944fbdc3f" />

## Langsmith dash

<img width="977" height="592" alt="Screenshot 2026-07-29 at 5 19 29 PM" src="https://github.com/user-attachments/assets/0fc2ac5e-1db2-4bb3-9f17-7c61339d04c8" />


## Notes

- Embeddings run locally, so re-indexing and retrieval cost nothing.
- Guardrails here are intentionally lightweight/heuristic — for production,
  consider a dedicated moderation model and a stricter faithfulness grader.
- The current browser history is presentation persistence, not LangGraph
  conversational memory. Server-side threads require a LangGraph checkpointer
  and authenticated storage.
