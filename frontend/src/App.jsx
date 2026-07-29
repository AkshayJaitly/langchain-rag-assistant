import { useEffect, useRef, useState } from "react";
import "./App.css";

// Backend base URL. Empty = same-origin (local dev uses the Vite proxy).
// On GitHub Pages set VITE_API_BASE to your hosted backend URL at build time.
// No API keys ever live here — they stay server-side on the backend.
const API = import.meta.env.VITE_API_BASE || "";

const EXAMPLES = [
  "Summarize the key points",
  "What are the main terms?",
  "What isn't covered?",
];

async function responseData(res) {
  const text = await res.text();
  if (!text) return {};
  try {
    return JSON.parse(text);
  } catch {
    if (!res.ok) throw new Error(text);
    throw new Error("The server returned an invalid response.");
  }
}

function guardrailKind(g) {
  if (g.startsWith("input")) return "danger";
  if (g.startsWith("output")) return "warn";
  if (g.startsWith("grounding")) return "warn";
  return "muted";
}

function Guardrails({ guardrails }) {
  if (!guardrails || guardrails.length === 0) {
    return (
      <span className="chip ok">
        <span className="chip-dot" /> guardrails passed
      </span>
    );
  }
  return (
    <div className="chips">
      {guardrails.map((g) => (
        <span key={g} className={`chip ${guardrailKind(g)}`}>
          {g.replace(/_/g, " ")}
        </span>
      ))}
    </div>
  );
}

/** Render [1], [2] citation markers as pills inside answer text. */
function AnswerText({ text }) {
  const parts = String(text).split(/(\[\d+\])/g);
  return (
    <div className="answer">
      {parts.map((p, i) =>
        /^\[\d+\]$/.test(p) ? (
          <span key={i} className="cite-pill">
            {p.slice(1, -1)}
          </span>
        ) : (
          <span key={i}>{p}</span>
        )
      )}
    </div>
  );
}

function Message({ msg }) {
  if (msg.role === "user") {
    return (
      <div className="row user">
        <div className="bubble user">{msg.content}</div>
      </div>
    );
  }
  return (
    <div className="row assistant">
      <div className="avatar">✦</div>
      <div className="assistant-card">
        <AnswerText text={msg.content} />
        <div className="meta">
          {msg.error ? (
            <span className="chip danger">request failed</span>
          ) : (
            <Guardrails guardrails={msg.guardrails} />
          )}
        </div>
        {msg.sources && msg.sources.length > 0 && (
          <details className="sources">
            <summary>
              <span className="src-count">{msg.sources.length}</span> source
              {msg.sources.length > 1 ? "s" : ""}
            </summary>
            <div className="source-list">
              {msg.sources.map((s) => (
                <div key={s.index} className="source">
                  <div className="source-head">
                    <span className="cite-pill sm">{s.index}</span>
                    <span className="src-name">{s.source}</span>
                    {s.page != null && <span className="src-page">p.{s.page}</span>}
                  </div>
                  <div className="snippet">{s.snippet}</div>
                </div>
              ))}
            </div>
          </details>
        )}
      </div>
    </div>
  );
}

export default function App() {
  const [docs, setDocs] = useState([]);
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [status, setStatus] = useState(null);
  const [health, setHealth] = useState(null);
  const [healthChecking, setHealthChecking] = useState(true);
  const [theme, setTheme] = useState(() => {
    const saved = localStorage.getItem("theme");
    if (saved === "light" || saved === "dark") return saved;
    return window.matchMedia("(prefers-color-scheme: dark)").matches
      ? "dark"
      : "light";
  });
  const scrollRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("theme", theme);
  }, [theme]);

  const refreshDocs = async () => {
    try {
      const res = await fetch(`${API}/api/documents`);
      const data = await responseData(res);
      setDocs(data.documents || []);
    } catch {
      /* backend not up yet */
    }
  };

  const refreshHealth = async () => {
    setHealthChecking(true);
    try {
      const res = await fetch(`${API}/api/health`);
      if (!res.ok) {
        setHealth(null);
        return false;
      }
      setHealth(await responseData(res));
      return true;
    } catch {
      setHealth(null);
      return false;
    } finally {
      setHealthChecking(false);
    }
  };

  useEffect(() => {
    refreshDocs();
    let cancelled = false;
    let retryTimer;
    const retryDelays = [0, 3000, 10000, 30000];

    const checkBackend = async (attempt) => {
      const isUp = await refreshHealth();
      if (!cancelled && !isUp && attempt + 1 < retryDelays.length) {
        retryTimer = window.setTimeout(
          () => checkBackend(attempt + 1),
          retryDelays[attempt + 1]
        );
      }
    };

    checkBackend(0);
    return () => {
      cancelled = true;
      window.clearTimeout(retryTimer);
    };
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages, loading]);

  const handleUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setStatus(null);
    const form = new FormData();
    form.append("file", file);
    try {
      const res = await fetch(`${API}/api/upload`, { method: "POST", body: form });
      const data = await responseData(res);
      if (!res.ok) throw new Error(data.detail || "Upload failed");
      setStatus({
        type: "ok",
        text: `Indexed “${data.filename}” · ${data.documents_ingested} doc(s)`,
      });
      refreshDocs();
    } catch (err) {
      setStatus({ type: "error", text: err.message });
    } finally {
      setUploading(false);
      e.target.value = "";
    }
  };

  const ask = async (q) => {
    if (!q || loading) return;
    setMessages((m) => [...m, { role: "user", content: q }]);
    setQuestion("");
    setLoading(true);
    try {
      const res = await fetch(`${API}/api/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q }),
      });
      const data = await responseData(res);
      if (!res.ok) throw new Error(data.detail || "Query failed");
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          content: data.answer,
          sources: data.sources,
          guardrails: data.guardrails,
        },
      ]);
    } catch (err) {
      setMessages((m) => [
        ...m,
        { role: "assistant", content: `Error: ${err.message}`, error: true },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const onSubmit = (e) => {
    e.preventDefault();
    ask(question.trim());
  };

  const backendUp = health != null;

  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">✦</div>
          <div>
            <div className="brand-name">RAG Assistant</div>
            <div className="brand-sub">grounded document Q&amp;A</div>
          </div>
        </div>

        <div className="stack-tags">
          {["LangChain", "LangGraph", "Chroma", "parent-child", "guardrails"].map(
            (t) => (
              <span key={t} className="tag">
                {t}
              </span>
            )
          )}
        </div>

        <label className={`upload ${uploading ? "busy" : ""}`}>
          <input
            type="file"
            accept=".pdf,.docx,.doc,.txt,.md"
            onChange={handleUpload}
            disabled={uploading}
            hidden
          />
          <span className="upload-badge">{uploading ? "◐" : "↑"}</span>
          <span className="upload-text">
            <span className="upload-title">
              {uploading ? "Embedding…" : "Upload document"}
            </span>
            <span className="upload-hint">PDF · Word · txt · md</span>
          </span>
        </label>

        {status && <div className={`toast ${status.type}`}>{status.text}</div>}

        <div className="section-label">
          Indexed <span className="count-pill">{docs.length}</span>
        </div>
        {docs.length === 0 ? (
          <p className="empty-note">No documents yet.</p>
        ) : (
          <ul className="doclist">
            {docs.map((d, i) => (
              <li key={`${d.filename}-${i}`}>
                <span className="doc-icon">▤</span>
                <span className="doc-name">{d.filename}</span>
                <span className="doc-count">{d.chunks}</span>
              </li>
            ))}
          </ul>
        )}
      </aside>

      <main className="chat">
        <header className="topbar">
          <div className="topbar-title">Chat</div>
          <div className="topbar-right">
            {messages.length > 0 && (
              <button className="ghost-btn" onClick={() => setMessages([])}>
                Clear
              </button>
            )}
            <button
              type="button"
              className={`status-pill ${backendUp ? "up" : "down"}`}
              onClick={refreshHealth}
              disabled={healthChecking}
              title={backendUp ? "Check backend again" : "Retry backend connection"}
            >
              <span className="status-dot" />
              {healthChecking
                ? "connecting…"
                : backendUp
                ? `${health.llm_provider} · ${health.llm_model}`
                : "backend offline · retry"}
            </button>
            <button
              className="icon-btn"
              onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
              title={theme === "dark" ? "Switch to light" : "Switch to dark"}
              aria-label="Toggle theme"
            >
              {theme === "dark" ? "☀" : "☾"}
            </button>
          </div>
        </header>

        <div className="messages" ref={scrollRef}>
          {messages.length === 0 && (
            <div className="hero">
              <div className="hero-mark">✦</div>
              <h2>Ask your documents</h2>
              <p>
                Answers are retrieved with the parent-child algorithm and screened
                by input &amp; output guardrails — every reply cites its sources.
              </p>
              <div className="example-chips">
                {EXAMPLES.map((ex) => (
                  <button
                    key={ex}
                    className="example"
                    onClick={() => ask(ex)}
                    disabled={loading}
                  >
                    {ex}
                  </button>
                ))}
              </div>
            </div>
          )}
          {messages.map((m, i) => (
            <Message key={i} msg={m} />
          ))}
          {loading && (
            <div className="row assistant">
              <div className="avatar">✦</div>
              <div className="assistant-card thinking">
                <div className="typing">
                  <span />
                  <span />
                  <span />
                </div>
              </div>
            </div>
          )}
        </div>

        <form className="composer" onSubmit={onSubmit}>
          <input
            ref={inputRef}
            type="text"
            value={question}
            placeholder="Ask a question about your documents…"
            onChange={(e) => setQuestion(e.target.value)}
            disabled={loading}
          />
          <button type="submit" disabled={loading || !question.trim()}>
            {loading ? "…" : "Ask"}
          </button>
        </form>
      </main>
    </div>
  );
}
