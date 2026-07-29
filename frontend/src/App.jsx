import { useEffect, useRef, useState } from "react";
import "./App.css";

// Backend base URL. Empty = same-origin (local dev uses the Vite proxy).
// On GitHub Pages set VITE_API_BASE to your hosted backend URL at build time.
// No API keys ever live here — they stay server-side on the backend.
const API = import.meta.env.VITE_API_BASE || "";
const HISTORY_KEY = "rag-chat-history-v1";
const PROFILE_KEY = "rag-profile-v1";
const DEFAULT_AVATARS = Array.from(
  { length: 6 },
  (_, index) => `${import.meta.env.BASE_URL}avatars/avatar-${index + 1}.png`
);

const EXAMPLES = [
  "Summarize the key points",
  "What are the main terms?",
  "What isn't covered?",
];

function loadStored(key, fallback) {
  try {
    const saved = localStorage.getItem(key);
    return saved ? JSON.parse(saved) : fallback;
  } catch {
    return fallback;
  }
}

function resizeProfileImage(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error("Could not read that image."));
    reader.onload = () => {
      const image = new Image();
      image.onerror = () => reject(new Error("That image could not be opened."));
      image.onload = () => {
        const size = Math.min(image.width, image.height);
        const sourceX = (image.width - size) / 2;
        const sourceY = (image.height - size) / 2;
        const canvas = document.createElement("canvas");
        canvas.width = 256;
        canvas.height = 256;
        const context = canvas.getContext("2d");
        context.drawImage(
          image,
          sourceX,
          sourceY,
          size,
          size,
          0,
          0,
          canvas.width,
          canvas.height
        );
        resolve(canvas.toDataURL("image/jpeg", 0.86));
      };
      image.src = reader.result;
    };
    reader.readAsDataURL(file);
  });
}

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

function Message({ msg, profile }) {
  if (msg.role === "user") {
    return (
      <div className="row user">
        <div className="bubble user">{msg.content}</div>
        <img
          className="message-user-avatar"
          src={profile.avatar}
          alt=""
          aria-hidden="true"
        />
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
  const [messages, setMessages] = useState(() =>
    loadStored(HISTORY_KEY, [])
  );
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [status, setStatus] = useState(null);
  const [health, setHealth] = useState(null);
  const [healthChecking, setHealthChecking] = useState(true);
  const [profile, setProfile] = useState(() =>
    loadStored(PROFILE_KEY, {
      name: "Guest",
      avatar: DEFAULT_AVATARS[0],
    })
  );
  const [profileMenuOpen, setProfileMenuOpen] = useState(false);
  const [profileModalOpen, setProfileModalOpen] = useState(false);
  const [draftName, setDraftName] = useState(profile.name);
  const [draftAvatar, setDraftAvatar] = useState(profile.avatar);
  const [profileError, setProfileError] = useState("");
  const [theme, setTheme] = useState(() => {
    const saved = localStorage.getItem("theme");
    if (saved === "light" || saved === "dark") return saved;
    return window.matchMedia("(prefers-color-scheme: dark)").matches
      ? "dark"
      : "light";
  });
  const scrollRef = useRef(null);
  const inputRef = useRef(null);
  const profileMenuRef = useRef(null);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("theme", theme);
  }, [theme]);

  useEffect(() => {
    try {
      localStorage.setItem(HISTORY_KEY, JSON.stringify(messages.slice(-100)));
    } catch {
      /* Storage can be unavailable in private browsing. */
    }
  }, [messages]);

  useEffect(() => {
    try {
      localStorage.setItem(PROFILE_KEY, JSON.stringify(profile));
    } catch {
      /* Custom images can exceed a browser's storage allowance. */
    }
  }, [profile]);

  useEffect(() => {
    const closeProfileMenu = (event) => {
      if (!profileMenuRef.current?.contains(event.target)) {
        setProfileMenuOpen(false);
      }
    };
    document.addEventListener("pointerdown", closeProfileMenu);
    return () => document.removeEventListener("pointerdown", closeProfileMenu);
  }, []);

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

  const openProfileEditor = () => {
    setDraftName(profile.name);
    setDraftAvatar(profile.avatar);
    setProfileError("");
    setProfileMenuOpen(false);
    setProfileModalOpen(true);
  };

  const handleProfileImage = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      setProfileError("Choose a PNG, JPG, WebP, or GIF image.");
      event.target.value = "";
      return;
    }
    if (file.size > 8 * 1024 * 1024) {
      setProfileError("Choose an image smaller than 8 MB.");
      event.target.value = "";
      return;
    }
    try {
      setDraftAvatar(await resizeProfileImage(file));
      setProfileError("");
    } catch (error) {
      setProfileError(error.message);
    } finally {
      event.target.value = "";
    }
  };

  const saveProfile = () => {
    setProfile({
      name: draftName.trim().slice(0, 32) || "Guest",
      avatar: draftAvatar || DEFAULT_AVATARS[0],
    });
    setProfileModalOpen(false);
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
            <div className="profile-wrap" ref={profileMenuRef}>
              <button
                type="button"
                className="profile-trigger"
                onClick={() => setProfileMenuOpen((open) => !open)}
                aria-label={`Open ${profile.name}'s profile menu`}
                aria-expanded={profileMenuOpen}
              >
                <img src={profile.avatar} alt="" />
                <span className="profile-name">{profile.name}</span>
                <span className="profile-chevron" aria-hidden="true">
                  ⌄
                </span>
              </button>
              {profileMenuOpen && (
                <div className="profile-menu">
                  <div className="profile-menu-head">
                    <img src={profile.avatar} alt="" />
                    <div>
                      <strong>{profile.name}</strong>
                      <span>Saved on this device</span>
                    </div>
                  </div>
                  <button type="button" onClick={openProfileEditor}>
                    <span aria-hidden="true">✎</span> Manage profile
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setMessages([]);
                      setProfileMenuOpen(false);
                    }}
                    disabled={messages.length === 0}
                  >
                    <span aria-hidden="true">↺</span> Clear chat history
                  </button>
                </div>
              )}
            </div>
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
            <Message key={i} msg={m} profile={profile} />
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

      {profileModalOpen && (
        <div
          className="profile-overlay"
          role="presentation"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) setProfileModalOpen(false);
          }}
        >
          <section
            className="profile-dialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="profile-dialog-title"
          >
            <div className="profile-dialog-head">
              <div>
                <span className="profile-kicker">Your space</span>
                <h2 id="profile-dialog-title">Choose your profile</h2>
                <p>Your name, avatar, and chat history stay on this device.</p>
              </div>
              <button
                type="button"
                className="dialog-close"
                onClick={() => setProfileModalOpen(false)}
                aria-label="Close profile editor"
              >
                ×
              </button>
            </div>

            <label className="profile-field">
              <span>Display name</span>
              <input
                type="text"
                value={draftName}
                maxLength={32}
                onChange={(event) => setDraftName(event.target.value)}
                placeholder="Guest"
              />
            </label>

            <div className="profile-field">
              <span>Pick an avatar</span>
              <div className="avatar-grid">
                {DEFAULT_AVATARS.map((avatar, index) => (
                  <button
                    key={avatar}
                    type="button"
                    className={`avatar-choice ${
                      draftAvatar === avatar ? "selected" : ""
                    }`}
                    onClick={() => {
                      setDraftAvatar(avatar);
                      setProfileError("");
                    }}
                    aria-label={`Choose avatar ${index + 1}`}
                    aria-pressed={draftAvatar === avatar}
                  >
                    <img src={avatar} alt={`Anime avatar ${index + 1}`} />
                    {draftAvatar === avatar && (
                      <span className="avatar-check" aria-hidden="true">
                        ✓
                      </span>
                    )}
                  </button>
                ))}
              </div>
            </div>

            <label className="custom-avatar">
              <input
                type="file"
                accept="image/png,image/jpeg,image/webp,image/gif"
                onChange={handleProfileImage}
                hidden
              />
              <span className="custom-avatar-icon" aria-hidden="true">
                ↑
              </span>
              <span>
                <strong>Upload your own</strong>
                <small>Square crop · up to 8 MB</small>
              </span>
              {draftAvatar?.startsWith("data:image") && (
                <img src={draftAvatar} alt="Your custom avatar preview" />
              )}
            </label>
            {profileError && <p className="profile-error">{profileError}</p>}

            <div className="profile-actions">
              <button
                type="button"
                className="profile-cancel"
                onClick={() => setProfileModalOpen(false)}
              >
                Cancel
              </button>
              <button type="button" className="profile-save" onClick={saveProfile}>
                Save profile
              </button>
            </div>
          </section>
        </div>
      )}
    </div>
  );
}
