import { useEffect, useRef, useState } from "react";
import { api, ApiError } from "../api/client";
import type { ParsedQuery, QueryType } from "../api/types";

interface Props {
  query: string;
  setQuery: (q: string) => void;
  hints: string[];
  onAppendHint: () => void;
  onClearHints: () => void;
  onSearch: () => void;
  loading: boolean;
  parsed: ParsedQuery | null;
  queryType: QueryType;
  inputRef: React.RefObject<HTMLTextAreaElement>;
  useLLM: boolean;
  onToggleLLM: () => void;
  expand: boolean;
  onToggleExpand: () => void;
  /** Keyframes to retrieve. Applied on the NEXT search, never on change. */
  topK: number;
  onTopK: (value: number) => void;
  /** What the results on screen were actually fetched with, or null before the
   *  first search — used to tell the operator the slider is not live yet. */
  appliedTopK: number | null;
  /** Folder-scope picker. Passed in rather than built here so the panel keeps
   *  owning only the query text and the knobs that shape one search. */
  scopeFilter?: React.ReactNode;
}

export function QueryPanel(props: Props) {
  const { query, setQuery, hints, onAppendHint, onClearHints, onSearch, loading, parsed, queryType, inputRef, useLLM, onToggleLLM, expand, onToggleExpand, topK, onTopK, appliedTopK, scopeFilter } = props;
  const [listening, setListening] = useState(false);
  const [voiceMsg, setVoiceMsg] = useState<string | null>(null);
  const [interim, setInterim] = useState("");

  const recRef = useRef<any>(null);
  const mediaRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const baseRef = useRef("");      // query text at the moment voice started
  const finalRef = useRef("");     // accumulated final transcript (Web Speech)
  const toggleRef = useRef<() => void>(() => {});

  const SpeechRecognition = (typeof window !== "undefined" &&
    ((window as any).SpeechRecognition || (window as any).webkitSpeechRecognition)) || null;
  // Brave ships the API object but disabled Google's backend → use Whisper there.
  const isBrave = typeof navigator !== "undefined" && !!(navigator as any).brave;
  const useSpeech = !!SpeechRecognition && !isBrave;

  // Auto-grow the textarea.
  useEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
    el.style.overflowY = el.scrollHeight > 200 ? "auto" : "hidden";
  }, [query, inputRef]);

  // cleanup on unmount
  useEffect(() => () => { recRef.current?.stop?.(); streamRef.current?.getTracks().forEach((t) => t.stop()); }, []);

  // Ctrl/Cmd+M toggles voice from anywhere.
  useEffect(() => { toggleRef.current = toggleVoice; });
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if ((e.ctrlKey || e.metaKey) && (e.key === "m" || e.key === "M")) {
        e.preventDefault();
        toggleRef.current();
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  async function finishViToEn(vi: string) {
    const base = baseRef.current;
    if (!vi.trim()) { setVoiceMsg(null); return; }
    setVoiceMsg("Đang dịch VI→EN…");
    try {
      const { text_en } = await api.translate(vi);
      setQuery((base ? base + " " : "") + (text_en || vi));
    } catch {
      setQuery((base ? base + " " : "") + vi);
    }
    setVoiceMsg(null);
  }

  function startSpeech() {
    baseRef.current = query;
    finalRef.current = "";
    setInterim("");
    setVoiceMsg(null);
    const rec = new SpeechRecognition();
    rec.lang = "vi-VN";
    rec.continuous = true;
    rec.interimResults = true;
    rec.onresult = (e: any) => {
      let fin = "", intr = "";
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const r = e.results[i];
        if (r.isFinal) fin += r[0].transcript;
        else intr += r[0].transcript;
      }
      if (fin) finalRef.current = (finalRef.current + " " + fin).trim();
      setInterim(intr);
      const live = (finalRef.current + " " + intr).trim();
      setQuery((baseRef.current ? baseRef.current + " " : "") + live);
    };
    rec.onerror = (e: any) => {
      const err = e?.error;
      if (err === "not-allowed" || err === "service-not-allowed") {
        setVoiceMsg("Cần cấp quyền micro cho trang này.");
      } else if (err === "network") {
        // Web Speech backend unreachable → fall back to server Whisper.
        try { rec.stop(); } catch { /* ignore */ }
        startWhisper();
      }
    };
    rec.onend = () => {
      setListening(false);
      setInterim("");
      const vi = finalRef.current.trim();
      if (vi) void finishViToEn(vi);
    };
    recRef.current = rec;
    try { rec.start(); setListening(true); } catch { setVoiceMsg("Không khởi động được micro."); }
  }

  async function startWhisper() {
    recRef.current = null;
    baseRef.current = query;
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch {
      setVoiceMsg("Cần cấp quyền micro cho trang này.");
      setListening(false);
      return;
    }
    streamRef.current = stream;
    chunksRef.current = [];
    const mr = new MediaRecorder(stream);
    mediaRef.current = mr;
    mr.ondataavailable = (e) => { if (e.data.size) chunksRef.current.push(e.data); };
    mr.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      setListening(false);
      const blob = new Blob(chunksRef.current, { type: mr.mimeType || "audio/webm" });
      if (!blob.size) { setVoiceMsg(null); return; }
      setVoiceMsg("Đang nhận dạng (Whisper)…");
      try {
        const { text, text_en } = await api.transcribe(blob); // already VI→EN translated
        const out = (text_en || text || "").trim();
        if (out) setQuery((baseRef.current ? baseRef.current + " " : "") + out);
        setVoiceMsg(out ? null : "Không nhận được giọng nói.");
      } catch (e) {
        setVoiceMsg(e instanceof ApiError && typeof e.detail === "string" ? e.detail : "Whisper không khả dụng (backend chưa bật?).");
      }
    };
    mr.start();
    mediaRef.current = mr;
    setListening(true);
    setVoiceMsg("● Đang ghi… bấm lại để dừng");
  }

  function toggleVoice() {
    if (listening) {
      if (recRef.current) { try { recRef.current.stop(); } catch { /* ignore */ } }
      else if (mediaRef.current && mediaRef.current.state !== "inactive") mediaRef.current.stop();
      return;
    }
    if (useSpeech) startSpeech();
    else startWhisper();
  }

  const rewrite = parsed?.translated_en_visual;
  const showRewrite = rewrite && rewrite.trim() && rewrite.trim() !== query.trim();

  return (
    <div className="panel">
      <h3>Query <span className="kbd">/</span></h3>
      <textarea
        ref={inputRef}
        className="query-input"
        data-testid="query-input"
        placeholder="Mô tả cảnh / nội dung cần tìm…  (Enter to search, Shift+Enter newline)"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            onSearch();
          }
        }}
      />
      <div className="row mt">
        <button className="btn primary" data-testid="search-btn" onClick={onSearch} disabled={loading}>
          {loading ? "Searching…" : "Search"} <span className="kbd">↵</span>
        </button>
        {queryType === "T-KIS" && (
          <button className="btn sm" onClick={onAppendHint} title="Combine this as an additional hint">
            + Append hint
          </button>
        )}
        <button
          className={`btn sm ${listening ? "danger" : ""}`}
          data-testid="voice-btn"
          onClick={toggleVoice}
          title={`Voice input (Ctrl+M) — ${useSpeech ? "browser speech, live" : "Whisper (server)"}`}
        >
          {listening ? "● rec" : "🎙 voice"}
        </button>
        <button
          className={`btn sm ${useLLM ? "primary" : "ghost"}`}
          data-testid="llm-toggle"
          onClick={onToggleLLM}
          title="LLM routing translates VI→EN and auto-routes channels, but adds several seconds per new query."
        >
          {useLLM ? "🧠 LLM on" : "LLM off (fast)"}
        </button>
        <button
          className={`btn sm ${expand ? "primary" : "ghost"}`}
          data-testid="expand-toggle"
          onClick={onToggleExpand}
          title="Query expansion: adds 2-3 visual paraphrases per query for better recall (slower — extra LLM call)."
        >
          {expand ? "🔎 Expand on" : "Expand off"}
        </button>
        {scopeFilter}
      </div>

      {/* Retrieval depth. Deliberately NOT live: moving it while a hundred
          results are on screen would re-run every channel on every drag. */}
      <div className="topk-row">
        <span className="k">keyframes</span>
        <input
          type="range"
          className="slider"
          min={20}
          max={1000}
          step={20}
          value={topK}
          data-testid="topk-slider"
          aria-label="Số keyframe truy hồi"
          onChange={(event) => onTopK(Number(event.target.value))}
        />
        <span className="mono topk-value" data-testid="topk-value">{topK}</span>
      </div>
      {appliedTopK != null && appliedTopK !== topK && (
        <div className="hint-text" data-testid="topk-dirty">
          Đang hiển thị {appliedTopK} keyframe — bấm <b>Search</b> để tải lại với {topK}.
        </div>
      )}
      {listening && interim && (
        <div className="hint-text" style={{ fontStyle: "italic", color: "var(--fg-dim)" }} data-testid="voice-interim">
          …{interim}
        </div>
      )}
      {voiceMsg && (
        <div className="hint-text" style={{ color: "var(--warn)" }} data-testid="voice-msg">{voiceMsg}</div>
      )}
      {hints.length > 0 && (
        <div className="row mt" style={{ fontSize: 11 }}>
          <span style={{ color: "var(--fg-faint)" }}>hints({hints.length}):</span>
          {hints.map((h, i) => (
            <span key={i} className="mono" style={{ color: "var(--fg-dim)" }}>
              ‹{h}›
            </span>
          ))}
          <button className="btn sm ghost" onClick={onClearHints}>clear</button>
        </div>
      )}
      {showRewrite && (
        <div className="row mt" style={{ fontSize: 11 }}>
          <span style={{ color: "var(--ch-speech)" }} title="Đã tự động áp dụng cho kênh vector (PE). Không cần bấm gì.">
            EN vector ✓
          </span>
          <span className="mono" style={{ color: "var(--ch-vector)" }}>{rewrite}</span>
          <button
            className="btn sm ghost"
            title="Chỉ chép bản tiếng Anh vào ô query để bạn chỉnh tay (không bắt buộc)."
            onClick={() => setQuery(rewrite!)}
          >
            edit
          </button>
        </div>
      )}
    </div>
  );
}
