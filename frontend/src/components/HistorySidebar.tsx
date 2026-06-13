import type { SubmitEntry } from "../api/types";
import { formatTime } from "../lib/media";

interface Props {
  history: SubmitEntry[];
}

export function HistorySidebar({ history }: Props) {
  return (
    <div className="panel" data-testid="history-sidebar">
      <h3>Submit history ({history.length})</h3>
      {history.length === 0 ? (
        <div className="empty" style={{ padding: 10 }}>No submissions yet.</div>
      ) : (
        [...history].reverse().map((h) => {
          const frames = h.payload.events?.length
            ? h.payload.events.map((e) => `f${e.frame_idx}`)
            : h.payload.frame_idx != null
            ? [`${h.payload.video_id} · f${h.payload.frame_idx}`]
            : [];
          return (
            <div className="hist-item" key={h.id}>
              <div className="row">
                <span className="badge image_pe">{h.query_type}</span>
                <span className={`status-pill ${h.status}`}>{h.status.replace("_", " ")}</span>
                {h.was_duplicate && <span className="badge warn">dup</span>}
              </div>
              <div className="mono" style={{ fontSize: 10, marginTop: 3, color: "var(--fg-dim)", wordBreak: "break-all" }}>
                {frames.join("  →  ")}
              </div>
              <div className="row" style={{ justifyContent: "space-between", marginTop: 2 }}>
                <span style={{ fontSize: 10, color: "var(--fg-faint)" }}>task {h.task_id}</span>
                <span style={{ fontSize: 10, color: "var(--fg-faint)" }} className="mono">
                  {formatTime((Date.now() / 1000 - h.ts))} ago
                </span>
              </div>
              {h.payload.answer && <div style={{ fontSize: 11, marginTop: 2 }}>ans: {h.payload.answer}</div>}
            </div>
          );
        })
      )}
    </div>
  );
}
