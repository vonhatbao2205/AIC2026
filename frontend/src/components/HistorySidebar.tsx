import { useState } from "react";
import type { SubmitEntry } from "../api/types";
import { formatTime } from "../lib/media";

interface Props {
  history: SubmitEntry[];
  /** The open DRES task, so "just this task" can be offered by name. */
  taskScope: string;
  taskLabel: string | null;
  /** Delete specific entries; no argument clears everything. */
  onDelete: (opts: { ids?: string[]; taskId?: string }) => void;
}

export function HistorySidebar({ history, taskScope, taskLabel, onDelete }: Props) {
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [confirming, setConfirming] = useState(false);
  const inTask = history.filter((h) => h.task_id === taskScope).length;

  function toggle(id: string) {
    setSelected((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  }

  function deleteSelected() {
    onDelete({ ids: [...selected] });
    setSelected(new Set());
  }

  return (
    <div className="panel" data-testid="history-sidebar">
      <h3>
        Submit history ({history.length})
        {history.length > 0 && (
          <button
            className="btn sm ghost"
            style={{ float: "right" }}
            onClick={() => setConfirming((c) => !c)}
            data-testid="history-clear"
          >
            clear…
          </button>
        )}
      </h3>

      {selected.size > 0 && (
        <div className="row hist-selection" data-testid="history-selection">
          <span>{selected.size} items selected</span>
          <div className="spacer" />
          <button className="btn sm" onClick={deleteSelected} data-testid="history-delete-selected">
            Delete selected
          </button>
          <button className="btn sm ghost" onClick={() => setSelected(new Set())}>deselect</button>
        </div>
      )}

      {confirming && (
        <div className="dup-warn" data-testid="history-clear-confirm">
          {/* The log is what the dedup guard remembers, so say what is lost. */}
          ⚠ Delete local history (DRES submissions remain). Deleted entries will
          no longer trigger duplicate warnings. A backup is saved next to the history file.
          <div className="row" style={{ gap: 8, marginTop: 8 }}>
            <button
              className="btn sm"
              disabled={inTask === 0}
              onClick={() => { onDelete({ taskId: taskScope }); setConfirming(false); }}
              data-testid="history-clear-task"
            >
              task {taskLabel ?? "current"} ({inTask})
            </button>
            <button
              className="btn sm"
              onClick={() => { onDelete({}); setSelected(new Set()); setConfirming(false); }}
              data-testid="history-clear-all"
            >
              all ({history.length})
            </button>
            <button className="btn sm ghost" onClick={() => setConfirming(false)}>cancel</button>
          </div>
        </div>
      )}

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
            <div className={`hist-item${selected.has(h.id) ? " selected" : ""}`} key={h.id}>
              <div className="row">
                <input
                  type="checkbox"
                  className="hist-pick"
                  checked={selected.has(h.id)}
                  onChange={() => toggle(h.id)}
                  aria-label={`Select submission ${h.task_name ?? h.task_id}`}
                  data-testid={`history-pick-${h.id}`}
                />
                <span className="badge image_pe">{h.query_type}</span>
                <span className={`status-pill ${h.status}`}>{h.status.replace("_", " ")}</span>
                {/* The DRES verdict is the only judgement that counts. */}
                {h.verdict && (
                  <span className={`status-pill ${h.verdict.toLowerCase()}`} data-testid="hist-verdict">
                    {h.verdict}
                  </span>
                )}
                {h.was_duplicate && <span className="badge warn">dup</span>}
                <div className="spacer" />
                <button
                  className="icon-btn sm hist-del"
                  title="Delete this history entry"
                  aria-label="Delete this entry"
                  onClick={() => onDelete({ ids: [h.id] })}
                  data-testid={`history-delete-${h.id}`}
                >
                  ✕
                </button>
              </div>
              <div className="mono" style={{ fontSize: 10, marginTop: 3, color: "var(--fg-dim)", wordBreak: "break-all" }}>
                {frames.join("  →  ")}
              </div>
              <div className="row" style={{ justifyContent: "space-between", marginTop: 2 }}>
                <span style={{ fontSize: 10, color: "var(--fg-faint)" }}>task {h.task_name ?? h.task_id}</span>
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
