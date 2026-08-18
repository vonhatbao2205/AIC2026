import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api/client";
import type { RetrievalDatabase } from "../api/types";
import type { ImportedQuestion } from "../lib/questions";
import { packSummary } from "../lib/questionPack";
import {
  MAX_ANSWER_LENGTH,
  MAX_ROWS_PER_QUESTION,
  findDuplicate,
  questionCsv,
  validateRows,
  type SubmissionRow,
} from "../lib/submission";
import { SubmissionFrameEditor, type CommitMode, type FrameEdit } from "./SubmissionFrameEditor";

export interface PackInfo {
  shared: boolean;
  sessionName: string | null;
  packHash: string | null;
  publishedBy: string | null;
  origin: "server" | "cache" | "local" | "none";
  loading: boolean;
  error: string | null;
}

export interface SubmissionSyncInfo {
  shared: boolean;
  status: "offline" | "connecting" | "live" | "error";
  pending: number;
  error: string | null;
  room: string;
  user: string;
  conflicts: string[];
  onDismissConflict: (rowId: string) => void;
}

interface Props {
  questions: ImportedQuestion[];
  rows: SubmissionRow[];
  onChangeRow: (rowId: string, patch: Partial<SubmissionRow>) => void;
  onDeleteRow: (rowId: string) => void;
  onAddRow: (questionId: string) => void;
  /** Copy a row (with an edited frame) into a new answer for the same question. */
  onCloneRow: (row: SubmissionRow) => void;
  onClearQuestion: (questionId: string) => void;
  onExport: () => void;
  onJumpToQuestion: (questionId: string) => void;
  exportError: string | null;
  /** Back to the console tab the operator came from. */
  onBack: () => void;
  backLabel: string;
  /** Import lives here too: this view is reachable with no tab open behind it. */
  onImport: (file: File) => void;
  importing: boolean;
  importError: string | null;
  sync: SubmissionSyncInfo;
  /** Parsed pack awaiting the publish decision. */
  pendingPack: ImportedQuestion[] | null;
  onApplyPack: () => void;
  onCancelPack: () => void;
  onSearchAll: () => void;
  searchingAll: boolean;
  pack: PackInfo;
  /** Used for rows that predate the per-row profile field. */
  defaultRetrievalDatabase: RetrievalDatabase;
}

function framesToText(frames: number[]): string {
  return frames.join(", ");
}

function parseFrames(text: string): number[] {
  return text
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean)
    .map((part) => Number(part))
    .filter((value) => Number.isFinite(value))
    .map((value) => Math.round(value));
}

/** Frame cell that keeps what was typed rather than what parses.
 *
 *  Driving the input straight off `number[]` swallows a separator the moment it
 *  is typed — "1200," re-renders as "1200" — which makes a multi-event TRAKE row
 *  impossible to enter. The raw text lives here; the parsed frames go up. */
function FramesInput(props: { value: number[]; onChange: (frames: number[]) => void; placeholder: string }) {
  const [text, setText] = useState(() => framesToText(props.value));
  const mine = useRef(text);

  useEffect(() => {
    // Re-sync only when the change came from somewhere other than this input.
    if (parseFrames(mine.current).join(",") !== props.value.join(",")) {
      const next = framesToText(props.value);
      mine.current = next;
      setText(next);
    }
  }, [props.value]);

  return (
    <input
      className="cell-input mono"
      value={text}
      placeholder={props.placeholder}
      onChange={(event) => {
        mine.current = event.target.value;
        setText(event.target.value);
        props.onChange(parseFrames(event.target.value));
      }}
    />
  );
}

/** Lazily resolved keyframe picture for exactly one row.
 *
 *  The table itself never loads media: a hundred answers would otherwise mean a
 *  hundred image requests on five machines. Only the frame being looked at is
 *  fetched, and only once the operator asks for it. */
function FramePreview(props: {
  videoId: string;
  frameIdx: number;
  keyframeId: string | null;
  retrievalDatabase: RetrievalDatabase;
  onClose: () => void;
  onOpenVideo: () => void;
}) {
  const { keyframeId, retrievalDatabase } = props;
  const [url, setUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setUrl(null);
    setError(null);
    if (!keyframeId) return;
    let cancelled = false;
    api
      .keyframe(keyframeId, retrievalDatabase)
      .then((info) => !cancelled && setUrl(info.keyframe_url))
      .catch(() => !cancelled && setError("Không resolve được ảnh keyframe."));
    return () => {
      cancelled = true;
    };
  }, [keyframeId, retrievalDatabase]);

  return (
    <div className="preview-overlay" data-testid="frame-preview">
      <div className="preview-head">
        <span className="mono">
          {props.videoId} · frame {props.frameIdx}
          {keyframeId ? ` · ${keyframeId}` : ""}
        </span>
        <div className="spacer" />
        <button className="btn sm ghost" onClick={props.onOpenVideo}>V · mở video</button>
        <button className="btn sm ghost" onClick={props.onClose}>P · đóng</button>
      </div>
      {keyframeId ? (
        error ? (
          <div className="dup-warn">⚠ {error}</div>
        ) : url ? (
          <img className="preview-img" src={url} alt={keyframeId} />
        ) : (
          <div className="empty">Đang tải ảnh…</div>
        )
      ) : (
        // A frame taken from a paused video need not be an extracted keyframe.
        // Showing the nearest one would quietly misrepresent what gets exported.
        <div className="empty" data-testid="preview-raw">
          RAW VIDEO FRAME — frame {props.frameIdx} không phải keyframe đã trích, nên không có
          ảnh tĩnh. Bấm <b>V</b> để mở video đúng tại frame này.
        </div>
      )}
    </div>
  );
}

export function SubmissionPanel(props: Props) {
  const { questions, rows, sync } = props;
  const fileRef = useRef<HTMLInputElement>(null);
  const [selected, setSelected] = useState<{ rowId: string; slot: number } | null>(null);
  const [previewOpen, setPreviewOpen] = useState(false);
  const [editorOpen, setEditorOpen] = useState(false);

  const byQuestion = useMemo(() => {
    const map = new Map<string, SubmissionRow[]>();
    for (const row of rows) {
      const list = map.get(row.questionId);
      if (list) list.push(row);
      else map.set(row.questionId, [row]);
    }
    return map;
  }, [rows]);

  const answered = questions.filter((question) => (byQuestion.get(question.id) ?? []).length > 0);
  const empty = questions.filter((question) => (byQuestion.get(question.id) ?? []).length === 0);
  const totalRows = rows.length;
  const orphans = rows.filter((row) => !questions.some((question) => question.id === row.questionId));

  /** Rows in the order they are rendered, for ↑/↓ navigation across questions. */
  const flatRows = useMemo(
    () => answered.flatMap((question) => byQuestion.get(question.id) ?? []),
    [answered, byQuestion],
  );
  const selectedRow = selected ? rows.find((row) => row.id === selected.rowId) ?? null : null;
  const slot = Math.min(selected?.slot ?? 0, Math.max(0, (selectedRow?.frames.length ?? 1) - 1));

  const move = useCallback(
    (delta: number) => {
      if (!flatRows.length) return;
      const index = flatRows.findIndex((row) => row.id === selected?.rowId);
      const next = Math.min(Math.max((index < 0 ? 0 : index) + delta, 0), flatRows.length - 1);
      setSelected({ rowId: flatRows[next].id, slot: 0 });
    },
    [flatRows, selected?.rowId],
  );

  useEffect(() => {
    function isTyping() {
      const element = document.activeElement;
      return element && (element.tagName === "INPUT" || element.tagName === "TEXTAREA");
    }
    function onKey(event: KeyboardEvent) {
      if (event.key === "Escape") {
        if (editorOpen) setEditorOpen(false);
        else if (previewOpen) setPreviewOpen(false);
        return;
      }
      if ((event.key === "e" || event.key === "E") && (event.ctrlKey || event.metaKey)) {
        event.preventDefault();
        props.onExport();
        return;
      }
      if (isTyping() || editorOpen) return;
      switch (event.key) {
        case "ArrowDown":
          event.preventDefault();
          move(1);
          break;
        case "ArrowUp":
          event.preventDefault();
          move(-1);
          break;
        case "ArrowRight":
          if (selectedRow && selectedRow.frames.length > 1) {
            event.preventDefault();
            setSelected({ rowId: selectedRow.id, slot: Math.min(slot + 1, selectedRow.frames.length - 1) });
          }
          break;
        case "ArrowLeft":
          if (selectedRow && selectedRow.frames.length > 1) {
            event.preventDefault();
            setSelected({ rowId: selectedRow.id, slot: Math.max(slot - 1, 0) });
          }
          break;
        case "p":
        case "P":
          if (selectedRow) setPreviewOpen((open) => !open);
          break;
        case "v":
        case "V":
          if (selectedRow) setEditorOpen(true);
          break;
        case "Delete":
          if (selectedRow) props.onDeleteRow(selectedRow.id);
          break;
        default:
          break;
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [move, selectedRow, slot, previewOpen, editorOpen, props]);

  /** Write a re-picked frame into the slot that was edited — either over the
   *  selected row, or as a second answer to the same question. */
  const commitFrameEdit = useCallback(
    (edit: FrameEdit, mode: CommitMode) => {
      if (!selectedRow) return;
      const frames = [...selectedRow.frames];
      const keyframeIds = [...(selectedRow.keyframeIds ?? [])];
      const ptsTimes = [...(selectedRow.ptsTimes ?? [])];
      while (keyframeIds.length < frames.length) keyframeIds.push(null);
      while (ptsTimes.length < frames.length) ptsTimes.push(null);
      frames[slot] = edit.frameIdx;
      keyframeIds[slot] = edit.keyframeId;
      ptsTimes[slot] = edit.ptsTime;
      if (mode === "new") {
        // A TRAKE alternative keeps the other events, so the whole row is
        // copied and only the edited slot differs.
        props.onCloneRow({ ...selectedRow, frames, keyframeIds, ptsTimes });
      } else {
        props.onChangeRow(selectedRow.id, { frames, keyframeIds, ptsTimes });
      }
      setEditorOpen(false);
    },
    [selectedRow, slot, props],
  );

  const syncLabel =
    !sync.shared
      ? "● local (chưa cấu hình Supabase)"
      : sync.pending > 0
        ? `● ${sync.pending} chờ đồng bộ`
        : sync.status === "live"
          ? "● Live"
          : sync.status === "error"
            ? "● mất kết nối"
            : "● đang kết nối…";

  return (
    <div className="submission-view" data-testid="submission-panel">
      <div className="submission-head">
        <button className="btn sm ghost" onClick={props.onBack} data-testid="submission-back">
          ← {props.backLabel}
        </button>
        <div>
          <h2 style={{ margin: 0, fontSize: 15 }}>Submission</h2>
          <div className="hint-text" style={{ margin: 0 }}>
            {totalRows} dòng · {answered.length}/{questions.length} câu đã có đáp án · export ra{" "}
            <code>submission/&lt;tên câu hỏi&gt;.csv</code>
          </div>
        </div>
        {props.pack.sessionName && (
          <span className="sync-pill" data-testid="pack-session" title={`pack ${props.pack.packHash}`}>
            ⛭ {props.pack.sessionName} · {props.pack.packHash}
          </span>
        )}
        <span
          className={`sync-pill ${sync.shared ? sync.status : "offline"}${sync.pending ? " pending" : ""}`}
          data-testid="sync-status"
          title={sync.shared ? `room ${sync.room} · ${sync.user}` : "Đặt VITE_SUPABASE_URL để bật sync"}
        >
          {syncLabel}
        </span>
        <div className="spacer" />
        <input
          ref={fileRef}
          type="file"
          accept=".zip,application/zip"
          style={{ display: "none" }}
          data-testid="submission-import-input"
          onChange={(event) => {
            const file = event.target.files?.[0];
            event.target.value = "";
            if (file) props.onImport(file);
          }}
        />
        <button
          className="btn sm ghost"
          onClick={props.onSearchAll}
          disabled={questions.length === 0 || props.searchingAll}
          data-testid="submission-search-all"
          title={`Mở ${questions.length} tab và search tất cả — thay toàn bộ tab đang mở`}
        >
          {props.searchingAll ? "Đang search…" : `⚡ Search tất cả (${questions.length})`}
        </button>
        <button
          className="btn sm"
          onClick={() => fileRef.current?.click()}
          disabled={props.importing}
          data-testid="submission-import"
        >
          {props.importing ? "Đang import…" : "⭳ Import câu hỏi"}
        </button>
        <button
          className="btn primary"
          onClick={props.onExport}
          disabled={totalRows === 0}
          data-testid="export-submission"
        >
          ⭱ Export submission.zip
        </button>
      </div>

      <div className="hint-text" style={{ marginTop: 0 }}>
        ↑↓ chọn dòng · ←→ chọn sự kiện (TRAKE) · <b>P</b> xem ảnh keyframe · <b>V</b> mở video để
        đổi frame · <b>Delete</b> xoá dòng · <b>Ctrl+E</b> export
      </div>

      {props.pendingPack && (() => {
        const summary = packSummary(props.pendingPack);
        return (
          <div className="pack-preview" data-testid="pack-preview">
            <div>
              <b>{summary.total} câu hỏi</b> · KIS {summary.byKind.kis} · QA {summary.byKind.qa} ·
              TRAKE {summary.byKind.trake}
            </div>
            <div className="hint-text" style={{ margin: 0 }}>
              {props.pack.shared
                ? "Publish sẽ thay gói câu hỏi của cả team và mở lại tab theo gói mới."
                : "Chưa cấu hình Supabase — gói này sẽ chỉ áp dụng trên máy bạn."}
            </div>
            <div className="row" style={{ gap: 8, marginTop: 6 }}>
              <button className="btn primary" data-testid="pack-publish" onClick={props.onApplyPack}>
                ⇪ Publish cho cả team
              </button>
              <button className="btn ghost" data-testid="pack-cancel" onClick={props.onCancelPack}>
                Huỷ
              </button>
            </div>
          </div>
        );
      })()}

      {props.pack.origin === "cache" && (
        <div className="dup-warn" data-testid="pack-cached">
          ⚠ Đang dùng gói câu hỏi từ cache — chưa xác nhận được với server. Nếu người khác vừa
          publish gói mới thì máy này có thể đang hiểu `question_id` khác cả team.
        </div>
      )}
      {props.pack.origin === "local" && props.pack.shared && (
        <div className="dup-warn" data-testid="pack-local-only">
          ⚠ Gói câu hỏi này chỉ nằm trên máy bạn (chưa publish) — đáp án vẫn ghi chung với team.
        </div>
      )}
      {props.pack.error && <div className="dup-warn">⚠ Pack: {props.pack.error}</div>}
      {sync.error && <div className="dup-warn" data-testid="sync-error">⚠ Sync: {sync.error}</div>}
      {sync.conflicts.map((rowId) => (
        <div className="dup-warn" key={rowId} data-testid="sync-conflict">
          ⚠ Dòng này vừa được người khác sửa nên thay đổi của bạn chưa được ghi. Bảng đang hiển thị
          bản mới nhất từ server.{" "}
          <button className="btn sm ghost" onClick={() => sync.onDismissConflict(rowId)}>OK</button>
        </div>
      ))}
      {props.importError && <div className="dup-warn">⚠ {props.importError}</div>}
      {props.exportError && <div className="dup-warn">⚠ {props.exportError}</div>}
      {orphans.length > 0 && (
        <div className="dup-warn" data-testid="submission-orphans">
          ⚠ {orphans.length} dòng thuộc câu hỏi không còn trong gói đã import — chúng sẽ KHÔNG được export.
        </div>
      )}

      {previewOpen && selectedRow && (
        <FramePreview
          videoId={selectedRow.videoId}
          frameIdx={selectedRow.frames[slot] ?? 0}
          keyframeId={selectedRow.keyframeIds?.[slot] ?? null}
          retrievalDatabase={selectedRow.retrievalDatabase ?? props.defaultRetrievalDatabase}
          onClose={() => setPreviewOpen(false)}
          onOpenVideo={() => setEditorOpen(true)}
        />
      )}

      {selectedRow && (
        <SubmissionFrameEditor
          open={editorOpen}
          videoId={selectedRow.videoId}
          frameIdx={selectedRow.frames[slot] ?? 0}
          keyframeId={selectedRow.keyframeIds?.[slot] ?? null}
          slot={slot}
          slotCount={selectedRow.frames.length}
          retrievalDatabase={selectedRow.retrievalDatabase ?? props.defaultRetrievalDatabase}
          onCommit={commitFrameEdit}
          onCancel={() => setEditorOpen(false)}
        />
      )}

      {questions.length === 0 && (
        <div className="empty">
          Chưa import gói câu hỏi nào — bấm “⭳ Import câu hỏi” ở trên để nạp file .zip.
        </div>
      )}

      {answered.map((question) => {
        const questionRows = byQuestion.get(question.id) ?? [];
        const problems = validateRows(questionRows, question);
        const problemsByRow = new Map<string, string[]>();
        for (const problem of problems) {
          problemsByRow.set(problem.rowId, [...(problemsByRow.get(problem.rowId) ?? []), problem.message]);
        }
        const csv = questionCsv(questionRows.slice(0, MAX_ROWS_PER_QUESTION), question.kind);
        return (
          <div className="submission-block" key={question.id} data-testid={`submission-${question.id}`}>
            <div className="submission-block-head">
              <button
                className="btn sm ghost"
                onClick={() => props.onJumpToQuestion(question.id)}
                title="Mở tab đang trả lời câu này"
              >
                {question.id}
              </button>
              <span className={`badge ${question.kind === "qa" ? "ocr" : question.kind === "trake" ? "speech" : "image_pe"}`}>
                {question.kind}
              </span>
              {question.eventCount && <span className="dres-dim">{question.eventCount} events</span>}
              <span className="dres-dim">
                {questionRows.length}/{MAX_ROWS_PER_QUESTION} dòng
              </span>
              {problems.length > 0 && (
                <span className="dres-warn" data-testid={`problems-${question.id}`}>
                  {problems.length} cảnh báo
                </span>
              )}
              <div className="spacer" />
              <button className="btn sm ghost" onClick={() => props.onAddRow(question.id)}>
                + dòng
              </button>
              <button className="btn sm ghost" onClick={() => props.onClearQuestion(question.id)}>
                xoá hết
              </button>
            </div>

            <table className="submission-table">
              <thead>
                <tr>
                  <th style={{ width: 28 }}>#</th>
                  <th style={{ width: 70 }}>ai</th>
                  <th style={{ width: 120 }}>video</th>
                  <th>{question.kind === "trake" ? `frames (${question.eventCount ?? "N"})` : "frame_idx"}</th>
                  <th style={{ width: 150 }}>keyframe</th>
                  {question.kind === "qa" && <th style={{ width: "26%" }}>answer</th>}
                  <th style={{ width: 34 }} />
                </tr>
              </thead>
              <tbody>
                {questionRows.map((row, index) => {
                  const rowProblems = problemsByRow.get(row.id) ?? [];
                  const duplicate = findDuplicate(questionRows, row, question.kind, row.id);
                  const isSelected = selected?.rowId === row.id;
                  const activeSlot = isSelected ? slot : 0;
                  return (
                    <tr
                      key={row.id}
                      className={[
                        rowProblems.length || duplicate ? "bad-row" : "",
                        isSelected ? "selected-row" : "",
                      ].join(" ").trim()}
                      onClick={() => setSelected({ rowId: row.id, slot: 0 })}
                      data-testid={`row-${question.id}-${index}`}
                    >
                      <td className="mono dim">
                        <span
                          className={`sync-dot ${row.syncState ?? "local"}`}
                          title={row.syncState ?? "local"}
                        />
                        {index + 1}
                      </td>
                      <td className="mono dim" data-testid={`row-user-${question.id}-${index}`}>
                        {row.submittedBy ?? "—"}
                      </td>
                      <td>
                        <input
                          className="cell-input mono"
                          value={row.videoId}
                          onChange={(event) => props.onChangeRow(row.id, { videoId: event.target.value })}
                          placeholder="L01_V028"
                        />
                      </td>
                      <td>
                        <FramesInput
                          value={row.frames}
                          onChange={(frames) => props.onChangeRow(row.id, { frames })}
                          placeholder={question.kind === "trake" ? "1200, 1850, 2100" : "25300"}
                        />
                      </td>
                      <td className="mono dim keyframe-cell">
                        {(row.keyframeIds?.[activeSlot] ?? null) ?? (
                          <span title="frame lấy trực tiếp từ video, không phải keyframe đã trích">raw</span>
                        )}
                      </td>
                      {question.kind === "qa" && (
                        <td>
                          <input
                            className="cell-input"
                            value={row.answer}
                            maxLength={MAX_ANSWER_LENGTH * 2}
                            onChange={(event) => props.onChangeRow(row.id, { answer: event.target.value })}
                            placeholder="đáp án (≤100 ký tự)"
                          />
                        </td>
                      )}
                      <td>
                        <button
                          className="rail-tab-close"
                          onClick={(event) => {
                            event.stopPropagation();
                            props.onDeleteRow(row.id);
                          }}
                          aria-label={`Xoá dòng ${index + 1}`}
                          data-testid={`delete-${question.id}-${index}`}
                        >
                          ×
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>

            {problems.length > 0 && (
              <ul className="submission-problems">
                {problems.map((problem, index) => (
                  <li key={`${problem.rowId}-${index}`}>{problem.message}</li>
                ))}
              </ul>
            )}
            <pre className="guard-json" data-testid={`csv-${question.id}`}>{csv}</pre>
          </div>
        );
      })}

      {empty.length > 0 && (
        <div className="submission-block">
          <div className="submission-block-head">
            <span className="dres-dim">Chưa có đáp án ({empty.length})</span>
          </div>
          <div className="submission-empty-list">
            {empty.map((question) => (
              <button
                key={question.id}
                className="btn sm ghost"
                onClick={() => props.onAddRow(question.id)}
                title={question.text.slice(0, 200)}
              >
                {question.id} +
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
