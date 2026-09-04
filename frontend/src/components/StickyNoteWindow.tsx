import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { RetrievalDatabase } from "../api/types";
import type { NoteDraft } from "../hooks/useStickyNotes";
import type { ImportedQuestion } from "../lib/questions";
import { findNoteDuplicate, type NoteCandidate, type NoteWindow } from "../lib/stickyNotes";
import { MAX_ANSWER_LENGTH, MAX_ROWS_PER_QUESTION, validateRows } from "../lib/submission";
import { hasRowDrag, planReorder, readRowDrag, setRowDrag } from "../lib/submissionOrder";
import { FramePreview } from "./FramePreview";
import { SubmissionFrameEditor, type CommitMode, type FrameEdit } from "./SubmissionFrameEditor";

interface Props {
  question: ImportedQuestion | null;
  candidates: NoteCandidate[];
  submissionCount: number;
  window: NoteWindow;
  defaultRetrievalDatabase: RetrievalDatabase;
  onWindow: (patch: Partial<NoteWindow>) => void;
  onAdd: (draft: NoteDraft) => void;
  onUpdate: (id: string, patch: Partial<NoteCandidate>) => void;
  onRemove: (id: string) => void;
  onReorder: (changes: ReturnType<typeof planReorder>) => void;
  onClear: () => void;
  onPush: () => void;
  /** Restore one saved TRAKE sequence into the active Search tab's event bar. */
  onRestoreTrake: (candidate: NoteCandidate) => void;
}

interface RowDraft {
  videoId?: string;
  frames?: string;
  answer?: string;
}

interface PointerGesture {
  mode: "move" | "resize";
  pointerId: number;
  startX: number;
  startY: number;
  x: number;
  y: number;
  w: number;
  h: number;
}

function framesToText(frames: number[]): string {
  return frames.join(", ");
}

function parseFrames(text: string): number[] {
  return text
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean)
    .map(Number)
    .filter(Number.isFinite)
    .map(Math.round);
}

/** Local, movable staging table for the question of the active Search tab. */
export function StickyNoteWindow(props: Props) {
  const { candidates, question } = props;
  const [selected, setSelected] = useState<{ id: string; slot: number } | null>(null);
  const [drafts, setDrafts] = useState<Record<string, RowDraft>>({});
  const [previewOpen, setPreviewOpen] = useState(false);
  const [editorOpen, setEditorOpen] = useState(false);
  const [dragging, setDragging] = useState<string | null>(null);
  const [gripped, setGripped] = useState<string | null>(null);
  const [dropTarget, setDropTarget] = useState<{ id: string; below: boolean } | null>(null);
  const gesture = useRef<PointerGesture | null>(null);
  const draftsRef = useRef(drafts);
  draftsRef.current = drafts;

  const selectedCandidate = selected
    ? candidates.find((candidate) => candidate.id === selected.id) ?? null
    : null;
  const slot = Math.min(
    selected?.slot ?? 0,
    Math.max(0, (selectedCandidate?.frames.length ?? 1) - 1),
  );
  const pending = useMemo(
    () => candidates.filter((candidate) => candidate.pushState === "draft"),
    [candidates],
  );
  const queuedCount = useMemo(
    () => candidates.filter((candidate) => candidate.pushState === "queued").length,
    [candidates],
  );
  const validationErrors = useMemo(() => {
    if (!question) return [];
    return validateRows(
      pending.map((candidate) => ({ ...candidate, questionId: question.id })),
      question,
    ).filter((problem) => problem.severity === "error");
  }, [pending, question]);
  const overCapacity = props.submissionCount + pending.length > MAX_ROWS_PER_QUESTION;

  useEffect(() => {
    setSelected(null);
    setDrafts({});
    setPreviewOpen(false);
    setEditorOpen(false);
  }, [question?.id]);

  useEffect(() => {
    if (selected && !candidates.some((candidate) => candidate.id === selected.id)) {
      setSelected(null);
      setPreviewOpen(false);
      setEditorOpen(false);
    }
  }, [candidates, selected]);

  const discardDraft = useCallback((id: string) => {
    setDrafts((current) => {
      if (!(id in current)) return current;
      const { [id]: _discarded, ...rest } = current;
      return rest;
    });
  }, []);

  const commitDraft = useCallback(
    (candidate: NoteCandidate): NoteCandidate => {
      const draft = draftsRef.current[candidate.id];
      discardDraft(candidate.id);
      if (!draft) return candidate;
      const patch: Partial<NoteCandidate> = {};
      const videoId = draft.videoId ?? candidate.videoId;
      if (videoId !== candidate.videoId) patch.videoId = videoId;
      if (draft.answer !== undefined && draft.answer !== candidate.answer) {
        patch.answer = draft.answer;
      }
      if (draft.frames !== undefined) {
        const frames = parseFrames(draft.frames);
        if (frames.join(",") !== candidate.frames.join(",")) {
          patch.frames = frames;
          // Provenance only survives in slots whose exact frame and video did.
          patch.keyframeIds = frames.map((frame, index) =>
            videoId === candidate.videoId && frame === candidate.frames[index]
              ? candidate.keyframeIds[index] ?? null
              : null,
          );
          patch.ptsTimes = frames.map((frame, index) =>
            videoId === candidate.videoId && frame === candidate.frames[index]
              ? candidate.ptsTimes[index] ?? null
              : null,
          );
        }
      }
      if (videoId !== candidate.videoId && !patch.frames) {
        patch.keyframeIds = candidate.frames.map(() => null);
        patch.ptsTimes = candidate.frames.map(() => null);
      }
      if (Object.keys(patch).length) props.onUpdate(candidate.id, patch);
      return { ...candidate, ...patch };
    },
    [discardDraft, props],
  );

  const onCellKey = useCallback(
    (event: React.KeyboardEvent<HTMLInputElement>, candidate: NoteCandidate) => {
      if (event.key === "Enter") {
        event.preventDefault();
        commitDraft(candidate);
      } else if (event.key === "Escape") {
        event.stopPropagation();
        discardDraft(candidate.id);
      }
    },
    [commitDraft, discardDraft],
  );

  const reorder = useCallback(
    (sourceId: string, targetId: string) => {
      if (sourceId === targetId) return;
      const from = candidates.findIndex((candidate) => candidate.id === sourceId);
      const to = candidates.findIndex((candidate) => candidate.id === targetId);
      if (from < 0 || to < 0) return;
      // A queued/synced row is history. Moving it would claim that Submission was
      // reordered too, even though this intentionally local window cannot make
      // that shared write. Editing its content re-arms it as a fresh candidate.
      if (candidates[from].pushState !== "draft" || candidates[to].pushState !== "draft") return;
      const changes = planReorder(candidates, from, to);
      if (changes.length) props.onReorder(changes);
      setSelected({ id: sourceId, slot: 0 });
    },
    [candidates, props],
  );

  const nudge = useCallback(
    (delta: number) => {
      if (!selectedCandidate || selectedCandidate.pushState !== "draft") return;
      const from = candidates.findIndex((candidate) => candidate.id === selectedCandidate.id);
      const to = from + delta;
      if (to < 0 || to >= candidates.length || candidates[to].pushState !== "draft") return;
      reorder(selectedCandidate.id, candidates[to].id);
    },
    [candidates, reorder, selectedCandidate],
  );

  const moveSelection = useCallback(
    (delta: number) => {
      if (!candidates.length) return;
      const current = candidates.findIndex((candidate) => candidate.id === selected?.id);
      const next = current < 0
        ? 0
        : Math.min(Math.max(current + delta, 0), candidates.length - 1);
      setSelected({ id: candidates[next].id, slot: 0 });
    },
    [candidates, selected?.id],
  );

  useEffect(() => {
    if (!props.window.open) return;
    const isTyping = () => {
      const element = document.activeElement;
      return element && (element.tagName === "INPUT" || element.tagName === "TEXTAREA");
    };
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        if (editorOpen) setEditorOpen(false);
        else if (previewOpen) setPreviewOpen(false);
        else props.onWindow({ open: false });
        return;
      }
      if (isTyping() || editorOpen) return;
      switch (event.key) {
        case "ArrowDown":
          event.preventDefault();
          if (event.altKey) nudge(1);
          else moveSelection(1);
          break;
        case "ArrowUp":
          event.preventDefault();
          if (event.altKey) nudge(-1);
          else moveSelection(-1);
          break;
        case "ArrowRight":
          if (selectedCandidate && selectedCandidate.frames.length > 1) {
            event.preventDefault();
            setSelected({ id: selectedCandidate.id, slot: Math.min(slot + 1, selectedCandidate.frames.length - 1) });
          }
          break;
        case "ArrowLeft":
          if (selectedCandidate && selectedCandidate.frames.length > 1) {
            event.preventDefault();
            setSelected({ id: selectedCandidate.id, slot: Math.max(slot - 1, 0) });
          }
          break;
        case "p":
        case "P":
          if (selectedCandidate) setPreviewOpen((open) => !open);
          break;
        case "v":
        case "V":
          if (selectedCandidate) setEditorOpen(true);
          break;
        case "Delete":
          if (selectedCandidate) props.onRemove(selectedCandidate.id);
          break;
        default:
          break;
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [editorOpen, moveSelection, nudge, previewOpen, props, selectedCandidate, slot]);

  useEffect(() => {
    if (!gripped) return;
    const release = () => setGripped(null);
    window.addEventListener("mouseup", release);
    return () => window.removeEventListener("mouseup", release);
  }, [gripped]);

  const beginGesture = (event: React.PointerEvent, mode: PointerGesture["mode"]) => {
    if (mode === "move" && (event.target as HTMLElement).closest("button")) return;
    gesture.current = {
      mode,
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      x: props.window.x,
      y: props.window.y,
      w: props.window.w,
      h: props.window.h,
    };
    event.currentTarget.setPointerCapture?.(event.pointerId);
    event.preventDefault();
  };

  const continueGesture = (event: React.PointerEvent) => {
    const current = gesture.current;
    if (!current || current.pointerId !== event.pointerId) return;
    const dx = event.clientX - current.startX;
    const dy = event.clientY - current.startY;
    props.onWindow(
      current.mode === "move"
        ? { x: current.x + dx, y: current.y + dy }
        : { w: current.w + dx, h: current.h + dy },
    );
  };

  const endGesture = (event: React.PointerEvent) => {
    if (gesture.current?.pointerId === event.pointerId) gesture.current = null;
    event.currentTarget.releasePointerCapture?.(event.pointerId);
  };

  const commitFrameEdit = useCallback(
    (edit: FrameEdit, mode: CommitMode) => {
      if (!selectedCandidate) return;
      const frames = [...selectedCandidate.frames];
      const keyframeIds = [...selectedCandidate.keyframeIds];
      const ptsTimes = [...selectedCandidate.ptsTimes];
      while (keyframeIds.length < frames.length) keyframeIds.push(null);
      while (ptsTimes.length < frames.length) ptsTimes.push(null);
      frames[slot] = edit.frameIdx;
      keyframeIds[slot] = edit.keyframeId;
      ptsTimes[slot] = edit.ptsTime;
      if (mode === "new") {
        const {
          id: _id,
          createdAt: _createdAt,
          pushState: _pushState,
          submissionRowId: _submissionRowId,
          ...draft
        } = selectedCandidate;
        props.onAdd({ ...draft, frames, keyframeIds, ptsTimes });
      } else {
        props.onUpdate(selectedCandidate.id, { frames, keyframeIds, ptsTimes });
      }
      setEditorOpen(false);
    },
    [props, selectedCandidate, slot],
  );

  if (!props.window.open) return null;

  return (
    <>
      <section
        className="sticky-note-window"
        style={{
          left: props.window.x,
          top: props.window.y,
          width: props.window.w,
          height: props.window.h,
        }}
        data-testid="sticky-note"
        aria-label="Sticky note candidates"
      >
        <header
          className="sticky-note-head"
          onPointerDown={(event) => beginGesture(event, "move")}
          onPointerMove={continueGesture}
          onPointerUp={endGesture}
          onPointerCancel={endGesture}
        >
          <span className="sticky-note-pin" aria-hidden="true">◆</span>
          <div className="sticky-note-title">
            <b>Sticky note</b>
            <span className="mono">{question?.id ?? "chưa chọn câu hỏi"}</span>
          </div>
          <span className="sticky-note-count">
            {pending.length > 0 && `${pending.length} nháp`}
            {pending.length > 0 && queuedCount > 0 && " · "}
            {queuedCount > 0 && `${queuedCount} đang đồng bộ`}
            {pending.length === 0 && queuedCount === 0 && `${candidates.length} candidate`}
          </span>
          <div className="spacer" />
          <button
            className="btn sm sticky-push"
            data-testid="sticky-push"
            disabled={!question || !pending.length || validationErrors.length > 0 || overCapacity}
            onClick={props.onPush}
            title="Chỉ chuyển các candidate chưa từng push"
          >
            ⇥ Push {pending.length || "all"}
          </button>
          <button
            className="sticky-note-close"
            aria-label="Đóng sticky note"
            data-testid="sticky-close"
            onClick={() => props.onWindow({ open: false })}
          >
            ×
          </button>
        </header>

        <div className="sticky-note-body">
          {!question ? (
            <div className="sticky-note-empty">Chọn một câu hỏi cho tab Search để tạo candidate.</div>
          ) : candidates.length === 0 ? (
            <div className="sticky-note-empty">
              Chưa có candidate. Dùng nút <b>+ Sticky</b> ở Detail hoặc Paused frame.
            </div>
          ) : (
            <>
              {validationErrors.length > 0 && (
                <div className="sticky-note-warning" data-testid="sticky-invalid">
                  ⚠ {validationErrors[0].message}
                  {validationErrors.length > 1 ? ` · +${validationErrors.length - 1} lỗi` : ""}
                </div>
              )}
              {overCapacity && (
                <div className="sticky-note-warning" data-testid="sticky-capacity">
                  ⚠ Submission chỉ còn {Math.max(0, MAX_ROWS_PER_QUESTION - props.submissionCount)} chỗ,
                  nhưng note có {pending.length} candidate chưa push.
                </div>
              )}
              {previewOpen && selectedCandidate && (
                <FramePreview
                  videoId={selectedCandidate.videoId}
                  frameIdx={selectedCandidate.frames[slot] ?? 0}
                  keyframeId={selectedCandidate.keyframeIds[slot] ?? null}
                  retrievalDatabase={selectedCandidate.retrievalDatabase ?? props.defaultRetrievalDatabase}
                  onClose={() => setPreviewOpen(false)}
                  onOpenVideo={() => setEditorOpen(true)}
                />
              )}
              <div className="sticky-note-table-wrap">
                <table className="submission-table sticky-note-table">
                  <thead>
                    <tr>
                      <th style={{ width: 42 }}>#</th>
                      <th style={{ width: 116 }}>video</th>
                      <th>frame</th>
                      <th style={{ width: 118 }}>keyframe</th>
                      {question.kind === "qa" && <th>answer</th>}
                      <th style={{ width: question.kind === "trake" ? 92 : 30 }} />
                    </tr>
                  </thead>
                  <tbody>
                    {candidates.map((candidate, index) => {
                      const draft = drafts[candidate.id];
                      const isSelected = candidate.id === selected?.id;
                      const activeSlot = isSelected ? slot : 0;
                      const duplicate = findNoteDuplicate(candidates, candidate, candidate.id);
                      const isDragging = dragging === candidate.id;
                      const target = dropTarget?.id === candidate.id && !isDragging ? dropTarget : null;
                      return (
                        <tr
                          key={candidate.id}
                          className={[
                            isSelected ? "selected-row" : "",
                            candidate.pushState === "synced" ? "sticky-note-pushed" : "",
                            candidate.pushState === "queued" ? "sticky-note-queued" : "",
                            duplicate ? "bad-row" : "",
                            isDragging ? "dragging-row" : "",
                            target ? (target.below ? "drop-below" : "drop-above") : "",
                          ].join(" ").trim()}
                          data-testid={`sticky-row-${index}`}
                          onClick={() => setSelected({ id: candidate.id, slot: 0 })}
                          draggable={gripped === candidate.id && candidate.pushState === "draft"}
                          onDragStart={(event) => {
                            if (candidate.pushState !== "draft") return;
                            setRowDrag(event, { questionId: question.id, rowId: candidate.id });
                            setDragging(candidate.id);
                          }}
                          onDragEnd={() => {
                            setDragging(null);
                            setDropTarget(null);
                            setGripped(null);
                          }}
                          onDragOver={(event) => {
                            if (!hasRowDrag(event) || candidate.pushState !== "draft") return;
                            event.preventDefault();
                            const source = candidates.findIndex((item) => item.id === dragging);
                            const below = source >= 0 && source < index;
                            setDropTarget({ id: candidate.id, below });
                          }}
                          onDrop={(event) => {
                            event.preventDefault();
                            const payload = readRowDrag(event);
                            setDragging(null);
                            setDropTarget(null);
                            setGripped(null);
                            if (payload?.questionId === question.id) reorder(payload.rowId, candidate.id);
                          }}
                        >
                          <td className="mono dim">
                            <span
                              className={`row-grip${candidate.pushState !== "draft" ? " disabled" : ""}`}
                              role="button"
                              aria-label={`Kéo candidate ${index + 1}`}
                              data-testid={`sticky-grip-${index}`}
                              title={candidate.pushState !== "draft" ? "Đã chuyển sang Submission — thứ hạng sửa ở đó" : "Kéo để đổi thứ tự"}
                              onMouseDown={() => candidate.pushState === "draft" && setGripped(candidate.id)}
                              onMouseUp={() => setGripped(null)}
                            >
                              ⠿
                            </span>
                            {index + 1}
                            {candidate.pushState === "synced" && (
                              <span
                                className="sticky-pushed-mark"
                                title="Supabase đã nhận; sửa nội dung sẽ đưa candidate vào lượt push kế tiếp"
                              >
                                ✓
                              </span>
                            )}
                            {candidate.pushState === "queued" && (
                              <span
                                className="sticky-queued-mark"
                                title="Đang chờ Supabase xác nhận; hệ thống sẽ tự thử lại"
                              >
                                ◌
                              </span>
                            )}
                          </td>
                          <td>
                            <input
                              className="cell-input mono"
                              value={draft?.videoId ?? candidate.videoId}
                              onChange={(event) => setDrafts((current) => ({
                                ...current,
                                [candidate.id]: { ...current[candidate.id], videoId: event.target.value },
                              }))}
                              onBlur={() => commitDraft(candidate)}
                              onKeyDown={(event) => onCellKey(event, candidate)}
                            />
                          </td>
                          <td>
                            <input
                              className="cell-input mono"
                              value={draft?.frames ?? framesToText(candidate.frames)}
                              onChange={(event) => setDrafts((current) => ({
                                ...current,
                                [candidate.id]: { ...current[candidate.id], frames: event.target.value },
                              }))}
                              onBlur={() => commitDraft(candidate)}
                              onKeyDown={(event) => onCellKey(event, candidate)}
                            />
                          </td>
                          <td className="mono dim keyframe-cell">
                            {candidate.keyframeIds[activeSlot] ?? <span title="raw video frame">raw</span>}
                          </td>
                          {question.kind === "qa" && (
                            <td>
                              <input
                                className="cell-input"
                                maxLength={MAX_ANSWER_LENGTH * 2}
                                value={draft?.answer ?? candidate.answer}
                                onChange={(event) => setDrafts((current) => ({
                                  ...current,
                                  [candidate.id]: { ...current[candidate.id], answer: event.target.value },
                                }))}
                                onBlur={() => commitDraft(candidate)}
                                onKeyDown={(event) => onCellKey(event, candidate)}
                                placeholder="answer"
                              />
                            </td>
                          )}
                          <td>
                            {question.kind === "trake" && (
                              <button
                                className="btn sm ghost sticky-restore-trake"
                                data-testid={`sticky-restore-trake-${index}`}
                                title="Nạp chuỗi này về TRAKE events"
                                onClick={(event) => {
                                  event.stopPropagation();
                                  props.onRestoreTrake(commitDraft(candidate));
                                }}
                              >
                                ↩ TRAKE
                              </button>
                            )}
                            <button
                              className="cell-delete"
                              aria-label={`Xoá candidate ${index + 1}`}
                              data-testid={`sticky-delete-${index}`}
                              onClick={(event) => {
                                event.stopPropagation();
                                props.onRemove(candidate.id);
                              }}
                            >
                              ×
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </div>

        <footer className="sticky-note-foot">
          <span>↑↓ chọn · Alt+↑↓ reorder · P ảnh · V video · Delete xoá</span>
          {candidates.length > 0 && (
            <button className="btn sm ghost" onClick={props.onClear}>xoá note</button>
          )}
        </footer>
        <div
          className="sticky-note-resize"
          aria-hidden="true"
          onPointerDown={(event) => beginGesture(event, "resize")}
          onPointerMove={continueGesture}
          onPointerUp={endGesture}
          onPointerCancel={endGesture}
        />
      </section>

      {selectedCandidate && (
        <SubmissionFrameEditor
          open={editorOpen}
          videoId={selectedCandidate.videoId}
          frameIdx={selectedCandidate.frames[slot] ?? 0}
          keyframeId={selectedCandidate.keyframeIds[slot] ?? null}
          slot={slot}
          slotCount={selectedCandidate.frames.length}
          retrievalDatabase={selectedCandidate.retrievalDatabase ?? props.defaultRetrievalDatabase}
          onCommit={commitFrameEdit}
          onCancel={() => setEditorOpen(false)}
        />
      )}
    </>
  );
}
