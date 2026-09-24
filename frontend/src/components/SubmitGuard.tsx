import type { AnswerMode, FrameResult, QueryType, SubmitPreview } from "../api/types";
import { formatTime } from "../lib/media";
import type { PausedFrame } from "./PausedFramePanel";
import type { TrakeSlot } from "./TrakePanel";

interface Props {
  open: boolean;
  queryType: QueryType;
  frame: FrameResult | null;
  pausedFrame: PausedFrame | null;
  frameIdx: number | null; // effective frame_idx that will be submitted
  trakeSlots: (TrakeSlot | null)[];
  answer: string;
  setAnswer: (v: string) => void;
  // ---- submission-table target (used when DRES is switched off) ----
  /** False = answers land in the submission table instead of the DRES server. */
  dresEnabled: boolean;
  /** Which imported question the row will be written under. */
  questionId: string | null;
  /** The exact CSV line that will be appended, or null if it cannot be built. */
  csvLine: string | null;
  csvError: string | null;
  // ---- DRES target: what the answer is sent as, and to which open task ----
  dresConfigured: boolean;
  evaluationName: string | null;
  /** Empty = take the open task name from the server at submit time. */
  taskNameOverride: string;
  setTaskNameOverride: (v: string) => void;
  answerMode: AnswerMode;
  setAnswerMode: (v: AnswerMode) => void;
  segmentPadMs: number;
  setSegmentPadMs: (v: number) => void;
  preview: SubmitPreview | null;
  previewError: string | null;
  previewLoading: boolean;
  duplicateId: string | null; // the already-submitted answer key, if any
  orderViolations: number[];
  submitting: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}

/** The final-round answer shape per task (HD-ChungKet-2026); the backend agrees. */
function defaultAnswerMode(queryType: QueryType): Exclude<AnswerMode, "auto"> {
  if (queryType === "QA") return "qa_text";
  if (queryType === "TRAKE") return "trake_text";
  return "temporal";
}

const ANSWER_MODE_LABELS: Record<Exclude<AnswerMode, "auto">, string> = {
  temporal: "media + time",
  qa_text: "text QA-answer-video-ms",
  trake_text: "text TR-video-frames",
  temporal_text: "media + time + text",
  text: "text only",
  item: "media item only",
};

/** DRES windows are milliseconds; show them as mm:ss.mmm so they stay checkable. */
function msLabel(ms: number | undefined): string {
  if (ms == null) return "—";
  return `${formatTime(ms / 1000)}.${String(Math.round(ms) % 1000).padStart(3, "0")}`;
}

export function SubmitGuard(props: Props) {
  const {
    open, queryType, frame, pausedFrame, frameIdx, trakeSlots, duplicateId,
    orderViolations, submitting, preview, previewError, previewLoading,
  } = props;
  if (!open) return null;

  const isTrake = queryType === "TRAKE";
  const filledSlots = trakeSlots.filter((s): s is TrakeSlot => s !== null);
  const ocr = frame?.evidence.find((e) => e.type === "ocr");
  const speech = frame?.evidence.find((e) => e.type === "speech");
  const audio = frame?.evidence.find((e) => e.type === "audio");

  const missingFrameIdx = !isTrake && !pausedFrame && !!frame && frameIdx == null;
  const answers = preview?.body.answerSets[0]?.answers ?? [];
  const resolvedMode = preview?.answer_mode ?? defaultAnswerMode(queryType);
  // Only the start/end forms have a segment to widen: the final-round QA text
  // carries one instant and TRAKE carries frame numbers.
  const hasSegment = resolvedMode === "temporal" || resolvedMode === "temporal_text";
  const taskLabel = preview?.task_name ?? props.taskNameOverride ?? "";
  // Two ways a TRAKE row can be malformed in a way the export cannot see, so the
  // guard is where they have to stop rather than where they are mentioned:
  //
  //   - Frames from more than one video. The row names ONE video (the first
  //     filled slot's) and lists frame indices under it, so a frame from another
  //     video is submitted as though it came from this one — a wrong answer that
  //     looks perfectly well-formed.
  //   - Events not increasing in time. The organiser's parser rejects the row,
  //     and one rejected row blocks the whole submission.
  const slotVideos = new Set(filledSlots.map((slot) => slot.video_id));
  const mixedVideos = isTrake && slotVideos.size > 1;
  const outOfOrder = isTrake && orderViolations.length > 0;
  // A format error means DRES would reject (or misread) this answer — the guard
  // holds it here instead of spending a wrong submit on it.
  const blocked =
    (isTrake ? filledSlots.length === 0 : (!pausedFrame && !frame) || missingFrameIdx) ||
    mixedVideos || outOfOrder ||
    Boolean(props.dresEnabled ? (previewError || previewLoading || !preview || !props.dresConfigured || preview.answer_mode_mismatch) : props.csvError);
  const thumbUrl = pausedFrame?.thumbnail ?? frame?.keyframe_url ?? null;
  const previewAlt = pausedFrame
    ? `Paused frame ${pausedFrame.frame_idx}`
    : frame?.submit_keyframe_id ?? "Submission frame";

  return (
    <div className="modal-backdrop" onClick={props.onCancel} data-testid="submit-guard">
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          {/* Both submit paths are now equally reachable, so the guard states
              which frame it holds instead of leaving it to the rows below. */}
          <h2 data-testid="guard-target">
            Confirm submission · {queryType}
            {!isTrake && ` · ${pausedFrame ? "paused raw frame" : "result keyframe"}`}
          </h2>
          <button className="btn sm ghost" onClick={props.onCancel}>esc</button>
        </div>
        <div className="modal-body">
          {duplicateId && (
            <div className="dup-warn" data-testid="dup-warn">
              ⚠ Duplicate: {duplicateId} was already submitted for{" "}
              {props.dresEnabled ? `task ${taskLabel || "current"}` : props.questionId ?? "this question"}.
            </div>
          )}
          {mixedVideos && (
            <div className="dup-warn" data-testid="guard-mixed-videos">
              ⛔ Events belong to {slotVideos.size} different videos ({[...slotVideos].join(", ")}).
              A TRAKE row must contain frames from ONE video.
            </div>
          )}
          {outOfOrder && (
            <div className="dup-warn" data-testid="guard-order-violation">
              ⛔ Out of order at E{orderViolations.join(", E")} — events must be chronological. Invalid
              ordering causes the parser to reject the row and block the submission.
            </div>
          )}
          {props.dresEnabled && previewError && (
            <div className="dup-warn" data-testid="guard-format-error">⚠ {previewError}</div>
          )}
          {!props.dresEnabled && props.csvError && (
            <div className="dup-warn" data-testid="guard-format-error">⚠ {props.csvError}</div>
          )}
          {preview?.warnings?.map((w) => (
            <div className="dup-warn" key={w}>⚠ {w}</div>
          ))}

          {/* ---- submission-table target ---- */}
          {!props.dresEnabled && (
            <div className="guard-dres" data-testid="guard-submission-target">
              <div className="row">
                <span className="k" style={{ width: 74, color: "var(--fg-faint)" }}>save to</span>
                <span className="v mono" data-testid="guard-question">
                  {props.questionId ? `${props.questionId}.csv` : "— no question assigned to this tab —"}
                </span>
              </div>
              <div className="row">
                <span className="k" style={{ width: 74, color: "var(--fg-faint)" }}>CSV row</span>
                <span className="v mono" data-testid="guard-csv-line" style={{ color: "var(--accent)" }}>
                  {props.csvLine ?? "—"}
                </span>
              </div>
            </div>
          )}

          {/* ---- DRES target ---- */}
          {props.dresEnabled && (
          <div className="guard-dres" data-testid="guard-dres-target">
            <div className="row">
              <span className="k" style={{ width: 74, color: "var(--fg-faint)" }}>evaluation</span>
              <span className="v mono" data-testid="guard-evaluation">
                {props.evaluationName ?? (props.dresConfigured ? "…" : "local only")}
              </span>
              {/* DRES's own task type decides the answer shape. */}
              {preview?.task_type && (
                <span className="badge image_pe" data-testid="guard-task-type">{preview.task_type}</span>
              )}
            </div>
            <div className="row">
              <span className="k" style={{ width: 74, color: "var(--fg-faint)" }}>task name</span>
              <input
                className="answer-input"
                data-testid="guard-task-name"
                value={props.taskNameOverride}
                onChange={(e) => props.setTaskNameOverride(e.target.value)}
                placeholder={preview?.task_name ?? "auto (current task)"}
                style={{ maxWidth: 220 }}
              />
              {preview?.task_name && !props.taskNameOverride && (
                <span className="badge image_pe" data-testid="guard-task-auto">{preview.task_name}</span>
              )}
            </div>
            <div className="row">
              <span className="k" style={{ width: 74, color: "var(--fg-faint)" }}>answer as</span>
              <select
                className={`dres-select${preview?.answer_mode_mismatch ? " bad" : ""}`}
                data-testid="guard-answer-mode"
                value={props.answerMode}
                onChange={(e) => props.setAnswerMode(e.target.value as AnswerMode)}
              >
                <option value="auto">auto ({ANSWER_MODE_LABELS[defaultAnswerMode(queryType)]})</option>
                <option value="qa_text">{ANSWER_MODE_LABELS.qa_text}</option>
                <option value="trake_text">{ANSWER_MODE_LABELS.trake_text}</option>
                <option value="temporal">media + time (start/end ms)</option>
                <option value="temporal_text">media + time + text (preliminary QA)</option>
                <option value="text">text only</option>
                <option value="item">media item only</option>
              </select>
              {hasSegment && (
                <>
                  <span className="k" style={{ color: "var(--fg-faint)" }}>± ms</span>
                  <input
                    className="answer-input"
                    data-testid="guard-pad-ms"
                    type="number"
                    min={0}
                    step={100}
                    value={props.segmentPadMs}
                    onChange={(e) => props.setSegmentPadMs(Math.max(0, Number(e.target.value) || 0))}
                    style={{ maxWidth: 90 }}
                  />
                </>
              )}
            </div>
          </div>
          )}

          {isTrake ? (
            <div>
              <div style={{ fontSize: 11, color: "var(--fg-faint)", marginBottom: 6 }}>
                Ordered frames to submit ({filledSlots.length}):
              </div>
              <div className="mono" style={{ fontSize: 12, marginBottom: 8, color: "var(--fg-dim)" }}>
                video <b style={{ color: "var(--fg)" }}>{filledSlots[0]?.video_id}</b>
              </div>
              <div className="trake-slots">
                {trakeSlots.map((s, i) =>
                  s ? (
                    <div key={i} className="slot filled" style={{ flex: "0 0 96px" }}>
                      <div className="slot-head">E{i + 1}</div>
                      {s.thumbnail ? <img src={s.thumbnail} alt="" /> : <div style={{ height: 54, background: "var(--bg-3)", borderRadius: 5 }} />}
                      <div className="slot-id">frame {s.frame_idx}</div>
                      <div className="mono" style={{ fontSize: 9 }}>{formatTime(s.pts_time)}</div>
                    </div>
                  ) : null,
                )}
              </div>
              <div className="mono" style={{ fontSize: 11, marginTop: 8 }} data-testid="trake-ordered-ids">
                {filledSlots.map((s) => `frame ${s.frame_idx}`).join("  →  ")}
              </div>
            </div>
          ) : pausedFrame || frame ? (
            <>
              {thumbUrl ? (
                <img className="guard-thumb" src={thumbUrl} alt={previewAlt} />
              ) : (
                <div className="guard-thumb guard-placeholder">no preview</div>
              )}
              <div className="kv">
                <span className="k">video</span>
                <span className="v" data-testid="guard-submit-id">
                  {pausedFrame?.video_id ?? frame?.video_id}
                </span>
                <span className="k">frame_idx</span>
                <span
                  className="v"
                  data-testid="guard-frame-idx"
                  style={{ color: frameIdx == null ? "var(--bad)" : "var(--accent)", fontWeight: 700 }}
                >
                  {frameIdx ?? "unknown"}
                </span>
                <span className="k">time</span>
                <span className="v">
                  {pausedFrame
                    ? `exact paused · ${formatTime(pausedFrame.pts_time)} · ${pausedFrame.pts_time.toFixed(3)}s`
                    : `#${frame?.keyframe_n} · ${formatTime(frame?.pts_time)}`}
                </span>
                <span className="k">{pausedFrame ? "source" : "btc id"}</span>
                <span className="v" style={{ fontSize: 11, color: "var(--fg-faint)" }}>
                  {pausedFrame
                    ? `raw paused frame · ${pausedFrame.fps.toFixed(3)} fps`
                    : frame?.submit_keyframe_id}
                </span>
              </div>
              {missingFrameIdx && (
                <div className="dup-warn">⚠ Could not determine frame_idx (missing pts_time/fps); this frame cannot be submitted.</div>
              )}
              {queryType === "QA" && (
                <div style={{ margin: "8px 0" }}>
                  <div className="k" style={{ color: "var(--fg-faint)", marginBottom: 3 }}>
                    answer (editable · NVILA suggestion or manual)
                  </div>
                  <input
                    className="answer-input"
                    data-testid="qa-answer"
                    value={props.answer}
                    onChange={(e) => props.setAnswer(e.target.value)}
                    placeholder="type the answer to submit"
                  />
                </div>
              )}
              {ocr?.text && <div className="evidence"><div className="e-time">OCR</div>{ocr.text}</div>}
              {speech?.text && <div className="evidence"><div className="e-time">ASR {formatTime(speech.start)}</div>{speech.text}</div>}
              {audio && <div className="evidence"><div className="e-time">AUDIO {String(audio.top1_label ?? "")}</div>{audio.text ?? ""}</div>}
            </>
          ) : (
            <div className="empty">Nothing selected.</div>
          )}

          {/* ---- exactly what DRES will receive ---- */}
          {props.dresEnabled && (
          <div className="guard-payload">
            <div className="k" style={{ color: "var(--fg-faint)", marginBottom: 4 }}>
              DRES payload {previewLoading && <span className="dres-dim">· building…</span>}
            </div>
            {answers.length > 0 && (
              <div className="mono" style={{ fontSize: 11, marginBottom: 4 }} data-testid="guard-answer-summary">
                {answers.map((a, i) => (
                  <div key={i}>
                    {a.mediaItemName && (
                      <>
                        {a.mediaItemName} · {msLabel(a.start)} → {msLabel(a.end)} ({a.start}–{a.end} ms)
                        {a.text != null && " · "}
                      </>
                    )}
                    {a.text != null && <>text: “{a.text}”</>}
                  </div>
                ))}
              </div>
            )}
            <pre className="guard-json" data-testid="guard-json">
              {preview ? JSON.stringify(preview.body, null, 2) : previewError ? "—" : "…"}
            </pre>
            {preview?.url && (
              <div className="mono" style={{ fontSize: 10, color: "var(--fg-faint)" }}>
                POST {preview.url}
              </div>
            )}
          </div>
          )}
        </div>
        <div className="modal-foot">
          <button className="btn ghost" onClick={props.onCancel}>Cancel</button>
          <button
            className="btn primary"
            data-testid="confirm-submit"
            disabled={blocked || submitting}
            onClick={props.onConfirm}
          >
            {duplicateId ? "Add despite duplicate" : props.dresEnabled ? "Confirm submit" : "Save to submission"}
          </button>
        </div>
      </div>
    </div>
  );
}
