import type { FrameResult, QueryType } from "../api/types";
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
  taskId: string;
  setTaskId: (v: string) => void;
  answer: string;
  setAnswer: (v: string) => void;
  duplicateId: string | null; // the already-submitted frame id, if any
  orderViolations: number[];
  submitting: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}

export function SubmitGuard(props: Props) {
  const { open, queryType, frame, pausedFrame, frameIdx, trakeSlots, duplicateId, orderViolations, submitting } = props;
  if (!open) return null;

  const isTrake = queryType === "TRAKE";
  const filledSlots = trakeSlots.filter((s): s is TrakeSlot => s !== null);
  const ocr = frame?.evidence.find((e) => e.type === "ocr");
  const speech = frame?.evidence.find((e) => e.type === "speech");
  const audio = frame?.evidence.find((e) => e.type === "audio");

  const missingFrameIdx = !isTrake && !pausedFrame && !!frame && frameIdx == null;
  const blocked = isTrake ? filledSlots.length === 0 : (!pausedFrame && !frame) || missingFrameIdx;
  const preview = pausedFrame?.thumbnail ?? frame?.keyframe_url ?? null;
  const previewAlt = pausedFrame
    ? `Paused frame ${pausedFrame.frame_idx}`
    : frame?.submit_keyframe_id ?? "Submission frame";

  return (
    <div className="modal-backdrop" onClick={props.onCancel} data-testid="submit-guard">
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <h2>Confirm submission · {queryType}</h2>
          <button className="btn sm ghost" onClick={props.onCancel}>esc</button>
        </div>
        <div className="modal-body">
          {duplicateId && (
            <div className="dup-warn" data-testid="dup-warn">
              ⚠ Duplicate: {duplicateId} was already submitted for task {props.taskId}.
            </div>
          )}
          {isTrake && orderViolations.length > 0 && (
            <div className="dup-warn">⚠ Order violation at E{orderViolations.join(", E")} — events must increase in time.</div>
          )}

          <div className="row" style={{ marginBottom: 10 }}>
            <span className="k" style={{ width: 70, color: "var(--fg-faint)" }}>task id</span>
            <input className="answer-input" value={props.taskId} onChange={(e) => props.setTaskId(e.target.value)} style={{ maxWidth: 200 }} />
          </div>

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
              {preview ? (
                <img className="guard-thumb" src={preview} alt={previewAlt} />
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
                  {frameIdx ?? "không xác định"}
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
                <div className="dup-warn">⚠ Không trích được frame_idx (thiếu pts_time/fps) — không thể nộp frame này.</div>
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
        </div>
        <div className="modal-foot">
          <button className="btn ghost" onClick={props.onCancel}>Cancel</button>
          <button
            className="btn primary"
            data-testid="confirm-submit"
            disabled={blocked || submitting}
            onClick={props.onConfirm}
          >
            {duplicateId ? "Submit anyway" : "Confirm submit"}
          </button>
        </div>
      </div>
    </div>
  );
}
