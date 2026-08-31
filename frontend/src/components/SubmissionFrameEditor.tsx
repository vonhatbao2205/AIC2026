import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError } from "../api/client";
import type { RetrievalDatabase } from "../api/types";
import { formatTime } from "../lib/media";
import { VideoViewer, type VideoViewerHandle } from "./VideoViewer";
import { isTypingTarget, seekDeltaForKey } from "../lib/videoSeek";

export interface FrameEdit {
  frameIdx: number;
  /** Null: the frame came off the video and is not an extracted keyframe. */
  keyframeId: string | null;
  ptsTime: number | null;
}

export type CommitMode = "replace" | "new";

interface Props {
  open: boolean;
  videoId: string;
  /** Frame currently stored for the slot being edited. */
  frameIdx: number;
  /** Keyframe id of that slot, when it has one — the cheap way to get fps. */
  keyframeId: string | null;
  /** 0 for KIS/QA; the event index for TRAKE. */
  slot: number;
  slotCount: number;
  retrievalDatabase: RetrievalDatabase;
  onCommit: (edit: FrameEdit, mode: CommitMode) => void;
  onCancel: () => void;
}

interface VideoSource {
  url: string;
  fps: number;
}

/** Resolved video sources, kept for the life of the page.
 *
 *  Re-opening the editor on a video already looked at must not pay for the
 *  lookup again — during a contest the same video is revisited constantly. */
const sourceCache = new Map<string, VideoSource>();

/** Re-pick the frame of one submission row by scrubbing its video.
 *
 *  The edit is a local draft until a commit button is pressed. Pushing every
 *  pause to the shared table would make the row jump around on four other
 *  screens while somebody is still hunting for the moment.
 */
export function SubmissionFrameEditor(props: Props) {
  const { open, videoId, frameIdx, keyframeId, retrievalDatabase } = props;
  const cacheKey = `${retrievalDatabase}|${videoId}`;
  const [source, setSource] = useState<VideoSource | null>(() => sourceCache.get(cacheKey) ?? null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [paused, setPaused] = useState<{ rawTime: number; frameIdx: number } | null>(null);
  const viewerRef = useRef<VideoViewerHandle>(null);

  const fps = source?.fps ?? 25;
  const originalTime = frameIdx / fps;

  useEffect(() => {
    if (!open) {
      setPaused(null);
      setLoadError(null);
      return;
    }
    const cached = sourceCache.get(cacheKey);
    if (cached) {
      setSource(cached);
      return;
    }
    let cancelled = false;
    // A keyframe id resolves to fps + video_url in one small response. The
    // timeline endpoint would answer too, but it ships every keyframe of the
    // video (thousands of rows) just to read two fields off the top — that was
    // the whole reason opening this editor felt slow.
    const lookup = keyframeId
      ? api.keyframe(keyframeId, retrievalDatabase).then((info) => ({
          url: info.video_url,
          fps: info.fps ?? 25,
        }))
      : api.timeline(videoId, retrievalDatabase).then((timeline) => ({
          url: timeline.video_url,
          fps: timeline.fps ?? 25,
        }));
    lookup
      .then((resolved) => {
        sourceCache.set(cacheKey, resolved);
        if (!cancelled) setSource(resolved);
      })
      .catch((error) => {
        if (cancelled) return;
        setLoadError(
          error instanceof ApiError ? `Không tải được video (${error.status})` : "Không tải được video",
        );
      });
    return () => {
      cancelled = true;
    };
  }, [open, cacheKey, videoId, keyframeId, retrievalDatabase]);

  // Same scrub keys as the search console's inline player: arrows +/-5s, 'a'/'d'
  // +/-1s. SubmissionPanel releases every key while this modal is open, so there
  // is nothing to arbitrate against here — but the typing guard stays, because
  // 'a' and 'd' must remain letters for any text box inside the modal.
  useEffect(() => {
    if (!open) return;
    function onKey(event: KeyboardEvent) {
      if (isTypingTarget()) return;
      const delta = seekDeltaForKey(event.key);
      if (delta == null) return;
      // Capture phase + stopPropagation: the <video controls> element handles
      // the arrows itself (its focused progress slider steps ~1% of duration),
      // and that runs before the event reaches window, so preventDefault alone
      // would leave the playhead jumping by our step plus the browser's.
      event.stopPropagation();
      event.preventDefault();
      viewerRef.current?.seekBy(delta);
    }
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [open]);

  const onPaused = useCallback(
    (rawTime: number) => {
      // Same derivation the search console uses, so a frame picked here and one
      // picked there mean the same thing.
      setPaused({ rawTime, frameIdx: Math.round(rawTime * fps) });
    },
    [fps],
  );

  if (!open) return null;

  const candidate: FrameEdit | null = paused
    ? { frameIdx: paused.frameIdx, keyframeId: null, ptsTime: paused.rawTime }
    : null;

  return (
    <div className="modal-backdrop" onClick={props.onCancel} data-testid="frame-editor">
      <div className="modal wide" onClick={(event) => event.stopPropagation()}>
        <div className="modal-head">
          <h2>
            Sửa frame · {videoId}
            {props.slotCount > 1 ? ` · sự kiện ${props.slot + 1}/${props.slotCount}` : ""}
          </h2>
          <button className="btn sm ghost" onClick={props.onCancel}>esc</button>
        </div>
        <div className="modal-body">
          {loadError && <div className="dup-warn">⚠ {loadError}</div>}
          {source ? (
            <VideoViewer
              ref={viewerRef}
              src={source.url}
              startTime={originalTime}
              onPaused={onPaused}
            />
          ) : (
            !loadError && <div className="empty">Đang mở video…</div>
          )}

          <div className="frame-edit-grid">
            <div>
              <div className="k">Đang lưu</div>
              <div className="v mono" data-testid="editor-original">
                frame {frameIdx} · {formatTime(originalTime)}
              </div>
            </div>
            <div>
              <div className="k">Frame đang dừng</div>
              <div className="v mono" data-testid="editor-candidate">
                {candidate
                  ? `frame ${candidate.frameIdx} · ${formatTime(candidate.ptsTime ?? 0)}`
                  : "— tạm dừng video để chọn —"}
              </div>
            </div>
          </div>
          <div className="hint-text">
            Tua bằng <b>←</b>/<b>→</b> (±5s) hoặc <b>a</b>/<b>d</b> (±1s). Tạm dừng video đúng
            khoảnh khắc cần nộp, rồi chọn ghi đè dòng hiện tại hay thêm một
            dòng mới cho cùng câu hỏi. Frame lấy ở đây là frame thật của video nên có thể không
            phải keyframe đã trích — khi đó preview sẽ báo raw frame thay vì hiện ảnh tĩnh.
          </div>
        </div>
        <div className="modal-foot">
          <button className="btn ghost" onClick={props.onCancel}>Huỷ</button>
          <button
            className="btn ghost"
            disabled={!candidate}
            data-testid="editor-commit-new"
            onClick={() => candidate && props.onCommit(candidate, "new")}
          >
            + Thêm dòng mới
          </button>
          <button
            className="btn primary"
            disabled={!candidate}
            data-testid="editor-commit"
            onClick={() => candidate && props.onCommit(candidate, "replace")}
          >
            Sửa dòng này
          </button>
        </div>
      </div>
    </div>
  );
}
