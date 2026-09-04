import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { RetrievalDatabase } from "../api/types";

interface Props {
  videoId: string;
  frameIdx: number;
  keyframeId: string | null;
  retrievalDatabase: RetrievalDatabase;
  onClose: () => void;
  onOpenVideo: () => void;
}

/** Lazily resolve the still image for one saved frame.
 *
 *  Both Submission and Sticky Note use this component. A raw video frame has no
 *  keyframe id, and must remain visibly raw rather than being replaced with a
 *  nearby extracted image that is not the candidate the operator chose. */
export function FramePreview(props: Props) {
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
        <div className="empty" data-testid="preview-raw">
          RAW VIDEO FRAME — frame {props.frameIdx} không phải keyframe đã trích, nên không có
          ảnh tĩnh. Bấm <b>V</b> để mở video đúng tại frame này.
        </div>
      )}
    </div>
  );
}
