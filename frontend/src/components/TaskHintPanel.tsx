import type { DresTaskHint } from "../api/types";

interface Props {
  hint: DresTaskHint | null;
  loading: boolean;
  error: string | null;
  /** True when the query box already holds this statement. */
  inQuery: boolean;
  onUseAsQuery: () => void;
  onRefresh: () => void;
}

/** Media hints (V-KIS) arrive base64-encoded; DRES may or may not prefix them. */
function dataUrl(contentType: string, content: string): string {
  if (content.startsWith("data:")) return content;
  const mime = contentType === "VIDEO" ? "video/mp4" : "image/png";
  return `data:${mime};base64,${content}`;
}

export function TaskHintPanel({ hint, loading, error, inQuery, onUseAsQuery, onRefresh }: Props) {
  if (!hint && !loading && !error) return null;
  const media = (hint?.elements ?? []).filter((e) => e.content_type !== "TEXT");

  return (
    <div className="panel task-hint" data-testid="task-hint-panel">
      <h3>
        Đề bài{hint?.task_name ? ` · ${hint.task_name}` : ""}
        <button className="btn sm ghost" onClick={onRefresh} style={{ float: "right" }} data-testid="task-hint-refresh">
          ⟳
        </button>
      </h3>

      {error && <div className="dres-warn" data-testid="task-hint-error">{error}</div>}
      {hint?.warnings?.map((w) => (
        <div className="dres-warn" key={w}>{w}</div>
      ))}

      {hint?.text ? (
        <>
          <div className="task-hint-text" data-testid="task-hint-text">{hint.text}</div>
          <div className="row" style={{ marginTop: 8, gap: 8 }}>
            <button
              className="btn sm"
              onClick={onUseAsQuery}
              disabled={inQuery}
              data-testid="task-hint-use"
              title="Đưa nguyên văn đề bài vào ô truy vấn"
            >
              {inQuery ? "đang ở ô truy vấn" : "→ dùng làm truy vấn"}
            </button>
            <button
              className="btn sm ghost"
              onClick={() => navigator.clipboard?.writeText(hint.text)}
              data-testid="task-hint-copy"
            >
              copy
            </button>
          </div>
        </>
      ) : loading ? (
        <div className="empty" style={{ padding: 10 }}>đang tải đề bài…</div>
      ) : !error ? (
        <div className="empty" style={{ padding: 10 }}>Task đang mở chưa có đề bài dạng chữ.</div>
      ) : null}

      {media.map((element, i) => (
        <div key={i} className="task-hint-media">
          {element.offset > 0 && <div className="e-time">sau {element.offset}s</div>}
          {element.content_type === "VIDEO" ? (
            <video src={dataUrl(element.content_type, element.content)} controls loop={hint?.loop} data-testid="task-hint-video" />
          ) : (
            <img src={dataUrl(element.content_type, element.content)} alt="task hint" data-testid="task-hint-image" />
          )}
        </div>
      ))}
    </div>
  );
}
