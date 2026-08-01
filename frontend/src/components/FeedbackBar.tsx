import type { FeedbackState } from "../api/types";

interface Props {
  feedback: FeedbackState;
  onRemove: (bucket: keyof FeedbackState, value: string) => void;
  onClear: () => void;
}

const BUCKETS: { key: keyof FeedbackState; mark: string; className: string; title: string }[] = [
  { key: "positive_frames", mark: "＋", className: "pos", title: "Seeds image-to-image similar search" },
  { key: "positive_videos", mark: "⬆", className: "pos", title: "Video aggregate score boosted" },
  { key: "negative_frames", mark: "✕", className: "neg", title: "Frame removed from results" },
  { key: "negative_videos", mark: "⬇", className: "neg", title: "Video aggregate score demoted" },
];

/** Nothing may re-rank silently: every active feedback item is listed here with
 *  its own remove button, plus a Clear-all. */
export function FeedbackBar({ feedback, onRemove, onClear }: Props) {
  const active = BUCKETS.flatMap(({ key, mark, className, title }) =>
    (feedback[key] as string[]).map((value) => ({ key, value, mark, className, title })),
  );
  if (active.length === 0) return null;

  return (
    <div className="panel feedback-bar" data-testid="feedback-bar">
      <div className="feedback-bar-head">
        <h3>Feedback active</h3>
        <button className="btn sm ghost" data-testid="feedback-clear" onClick={onClear}>
          Clear
        </button>
      </div>
      <div className="feedback-chips">
        {active.map(({ key, value, mark, className, title }) => (
          <button
            key={`${key}:${value}`}
            className={`feedback-chip ${className}`}
            title={`${title} · click to remove`}
            data-testid="feedback-chip"
            onClick={() => onRemove(key, value)}
          >
            <span className="feedback-mark">{mark}</span>
            <span className="mono">{value}</span>
            <span className="feedback-x">×</span>
          </button>
        ))}
      </div>
    </div>
  );
}
