// Client-side mirror of the backend canonical id rules, used for the local
// duplicate-submit pre-check (the backend remains the source of truth).

const SHARD_RE = /^([A-Za-z]+\d+)(?:_[A-Za-z0-9]+)+$/;

export function normalizeCategory(raw: string): string {
  const m = SHARD_RE.exec(raw.trim());
  return m ? m[1] : raw.trim();
}

export function normalizeSubmitId(submitKeyframeId: string): string {
  const parts = submitKeyframeId.trim().split("/");
  if (parts.length !== 3) return submitKeyframeId.trim();
  let [cat, video, frame] = parts;
  if (frame.endsWith(".jpg")) frame = frame.slice(0, -4);
  const n = parseInt(frame, 10);
  if (Number.isNaN(n)) return submitKeyframeId.trim();
  return `${normalizeCategory(cat)}/${video}/${String(n).padStart(3, "0")}`;
}
