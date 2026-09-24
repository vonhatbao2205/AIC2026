// R2 media URL helpers. The backend already returns built URLs; these mirror the
// convention for any client-side need. Never rely on local/Kaggle paths.

/** Media folder of a video: "L21_V001" -> "L21". The batch-2 camera and cycling
 *  series join group and number with a hyphen ("N001-V001" -> "N001"). */
export function groupFromVideoId(videoId: string): string {
  return videoId.split(/[_-]/)[0];
}

export function keyframeUrl(baseUrl: string, videoId: string, keyframeN: number): string {
  const group = groupFromVideoId(videoId);
  const frame = String(keyframeN).padStart(3, "0");
  return `${baseUrl.replace(/\/$/, "")}/Keyframes/Keyframes_${group}/${videoId}/${frame}.jpg`;
}

export function keyframeUrlFromSubmitId(baseUrl: string, submitKeyframeId: string): string {
  const [, videoId, frame] = submitKeyframeId.split("/");
  const frameName = frame.endsWith(".jpg") ? frame : `${frame}.jpg`;
  const group = groupFromVideoId(videoId);
  return `${baseUrl.replace(/\/$/, "")}/Keyframes/Keyframes_${group}/${videoId}/${frameName}`;
}

/** The same media asset under the fallback origin, or null when the swap does not apply.
 *
 *  A dead origin can affect every asset it serves at once, so the retry is an
 *  origin swap rather than a separately stored alternative for every URL.
 *  Returns null when there is no fallback configured, when the URL does not
 *  actually come from the primary origin (nothing to swap), or when the two
 *  origins are the same, which would dress a single point of failure up as
 *  redundancy and retry against the host that just failed. */
export function swapMediaOrigin(
  url: string,
  primary: string | undefined,
  fallback: string | undefined,
): string | null {
  if (!url || !primary || !fallback) return null;
  const from = primary.replace(/\/$/, "");
  const to = fallback.replace(/\/$/, "");
  if (from === to || !url.startsWith(from)) return null;
  return to + url.slice(from.length);
}

/** Backward-compatible, video-specific name used by the video player. */
export const swapVideoOrigin = swapMediaOrigin;

export function videoUrl(baseUrl: string, videoId: string): string {
  const group = groupFromVideoId(videoId);
  // N camera videos play from their repaired copies (backend `media.py`).
  const root = group.startsWith("N") ? "Videos_Web" : "Videos";
  return `${baseUrl.replace(/\/$/, "")}/${root}/Videos_${group}/${videoId}.mp4`;
}

/** Where a frame sits on the video's timeline, for chronological ordering.
 *
 *  `pts_time` is the real answer; `frame_idx` and `keyframe_n` are the fallbacks
 *  for a frame the keyframe-map enrichment could not fill in (it stops at the
 *  retrieval-depth cap). All three increase with time within one video, so the
 *  ordering stays correct whichever one a frame carries. */
export function frameTimeKey(frame: {
  pts_time?: number | null;
  frame_idx?: number | null;
  keyframe_n?: number | null;
}): number | null {
  if (frame.pts_time != null && !Number.isNaN(frame.pts_time)) return frame.pts_time;
  if (frame.frame_idx != null && !Number.isNaN(frame.frame_idx)) return frame.frame_idx;
  if (frame.keyframe_n != null && !Number.isNaN(frame.keyframe_n)) return frame.keyframe_n;
  return null;
}

/** Frames in chronological order, earliest first.
 *
 *  Frames with no time at all keep their relevance order at the end rather than
 *  being scattered through the strip: an unknown position is not position zero.
 *  Never sorts in place — the caller's array is the retrieval ranking. */
export function sortFramesByTime<T extends { pts_time?: number | null; frame_idx?: number | null; keyframe_n?: number | null }>(
  frames: readonly T[],
): T[] {
  return frames
    .map((frame, index) => ({ frame, index, key: frameTimeKey(frame) }))
    .sort((a, b) => {
      if (a.key == null && b.key == null) return a.index - b.index;
      if (a.key == null) return 1;
      if (b.key == null) return -1;
      // Ties keep the retrieval order, so the sort is stable across browsers.
      return a.key - b.key || a.index - b.index;
    })
    .map((item) => item.frame);
}

/** 1-based rank of the strongest `count` frames by fused score, keyed by id.
 *
 *  Chronological order is the right way to READ a scene but it throws away the
 *  one thing the ranking said: which frame is the reason this video is on screen
 *  at all. Marking the top few puts that back without giving up the reading
 *  order. Linear, because a group can hold hundreds of frames and only three of
 *  them are wanted. */
export function topRelevanceRanks<T extends { submit_keyframe_id: string; score: number }>(
  frames: readonly T[],
  count = 3,
): Map<string, number> {
  if (count <= 0) return new Map();
  const best: T[] = [];
  for (const frame of frames) {
    let at = best.length;
    while (at > 0 && best[at - 1].score < frame.score) at -= 1;
    if (at >= count) continue;
    best.splice(at, 0, frame);
    if (best.length > count) best.pop();
  }
  return new Map(best.map((frame, index) => [frame.submit_keyframe_id, index + 1]));
}

export function formatTime(seconds: number | null | undefined): string {
  if (seconds == null || Number.isNaN(seconds)) return "--:--";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}
