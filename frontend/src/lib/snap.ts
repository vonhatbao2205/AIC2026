import type { TimelineKeyframe } from "../api/types";

export interface SnapResult {
  submit_keyframe_id: string;
  keyframe_n: number;
  frame_idx: number;
  pts_time: number;
  raw_time: number;
  raw_frame_idx: number;
  delta_frames: number;
  delta_seconds: number;
  far: boolean;
}

// Snap a raw paused time to the nearest BTC keyframe.
// Primary key: frame_idx (round(rawTime*fps)); fallback: nearest pts_time.
export function snapToKeyframe(
  rawTime: number,
  fps: number,
  keyframes: TimelineKeyframe[],
  farThresholdSeconds = 2.0,
): SnapResult | null {
  if (!keyframes.length) return null;
  const rawFrameIdx = fps ? Math.round(rawTime * fps) : 0;

  const hasFrameIdx = keyframes.every((k) => k.frame_idx != null);
  let nearest: TimelineKeyframe;
  if (hasFrameIdx) {
    nearest = keyframes.reduce((best, k) =>
      Math.abs((k.frame_idx as number) - rawFrameIdx) < Math.abs((best.frame_idx as number) - rawFrameIdx) ? k : best,
    );
  } else {
    nearest = keyframes.reduce((best, k) =>
      Math.abs((k.pts_time ?? 0) - rawTime) < Math.abs((best.pts_time ?? 0) - rawTime) ? k : best,
    );
  }

  const kfFrameIdx = nearest.frame_idx ?? Math.round((nearest.pts_time ?? 0) * fps);
  const kfPts = nearest.pts_time ?? 0;
  const deltaFrames = kfFrameIdx - rawFrameIdx;
  const deltaSeconds = kfPts - rawTime;
  return {
    submit_keyframe_id: nearest.submit_keyframe_id,
    keyframe_n: nearest.keyframe_n,
    frame_idx: kfFrameIdx,
    pts_time: kfPts,
    raw_time: rawTime,
    raw_frame_idx: rawFrameIdx,
    delta_frames: deltaFrames,
    delta_seconds: deltaSeconds,
    far: Math.abs(deltaSeconds) > farThresholdSeconds,
  };
}

// Validate strictly increasing pts_time across filled TRAKE slots.
// Returns 1-based indices that violate ordering.
export function validateIncreasingOrder(ptsTimes: (number | null)[]): number[] {
  const violations: number[] = [];
  let last: number | null = null;
  ptsTimes.forEach((t, i) => {
    if (t == null) return;
    if (last != null && t <= last) violations.push(i + 1);
    last = t;
  });
  return violations;
}
