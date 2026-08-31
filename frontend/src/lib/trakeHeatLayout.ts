import type { TrakeHeatPeak } from "../api/types";

/** Frame size in the heat rows. Big enough to read the shot and to hit with a
 *  trackpad, which is the whole reason these are frames and not marks. */
export const HEAT_THUMB_W = 72;
export const HEAT_THUMB_H = 44;
/** Vertical pitch of a lane, and the padding the track adds around them. */
export const HEAT_LANE_H = HEAT_THUMB_H + 6;
export const HEAT_TRACK_PAD = 8;
/** Clear space between two frames on the same lane. */
const MIN_GAP = 5;
/** Beyond this the row is taller than it is useful; extra crowding is resolved
 *  by nudging along the lane instead, with the true-time tick left behind. */
const MAX_LANES = 2;
/** Used until the real track has been measured (and in jsdom, which has no
 *  layout). Only the first paint uses it. */
export const FALLBACK_TRACK_W = 820;

export interface PlacedPeak {
  peak: TrakeHeatPeak;
  /** Centre of the frame, px from the left of the track. */
  x: number;
  /** Where the moment actually is — the tick, drawn whether or not `x` moved. */
  trueX: number;
  lane: number;
  /** True when crowding pushed the frame off its own timestamp. */
  shifted: boolean;
}

export interface HeatLayout {
  placed: PlacedPeak[];
  lanes: number;
  trackHeight: number;
}

/** Lay one event's candidates out along a track so that every one of them can
 *  be clicked.
 *
 *  Positioning frames purely by timestamp is honest and unusable: three
 *  candidates eleven seconds apart in a ten-minute video land on top of each
 *  other, and only the last one painted can be hit — the two underneath are
 *  unreachable, which is exactly the case where the operator most wants to
 *  compare them.
 *
 *  So: frames stack into a second lane when they collide, and only if both lanes
 *  are crowded does a frame get nudged along its lane. The exact time is never
 *  lost — every frame keeps a `trueX` tick at its real position, so the row still
 *  reads as the temporal pattern it is while the frames stay individually
 *  reachable. */
export function layoutHeatPeaks(
  peaks: readonly TrakeHeatPeak[],
  span: number,
  trackWidth: number,
): HeatLayout {
  const step = HEAT_THUMB_W + MIN_GAP;
  // Inset by half a frame at each end so the first and last are fully on track.
  const usable = Math.max(trackWidth - HEAT_THUMB_W, 1);
  const safeSpan = span > 0 ? span : 1;
  const ordered = [...peaks].sort(
    (a, b) => a.pts_time - b.pts_time || a.submit_keyframe_id.localeCompare(b.submit_keyframe_id),
  );

  const laneLastX: number[] = [];
  const placed: PlacedPeak[] = [];
  for (const peak of ordered) {
    const ratio = Math.min(1, Math.max(0, peak.pts_time / safeSpan));
    const trueX = HEAT_THUMB_W / 2 + ratio * usable;
    let lane = laneLastX.findIndex((last) => trueX - last >= step);
    let x = trueX;
    if (lane === -1) {
      if (laneLastX.length < MAX_LANES) {
        lane = laneLastX.length;
        laneLastX.push(Number.NEGATIVE_INFINITY);
      } else {
        // Both lanes are busy here: give it to whichever has more room behind it
        // and slide it clear. The tick stays put.
        lane = laneLastX.indexOf(Math.min(...laneLastX));
        x = laneLastX[lane] + step;
      }
    }
    laneLastX[lane] = x;
    placed.push({ peak, x, trueX, lane, shifted: Math.abs(x - trueX) > 0.5 });
  }

  const lanes = Math.max(1, laneLastX.length);
  return { placed, lanes, trackHeight: lanes * HEAT_LANE_H + HEAT_TRACK_PAD };
}
