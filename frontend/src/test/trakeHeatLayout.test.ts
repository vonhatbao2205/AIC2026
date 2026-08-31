import { describe, expect, it } from "vitest";
import type { TrakeHeatPeak } from "../api/types";
import { HEAT_THUMB_W, layoutHeatPeaks } from "../lib/trakeHeatLayout";

function peak(pts: number, id = `k${pts}`): TrakeHeatPeak {
  return {
    event_index: 1,
    submit_keyframe_id: id,
    keyframe_n: 1,
    frame_idx: Math.round(pts * 25),
    pts_time: pts,
    score: 0.03,
    strength: 0.8,
    via_fill: false,
    selected_by_dp: false,
    keyframe_url: "https://media.test/k.jpg",
  };
}

const TRACK = 900;
const SPAN = 600; // a ten-minute video

/** Two frames are individually clickable only if their boxes do not cover each
 *  other — same lane means they must be a frame's width apart. */
function overlapping(a: { x: number; lane: number }, b: { x: number; lane: number }): boolean {
  return a.lane === b.lane && Math.abs(a.x - b.x) < HEAT_THUMB_W;
}

describe("TRAKE heat row layout", () => {
  it("keeps well-separated candidates on one lane at their own timestamps", () => {
    const { placed, lanes } = layoutHeatPeaks(
      [peak(30), peak(200), peak(400), peak(560)],
      SPAN,
      TRACK,
    );
    expect(lanes).toBe(1);
    expect(placed.every((item) => !item.shifted)).toBe(true);
    expect(placed.map((item) => Math.round(item.x))).toEqual(
      placed.map((item) => Math.round(item.trueX)),
    );
  });

  it("stacks a tight cluster instead of burying it", () => {
    // 05:33 / 05:44 / 06:15 in a ten-minute video: eleven seconds apart is ~16px,
    // so on one lane the last frame painted is the only one that can be hit and
    // the two underneath — exactly the ones worth comparing — are unreachable.
    const { placed, lanes } = layoutHeatPeaks(
      [peak(333), peak(344), peak(375)],
      SPAN,
      TRACK,
    );
    expect(lanes).toBe(2);
    for (let i = 0; i < placed.length; i += 1) {
      for (let j = i + 1; j < placed.length; j += 1) {
        expect(overlapping(placed[i], placed[j])).toBe(false);
      }
    }
  });

  it("never loses the true timestamp when a frame has to be nudged", () => {
    // Four candidates inside twelve seconds: both lanes are busy, so the layout
    // slides one clear — and the tick stays on the real moment, or the row would
    // be lying about when the event fires.
    const { placed } = layoutHeatPeaks(
      [peak(300), peak(304), peak(308), peak(312)],
      SPAN,
      TRACK,
    );
    const shifted = placed.filter((item) => item.shifted);
    expect(shifted.length).toBeGreaterThan(0);
    for (const item of placed) {
      const expected = HEAT_THUMB_W / 2 + (item.peak.pts_time / SPAN) * (TRACK - HEAT_THUMB_W);
      expect(item.trueX).toBeCloseTo(expected, 6);
    }
  });

  it("gives every candidate its own reachable box, however crowded", () => {
    const peaks = [0, 3, 6, 9, 12, 15, 18, 21].map((t) => peak(300 + t));
    const { placed } = layoutHeatPeaks(peaks, SPAN, TRACK);
    expect(placed).toHaveLength(8);
    for (let i = 0; i < placed.length; i += 1) {
      for (let j = i + 1; j < placed.length; j += 1) {
        expect(overlapping(placed[i], placed[j])).toBe(false);
      }
    }
  });

  it("keeps the first and last frames fully inside the track", () => {
    const { placed } = layoutHeatPeaks([peak(0), peak(SPAN)], SPAN, TRACK);
    expect(placed[0].x - HEAT_THUMB_W / 2).toBeGreaterThanOrEqual(0);
    expect(placed[1].x + HEAT_THUMB_W / 2).toBeLessThanOrEqual(TRACK);
  });

  it("orders by time whatever order the backend sent", () => {
    const { placed } = layoutHeatPeaks([peak(400), peak(30), peak(200)], SPAN, TRACK);
    expect(placed.map((item) => item.peak.pts_time)).toEqual([30, 200, 400]);
  });

  it("survives a degenerate span without dividing by zero", () => {
    const { placed, lanes } = layoutHeatPeaks([peak(0), peak(0, "b")], 0, TRACK);
    expect(lanes).toBe(2);
    expect(placed.every((item) => Number.isFinite(item.x))).toBe(true);
  });
});
