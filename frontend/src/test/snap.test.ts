import { describe, expect, it } from "vitest";
import { snapToKeyframe, validateIncreasingOrder } from "../lib/snap";
import { normalizeSubmitId } from "../lib/identity";
import type { TimelineKeyframe } from "../api/types";

const KFS: TimelineKeyframe[] = [
  { submit_keyframe_id: "K01/K01_V001/001", keyframe_n: 1, frame_idx: 0, pts_time: 0, keyframe_url: "" },
  { submit_keyframe_id: "K01/K01_V001/002", keyframe_n: 2, frame_idx: 150, pts_time: 6, keyframe_url: "" },
  { submit_keyframe_id: "K01/K01_V001/003", keyframe_n: 3, frame_idx: 300, pts_time: 12, keyframe_url: "" },
];

describe("snapToKeyframe", () => {
  it("snaps to nearest frame_idx", () => {
    const r = snapToKeyframe(5.0, 25, KFS)!;
    expect(r.submit_keyframe_id).toBe("K01/K01_V001/002");
    expect(r.raw_frame_idx).toBe(125);
    expect(r.delta_frames).toBe(25);
    expect(r.far).toBe(false);
  });

  it("flags far snap", () => {
    const r = snapToKeyframe(9.0, 25, KFS)!;
    expect(r.far).toBe(true);
  });

  it("falls back to pts_time when frame_idx missing", () => {
    const kfs = KFS.map((k) => ({ ...k, frame_idx: null }));
    const r = snapToKeyframe(5.0, 25, kfs)!;
    expect(r.submit_keyframe_id).toBe("K01/K01_V001/002");
  });

  it("returns null on empty", () => {
    expect(snapToKeyframe(5, 25, [])).toBeNull();
  });
});

describe("validateIncreasingOrder", () => {
  it("passes increasing", () => {
    expect(validateIncreasingOrder([1, 2, 3])).toEqual([]);
  });
  it("catches out-of-order E2", () => {
    expect(validateIncreasingOrder([10, 5, 30])).toEqual([2]);
  });
  it("ignores empty slots", () => {
    expect(validateIncreasingOrder([10, null, 30])).toEqual([]);
  });
});

describe("normalizeSubmitId", () => {
  it("normalizes shard + pads", () => {
    expect(normalizeSubmitId("L26_a/L26_V001/14")).toBe("L26/L26_V001/014");
  });
  it("leaves canonical id unchanged", () => {
    expect(normalizeSubmitId("K01/K01_V001/001")).toBe("K01/K01_V001/001");
  });
});
