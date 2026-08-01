import { describe, expect, it } from "vitest";
import type { FrameResult, VideoGroup } from "../api/types";
import { buildQaCandidates, findQaFrame, selectQaCandidateFrames } from "../lib/qa";

function frame(video: string, n: number, score: number, pts: number): FrameResult {
  return {
    image_id: `K01/${video}/${String(n).padStart(3, "0")}`,
    submit_keyframe_id: `K01/${video}/${String(n).padStart(3, "0")}`,
    video_id: video,
    keyframe_n: n,
    frame_idx: Math.round(pts * 25),
    fps: 25,
    pts_time: pts,
    score,
    channels: ["image_pe"],
    per_channel_score: { image_pe: score },
    keyframe_url: "https://media.test/frame.jpg",
    video_url: "https://media.test/video.mp4",
    evidence: [],
  };
}

function group(video: string, frames: FrameResult[]): VideoGroup {
  return {
    video_id: video,
    video_score: frames[0]?.score ?? 0,
    max_score: frames[0]?.score ?? 0,
    mean_top_score: frames[0]?.score ?? 0,
    frame_count: frames.length,
    timestamp_dispersion: 0,
    ambiguous: false,
    channels: ["image_pe"],
    video_url: "https://media.test/video.mp4",
    frames,
  };
}

describe("QA visual evidence packing", () => {
  it("packs frames from the highest-scoring video before lower-ranked videos", () => {
    const a = [frame("K01_V001", 1, .99, 0), frame("K01_V001", 2, .98, 5), frame("K01_V001", 3, .97, 10)];
    const b = [frame("K01_V002", 1, .90, 0), frame("K01_V002", 2, .89, 5)];
    const c = [frame("K01_V003", 1, .80, 0)];
    const candidates = selectQaCandidateFrames(
      [group("K01_V001", a), group("K01_V002", b), group("K01_V003", c)],
      null,
      5,
      2,
    );
    expect(candidates.map((item) => item.video_id)).toEqual([
      "K01_V001", "K01_V001", "K01_V002", "K01_V002", "K01_V003",
    ]);
  });

  it("places the inspected frame at C01, then resumes ranked-video priority", () => {
    const a = [frame("K01_V001", 1, .99, 0), frame("K01_V001", 2, .98, 5), frame("K01_V001", 3, .70, 10)];
    const b = [frame("K01_V002", 1, .90, 0)];
    const candidates = buildQaCandidates([group("K01_V001", a), group("K01_V002", b)], a[2], 3, 2);
    expect(candidates.map((item) => item.submit_keyframe_id)).toEqual([
      a[2].submit_keyframe_id,
      a[0].submit_keyframe_id,
      b[0].submit_keyframe_id,
    ]);
    expect(candidates[0].submit_keyframe_id).toBe(a[2].submit_keyframe_id);
  });

  it("keeps a selected frame from a lower-ranked video at C01 without losing top-video priority", () => {
    const a = [frame("K01_V001", 1, .99, 0), frame("K01_V001", 2, .98, 5)];
    const b = [frame("K01_V002", 1, .90, 0), frame("K01_V002", 2, .89, 5)];
    const candidates = selectQaCandidateFrames(
      [group("K01_V002", b), group("K01_V001", a)],
      b[1],
      4,
      2,
    );
    expect(candidates.map((item) => item.submit_keyframe_id)).toEqual([
      b[1].submit_keyframe_id,
      a[0].submit_keyframe_id,
      a[1].submit_keyframe_id,
      b[0].submit_keyframe_id,
    ]);
    expect(candidates.filter((item) => item.video_id === "K01_V002")).toHaveLength(2);
  });

  it("finds a hotspot location in grouped results", () => {
    const target = frame("K01_V002", 7, .7, 12);
    expect(findQaFrame([group("K01_V001", []), group("K01_V002", [target])], target.submit_keyframe_id))
      .toEqual({ video: 1, frame: 0 });
  });
});
