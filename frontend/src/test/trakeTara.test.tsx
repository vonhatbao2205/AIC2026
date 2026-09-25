import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { TrakeHeatPeak, TrakeVideoResult } from "../api/types";
import { TrakeVideoResults } from "../components/TrakeVideoResults";

function peak(event_index: number, pts_time: number, channels: TrakeHeatPeak["channels"]): TrakeHeatPeak {
  return {
    event_index,
    submit_keyframe_id: `N042/N042-V002/${String(pts_time).padStart(3, "0")}`,
    keyframe_n: pts_time,
    frame_idx: pts_time * 25,
    pts_time,
    score: 0.02,
    strength: 0.8,
    via_fill: false,
    selected_by_dp: true,
    keyframe_url: "https://media.test/k.jpg",
    channels,
  };
}

const video: TrakeVideoResult = {
  video_id: "N042-V002",
  video_url: "https://media.test/N042-V002.mp4",
  trake_video_score: 0.9,
  coverage: 2,
  confident_coverage: 2,
  filled_events: 0,
  evidence_coverage: 2,
  chain_quality: 0.8,
  mean_quality: 0.8,
  min_quality: 0.8,
  complete: true,
  warning: null,
  min_event_gap: 20,
  duration_s: 120,
  preliminary_rank: 1,
  events: [
    { event_index: 1, has_candidate: true, in_chain: true, representative: peak(1, 10, ["tara"]), peaks: [peak(1, 10, ["tara"])] },
    { event_index: 2, has_candidate: true, in_chain: true, representative: peak(2, 30, ["image_pe"]), peaks: [peak(2, 30, ["image_pe"])] },
  ],
  best_chain: [],
};

describe("TRAKE with TARA", () => {
  it("tags the events a TARA clip found", () => {
    render(
      <TrakeVideoResults
        videos={[video]}
        eventCount={2}
        selectedVideoId={null}
        activeEventIndex={1}
        overrides={{}}
        loading={false}
        onSelectVideo={() => {}}
        onPickMoment={() => {}}
        onOverrideEvent={() => {}}
        onClearOverride={() => {}}
        onQuickSubmit={() => {}}
      />,
    );
    expect(screen.getByTestId("trake-event-tara-1")).toHaveTextContent("TARA");
    expect(screen.queryByTestId("trake-event-tara-2")).not.toBeInTheDocument();
  });
});
