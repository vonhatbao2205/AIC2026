import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { DetailPanel } from "../components/DetailPanel";
import { QueryUnderstanding } from "../components/QueryUnderstanding";
import type { FrameResult, ParsedQuery, TrafficFilterInfo } from "../api/types";
import { overlayText } from "../lib/overlay";

const parsed = {
  query_type: "T-KIS",
  confidence: 0.6,
  original_query: "",
  normalized_vi: "",
  translated_en_visual: "",
  operator_summary_vi: "",
  channels: {},
  filters: { video_ids: [], categories: [], must_include: [], must_not_include: [] },
} as unknown as ParsedQuery;

const traffic: TrafficFilterInfo = {
  mode: "auto",
  active: true,
  streets: ["Nguyễn Trãi", "Cống Quỳnh"],
  cameras: [
    { id: "nguyentraicongquynh1", label: "Nguyễn Trãi – Cống Quỳnh (1)", banner: "Nguyen Trai - Cong Quynh 1", videos: ["N092-V002"] },
  ],
  dates: ["2026-06-15"],
  time: { from: "18:55", to: "20:04", text: "19 giờ" },
  race_stage: null,
  videos: 1,
  keyframes: 214,
  warnings: ["No traffic camera recorded on 03/02; date ignored."],
  reason_en: "Traffic cameras: 1 camera(s) … Other folders are not affected.",
};

describe("traffic-camera filter", () => {
  it("shows what the query's cues narrowed and switches them off for the next search", () => {
    const onMode = vi.fn();
    const { rerender } = render(
      <QueryUnderstanding parsed={parsed} traffic={traffic} trafficMode="auto" onTrafficMode={onMode} />,
    );
    expect(screen.getByText(/Nguyễn Trãi – Cống Quỳnh \(1\)/)).toBeInTheDocument();
    expect(screen.getByText(/2026-06-15/)).toBeInTheDocument();
    expect(screen.getByText(/18:55–20:04/)).toBeInTheDocument();
    expect(screen.getByText(/03\/02; date ignored/)).toBeInTheDocument();
    fireEvent.click(screen.getByTestId("traffic-toggle"));
    expect(onMode).toHaveBeenCalledWith("off");

    rerender(<QueryUnderstanding parsed={parsed} traffic={traffic} trafficMode="off" onTrafficMode={onMode} />);
    expect(screen.getByText(/Applies to the next/)).toBeInTheDocument();
    expect(screen.getByTestId("traffic-toggle")).toHaveTextContent("Apply");
  });

  it("stays out of the way when the query names no camera, date, time or stage", () => {
    const idle = { ...traffic, active: false, cameras: [], dates: [], time: null, warnings: [] };
    render(<QueryUnderstanding parsed={parsed} traffic={idle} trafficMode="auto" onTrafficMode={vi.fn()} />);
    expect(screen.queryByTestId("traffic-filter")).toBeNull();
  });

  it("puts the banner reading of a camera frame next to its keyframe", () => {
    const frame = {
      image_id: "N092/N092-V002/111",
      submit_keyframe_id: "N092/N092-V002/111",
      video_id: "N092-V002",
      keyframe_n: 111,
      frame_idx: 5000,
      fps: 25,
      pts_time: 200,
      score: 0.1,
      channels: ["ocr"],
      per_channel_score: { ocr: 1 },
      keyframe_url: "",
      video_url: "",
      evidence: [{ type: "ocr", score: 3, text: "CHAGEE", banner_camera: "Nguyen Trai - Cong Quynh 1", text_ticker: "tin khác" }],
      overlay: { camera: "Nguyễn Trãi – Cống Quỳnh (1)", banner_date: "2026-06-15", clock: "19:05:08" },
    } as FrameResult;
    render(<DetailPanel frame={frame} queryType="T-KIS" onSubmit={vi.fn()} onAddToSticky={vi.fn()} onCopyId={vi.fn()} />);
    expect(screen.getByTestId("frame-overlay")).toHaveTextContent("Nguyễn Trãi – Cống Quỳnh (1) · 2026-06-15 · 19:05:08");
    expect(screen.getByText(/ticker: tin khác/)).toBeInTheDocument();
    expect(screen.getByText(/🚦Nguyen Trai - Cong Quynh 1/)).toBeInTheDocument();
  });

  it("formats a race HUD reading", () => {
    expect(overlayText({ race_stage: 6, race_time: "01:23:45" })).toBe("Stage 6 · race 01:23:45");
    expect(overlayText(undefined)).toBe("");
  });
});
