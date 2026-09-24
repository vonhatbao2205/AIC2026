import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AvsPanel } from "../components/AvsPanel";
import { api } from "../api/client";
import type { AnswerGenerateResponse, FrameResult, SubmitEntry, SubmitPreview, VideoGroup } from "../api/types";
import { autoEvaluationId } from "../lib/dres";

const frame: FrameResult = { image_id: "A/1", submit_keyframe_id: "A/1", video_id: "A", keyframe_n: 1,
  frame_idx: 100, fps: 25, pts_time: 4, score: 1, channels: [], per_channel_score: {}, evidence: [], keyframe_url: "", video_url: "A.mp4" };
const groups = [{ video_id: "A", frames: [frame] }] as VideoGroup[];
const props = { groups, query: "red cars", database: "infoshotpp" as const, imageModels: ["pe" as const],
  scopeId: "avs-1", dresEnabled: false, evaluationId: null, taskName: "", selected: frame, onInspect: vi.fn() };
beforeEach(() => localStorage.clear());
afterEach(() => vi.restoreAllMocks());

describe("AVS review and submission", () => {
  it("persists accepted moments across query refinements and warns about exact/near duplicates", () => {
    const view = render(<AvsPanel {...props} />);
    fireEvent.click(screen.getByText("Accept selected frame"));
    expect(screen.getByText("Accepted basket · 1")).toBeInTheDocument();
    fireEvent.click(screen.getByText("Accept selected frame"));
    expect(screen.getByRole("alert")).toHaveTextContent("Duplicate:");
    view.rerender(<AvsPanel {...props} query="blue cars" selected={{ ...frame, frame_idx: 125, pts_time: 5 }} />);
    fireEvent.click(screen.getByText("Accept selected frame"));
    expect(screen.getByRole("alert")).toHaveTextContent("within 2 seconds");
    fireEvent.click(screen.getByText("Accept nearby moment anyway"));
    expect(screen.getByText("Accepted basket · 2")).toBeInTheDocument();
    view.unmount();
    render(<AvsPanel {...props} />);
    expect(screen.getByText("Accepted basket · 2")).toBeInTheDocument();
  });

  it("keeps different tasks' baskets separate", () => {
    const view = render(<AvsPanel key="a" {...props} />);
    fireEvent.click(screen.getByText("Accept selected frame"));
    view.rerender(<AvsPanel key="b" {...props} scopeId="avs-2" />);
    expect(screen.getByText("Accepted basket · 0")).toBeInTheDocument();
    view.rerender(<AvsPanel key="a" {...props} />);
    expect(screen.getByText("Accepted basket · 1")).toBeInTheDocument();
  });

  it("discards an old queue response after results change", async () => {
    let resolve!: (r: AnswerGenerateResponse) => void;
    vi.spyOn(api, "generateAnswers").mockImplementation(() => new Promise(r => { resolve = r; }));
    const view = render(<AvsPanel {...props} />);
    fireEvent.click(screen.getByText("Prepare review queue"));
    expect(api.generateAnswers).toHaveBeenCalledWith(expect.objectContaining({ query_type_hint: "AVS", groups }));
    view.rerender(<AvsPanel {...props} groups={[]} />);
    await act(async () => resolve({ answers: [{ video_id: "A", frames: [100] }] } as AnswerGenerateResponse));
    expect(screen.getByText("Review queue · 0")).toBeInTheDocument();
  });

  it("requires a preview and explicit confirmation before posting; records the returned verdict", async () => {
    const preview = { evaluation_id: "e1", task_name: "avs-01", answer_mode_mismatch: false,
      duplicate_key: null, warnings: [], body: { answerSets: [{ taskName: "avs-01", answers: [{ mediaItemName: "A", start: 4000, end: 4000 }] }] } } as unknown as SubmitPreview;
    vi.spyOn(api, "submitPreview").mockResolvedValue(preview);
    vi.spyOn(api, "submit").mockResolvedValue({ status: "dres_ok", verdict: "CORRECT" } as SubmitEntry);
    render(<AvsPanel {...props} dresEnabled evaluationId="e1" taskName="avs-01" />);
    fireEvent.click(screen.getByText("Accept selected frame"));
    expect(api.submit).not.toHaveBeenCalled();
    fireEvent.click(screen.getByText("Preview DRES"));
    const modal = await screen.findByRole("dialog");
    expect(modal).toHaveTextContent('"start": 4000');
    expect(api.submit).not.toHaveBeenCalled();
    fireEvent.click(within(modal).getByText("Confirm AVS submission"));
    await waitFor(() => expect(screen.getByText("DRES: CORRECT")).toBeInTheDocument());
    expect(api.submit).toHaveBeenCalledOnce();
    expect(api.submit).toHaveBeenCalledWith(expect.objectContaining({ query_type: "AVS", require_dres: true, expected_task_name: "avs-01" }));
    expect(screen.getByText("Preview DRES")).toBeDisabled();
  });

  it("routes AVS using task templates when run names are generic", () => {
    expect(autoEvaluationId([
      { id: "e1", name: "final", status: "ACTIVE", task_templates: [{ taskType: "Ad-hoc Video Search" }] },
      { id: "e2", name: "textual", status: "ACTIVE", task_templates: [] },
    ] as never, "AVS")).toBe("e1");
  });
});
