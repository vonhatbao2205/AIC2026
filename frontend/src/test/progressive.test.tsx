import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ProgressivePanel } from "../components/ProgressivePanel";
import App from "../App";
import { api } from "../api/client";
import type { FrameResult, VideoGroup, ProgressiveConfig, ProgressiveSnapshot } from "../api/types";

const config: ProgressiveConfig = { retrieval_database: "btc", task_id: "manual", image_models: ["pe"], scope: { mode: "all", categories: [] }, translate: false, hybrid: false, top_k: 200 };
const snapshot = (id: string, revision = 0): ProgressiveSnapshot => ({ session_id: id, revision, committed_revision: revision, pending_revision: null, turn: revision, groups: [] });
afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); });

describe("PHM session ownership", () => {
  it("keeps the selected frame by ID when both video and frame rankings move", async () => {
    localStorage.clear();
    vi.stubGlobal("fetch", vi.fn(async (url: string) => {
      const body = url.includes("/api/config") ? { configured: true }
        : url.includes("/api/health") ? { ok: true, mode: "mock", capabilities: {}, warnings: [], services: {} }
        : url.includes("/api/submit/history") ? { history: [] }
        : { categories: [], groups: [], topics: [] };
      return { ok: true, json: async () => body };
    }));
    const frame = (v: string, n: number): FrameResult => ({ image_id: `${v}/${n}`, submit_keyframe_id: `${v}/${n}`, video_id: v, keyframe_n: n, frame_idx: n * 25, fps: 25, pts_time: n, score: 1 / n, channels: [], per_channel_score: {}, keyframe_url: "", video_url: "", evidence: [] });
    const f1 = frame("A", 1), f2 = frame("A", 2), fb = frame("B", 1);
    const group = (v: string, frames: FrameResult[]): VideoGroup => ({ video_id: v, frames, video_score: 1, max_score: 1, mean_top_score: 1, frame_count: frames.length, timestamp_dispersion: 0, ambiguous: false, channels: [], video_url: "", progressive: { trajectory: [1], memory_score: 1, cumulative_rank: 1, dispersed: false, moment_source: "cumulative", hint_evidence: [] } });
    vi.spyOn(api, "createProgressive").mockResolvedValue(snapshot("one"));
    vi.spyOn(api, "closeProgressive").mockResolvedValue({ closed: true });
    vi.spyOn(api, "updateProgressive").mockResolvedValueOnce({ ...snapshot("one", 1), groups: [group("A", [f1, f2]), group("B", [fb])] })
      .mockResolvedValueOnce({ ...snapshot("one", 2), groups: [group("B", [fb]), group("A", [f2, f1])] });
    render(<App />);
    fireEvent.click(screen.getByRole("checkbox", { name: "Progressive Hint Memory" }));
    fireEvent.change(screen.getByLabelText("Hint mới"), { target: { value: "first hint" } });
    fireEvent.click(screen.getByRole("button", { name: /Áp dụng/ }));
    await waitFor(() => expect(screen.getAllByTestId("frame-thumb")).toHaveLength(3));
    fireEvent.click(screen.getAllByTestId("frame-thumb")[1]);
    expect(screen.getByTestId("submit-id")).toHaveTextContent("A/2");
    // Parking a tab must not destroy the panel's session or its ledger.
    fireEvent.click(screen.getByTestId("rail-add"));
    expect(api.closeProgressive).not.toHaveBeenCalled();
    fireEvent.click(screen.getByTestId("rail-tab-0"));
    expect(screen.getByLabelText("Nội dung hint 1")).toHaveValue("first hint");
    fireEvent.change(screen.getByLabelText("Hint mới"), { target: { value: "second hint" } });
    fireEvent.click(screen.getByRole("button", { name: /Áp dụng/ }));
    await waitFor(() => expect(screen.getByText(/lượt 2 · revision 2/)).toBeInTheDocument());
    expect(screen.getByTestId("submit-id")).toHaveTextContent("A/2");
    expect(screen.getAllByTestId("video-group")[1]).toHaveClass("selected");
    expect(screen.getAllByTestId("frame-thumb")[1]).toHaveClass("selected");
  });

  it("edits the same hint ID, disables it and applies the full ledger", async () => {
    vi.spyOn(api, "createProgressive").mockResolvedValue(snapshot("one"));
    vi.spyOn(api, "closeProgressive").mockResolvedValue({ closed: true });
    const update = vi.spyOn(api, "updateProgressive").mockImplementation(async (_, body) => snapshot("one", body.expected_revision + 1));
    const accept = vi.fn();
    render(<ProgressivePanel config={config} initialText="red car" onSnapshot={accept} onBusy={() => {}} />);
    fireEvent.click(screen.getByRole("button", { name: /Áp dụng/ }));
    await waitFor(() => expect(accept).toHaveBeenCalledTimes(1));
    const id = update.mock.calls[0][1].hints[0].hint_id;
    fireEvent.change(screen.getByLabelText("Nội dung hint 1"), { target: { value: "blue car" } });
    fireEvent.click(screen.getByRole("button", { name: /Áp dụng/ }));
    await waitFor(() => expect(accept).toHaveBeenCalledTimes(2));
    expect(update.mock.calls[1][1].hints[0]).toMatchObject({ hint_id: id, raw_text: "blue car" });
    expect(update.mock.calls[1][1].expected_revision).toBe(1);
    fireEvent.click(screen.getByRole("checkbox"));
    fireEvent.click(screen.getByRole("button", { name: /Áp dụng/ }));
    await waitFor(() => expect(accept).toHaveBeenCalledTimes(3));
    expect(update.mock.calls[2][1].hints[0].enabled).toBe(false);
  });

  it("retries a lost response using the same request ID and body", async () => {
    vi.spyOn(api, "createProgressive").mockResolvedValue(snapshot("one"));
    vi.spyOn(api, "closeProgressive").mockResolvedValue({ closed: true });
    const update = vi.spyOn(api, "updateProgressive").mockRejectedValueOnce(new TypeError("network"))
      .mockResolvedValueOnce(snapshot("one", 1));
    render(<ProgressivePanel config={config} initialText="red car" onSnapshot={() => {}} onBusy={() => {}} />);
    fireEvent.click(screen.getByRole("button", { name: /Áp dụng/ }));
    fireEvent.click(await screen.findByRole("button", { name: "Retry request" }));
    await waitFor(() => expect(update).toHaveBeenCalledTimes(2));
    expect(update.mock.calls[0][1]).toEqual(update.mock.calls[1][1]);
  });

  it("aborts and closes an unmounted session; its late response cannot touch the new tab", async () => {
    vi.spyOn(api, "createProgressive").mockResolvedValueOnce(snapshot("old")).mockResolvedValueOnce(snapshot("new"));
    const close = vi.spyOn(api, "closeProgressive").mockResolvedValue({ closed: true });
    let resolveOld!: (s: ProgressiveSnapshot) => void;
    const old = new Promise<ProgressiveSnapshot>(r => { resolveOld = r; });
    const update = vi.spyOn(api, "updateProgressive").mockReturnValueOnce(old).mockResolvedValueOnce(snapshot("new", 1));
    const accept = vi.fn();
    const view = render(<ProgressivePanel key="old" config={config} initialText="old hint" onSnapshot={accept} onBusy={() => {}} />);
    fireEvent.click(screen.getByRole("button", { name: /Áp dụng/ }));
    await waitFor(() => expect(update).toHaveBeenCalledTimes(1));
    view.rerender(<ProgressivePanel key="new" config={config} initialText="new hint" onSnapshot={accept} onBusy={() => {}} />);
    expect(close).toHaveBeenCalledWith("old");
    expect(update.mock.calls[0][2]?.aborted).toBe(true);
    fireEvent.click(screen.getByRole("button", { name: /Áp dụng/ }));
    await waitFor(() => expect(accept).toHaveBeenCalledTimes(1));
    await act(async () => resolveOld(snapshot("old", 1)));
    expect(accept).toHaveBeenCalledTimes(1);
    expect(accept.mock.calls[0][0].session_id).toBe("new");
  });
});
