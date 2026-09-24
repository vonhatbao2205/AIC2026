import { act, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "../api/client";
import type { ParsedQuery, PeLiveReport, PeQueryReport } from "../api/types";
import { PeTokenMeter } from "../components/PeTokenMeter";
import { QueryUnderstanding } from "../components/QueryUnderstanding";

const base: PeLiveReport = {
  context_length: 72, limit: 70, exact: true, text: "", tokens: 0, range: null,
  kept: "", dropped: "", at_risk: "", status: "ok",
};

afterEach(() => {
  vi.restoreAllMocks();
  vi.useRealTimers();
});

async function settle() {
  await act(async () => {
    vi.advanceTimersByTime(400);
  });
  await act(async () => {});
}

describe("PE token meter", () => {
  it("counts once the operator pauses, with hints in front, and shows what PE ignores", async () => {
    vi.useFakeTimers();
    const spy = vi.spyOn(api, "peTokens").mockResolvedValue({
      ...base, tokens: 94, status: "over", dropped: "phía sau là xe hỗ trợ",
    });
    const { rerender } = render(<PeTokenMeter query="một" hints={["gợi ý"]} translate={false} active />);
    rerender(<PeTokenMeter query="một người" hints={["gợi ý"]} translate={false} active />);
    await settle();
    // Debounced: only the text the operator paused on is counted.
    expect(spy).toHaveBeenCalledTimes(1);
    expect(spy.mock.calls[0][0]).toEqual({ query: "một người", previous_hints: ["gợi ý"], translate: false });
    expect(screen.getByTestId("pe-meter")).toHaveTextContent("PE 94/70 tokens · PE ignores: “phía sau là xe hỗ trợ”");
    expect(screen.getByTestId("pe-meter")).toHaveClass("over");
  });

  it("marks an estimate for the translated query and what is at risk", async () => {
    vi.useFakeTimers();
    vi.spyOn(api, "peTokens").mockResolvedValue({
      ...base, exact: false, tokens: 69, range: [58, 91], status: "may_exceed", at_risk: "hai bên đường",
    });
    render(<PeTokenMeter query="một câu dài" translate active llm />);
    await settle();
    const meter = screen.getByTestId("pe-meter");
    expect(meter).toHaveTextContent("PE ≈69/70 tokens (58–91)");
    expect(meter).toHaveTextContent("may exceed after translation — at risk: “hai bên đường”");
    expect(meter).toHaveTextContent("the LLM rewrites the visual query");
  });

  it("stays silent when PE is not searched", async () => {
    vi.useFakeTimers();
    const spy = vi.spyOn(api, "peTokens").mockResolvedValue({ ...base, tokens: 5 });
    render(<PeTokenMeter query="a man" translate active={false} />);
    await settle();
    expect(spy).not.toHaveBeenCalled();
    expect(screen.queryByTestId("pe-meter")).toBeNull();
  });
});

describe("PE report after a search", () => {
  const parsed = {
    query_type: "T-KIS", confidence: 0.6, original_query: "", normalized_vi: "", translated_en_visual: "",
    operator_summary_vi: "", channels: {},
    filters: { video_ids: [], categories: [], must_include: [], must_not_include: [] },
  } as unknown as ParsedQuery;

  it("says which part of the sent query PE never read", () => {
    const report: PeQueryReport = {
      context_length: 72, limit: 70, truncated: 1,
      queries: [{ text: "…", tokens: 94, limit: 70, truncated: true, kept: "…", dropped: "a white support car behind" }],
    };
    render(<QueryUnderstanding parsed={parsed} peTokens={report} />);
    expect(screen.getByTestId("pe-report")).toHaveTextContent("read only the first 70 tokens of 1/1 visual query");
    expect(screen.getByTestId("pe-report")).toHaveTextContent("94/70 tokens · ignored: “a white support car behind”");
  });

  it("says nothing when every query fit", () => {
    const report: PeQueryReport = {
      context_length: 72, limit: 70, truncated: 0,
      queries: [{ text: "a man", tokens: 2, limit: 70, truncated: false, kept: "a man", dropped: "" }],
    };
    render(<QueryUnderstanding parsed={parsed} peTokens={report} />);
    expect(screen.queryByTestId("pe-report")).toBeNull();
  });
});
