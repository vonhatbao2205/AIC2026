import { act, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SearchThinking } from "../components/SearchThinking";

describe("SearchThinking", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-01-01T00:00:00Z"));
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.useRealTimers();
  });

  it("starts with a quiet two-line search status", () => {
    render(<SearchThinking />);

    expect(screen.getByRole("status", { name: "Searching" })).toBeInTheDocument();
    const logo = screen.getByTestId("search-thinking-logo").querySelector("img");
    expect(logo).toBeInTheDocument();
    expect(logo?.getAttribute("src")).toContain("chatgpt-seeklogo.png");
    expect(screen.queryByText("✦")).not.toBeInTheDocument();
    expect(screen.getByTestId("search-thinking-verb")).toHaveTextContent("Pondering…");
    expect(screen.getByTestId("search-thinking-time")).toHaveTextContent("0s");
    expect(screen.getByTestId("search-thinking-subline")).toHaveTextContent("Looking at the clues…");
  });

  it("rotates the decorative verb without immediately repeating it", () => {
    vi.spyOn(Math, "random").mockReturnValue(0);
    render(<SearchThinking />);

    act(() => vi.advanceTimersByTime(1_800));

    expect(screen.getByTestId("search-thinking-verb")).toHaveTextContent("Ruminating…");
  });

  it("advances the subline more slowly and shows elapsed time", () => {
    render(<SearchThinking />);

    act(() => vi.advanceTimersByTime(9_000));

    expect(screen.getByTestId("search-thinking-time")).toHaveTextContent("9s");
    expect(screen.getByTestId("search-thinking-subline")).toHaveTextContent("Connecting a few possibilities…");
  });
});
