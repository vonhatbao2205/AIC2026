import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { ScopeCatalogue } from "../api/types";
import { ScopeFilter } from "../components/ScopeFilter";
import { toggleVideo } from "../lib/scope";

const N = Array.from({ length: 100 }, (_, i) => `N${String(i + 1).padStart(3, "0")}`);
const STAGES = Array.from({ length: 12 }, (_, i) => `S01-V${String(i + 1).padStart(3, "0")}`);
const CATALOGUE: ScopeCatalogue = {
  retrieval_database: "infoshotpp",
  categories: [
    ...["L23", ...N].map((category) => ({
      category, label_en: `Folder ${category}`, label_vi: `Thư mục ${category}`, open_subject: false,
    })),
    {
      category: "S01", label_en: "Cycling — 2026 Television Cup", label_vi: "Đua xe đạp", open_subject: false,
      videos: STAGES.map((video_id, i) => ({
        video_id, label_en: `Stage ${i + 1}`, label_vi: `Chặng ${i + 1}`, summary: `Summary of stage ${i + 1}`,
      })),
    },
  ],
  groups: [
    { id: "L", label_vi: "L", label_en: "L21–L30", categories: ["L23"], collapsed: false },
    { id: "N", label_vi: "N", label_en: "N001–N100 · traffic cameras (batch 2)", categories: N, collapsed: true },
    { id: "S", label_vi: "S", label_en: "S01 · cycling (batch 2)", categories: ["S01"], collapsed: true },
  ],
  topics: [],
};

describe("collapsed scope group", () => {
  it("shows only the group checkbox and still selects every folder in it", async () => {
    const user = userEvent.setup();
    const onSelected = vi.fn();
    render(
      <ScopeFilter
        catalogue={CATALOGUE}
        mode="manual"
        onMode={() => {}}
        selected={["L23"]}
        onSelected={onSelected}
        applied={null}
        error={null}
      />,
    );
    await user.click(screen.getByTestId("scope-toggle"));

    expect(screen.getByTestId("scope-cat-L23")).toBeInTheDocument();
    expect(screen.queryByTestId("scope-cat-N001")).not.toBeInTheDocument();
    await user.click(screen.getByTestId("scope-group-N"));
    expect(onSelected).toHaveBeenLastCalledWith(["L23", ...N]);
  });

  function renderManual(selected: string[], onSelected = vi.fn()) {
    render(
      <ScopeFilter
        catalogue={CATALOGUE}
        mode="manual"
        onMode={() => {}}
        selected={selected}
        onSelected={onSelected}
        applied={null}
        error={null}
      />,
    );
    return onSelected;
  }

  it("unfolds on demand so a single camera can be searched on its own", async () => {
    const user = userEvent.setup();
    const onSelected = vi.fn();
    render(
      <ScopeFilter
        catalogue={CATALOGUE}
        mode="manual"
        onMode={() => {}}
        selected={[]}
        onSelected={onSelected}
        applied={null}
        error={null}
      />,
    );
    await user.click(screen.getByTestId("scope-toggle"));
    const expand = screen.getByTestId("scope-expand-N");
    expect(expand).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByTestId("scope-expand-L")).not.toBeInTheDocument();

    await user.click(expand);
    expect(expand).toHaveAttribute("aria-expanded", "true");
    expect(screen.getAllByTestId(/^scope-cat-N\d{3}$/)).toHaveLength(100);
    expect(screen.getByTestId("scope-cat-N042").closest("label")).toHaveTextContent("Folder N042");

    await user.click(screen.getByTestId("scope-cat-N042"));
    expect(onSelected).toHaveBeenLastCalledWith(["N042"]);

    await user.click(expand);
    expect(screen.queryByTestId("scope-cat-N042")).not.toBeInTheDocument();
  });

  it("half-ticks a folded group and counts the folders picked inside it", async () => {
    const user = userEvent.setup();
    render(
      <ScopeFilter
        catalogue={CATALOGUE}
        mode="manual"
        onMode={() => {}}
        selected={["N007", "N042"]}
        onSelected={() => {}}
        applied={null}
        error={null}
      />,
    );
    await user.click(screen.getByTestId("scope-toggle"));

    const head = screen.getByTestId("scope-group-N") as HTMLInputElement;
    expect(head.checked).toBe(false);
    expect(head.indeterminate).toBe(true);
    expect(screen.getByTestId("scope-expand-N")).toHaveTextContent("2/100");
  });

  it("unfolds the race into its twelve stages so one stage can be searched alone", async () => {
    const user = userEvent.setup();
    const onSelected = renderManual([]);
    await user.click(screen.getByTestId("scope-toggle"));
    expect(screen.queryByTestId("scope-video-S01-V006")).not.toBeInTheDocument();
    expect(screen.getByTestId("scope-expand-S")).toHaveTextContent("12");

    await user.click(screen.getByTestId("scope-expand-S"));
    expect(screen.getAllByTestId(/^scope-video-S01-V\d{3}$/)).toHaveLength(12);
    const stage6 = screen.getByTestId("scope-video-S01-V006").closest("label")!;
    expect(stage6).toHaveTextContent("Stage 6");
    expect(stage6).toHaveAttribute("title", "Summary of stage 6");

    await user.click(screen.getByTestId("scope-video-S01-V006"));
    expect(onSelected).toHaveBeenLastCalledWith(["S01-V006"]);
  });

  it("shows a single stage as a partial race", async () => {
    const user = userEvent.setup();
    renderManual(["S01-V006"]);
    await user.click(screen.getByTestId("scope-toggle"));
    await user.click(screen.getByTestId("scope-expand-S"));

    expect(screen.getByTestId("scope-expand-S")).toHaveTextContent("1/12");
    expect((screen.getByTestId("scope-group-S") as HTMLInputElement).indeterminate).toBe(true);
    expect((screen.getByTestId("scope-cat-S01") as HTMLInputElement).indeterminate).toBe(true);
    expect(screen.getByTestId("scope-video-S01-V006")).toBeChecked();
    expect(screen.getByTestId("scope-video-S01-V007")).not.toBeChecked();
  });

  it("ticking the whole race drops the single stages it already covers", async () => {
    const user = userEvent.setup();
    const onSelected = renderManual(["L23", "S01-V006"]);
    await user.click(screen.getByTestId("scope-toggle"));

    await user.click(screen.getByTestId("scope-group-S"));
    expect(onSelected).toHaveBeenLastCalledWith(["L23", "S01"]);
  });
});

describe("toggleVideo", () => {
  it("unticking a stage out of the whole race keeps the other eleven", () => {
    expect(toggleVideo(["S01"], "S01", "S01-V006", CATALOGUE)).toEqual(STAGES.filter((id) => id !== "S01-V006"));
  });

  it("ticking the last missing stage folds them back into the folder", () => {
    const eleven = STAGES.filter((id) => id !== "S01-V012");
    expect(toggleVideo(["L23", ...eleven], "S01", "S01-V012", CATALOGUE)).toEqual(["L23", "S01"]);
  });

  it("unticks a stage picked on its own", () => {
    expect(toggleVideo(["S01-V006", "S01-V007"], "S01", "S01-V006", CATALOGUE)).toEqual(["S01-V007"]);
  });
});
