import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { ScopeCatalogue } from "../api/types";
import { ScopeFilter } from "../components/ScopeFilter";

const N = Array.from({ length: 100 }, (_, i) => `N${String(i + 1).padStart(3, "0")}`);
const CATALOGUE: ScopeCatalogue = {
  retrieval_database: "infoshotpp",
  categories: ["L23", ...N].map((category) => ({
    category, label_en: `Folder ${category}`, label_vi: `Thư mục ${category}`, open_subject: false,
  })),
  groups: [
    { id: "L", label_vi: "L", label_en: "L21–L30", categories: ["L23"], collapsed: false },
    { id: "N", label_vi: "N", label_en: "N001–N100 · traffic cameras (batch 2)", categories: N, collapsed: true },
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
});
