import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import { useStickyNotes } from "../hooks/useStickyNotes";
import {
  applyOrder,
  keepOnlyPack,
  noteKey,
  pushPlan,
  type NoteCandidate,
} from "../lib/stickyNotes";
import { planReorder } from "../lib/submissionOrder";

const BASE = Date.parse("2026-09-04T10:00:00.000Z");

function candidate(index: number, pushedAt: string | null = null): NoteCandidate {
  return {
    id: `note-${index}`,
    videoId: "K01_V001",
    frames: [index * 25],
    answer: "",
    keyframeIds: [`K01/K01_V001/${String(index).padStart(3, "0")}`],
    ptsTimes: [index],
    retrievalDatabase: "btc",
    createdAt: new Date(BASE + index).toISOString(),
    pushedAt,
  };
}

beforeEach(() => localStorage.clear());

describe("sticky note storage model", () => {
  it("separates questions inside a pack and removes drafts from old packs", () => {
    const store = {
      [noteKey("pack-a", "q-1")]: [candidate(1)],
      [noteKey("pack-a", "q-2")]: [candidate(2)],
      [noteKey("pack-b", "q-1")]: [candidate(3)],
    };

    expect(noteKey("pack-a", "q-1")).not.toBe(noteKey("pack-a", "q-2"));
    expect(Object.keys(keepOnlyPack(store, "pack-b"))).toEqual(["pack-b::q-1"]);
    // Startup has no pack briefly; that must never erase the last known draft.
    expect(keepOnlyPack(store, null)).toBe(store);
  });

  it("ignores malformed localStorage rows instead of taking Search down", async () => {
    localStorage.setItem("aic26_sticky_notes", JSON.stringify({
      "pack-a::q-1": { not: "an array" },
      "pack-a::q-2": [null, { frames: [25] }],
    }));
    const hook = renderHook(() => useStickyNotes("pack-a"));
    expect(hook.result.current.candidatesFor("q-1")).toEqual([]);
    expect(hook.result.current.candidatesFor("q-2")).toEqual([]);
  });

  it("pushes only candidates not sent before and keeps their order timestamps", () => {
    const list = [candidate(1, "2026-09-04T11:00:00.000Z"), candidate(2), candidate(3)];
    const plan = pushPlan(list);

    expect(plan.map((item) => item.id)).toEqual(["note-2", "note-3"]);
    expect(plan.map((item) => item.createdAt)).toEqual([
      new Date(BASE + 2).toISOString(),
      new Date(BASE + 3).toISOString(),
    ]);
  });

  it("uses the Submission ordering algorithm for draft reorder", () => {
    const list = [candidate(1), candidate(2), candidate(3)];
    const changes = planReorder(list, 2, 0);
    expect(applyOrder(list, changes).map((item) => item.id)).toEqual([
      "note-3",
      "note-1",
      "note-2",
    ]);
  });
});

describe("useStickyNotes", () => {
  it("persists per-pack candidates and only re-arms a pushed candidate after an edit", async () => {
    const { result, unmount } = renderHook(() => useStickyNotes("pack-a"));
    act(() => {
      result.current.add("q-1", {
        videoId: "K01_V001",
        frames: [25],
        answer: "",
        keyframeIds: ["K01/K01_V001/001"],
        ptsTimes: [1],
        retrievalDatabase: "btc",
      });
    });
    const id = result.current.candidatesFor("q-1")[0].id;
    act(() => result.current.markPushed("q-1", [id]));
    expect(result.current.pending("q-1")).toHaveLength(0);
    const firstCreatedAt = result.current.candidatesFor("q-1")[0].createdAt;

    act(() => result.current.update("q-1", id, { answer: "new answer" }));
    expect(result.current.pending("q-1").map((item) => item.id)).toEqual([id]);
    expect(result.current.candidatesFor("q-1")[0].createdAt).not.toBe(firstCreatedAt);
    await waitFor(() => expect(localStorage.getItem("aic26_sticky_notes")).toContain("new answer"));

    unmount();
    const restored = renderHook(() => useStickyNotes("pack-a"));
    expect(restored.result.current.candidatesFor("q-1")[0].answer).toBe("new answer");
  });

  it("cleans old local drafts when a different question pack becomes active", async () => {
    const hook = renderHook(({ pack }) => useStickyNotes(pack), {
      initialProps: { pack: "pack-a" as string | null },
    });
    act(() => {
      hook.result.current.add("same-question-id", {
        videoId: "K01_V001",
        frames: [25],
        answer: "",
        keyframeIds: [null],
        ptsTimes: [1],
        retrievalDatabase: "btc",
      });
    });

    hook.rerender({ pack: "pack-b" });
    await waitFor(() =>
      expect(hook.result.current.candidatesFor("same-question-id")).toHaveLength(0),
    );
    expect(localStorage.getItem("aic26_sticky_notes")).not.toContain("pack-a");
  });

  it("persists the movable window geometry", async () => {
    const first = renderHook(() => useStickyNotes("pack-a"));
    act(() => first.result.current.setWindow({ x: 211, y: 133, w: 640, h: 420, open: true }));
    await waitFor(() => expect(localStorage.getItem("aic26_sticky_window")).toContain('"x":211'));
    first.unmount();

    const second = renderHook(() => useStickyNotes("pack-a"));
    expect(second.result.current.window).toMatchObject({ x: 211, y: 133, w: 640, h: 420, open: true });
  });
});
