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

function candidate(index: number, pushState: NoteCandidate["pushState"] = "draft"): NoteCandidate {
  return {
    id: `note-${index}`,
    videoId: "K01_V001",
    frames: [index * 25],
    answer: "",
    keyframeIds: [`K01/K01_V001/${String(index).padStart(3, "0")}`],
    ptsTimes: [index],
    retrievalDatabase: "btc",
    createdAt: new Date(BASE + index).toISOString(),
    ...(pushState === "draft" ? {} : { submissionRowId: `submission-${index}` }),
    pushState,
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
    const list = [candidate(1, "synced"), candidate(2), candidate(3)];
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
  it("persists queued ids, confirms sync separately, and re-arms only after an edit", async () => {
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
    let queued: NoteCandidate[] = [];
    act(() => {
      queued = result.current.queue("q-1", [{
        candidateId: id,
        submissionRowId: "11111111-1111-4111-8111-111111111111",
      }]);
    });
    expect(queued[0]).toMatchObject({
      pushState: "queued",
      submissionRowId: "11111111-1111-4111-8111-111111111111",
    });
    expect(result.current.pending("q-1")).toHaveLength(0);
    expect(localStorage.getItem("aic26_sticky_notes")).toContain('"pushState":"queued"');
    act(() => result.current.markSynced("q-1", ["11111111-1111-4111-8111-111111111111"]));
    expect(result.current.candidatesFor("q-1")[0].pushState).toBe("synced");
    const firstCreatedAt = result.current.candidatesFor("q-1")[0].createdAt;

    act(() => result.current.update("q-1", id, { answer: "new answer" }));
    expect(result.current.pending("q-1").map((item) => item.id)).toEqual([id]);
    expect(result.current.candidatesFor("q-1")[0]).toMatchObject({
      pushState: "draft",
      answer: "new answer",
    });
    expect(result.current.candidatesFor("q-1")[0]).not.toHaveProperty("submissionRowId");
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
