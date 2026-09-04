import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { StickyNoteWindow } from "../components/StickyNoteWindow";
import type { ImportedQuestion } from "../lib/questions";
import { applyOrder, type NoteCandidate } from "../lib/stickyNotes";

const QUESTION: ImportedQuestion = {
  id: "query-pack-1-kis",
  order: 1,
  kind: "kis",
  queryType: "T-KIS",
  text: "find a person",
  eventCount: null,
};

const candidates: NoteCandidate[] = [1, 2, 3].map((index) => ({
  id: `candidate-${index}`,
  videoId: "K01_V001",
  frames: [index * 25],
  answer: "",
  keyframeIds: index === 1 ? [null] : [`K01/K01_V001/00${index}`],
  ptsTimes: [index],
  retrievalDatabase: "btc",
  createdAt: new Date(Date.parse("2026-09-04T10:00:00.000Z") + index).toISOString(),
  pushState: "draft",
}));

function dataTransfer() {
  const values = new Map<string, string>();
  const types: string[] = [];
  return {
    types,
    effectAllowed: "",
    dropEffect: "",
    setData(type: string, value: string) {
      if (!values.has(type)) types.push(type);
      values.set(type, value);
    },
    getData(type: string) {
      return values.get(type) ?? "";
    },
    setDragImage() {},
  };
}

function pointer(type: string, pointerId: number, clientX: number, clientY: number): Event {
  const event = new Event(type, { bubbles: true, cancelable: true });
  Object.defineProperties(event, {
    pointerId: { value: pointerId },
    clientX: { value: clientX },
    clientY: { value: clientY },
  });
  return event;
}

function renderNote(overrides: Partial<React.ComponentProps<typeof StickyNoteWindow>> = {}) {
  const props: React.ComponentProps<typeof StickyNoteWindow> = {
    question: QUESTION,
    candidates,
    submissionCount: 0,
    window: { x: 120, y: 100, w: 620, h: 400, open: true },
    defaultRetrievalDatabase: "btc",
    onWindow: vi.fn(),
    onAdd: vi.fn(),
    onUpdate: vi.fn(),
    onRemove: vi.fn(),
    onReorder: vi.fn(),
    onClear: vi.fn(),
    onPush: vi.fn(),
    onRestoreTrake: vi.fn(),
    ...overrides,
  };
  return { ...render(<StickyNoteWindow {...props} />), props };
}

beforeEach(() => {
  vi.stubGlobal("fetch", vi.fn(async () => ({
    ok: true,
    status: 200,
    json: async () => ({
      video_id: "K01_V001",
      video_url: "https://media.test/K01_V001.mp4",
      fps: 25,
      duration: 20,
      keyframes: [],
      speech_segments: [],
      ocr_markers: [],
      audio_windows: [],
      heatmap: [],
    }),
  })));
});

afterEach(() => vi.unstubAllGlobals());

describe("StickyNoteWindow", () => {
  it("previews raw frames and opens the same video frame editor as Submission", async () => {
    renderNote();
    fireEvent.click(screen.getByTestId("sticky-row-0"));

    fireEvent.keyDown(window, { key: "p" });
    expect(await screen.findByTestId("preview-raw")).toHaveTextContent("frame 25");
    fireEvent.keyDown(window, { key: "p" });
    await waitFor(() => expect(screen.queryByTestId("preview-raw")).not.toBeInTheDocument());

    fireEvent.keyDown(window, { key: "v" });
    expect(await screen.findByTestId("frame-editor")).toHaveTextContent("K01_V001");
    expect(screen.getByTestId("editor-original")).toHaveTextContent("frame 25");
  });

  it("commits local cell edits, clears stale keyframe provenance, and deletes locally", () => {
    const onUpdate = vi.fn();
    const onRemove = vi.fn();
    renderNote({ onUpdate, onRemove });
    const row = screen.getByTestId("sticky-row-1");
    const [video, frames] = within(row).getAllByRole("textbox");

    fireEvent.change(video, { target: { value: "K02_V009" } });
    fireEvent.change(frames, { target: { value: "999" } });
    fireEvent.blur(frames);

    expect(onUpdate).toHaveBeenCalledWith("candidate-2", expect.objectContaining({
      videoId: "K02_V009",
      frames: [999],
      keyframeIds: [null],
      ptsTimes: [null],
    }));
    fireEvent.click(within(row).getByRole("button", { name: "Xoá candidate 2" }));
    expect(onRemove).toHaveBeenCalledWith("candidate-2");
  });

  it("reorders candidates by drag and emits the exact Submission-style plan", () => {
    const onReorder = vi.fn();
    renderNote({ onReorder });
    const transfer = dataTransfer();
    fireEvent.mouseDown(screen.getByTestId("sticky-grip-2"));
    fireEvent.dragStart(screen.getByTestId("sticky-row-2"), { dataTransfer: transfer });
    fireEvent.dragOver(screen.getByTestId("sticky-row-0"), { dataTransfer: transfer });
    fireEvent.drop(screen.getByTestId("sticky-row-0"), { dataTransfer: transfer });

    expect(onReorder).toHaveBeenCalledTimes(1);
    expect(applyOrder(candidates, onReorder.mock.calls[0][0]).map((item) => item.id)).toEqual([
      "candidate-3",
      "candidate-1",
      "candidate-2",
    ]);
  });

  it("reports persisted geometry patches while moving and resizing", () => {
    const onWindow = vi.fn();
    renderNote({ onWindow });
    const header = document.querySelector(".sticky-note-head") as HTMLElement;
    fireEvent(header, pointer("pointerdown", 1, 130, 110));
    fireEvent(header, pointer("pointermove", 1, 180, 150));
    expect(onWindow).toHaveBeenCalledWith({ x: 170, y: 140 });

    const resize = document.querySelector(".sticky-note-resize") as HTMLElement;
    fireEvent(resize, pointer("pointerdown", 2, 740, 500));
    fireEvent(resize, pointer("pointermove", 2, 800, 550));
    expect(onWindow).toHaveBeenCalledWith({ w: 680, h: 450 });
  });

  it("shows queued separately and displays a check only after server sync", () => {
    const queued: NoteCandidate = {
      ...candidates[0],
      pushState: "queued",
      submissionRowId: "11111111-1111-4111-8111-111111111111",
    };
    const synced: NoteCandidate = {
      ...candidates[1],
      pushState: "synced",
      submissionRowId: "22222222-2222-4222-8222-222222222222",
    };
    renderNote({ candidates: [queued, synced] });

    expect(screen.getByTitle("Đang chờ Supabase xác nhận; hệ thống sẽ tự thử lại")).toHaveTextContent("◌");
    expect(screen.getByTitle("Supabase đã nhận; sửa nội dung sẽ đưa candidate vào lượt push kế tiếp")).toHaveTextContent("✓");
    expect(screen.getByTestId("sticky-push")).toBeDisabled();
  });

  it("offers a reverse load action for a TRAKE candidate", () => {
    const onRestoreTrake = vi.fn();
    renderNote({
      question: { ...QUESTION, kind: "trake", queryType: "TRAKE", eventCount: 1 },
      onRestoreTrake,
    });

    fireEvent.click(screen.getByTestId("sticky-restore-trake-0"));
    expect(onRestoreTrake).toHaveBeenCalledWith(candidates[0]);
  });
});
