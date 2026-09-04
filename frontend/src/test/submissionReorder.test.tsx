/**
 * Dragging an answer to a new rank.
 *
 * Rank is not decoration: the preliminary round scores
 * `Final = (R@1 + R@5 + R@20 + R@50 + R@100) / 5`, a max over the first k
 * answers, so the same frame at rank 1 scores the whole query and at rank 100
 * scores a fifth of it. The panel therefore has to hand the shared store a plan
 * that produces EXACTLY the order the operator dropped the row into — and has to
 * refuse anything that would re-attribute an answer instead of reordering one.
 */
import { fireEvent, render, screen, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { SubmissionPanel } from "../components/SubmissionPanel";
import type { OrderChange } from "../lib/submissionOrder";
import { sortRows } from "../lib/sharedSubmission";
import type { SubmissionRow } from "../lib/submission";
import type { ImportedQuestion } from "../lib/questions";

const QUESTION: ImportedQuestion = {
  id: "query-p1-3-kis",
  order: 3,
  kind: "kis",
  queryType: "T-KIS",
  text: "một người leo lên cột điện",
  eventCount: null,
};

const OTHER: ImportedQuestion = { ...QUESTION, id: "query-p1-4-kis", order: 4 };

const BASE = Date.parse("2026-09-04T10:00:00.000Z");

function answers(count: number, questionId = QUESTION.id): SubmissionRow[] {
  return Array.from({ length: count }, (_, index) => ({
    id: `${questionId}-row-${index + 1}`,
    questionId,
    videoId: "L27_V013",
    frames: [18000 + index],
    answer: "",
    source: "generated" as const,
    createdAt: new Date(BASE + index).toISOString(),
    syncState: "synced" as const,
  }));
}

/** Minimal stand-in for the browser's DataTransfer — jsdom has none. */
function dataTransfer() {
  const store = new Map<string, string>();
  const types: string[] = [];
  return {
    types,
    effectAllowed: "",
    dropEffect: "",
    setData(type: string, value: string) {
      if (!store.has(type)) types.push(type);
      store.set(type, value);
    },
    getData(type: string) {
      return store.get(type) ?? "";
    },
    setDragImage() {},
  };
}

const onReorderRows = vi.fn();

function renderPanel(rows: SubmissionRow[], questions: ImportedQuestion[] = [QUESTION]) {
  return render(
    <SubmissionPanel
      questions={questions}
      rows={sortRows(rows)}
      onChangeRow={vi.fn()}
      onReorderRows={onReorderRows}
      onDeleteRow={vi.fn()}
      onAddRow={vi.fn()}
      onCloneRow={vi.fn()}
      onClearQuestion={vi.fn()}
      onFillAnswer={vi.fn()}
      onExport={vi.fn()}
      onJumpToQuestion={vi.fn()}
      exportError={null}
      onBack={vi.fn()}
      backLabel="T-KIS"
      onImport={vi.fn()}
      importing={false}
      importError={null}
      sync={{
        shared: true,
        status: "live",
        pending: 0,
        error: null,
        room: "test-room",
        user: "Tester",
        conflicts: [],
        onDismissConflict: vi.fn(),
      }}
      pendingPack={null}
      onApplyPack={vi.fn()}
      onCancelPack={vi.fn()}
      onSearchAll={vi.fn()}
      searchingAll={false}
      onUndo={vi.fn(() => null)}
      undoLabel={null}
      pack={{
        shared: true,
        sessionName: "query-p1",
        packHash: "3f9a1c04",
        publishedBy: "Tester",
        origin: "server",
        loading: false,
        error: null,
      }}
      defaultRetrievalDatabase="btc"
      autoGen={{
        running: false,
        progress: null,
        log: [],
        error: null,
        onStart: vi.fn(),
        onCancel: vi.fn(),
      }}
      answerImport={{
        pending: null,
        importing: false,
        error: null,
        onPick: vi.fn(),
        onApply: vi.fn(),
        onCancel: vi.fn(),
      }}
    />,
  );
}

/** The order the panel would show after the store applied the emitted plan. */
function resultingOrder(rows: SubmissionRow[], plan: OrderChange[]): string[] {
  const wanted = new Map(plan.map((change) => [change.id, change.createdAt]));
  return sortRows(
    rows.map((row) => (wanted.has(row.id) ? { ...row, createdAt: wanted.get(row.id) } : row)),
  ).map((row) => row.id);
}

function drag(sourceIndex: number, targetIndex: number, questionId = QUESTION.id) {
  const grip = screen.getByTestId(`grip-${questionId}-${sourceIndex}`);
  const source = screen.getByTestId(`row-${questionId}-${sourceIndex}`);
  const target = screen.getByTestId(`row-${questionId}-${targetIndex}`);
  const transfer = dataTransfer();
  fireEvent.mouseDown(grip);
  fireEvent.dragStart(source, { dataTransfer: transfer });
  fireEvent.dragOver(target, { dataTransfer: transfer });
  return { source, target, transfer };
}

beforeEach(() => {
  onReorderRows.mockClear();
});

describe("dragging an answer to a new rank", () => {
  it("moves rank 7 to rank 1 and pushes the rest down one", () => {
    const rows = answers(10);
    renderPanel(rows);

    const { target, transfer } = drag(6, 0);
    fireEvent.drop(target, { dataTransfer: transfer });

    expect(onReorderRows).toHaveBeenCalledTimes(1);
    const [plan, label] = onReorderRows.mock.calls[0];
    expect(resultingOrder(rows, plan)).toEqual([
      `${QUESTION.id}-row-7`,
      `${QUESTION.id}-row-1`,
      `${QUESTION.id}-row-2`,
      `${QUESTION.id}-row-3`,
      `${QUESTION.id}-row-4`,
      `${QUESTION.id}-row-5`,
      `${QUESTION.id}-row-6`,
      `${QUESTION.id}-row-8`,
      `${QUESTION.id}-row-9`,
      `${QUESTION.id}-row-10`,
    ]);
    expect(label).toContain("hạng 7 → 1");
  });

  it("only rewrites the rows the move passes", () => {
    // A hundred-row question must not send a hundred timestamps because one
    // answer moved six places.
    renderPanel(answers(100));
    // The question collapses to ten rows until asked; expand it so rank 7 and
    // rank 1 are both on screen.
    fireEvent.click(screen.getByTestId(`expand-${QUESTION.id}`));

    const { target, transfer } = drag(6, 0);
    fireEvent.drop(target, { dataTransfer: transfer });

    expect(onReorderRows.mock.calls[0][0]).toHaveLength(7);
  });

  it("refuses a drop coming from another question", () => {
    // Moving a row between questions is not a reorder — it would silently
    // re-attribute an answer to a question it was never an answer to.
    const rows = [...answers(3), ...answers(3, OTHER.id)];
    renderPanel(rows, [QUESTION, OTHER]);

    const grip = screen.getByTestId(`grip-${OTHER.id}-0`);
    const source = screen.getByTestId(`row-${OTHER.id}-0`);
    const target = screen.getByTestId(`row-${QUESTION.id}-0`);
    const transfer = dataTransfer();
    fireEvent.mouseDown(grip);
    fireEvent.dragStart(source, { dataTransfer: transfer });
    fireEvent.drop(target, { dataTransfer: transfer });

    expect(onReorderRows).not.toHaveBeenCalled();
  });

  it("does not write when the row is dropped back on itself", () => {
    renderPanel(answers(5));
    const { source, transfer } = drag(2, 2);
    fireEvent.drop(source, { dataTransfer: transfer });
    expect(onReorderRows).not.toHaveBeenCalled();
  });

  it("marks where the row will land, on the side it is travelling from", () => {
    // The line is the whole affordance: a filled highlight would read as
    // "replace this answer" rather than "insert here".
    renderPanel(answers(5));

    drag(3, 1);
    expect(screen.getByTestId(`row-${QUESTION.id}-1`).className).toContain("drop-above");
    expect(screen.getByTestId(`row-${QUESTION.id}-3`).className).toContain("dragging-row");

    fireEvent.dragEnd(screen.getByTestId(`row-${QUESTION.id}-3`));
    expect(screen.getByTestId(`row-${QUESTION.id}-1`).className).not.toContain("drop-above");
  });

  it("marks the underside when the row is travelling down", () => {
    renderPanel(answers(5));
    drag(0, 3);
    expect(screen.getByTestId(`row-${QUESTION.id}-3`).className).toContain("drop-below");
  });

  it("leaves a row undraggable until its grip is held", () => {
    // A permanently draggable `<tr>` swallows click-and-drag inside its own
    // inputs, so a video id could no longer be selected with the mouse.
    renderPanel(answers(3));
    const row = screen.getByTestId(`row-${QUESTION.id}-1`);
    expect(row).not.toHaveAttribute("draggable", "true");
    fireEvent.mouseDown(screen.getByTestId(`grip-${QUESTION.id}-1`));
    expect(row).toHaveAttribute("draggable", "true");
    fireEvent.mouseUp(screen.getByTestId(`grip-${QUESTION.id}-1`));
    expect(row).not.toHaveAttribute("draggable", "true");
  });
});

describe("Alt+↑/↓", () => {
  it("moves the selected answer one rank", () => {
    // Dragging rank 60 to rank 1 through an auto-scrolling table is not a thing
    // anyone can do under a clock.
    const rows = answers(5);
    renderPanel(rows);
    fireEvent.click(screen.getByTestId(`row-${QUESTION.id}-3`));

    fireEvent.keyDown(window, { key: "ArrowUp", altKey: true });

    const [plan] = onReorderRows.mock.calls[0];
    expect(resultingOrder(rows, plan).slice(0, 4)).toEqual([
      `${QUESTION.id}-row-1`,
      `${QUESTION.id}-row-2`,
      `${QUESTION.id}-row-4`,
      `${QUESTION.id}-row-3`,
    ]);
  });

  it("does nothing at the ends of the list", () => {
    renderPanel(answers(3));
    fireEvent.click(screen.getByTestId(`row-${QUESTION.id}-0`));
    fireEvent.keyDown(window, { key: "ArrowUp", altKey: true });
    expect(onReorderRows).not.toHaveBeenCalled();
  });

  it("leaves plain ↑/↓ as row selection", () => {
    renderPanel(answers(3));
    fireEvent.click(screen.getByTestId(`row-${QUESTION.id}-0`));
    fireEvent.keyDown(window, { key: "ArrowDown" });
    expect(onReorderRows).not.toHaveBeenCalled();
    expect(screen.getByTestId(`row-${QUESTION.id}-1`).className).toContain("selected-row");
  });

  it("opens the fold when the answer moves past it", () => {
    // Rank 11 is out of sight on a collapsed question, and a move that appears
    // to do nothing is worse than no move at all.
    const rows = answers(20);
    renderPanel(rows);
    fireEvent.click(screen.getByTestId(`row-${QUESTION.id}-9`));

    fireEvent.keyDown(window, { key: "ArrowDown", altKey: true });

    expect(onReorderRows).toHaveBeenCalledTimes(1);
    expect(screen.getByTestId(`row-${QUESTION.id}-10`)).toBeTruthy();
    expect(
      within(screen.getByTestId(`submission-${QUESTION.id}`)).getAllByTestId(
        new RegExp(`^row-${QUESTION.id}-\\d+$`),
      ),
    ).toHaveLength(20);
  });
});
