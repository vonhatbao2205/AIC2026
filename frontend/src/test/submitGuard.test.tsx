/**
 * The submit guard as the LAST line, not a second opinion.
 *
 * The console refuses a foreign frame at the moment it is dropped into a slot,
 * so the mixed-video state is not reachable from the UI. That is exactly why the
 * guard needs its own test: it is the layer that has to hold if any future path
 * — a shared submission import, a restored draft, a new drag source — ever
 * assembles a chain the assign-time check never saw.
 */
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { SubmitGuard } from "../components/SubmitGuard";
import type { TrakeSlot } from "../components/TrakePanel";

function slot(videoId: string, frameIdx: number, pts: number): TrakeSlot {
  return { video_id: videoId, frame_idx: frameIdx, pts_time: pts, thumbnail: null };
}

function renderGuard(trakeSlots: (TrakeSlot | null)[], orderViolations: number[] = []) {
  return render(
    <SubmitGuard
      open
      queryType="TRAKE"
      frame={null}
      pausedFrame={null}
      frameIdx={null}
      trakeSlots={trakeSlots}
      answer=""
      setAnswer={() => {}}
      dresEnabled={false}
      questionId="q-1"
      csvLine="K19_V028,100,200"
      csvError={null}
      dresConfigured={false}
      evaluationName={null}
      taskNameOverride=""
      setTaskNameOverride={() => {}}
      answerMode="auto"
      setAnswerMode={() => {}}
      segmentPadMs={500}
      setSegmentPadMs={() => {}}
      preview={null}
      previewError={null}
      previewLoading={false}
      duplicateId={null}
      orderViolations={orderViolations}
      submitting={false}
      onConfirm={() => {}}
      onCancel={() => {}}
    />,
  );
}

describe("submit guard · TRAKE row validity", () => {
  it("lets a single-video chain in chronological order through", () => {
    renderGuard([slot("K19_V028", 100, 4), slot("K19_V028", 500, 20)]);
    expect(screen.getByTestId("confirm-submit")).toBeEnabled();
    expect(screen.queryByTestId("guard-mixed-videos")).not.toBeInTheDocument();
  });

  it("refuses a chain whose events come from different videos", () => {
    // The row names ONE video — the first filled slot's — and lists frame indices
    // under it, so the K19_V099 frame would be submitted as if it came from
    // K19_V028: a wrong answer that is perfectly well-formed on the wire, which
    // is why nothing downstream can catch it.
    renderGuard([slot("K19_V028", 100, 4), slot("K19_V099", 500, 20)]);
    expect(screen.getByTestId("guard-mixed-videos")).toHaveTextContent("K19_V028");
    expect(screen.getByTestId("guard-mixed-videos")).toHaveTextContent("K19_V099");
    expect(screen.getByTestId("confirm-submit")).toBeDisabled();
  });

  it("refuses a chain that is not chronological", () => {
    // The organiser's parser rejects a row out of order, and one rejected row
    // blocks the whole submission — so a warning is not enough.
    renderGuard([slot("K19_V028", 500, 20), slot("K19_V028", 100, 4)], [2]);
    expect(screen.getByTestId("guard-order-violation")).toHaveTextContent("E2");
    expect(screen.getByTestId("confirm-submit")).toBeDisabled();
  });

  it("still refuses an empty chain", () => {
    renderGuard([null, null]);
    expect(screen.getByTestId("confirm-submit")).toBeDisabled();
  });
});
