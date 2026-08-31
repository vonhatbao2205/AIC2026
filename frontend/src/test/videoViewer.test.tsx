import { fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { VideoViewer } from "../components/VideoViewer";

/** jsdom has no `requestVideoFrameCallback`, so the component's accurate-time
 *  path is never exercised by default — which is exactly the path the TRAKE
 *  frame-pick depends on. Install a driveable fake and step frames by hand. */
function installVideoFrameCallback() {
  let nextId = 1;
  const pending = new Map<number, (now: number, md: { mediaTime: number }) => void>();
  const proto = HTMLVideoElement.prototype as unknown as Record<string, unknown>;
  proto.requestVideoFrameCallback = function (cb: (now: number, md: { mediaTime: number }) => void) {
    const id = nextId++;
    pending.set(id, cb);
    return id;
  };
  proto.cancelVideoFrameCallback = function (id: number) {
    pending.delete(id);
  };
  return {
    /** Present one frame at `mediaTime`. No-op when tracking has been cancelled. */
    present(mediaTime: number) {
      const entry = [...pending.entries()].pop();
      if (!entry) return false;
      const [id, cb] = entry;
      pending.delete(id);
      cb(performance.now(), { mediaTime });
      return true;
    },
    get tracking() {
      return pending.size > 0;
    },
  };
}

/** The console re-creates `onPaused` whenever the state it closes over moves —
 *  the timeline finishing its fetch is enough. `bump` reproduces that. */
function Harness({ onPaused }: { onPaused: (t: number, thumb: string | null) => void }) {
  const [tick, setTick] = useState(0);
  return (
    <>
      <button onClick={() => setTick((t) => t + 1)}>bump</button>
      <VideoViewer
        src="https://media.test/v.mp4"
        startTime={0}
        // A new identity every render, like the real `useCallback` whose deps moved.
        onPaused={(t, thumb) => onPaused(t + tick * 0, thumb)}
        onTime={() => {}}
      />
    </>
  );
}

describe("VideoViewer paused-frame accuracy", () => {
  let rvfc: ReturnType<typeof installVideoFrameCallback>;

  beforeEach(() => {
    rvfc = installVideoFrameCallback();
  });

  it("keeps tracking the presented frame after the parent re-creates its callbacks", () => {
    const onPaused = vi.fn();
    render(<Harness onPaused={onPaused} />);
    const video = screen.getByTestId("video-viewer") as HTMLVideoElement;

    fireEvent.play(video);
    expect(rvfc.present(83)).toBe(true); // operator is watching E1 at 1:23

    // The timeline fetch resolves -> onPaused/onTime get new identities. This
    // used to tear the listeners down and cancel the frame callback, freezing
    // the tracked time at 83s with nothing to restart it.
    fireEvent.click(screen.getByText("bump"));
    expect(rvfc.tracking).toBe(true);

    expect(rvfc.present(150)).toBe(true); // ...then seeks on to E3 at 2:30
    fireEvent.pause(video);

    expect(onPaused).toHaveBeenCalledTimes(1);
    expect(onPaused.mock.calls[0][0]).toBe(150); // E3, not the stale E1
  });

  it("reports the seek target when no frame has been presented since the seek", () => {
    const onPaused = vi.fn();
    render(<Harness onPaused={onPaused} />);
    const video = screen.getByTestId("video-viewer") as HTMLVideoElement;

    fireEvent.play(video);
    rvfc.present(83);

    // Jump to another event and pause before the next frame is presented: the
    // element's own clock is the authority, not the frame we seeked away from.
    Object.defineProperty(video, "currentTime", { configurable: true, writable: true, value: 150 });
    fireEvent.seeked(video);
    fireEvent.pause(video);

    expect(onPaused.mock.calls[0][0]).toBe(150);
  });
});
