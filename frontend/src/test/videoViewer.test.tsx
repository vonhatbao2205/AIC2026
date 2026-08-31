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
function Harness({
  onPaused,
  fallbackSrc,
  onFallback,
}: {
  onPaused: (t: number, thumb: string | null) => void;
  fallbackSrc?: string | null;
  onFallback?: (src: string) => void;
}) {
  const [tick, setTick] = useState(0);
  return (
    <>
      <button onClick={() => setTick((t) => t + 1)}>bump</button>
      <VideoViewer
        src="https://media.test/v.mp4"
        fallbackSrc={fallbackSrc}
        onFallback={onFallback}
        startTime={0}
        // A new identity every render, like the real `useCallback` whose deps moved.
        onPaused={(t, thumb) => onPaused(t + tick * 0, thumb)}
        onTime={() => {}}
      />
    </>
  );
}

/** jsdom never actually plays, so `paused` has to be stated. */
function playing(video: HTMLVideoElement) {
  Object.defineProperty(video, "paused", { configurable: true, get: () => false });
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

    playing(video);
    fireEvent.play(video);
    rvfc.present(83);

    // Jump to another event and pause before the next frame is presented: the
    // element's own clock is the authority, not the frame we seeked away from.
    Object.defineProperty(video, "currentTime", { configurable: true, writable: true, value: 150 });
    fireEvent.seeked(video);
    fireEvent.pause(video);

    expect(onPaused.mock.calls[0][0]).toBe(150);
  });

  it("captures the frame on a seek that lands while the video is paused", () => {
    const onPaused = vi.fn();
    render(<Harness onPaused={onPaused} />);
    const video = screen.getByTestId("video-viewer") as HTMLVideoElement;
    expect(video.paused).toBe(true);

    // Fine-tuning a frame — 'a'/'d', the arrows, the timeline, a heat peak — is
    // a seek on a paused video. Tying capture to the `pause` event alone meant
    // the operator had to play the clip and pause it again, and what came back
    // was wherever the video had run to, not the frame they had chosen.
    Object.defineProperty(video, "currentTime", { configurable: true, writable: true, value: 42.5 });
    fireEvent.seeked(video);

    expect(onPaused).toHaveBeenCalledTimes(1);
    expect(onPaused.mock.calls[0][0]).toBe(42.5);
  });

  it("leaves capture to the pause handler while the video is playing", () => {
    const onPaused = vi.fn();
    render(<Harness onPaused={onPaused} />);
    const video = screen.getByTestId("video-viewer") as HTMLVideoElement;

    playing(video);
    fireEvent.play(video);
    // Seeking mid-playback is navigation, not a frame pick; capturing there would
    // arm a submit target the operator is about to play straight past.
    Object.defineProperty(video, "currentTime", { configurable: true, writable: true, value: 42.5 });
    fireEvent.seeked(video);

    expect(onPaused).not.toHaveBeenCalled();
  });

  it("opens parked on the frame instead of playing away from it", () => {
    render(<Harness onPaused={() => {}} />);
    // Autoplay walked the video off the very frame it was opened to show, which
    // is what forced the play-then-pause dance in the first place.
    expect(screen.getByTestId("video-viewer")).not.toHaveAttribute("autoplay");
  });
});

describe("VideoViewer origin fallback", () => {
  const FALLBACK = "https://fallback.test/v.mp4";

  it("retries the other origin when the source fails to load", () => {
    const onFallback = vi.fn();
    render(<Harness onPaused={() => {}} fallbackSrc={FALLBACK} onFallback={onFallback} />);
    const video = screen.getByTestId("video-viewer") as HTMLVideoElement;
    expect(video.getAttribute("src")).toBe("https://media.test/v.mp4");

    // A dead origin kills every video at once, so the element's media error is
    // the signal to swap rather than to give up.
    fireEvent.error(video);

    expect(screen.getByTestId("video-viewer").getAttribute("src")).toBe(FALLBACK);
    expect(onFallback).toHaveBeenCalledWith(FALLBACK);
  });

  it("carries the playhead across the swap", () => {
    render(<Harness onPaused={() => {}} fallbackSrc={FALLBACK} />);
    const video = screen.getByTestId("video-viewer") as HTMLVideoElement;
    Object.defineProperty(video, "currentTime", { configurable: true, writable: true, value: 137 });

    fireEvent.error(video);
    // The new source seeks back once its metadata is in: a dead origin should
    // cost a reload, not the operator's place in the clip.
    fireEvent.loadedMetadata(screen.getByTestId("video-viewer"));

    expect((screen.getByTestId("video-viewer") as HTMLVideoElement).currentTime).toBe(137);
  });

  it("swaps at most once, so a dead fallback cannot loop", () => {
    const onFallback = vi.fn();
    render(<Harness onPaused={() => {}} fallbackSrc={FALLBACK} onFallback={onFallback} />);
    const video = screen.getByTestId("video-viewer") as HTMLVideoElement;

    fireEvent.error(video);
    fireEvent.error(screen.getByTestId("video-viewer"));
    fireEvent.error(screen.getByTestId("video-viewer"));

    expect(onFallback).toHaveBeenCalledTimes(1);
    expect(screen.getByTestId("video-viewer").getAttribute("src")).toBe(FALLBACK);
  });

  it("stays put when no fallback is configured", () => {
    const onFallback = vi.fn();
    render(<Harness onPaused={() => {}} fallbackSrc={null} onFallback={onFallback} />);
    fireEvent.error(screen.getByTestId("video-viewer"));

    expect(screen.getByTestId("video-viewer").getAttribute("src")).toBe("https://media.test/v.mp4");
    expect(onFallback).not.toHaveBeenCalled();
  });
});
