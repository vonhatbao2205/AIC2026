import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import { seekElementBy } from "../lib/videoSeek";

export interface VideoViewerHandle {
  toggle: () => void;
  seek: (t: number) => void;
  /** Nudge the playhead by `delta` seconds, clamped inside the media. */
  seekBy: (delta: number) => void;
  currentTime: () => number;
}

interface Props {
  src: string;
  /** Same video under the other origin, tried ONCE if `src` fails to load.
   *  The playhead is carried across, so a dead origin costs a reload rather than
   *  the operator's place in the clip. */
  fallbackSrc?: string | null;
  /** Fired when the fallback was used, so the console can stop starting new
   *  videos on an origin it now knows is down. */
  onFallback?: (src: string) => void;
  // Seek target (keyframe pts). Applied on load and whenever it changes, so the
  // video jumps to the selected keyframe instead of replaying from the start.
  startTime?: number | null;
  // Called with the frame the operator is looking at: on pause, and on every
  // seek that lands while the video is ALREADY paused. The second case is what
  // makes a frame pick exact — see the note on `handleSeeked` below.
  onPaused?: (rawTime: number, thumbnail: string | null) => void;
  // Live playhead reporting for the timeline.
  onTime?: (t: number) => void;
}

// Uses requestVideoFrameCallback (when available) to track the latest presented
// mediaTime, so the paused frame is computed from the actual displayed frame.
export const VideoViewer = forwardRef<VideoViewerHandle, Props>(function VideoViewer(
  { src, fallbackSrc, startTime, onPaused, onTime, onFallback },
  ref,
) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const latestMediaTime = useRef(0);
  const hasPresentedFrame = useRef(false);
  const rvfcId = useRef<number | null>(null);
  // The listeners below are attached ONCE per source and read the callbacks
  // through these refs. Depending on the callbacks directly meant that any
  // parent state they close over — the timeline finishing its fetch was enough —
  // tore the listeners down mid-playback. The teardown cancelled the
  // requestVideoFrameCallback loop, nothing restarted it (that only happens on a
  // `play` event, and the video was already playing), and `latestMediaTime`
  // froze. Pausing then reported the instant tracking died instead of the
  // instant on screen: seek to E3, pause, and the captured frame was E1's.
  const onPausedRef = useRef(onPaused);
  const onTimeRef = useRef(onTime);
  const onFallbackRef = useRef(onFallback);
  // Read inside the listener effect, which is keyed on the source alone.
  const startTimeRef = useRef(startTime);
  startTimeRef.current = startTime ?? 0;
  // Where to land after an origin swap; consumed by the seek effect below.
  const resumeAt = useRef<number | null>(null);
  onPausedRef.current = onPaused;
  onTimeRef.current = onTime;
  onFallbackRef.current = onFallback;

  // Which URL the element is actually playing. It starts at `src` and moves to
  // the fallback at most once per source; `activeSrc` rather than `src` is what
  // the <video> reads, so a swap does not fight the prop on the next render.
  const [activeSrc, setActiveSrc] = useState(src);
  const triedFallback = useRef(false);
  // A source change is a new decision: the fallback becomes available again.
  const lastSrc = useRef(src);
  if (lastSrc.current !== src) {
    lastSrc.current = src;
    triedFallback.current = false;
    if (activeSrc !== src) setActiveSrc(src);
  }

  useImperativeHandle(ref, () => ({
    toggle: () => {
      const v = videoRef.current;
      if (!v) return;
      if (v.paused) void v.play();
      else v.pause();
    },
    seek: (t: number) => {
      const v = videoRef.current;
      if (v) v.currentTime = t;
    },
    seekBy: (delta: number) => seekElementBy(videoRef.current, delta),
    currentTime: () => videoRef.current?.currentTime ?? 0,
  }));

  // Seek to the selected keyframe time on load and whenever it changes — or to
  // where playback actually was, when the source changed underneath because the
  // origin died. ONE seek path on purpose: an origin swap and a `startTime`
  // change both land on the same `loadedmetadata`, and two listeners racing
  // there meant the later one won, sending the operator back to the keyframe
  // instead of resuming where they were.
  useEffect(() => {
    const v = videoRef.current;
    const target = resumeAt.current ?? startTime;
    resumeAt.current = null;
    if (!v || target == null) return;
    const doSeek = () => {
      try {
        v.currentTime = target;
      } catch {
        /* not seekable yet */
      }
    };
    if (v.readyState >= 1) doSeek();
    else v.addEventListener("loadedmetadata", doSeek, { once: true });
    return () => v.removeEventListener("loadedmetadata", doSeek);
  }, [startTime, activeSrc]);

  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    const vAny = v as HTMLVideoElement & {
      requestVideoFrameCallback?: (cb: (now: number, md: { mediaTime: number }) => void) => number;
      cancelVideoFrameCallback?: (id: number) => void;
    };
    const supportsRVFC = typeof vAny.requestVideoFrameCallback === "function";

    const step = (_now: number, metadata: { mediaTime: number }) => {
      latestMediaTime.current = metadata.mediaTime;
      hasPresentedFrame.current = true;
      onTimeRef.current?.(metadata.mediaTime);
      rvfcId.current = vAny.requestVideoFrameCallback!(step);
    };

    function startTracking() {
      hasPresentedFrame.current = false;
      if (supportsRVFC && rvfcId.current == null) {
        rvfcId.current = vAny.requestVideoFrameCallback!(step);
      }
    }
    function stopTracking() {
      if (supportsRVFC && rvfcId.current != null) {
        vAny.cancelVideoFrameCallback!(rvfcId.current);
        rvfcId.current = null;
      }
    }
    function captureThumbnail(): string | null {
      try {
        const w = v!.videoWidth, h = v!.videoHeight;
        if (!w || !h) return null;
        const scale = Math.min(1, 320 / w);
        const canvas = document.createElement("canvas");
        canvas.width = Math.round(w * scale);
        canvas.height = Math.round(h * scale);
        const ctx = canvas.getContext("2d");
        if (!ctx) return null;
        ctx.drawImage(v!, 0, 0, canvas.width, canvas.height);
        return canvas.toDataURL("image/jpeg", 0.7); // throws if the frame is CORS-tainted
      } catch {
        return null;
      }
    }
    function handlePause() {
      const rawTime = supportsRVFC && hasPresentedFrame.current
        ? latestMediaTime.current
        : v!.currentTime;
      onPausedRef.current?.(rawTime, captureThumbnail());
      stopTracking();
    }
    function handleTimeUpdate() {
      if (!supportsRVFC) onTimeRef.current?.(v!.currentTime);
    }
    // A seek invalidates the last presented frame: until a new one arrives,
    // `currentTime` is the authority on where the video is. Without this, pausing
    // right after a seek captured the frame the operator had seeked AWAY from.
    //
    // A seek that lands while the video is ALREADY PAUSED also IS the frame pick.
    // Capture had been tied to the `pause` event alone, so choosing a frame meant
    // playing the clip and pausing it again — and the frame you get back is
    // wherever the video had run to by the time the pause took effect, not the
    // one you seeked to. Every fine-tune (`a`/`d`, the arrows, the timeline, a
    // heat peak) is a seek on a paused video, so this is the path that has to be
    // exact.
    function handleSeeked() {
      hasPresentedFrame.current = false;
      latestMediaTime.current = v!.currentTime;
      onTimeRef.current?.(v!.currentTime);
      if (v!.paused) onPausedRef.current?.(v!.currentTime, captureThumbnail());
    }

    // A dead origin surfaces as a media error on the element. Retry the same
    // video under the other origin, carrying the playhead across so the operator
    // keeps their place; once only, or a fallback that is also down would loop.
    function handleError() {
      if (triedFallback.current || !fallbackSrc || fallbackSrc === v!.src) return;
      triedFallback.current = true;
      resumeAt.current = v!.currentTime || startTimeRef.current || 0;
      setActiveSrc(fallbackSrc);
      onFallbackRef.current?.(fallbackSrc);
    }

    v.addEventListener("play", startTracking);
    v.addEventListener("pause", handlePause);
    v.addEventListener("seeked", handleSeeked);
    v.addEventListener("error", handleError);
    v.addEventListener("timeupdate", handleTimeUpdate);
    // React can re-run this effect while the video is already playing (a source
    // swap, a Fast Refresh); `play` will not fire again, so pick tracking back up.
    if (!v.paused) startTracking();
    return () => {
      v.removeEventListener("play", startTracking);
      v.removeEventListener("pause", handlePause);
      v.removeEventListener("seeked", handleSeeked);
      v.removeEventListener("error", handleError);
      v.removeEventListener("timeupdate", handleTimeUpdate);
      stopTracking();
    };
  }, [activeSrc, fallbackSrc]);

  return (
    // Deliberately NOT autoplaying. The video is opened to look at one frame —
    // the selected keyframe, or the heat peak just clicked — and playing from it
    // immediately walks away from that frame, so picking it meant chasing the
    // playhead back with a pause. It now opens parked on the frame, already
    // captured; `Space` starts it whenever the operator actually wants to watch.
    <video
      ref={videoRef}
      className="inline-video"
      src={activeSrc}
      controls
      preload="auto"
      crossOrigin="anonymous"
      data-testid="video-viewer"
    />
  );
});
