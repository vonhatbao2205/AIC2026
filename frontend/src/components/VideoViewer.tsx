import { forwardRef, useEffect, useImperativeHandle, useRef } from "react";
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
  fps: number;
  // Seek target (keyframe pts). Applied on load and whenever it changes, so the
  // video jumps to the selected keyframe instead of replaying from the start.
  startTime?: number | null;
  // Called on pause with the most accurate paused mediaTime available plus a
  // JPEG thumbnail captured from the exact paused frame (null if CORS-tainted).
  onPaused?: (rawTime: number, thumbnail: string | null) => void;
  // Live playhead reporting for the timeline.
  onTime?: (t: number) => void;
}

// Uses requestVideoFrameCallback (when available) to track the latest presented
// mediaTime, so the paused frame is computed from the actual displayed frame.
export const VideoViewer = forwardRef<VideoViewerHandle, Props>(function VideoViewer(
  { src, fps, startTime, onPaused, onTime },
  ref,
) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const latestMediaTime = useRef(0);
  const hasPresentedFrame = useRef(false);
  const rvfcId = useRef<number | null>(null);

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

  // Seek to the selected keyframe time on load and whenever it changes.
  useEffect(() => {
    const v = videoRef.current;
    if (!v || startTime == null) return;
    const doSeek = () => {
      try {
        v.currentTime = startTime;
      } catch {
        /* not seekable yet */
      }
    };
    if (v.readyState >= 1) doSeek();
    else v.addEventListener("loadedmetadata", doSeek, { once: true });
    return () => v.removeEventListener("loadedmetadata", doSeek);
  }, [startTime, src]);

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
      onTime?.(metadata.mediaTime);
      rvfcId.current = vAny.requestVideoFrameCallback!(step);
    };

    function startTracking() {
      hasPresentedFrame.current = false;
      if (supportsRVFC) rvfcId.current = vAny.requestVideoFrameCallback!(step);
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
      onPaused?.(rawTime, captureThumbnail());
      stopTracking();
    }
    function handleTimeUpdate() {
      if (!supportsRVFC) onTime?.(v!.currentTime);
    }

    v.addEventListener("play", startTracking);
    v.addEventListener("pause", handlePause);
    v.addEventListener("timeupdate", handleTimeUpdate);
    return () => {
      v.removeEventListener("play", startTracking);
      v.removeEventListener("pause", handlePause);
      v.removeEventListener("timeupdate", handleTimeUpdate);
      stopTracking();
    };
  }, [src, fps, onPaused, onTime]);

  return (
    <video
      ref={videoRef}
      className="inline-video"
      src={src}
      controls
      autoPlay
      preload="auto"
      crossOrigin="anonymous"
      data-testid="video-viewer"
    />
  );
});
