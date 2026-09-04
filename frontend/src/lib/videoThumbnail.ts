/** Capture the frame currently presented by a video element.
 *
 * This is shared by the visible paused-frame picker and Sticky TRAKE restore so
 * both paths have the same definition of an "exact raw frame". The data URL is
 * deliberately ephemeral UI state; callers must not persist it in localStorage.
 */
export function captureVideoThumbnail(video: HTMLVideoElement, maxWidth = 320): string | null {
  try {
    const width = video.videoWidth;
    const height = video.videoHeight;
    if (!width || !height) return null;
    const scale = Math.min(1, maxWidth / width);
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(width * scale);
    canvas.height = Math.round(height * scale);
    const context = canvas.getContext("2d");
    if (!context) return null;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.7);
  } catch {
    // Cross-origin video without the correct CORS response taints the canvas.
    return null;
  }
}

function waitForMedia(
  video: HTMLVideoElement,
  event: "loadeddata" | "seeked",
  timeoutMs = 15_000,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const timeout = window.setTimeout(() => finish(new Error(`Video ${event} timeout`)), timeoutMs);
    const onReady = () => finish();
    const onError = () => finish(new Error("Video load failed"));
    const finish = (error?: Error) => {
      window.clearTimeout(timeout);
      video.removeEventListener(event, onReady);
      video.removeEventListener("error", onError);
      if (error) reject(error);
      else resolve();
    };
    video.addEventListener(event, onReady, { once: true });
    video.addEventListener("error", onError, { once: true });
  });
}

/** Load one video and capture its exact decoded frames at the requested media
 * times. A single hidden element is reused so restoring N TRAKE events does not
 * download the same MP4 N times. */
export async function extractVideoThumbnails(
  videoUrl: string,
  times: readonly number[],
): Promise<(string | null)[]> {
  if (!times.length) return [];
  const video = document.createElement("video");
  video.crossOrigin = "anonymous";
  video.preload = "auto";
  video.muted = true;
  video.playsInline = true;
  video.style.cssText =
    "position:fixed;width:1px;height:1px;left:-10000px;top:-10000px;opacity:0;pointer-events:none";
  document.body.appendChild(video);

  try {
    const loaded = waitForMedia(video, "loadeddata");
    video.src = videoUrl;
    video.load();
    await loaded;

    const thumbnails: (string | null)[] = [];
    for (const requested of times) {
      const time = Number.isFinite(requested)
        ? Math.max(0, Math.min(requested, Number.isFinite(video.duration) ? video.duration : requested))
        : 0;
      if (Math.abs(video.currentTime - time) > 0.0005) {
        const seeked = waitForMedia(video, "seeked");
        video.currentTime = time;
        await seeked;
      }
      // This is the same drawImage capture used by the visible pause/seek path.
      thumbnails.push(captureVideoThumbnail(video));
    }
    return thumbnails;
  } finally {
    video.pause();
    video.removeAttribute("src");
    video.load();
    video.remove();
  }
}
