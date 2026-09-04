import { swapMediaOrigin } from "./media";

/** Retry any failed keyframe once under its configured fallback origin.
 *
 * A delegated capture listener covers keyframes rendered anywhere in the app,
 * including images added after installation. Image error events do not bubble,
 * but they do travel through the capture phase. */
export function installKeyframeFallback(
  primary: string | undefined,
  fallback: string | undefined,
): () => void {
  if (!primary || !fallback) return () => {};

  function onError(event: Event): void {
    const img = event.target;
    if (!(img instanceof HTMLImageElement) || img.dataset.originRetried) return;

    const next = swapMediaOrigin(img.currentSrc || img.src, primary, fallback);
    if (!next) return;

    // Mark before assigning src: a synchronously failing fallback must not loop.
    img.dataset.originRetried = "1";
    img.src = next;
  }

  document.addEventListener("error", onError, true);
  return () => document.removeEventListener("error", onError, true);
}
