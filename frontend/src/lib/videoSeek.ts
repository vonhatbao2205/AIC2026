/** Keyboard scrubbing for every inline video in the console.
 *
 *  Two surfaces show a video the operator scrubs against the clock — the search
 *  console's inline player and the submission tab's frame editor — and they must
 *  answer the same keys with the same steps. Keeping the mapping here is what
 *  stops one of them drifting to a different step size after a later edit.
 */

/** Coarse step for the arrows: big enough to cross a shot, small enough to land
 *  inside the next one. */
export const VIDEO_COARSE_STEP_S = 5;
/** Fine step for 'a'/'d', used when hunting the exact instant to submit. */
export const VIDEO_FINE_STEP_S = 1;

/** Seconds to scrub for `key`, or null when the key is not a scrub key.
 *
 *  Capitals are included because Caps Lock is not a mode in this console; the
 *  callers still have to apply their own typing guard, since 'a' and 'd' are
 *  ordinary letters that must reach a focused text box untouched.
 */
export function seekDeltaForKey(key: string): number | null {
  switch (key) {
    case "ArrowRight":
      return VIDEO_COARSE_STEP_S;
    case "ArrowLeft":
      return -VIDEO_COARSE_STEP_S;
    case "d":
    case "D":
      return VIDEO_FINE_STEP_S;
    case "a":
    case "A":
      return -VIDEO_FINE_STEP_S;
    default:
      return null;
  }
}

/** Nudge a video element by `delta` seconds, clamped inside the media.
 *
 *  Clamped here rather than left to the element: a negative `currentTime` throws
 *  in some browsers, and seeking past the end can leave the video stalled at
 *  `ended` instead of showing the last frame the operator wants. `duration` is
 *  NaN until metadata loads, so the upper bound is simply not applied yet.
 */
export function seekElementBy(video: HTMLVideoElement | null, delta: number): void {
  if (!video) return;
  const end = Number.isFinite(video.duration) ? video.duration : Infinity;
  video.currentTime = Math.min(Math.max(video.currentTime + delta, 0), end);
}

/** True while a text box owns the keyboard, so letter shortcuts stay letters. */
export function isTypingTarget(): boolean {
  const el = document.activeElement;
  return !!el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA");
}
