import { describe, expect, it } from "vitest";
import {
  VIDEO_COARSE_STEP_S,
  VIDEO_FINE_STEP_S,
  seekDeltaForKey,
  seekElementBy,
} from "../lib/videoSeek";

describe("video scrub key mapping", () => {
  it("maps arrows to the coarse step and a/d to the fine step, in both cases", () => {
    expect(seekDeltaForKey("ArrowRight")).toBe(VIDEO_COARSE_STEP_S);
    expect(seekDeltaForKey("ArrowLeft")).toBe(-VIDEO_COARSE_STEP_S);
    expect(seekDeltaForKey("d")).toBe(VIDEO_FINE_STEP_S);
    expect(seekDeltaForKey("D")).toBe(VIDEO_FINE_STEP_S);
    expect(seekDeltaForKey("a")).toBe(-VIDEO_FINE_STEP_S);
    expect(seekDeltaForKey("A")).toBe(-VIDEO_FINE_STEP_S);
  });

  it("claims no other key", () => {
    for (const key of ["v", "k", "s", "w", "ArrowUp", "ArrowDown", "Enter", " ", "Escape"]) {
      expect(seekDeltaForKey(key)).toBeNull();
    }
  });

  it("keeps the coarse step larger than the fine one", () => {
    expect(VIDEO_COARSE_STEP_S).toBeGreaterThan(VIDEO_FINE_STEP_S);
  });
});

describe("seekElementBy", () => {
  function video(currentTime: number, duration: number) {
    const el = document.createElement("video");
    el.currentTime = currentTime;
    Object.defineProperty(el, "duration", { configurable: true, value: duration });
    return el;
  }

  it("never rewinds past the start", () => {
    const el = video(2, 100);
    seekElementBy(el, -5);
    expect(el.currentTime).toBe(0);
  });

  it("never runs past the end", () => {
    const el = video(98, 100);
    seekElementBy(el, 5);
    expect(el.currentTime).toBe(100);
  });

  it("still seeks forward before metadata gives a duration", () => {
    const el = video(10, NaN);
    seekElementBy(el, 5);
    expect(el.currentTime).toBeCloseTo(15, 3);
  });

  it("does nothing without an element", () => {
    expect(() => seekElementBy(null, 5)).not.toThrow();
  });
});
