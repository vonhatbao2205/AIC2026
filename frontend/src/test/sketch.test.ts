import { afterEach, describe, expect, it } from "vitest";
import { hexToHsv, hexToRgb, hsvToHex, normalizeHex, rgbToHex } from "../lib/color";
import {
  type SketchDoc,
  type SketchOp,
  HISTORY_LIMIT,
  MAX_SAVED_SKETCHES,
  MAX_SIZE,
  MIN_SIZE,
  addOp,
  commitDoc,
  composite,
  constrainEnd,
  drawOp,
  emptyDoc,
  floodFill,
  historyOf,
  isAppend,
  isBlankDoc,
  isMeaningful,
  loadSketch,
  nextSize,
  ownsSketchKey,
  parseDoc,
  redo,
  restoreHistory,
  sampleColor,
  saveSketch,
  stabilize,
  undo,
} from "../lib/sketch";

/** jsdom has no canvas: a context stand-in that records what is drawn, with the
 *  state (composite mode, alpha, shadow) in force at each paint call. */
function fakeContext(width = 100, height = 50) {
  const calls: { name: string; state: Record<string, unknown>; args: unknown[] }[] = [];
  const state: Record<string, unknown> = {
    globalCompositeOperation: "source-over", globalAlpha: 1, shadowBlur: 0, shadowOffsetX: 0,
    fillStyle: "#000", strokeStyle: "#000", lineWidth: 1, lineCap: "butt", lineJoin: "miter",
  };
  const stack: Record<string, unknown>[] = [];
  const record = (name: string) => (...args: unknown[]) => {
    calls.push({ name, state: { ...state }, args });
  };
  const ctx = new Proxy(
    {
      canvas: { width, height },
      save: () => stack.push({ ...state }),
      restore: () => Object.assign(state, stack.pop()),
      beginPath: () => {}, closePath: () => {}, moveTo: () => {}, lineTo: () => {},
      quadraticCurveTo: () => {}, rect: () => {}, ellipse: () => {}, translate: () => {},
      setLineDash: () => {}, clearRect: record("clearRect"),
      stroke: record("stroke"), fill: record("fill"), fillRect: record("fillRect"),
      drawImage: record("drawImage"),
    } as Record<string, unknown>,
    {
      get: (target, key: string) => (key in target ? target[key] : state[key]),
      set: (_target, key: string, value) => {
        state[key] = value;
        return true;
      },
    },
  ) as unknown as CanvasRenderingContext2D;
  return { ctx, calls };
}

/** A transparent `width × height` layer. */
function layer(width: number, height: number): ImageData {
  return { data: new Uint8ClampedArray(width * height * 4), width, height, colorSpace: "srgb" } as ImageData;
}

function setPixel(image: ImageData, x: number, y: number, [r, g, b, a]: number[]) {
  const o = (y * image.width + x) * 4;
  image.data.set([r, g, b, a], o);
}

function pixel(image: ImageData, x: number, y: number): number[] {
  const o = (y * image.width + x) * 4;
  return Array.from(image.data.slice(o, o + 4));
}

const WHITE = { r: 255, g: 255, b: 255 };
const RED = { r: 224, g: 32, b: 32 };

const stroke = (x = 10): SketchOp => ({
  kind: "stroke", tip: "round", color: "#1e50dc", alpha: 1, size: 12, points: [x, 10, x + 20, 30],
});

describe("colour math", () => {
  it("normalizes hex input the way an operator types it", () => {
    expect(normalizeHex("2489B0")).toBe("#2489b0");
    expect(normalizeHex("#abc")).toBe("#aabbcc");
    expect(normalizeHex("#12345")).toBeNull();
    expect(normalizeHex("blue")).toBeNull();
  });

  it("matches the Photoshop HSB reading of a colour", () => {
    // The picker in the reference screenshot: #2489b0 = R36 G137 B176.
    expect(hexToRgb("#2489b0")).toEqual({ r: 36, g: 137, b: 176 });
    const hsv = hexToHsv("#2489b0");
    expect(Math.round(hsv.h)).toBe(197);
    expect(Math.round(hsv.s)).toBe(80);
    expect(Math.round(hsv.v)).toBe(69);
  });

  it("round-trips RGB ↔ HSV without drift", () => {
    for (const hex of ["#000000", "#ffffff", "#e02020", "#2489b0", "#6aa84f", "#8a3ab9", "#7f7f7f"]) {
      expect(hsvToHex(hexToHsv(hex))).toBe(hex);
    }
    expect(rgbToHex({ r: 300, g: -4, b: 12.4 })).toBe("#ff000c");
  });
});

describe("sketch history", () => {
  it("undoes and redoes whole operations", () => {
    let history = historyOf(emptyDoc());
    history = commitDoc(history, addOp(history.present, stroke(1)));
    history = commitDoc(history, addOp(history.present, stroke(2)));

    history = undo(history);
    expect(history.present.ops).toHaveLength(1);
    history = redo(history);
    expect(history.present.ops).toHaveLength(2);
    // A new edit after an undo discards the redo branch.
    history = commitDoc(undo(history), addOp(undo(history).present, stroke(3)));
    expect(history.future).toHaveLength(0);
    expect(redo(history)).toBe(history);
  });

  it("makes clear and background changes undoable too", () => {
    let history = commitDoc(historyOf(emptyDoc()), addOp(emptyDoc(), stroke()));
    history = commitDoc(history, { ...history.present, background: "#87c3ec" });
    history = commitDoc(history, emptyDoc());

    expect(isBlankDoc(history.present)).toBe(true);
    history = undo(history);
    expect(history.present.background).toBe("#87c3ec");
    expect(history.present.ops).toHaveLength(1);
  });

  it("caps the undo depth", () => {
    let history = historyOf(emptyDoc());
    for (let i = 0; i < HISTORY_LIMIT + 20; i += 1) history = commitDoc(history, addOp(history.present, stroke(i)));
    expect(history.past).toHaveLength(HISTORY_LIMIT);
  });

  it("keeps every restored operation undoable after a reload", () => {
    const doc: SketchDoc = { background: "#ffffff", ops: [stroke(1), stroke(2), stroke(3)] };
    let history = restoreHistory(doc);

    history = undo(history);
    expect(history.present.ops).toEqual([stroke(1), stroke(2)]);
    expect(restoreHistory(null).present).toEqual(emptyDoc());
  });

  it("recognizes an append so the cached layer draws one op, not the document", () => {
    const base = addOp(emptyDoc(), stroke(1));
    const next = addOp(base, stroke(2));

    expect(isAppend(base, next)).toBe(true);
    expect(isAppend(next, base)).toBe(false);
    expect(isAppend(base, { ...next, background: "#000000" })).toBe(false);
    expect(isAppend(null, next)).toBe(false);
  });

  it("treats a coloured background alone as something to search", () => {
    expect(isBlankDoc(emptyDoc())).toBe(true);
    expect(isBlankDoc({ background: "#87c3ec", ops: [] })).toBe(false);
  });
});

describe("bucket fill", () => {
  it("fills the region bounded by a stroke and stops at it", () => {
    const image = layer(12, 6);
    // A 2-px black wall at x = 5..6.
    for (let y = 0; y < 6; y += 1) {
      setPixel(image, 5, y, [0, 0, 0, 255]);
      setPixel(image, 6, y, [0, 0, 0, 255]);
    }

    expect(floodFill(image, WHITE, 1, 1, RED, 32)).toBe(true);

    expect(pixel(image, 0, 0)).toEqual([224, 32, 32, 255]);
    expect(pixel(image, 4, 5)).toEqual([224, 32, 32, 255]);
    // The wall's near edge is taken by the 1-px halo guard, its far edge is not.
    expect(pixel(image, 5, 3)).toEqual([224, 32, 32, 255]);
    expect(pixel(image, 6, 3)).toEqual([0, 0, 0, 255]);
    expect(pixel(image, 9, 3)[3]).toBe(0);
  });

  it("decides regions by what is seen: transparent layer over the background", () => {
    const image = layer(5, 1);
    // Faint grey that composites to near-white, then a clearly different block
    // (2 px wide: its first pixel is the halo the fill is allowed to take).
    setPixel(image, 1, 0, [200, 200, 200, 40]);
    setPixel(image, 3, 0, [0, 0, 255, 255]);
    setPixel(image, 4, 0, [0, 0, 255, 255]);

    floodFill(image, WHITE, 0, 0, RED, 32);

    expect(pixel(image, 1, 0)).toEqual([224, 32, 32, 255]);
    expect(pixel(image, 4, 0)).toEqual([0, 0, 255, 255]);
  });

  it("does nothing when the seed already has the colour or lies outside", () => {
    const image = layer(3, 3);
    expect(floodFill(image, RED, 1, 1, RED, 10)).toBe(false);
    expect(floodFill(image, WHITE, -1, 1, RED, 10)).toBe(false);
    expect(floodFill(image, WHITE, 3, 1, RED, 10)).toBe(false);
  });

  it("samples the colour an operator sees for the eyedropper", () => {
    const image = layer(2, 1);
    setPixel(image, 0, 0, [0, 0, 0, 128]); // half-transparent black over white

    expect(sampleColor(image, "#ffffff", 0, 0)).toBe("#7f7f7f");
    expect(sampleColor(image, "#ffffff", 1, 0)).toBe("#ffffff");
    expect(sampleColor(image, "#ffffff", 99, 99)).toBe("#ffffff"); // clamped to the edge
  });
});

describe("sketch rendering", () => {
  it("cuts the eraser out of the layer instead of painting over it", () => {
    const { ctx, calls } = fakeContext();
    drawOp(ctx, { kind: "erase", size: 20, points: [5, 5, 30, 30] }, "#ffffff");

    const painted = calls.find((call) => call.name === "stroke");
    expect(painted?.state.globalCompositeOperation).toBe("destination-out");
  });

  it("draws the soft tip as a blurred shadow and honours opacity", () => {
    const { ctx, calls } = fakeContext();
    drawOp(ctx, { kind: "stroke", tip: "soft", color: "#e02020", alpha: 0.5, size: 40, points: [5, 5, 30, 30] }, "#fff");

    const painted = calls.find((call) => call.name === "stroke");
    expect(painted?.state.shadowBlur).toBeGreaterThan(0);
    expect(painted?.state.shadowOffsetX).not.toBe(0);
    expect(painted?.state.globalAlpha).toBe(0.5);
  });

  it("fills or outlines shapes as chosen", () => {
    const filled = fakeContext();
    drawOp(filled.ctx, { kind: "rect", filled: true, color: "#000000", alpha: 1, size: 4, points: [1, 1, 20, 10] }, "#fff");
    const outline = fakeContext();
    drawOp(outline.ctx, { kind: "ellipse", filled: false, color: "#000000", alpha: 1, size: 4, points: [1, 1, 20, 10] }, "#fff");

    expect(filled.calls.map((call) => call.name)).toContain("fill");
    expect(outline.calls.map((call) => call.name)).toContain("stroke");
    expect(outline.calls.map((call) => call.name)).not.toContain("fill");
  });

  it("flattens the layer onto an opaque background for export", () => {
    const { ctx, calls } = fakeContext();
    composite(ctx, {} as CanvasImageSource, "#87c3ec");

    expect(calls[0].name).toBe("fillRect");
    expect(calls[0].state.fillStyle).toBe("#87c3ec");
    expect(calls[1].name).toBe("drawImage");
  });
});

describe("pointer helpers", () => {
  it("snaps lines to 15° and boxes to squares with Shift", () => {
    const [x, y] = constrainEnd("line", 0, 0, 100, 4, true);
    expect(y).toBeCloseTo(0, 6);
    expect(x).toBeCloseTo(Math.hypot(100, 4), 6);
    expect(constrainEnd("rect", 10, 10, 40, 20, true)).toEqual([40, 40]);
    expect(constrainEnd("ellipse", 10, 10, 0, 30, true)).toEqual([-10, 30]);
    expect(constrainEnd("rect", 10, 10, 40, 20, false)).toEqual([40, 20]);
  });

  it("discards clicks that drew no shape", () => {
    expect(isMeaningful({ kind: "rect", filled: true, color: "#000000", alpha: 1, size: 2, points: [5, 5, 5, 30] })).toBe(false);
    expect(isMeaningful({ kind: "line", tip: "round", color: "#000000", alpha: 1, size: 2, points: [5, 5, 6, 5] })).toBe(false);
    expect(isMeaningful({ kind: "lasso", color: "#000000", alpha: 1, points: [1, 1, 5, 5] })).toBe(false);
    // A single dab of the brush is a dot, and is kept.
    expect(isMeaningful({ kind: "stroke", tip: "round", color: "#000000", alpha: 1, size: 2, points: [5, 5] })).toBe(true);
  });

  it("steadies the pointer by the chosen amount", () => {
    expect(stabilize([0, 0], [10, 10], 0)).toEqual([10, 10]);
    expect(stabilize([0, 0], [10, 10], 0.5)).toEqual([5, 5]);
  });

  it("keeps the brush size inside its bounds", () => {
    expect(nextSize(MIN_SIZE, -1)).toBe(MIN_SIZE);
    expect(nextSize(MAX_SIZE, 1)).toBe(MAX_SIZE);
    expect(nextSize(20, 1)).toBe(24);
  });
});

describe("sketch keyboard scope", () => {
  function owns(init: KeyboardEventInit, inside = true) {
    const root = document.createElement("div");
    if (inside) root.setAttribute("data-sketch-scope", "");
    const target = document.createElement("canvas");
    root.appendChild(target);
    document.body.appendChild(root);
    let result: boolean | null = null;
    target.addEventListener("keydown", (event) => { result = ownsSketchKey(event); });
    target.dispatchEvent(new KeyboardEvent("keydown", { bubbles: true, ...init }));
    root.remove();
    return result;
  }

  it("claims tool keys, undo and Enter inside the canvas", () => {
    expect(owns({ key: "b" })).toBe(true);
    expect(owns({ key: "]" })).toBe(true);
    expect(owns({ key: "z", ctrlKey: true })).toBe(true);
    // Plain Enter must never open the submit guard while drawing.
    expect(owns({ key: "Enter" })).toBe(true);
  });

  it("leaves navigation keys to the console and ignores focus elsewhere", () => {
    expect(owns({ key: "ArrowDown" })).toBe(false);
    expect(owns({ key: "v" })).toBe(false);
    expect(owns({ key: "b" }, false)).toBe(false);
  });
});

describe("sketch autosave", () => {
  afterEach(() => localStorage.clear());

  it("round-trips a drawing per question", () => {
    const doc: SketchDoc = { background: "#87c3ec", ops: [stroke(1), { kind: "fill", color: "#6aa84f", tolerance: 32, points: [3, 4] }] };
    expect(saveSketch("q-1", doc)).toBe(true);

    expect(loadSketch("q-1")).toEqual(doc);
    expect(loadSketch("q-2")).toBeNull();
  });

  it("frees the slot of a blank drawing", () => {
    saveSketch("q-1", addOp(emptyDoc(), stroke()));
    saveSketch("q-1", emptyDoc());

    expect(loadSketch("q-1")).toBeNull();
  });

  it("evicts the oldest drawings past the cap", () => {
    for (let i = 0; i <= MAX_SAVED_SKETCHES; i += 1) saveSketch(`q-${i}`, addOp(emptyDoc(), stroke(i)));

    expect(loadSketch("q-0")).toBeNull();
    expect(loadSketch(`q-${MAX_SAVED_SKETCHES}`)).not.toBeNull();
  });

  it("drops invalid stored operations instead of the whole drawing", () => {
    const doc = parseDoc({
      background: "#FFF",
      ops: [
        stroke(),
        { kind: "stroke", color: "not-a-colour", size: 4, points: [1, 1] },
        { kind: "rect", color: "#000000", size: 4, points: [1, 2, 3] },
        { kind: "teleport" },
        { kind: "erase", size: 9999, points: [1, 1] },
      ],
    });

    expect(doc?.background).toBe("#ffffff");
    expect(doc?.ops.map((op) => op.kind)).toEqual(["stroke", "erase"]);
    expect(doc?.ops[1]).toMatchObject({ size: MAX_SIZE });
    expect(parseDoc({ ops: [] })).toBeNull();
    expect(parseDoc("junk")).toBeNull();
  });
});
