import { type Rgb, hexToRgb, normalizeHex, rgbToHex } from "./color";

/** V-KIS sketch engine: the drawing model, its undo history, rendering and
 *  autosave. Kept free of React so every rule here is unit-testable.
 *
 *  The drawing is a list of operations replayed onto a transparent layer that
 *  sits over a solid background. The search sends that layer flattened onto the
 *  background as one opaque PNG: the PE server drops alpha with
 *  `convert("RGB")`, which would turn anything transparent black. */

/** Backing-store size. 16:9 like the keyframes: PE squashes both to 448×448
 *  (no centre crop), so a 16:9 drawing maps onto 16:9 frames the same way. */
export const SKETCH_WIDTH = 1024;
export const SKETCH_HEIGHT = 576;
export const DEFAULT_BACKGROUND = "#ffffff";
export const MIN_SIZE = 1;
export const MAX_SIZE = 240;

export type BrushTip = "round" | "square" | "soft";
export type SketchTool = "brush" | "eraser" | "line" | "rect" | "ellipse" | "lasso" | "fill" | "picker";

interface Paint {
  color: string; // #rrggbb
  alpha: number; // 0..1
}

/** One undoable drawing operation. Coordinates are backing-store pixels, stored
 *  flat (`[x0, y0, x1, y1, …]`) and rounded so an autosave stays small. */
export type SketchOp =
  | ({ kind: "stroke"; tip: BrushTip; size: number; points: number[] } & Paint)
  | { kind: "erase"; size: number; points: number[] }
  | ({ kind: "line"; tip: BrushTip; size: number; points: number[] } & Paint)
  | ({ kind: "rect" | "ellipse"; filled: boolean; size: number; points: number[] } & Paint)
  | ({ kind: "lasso"; points: number[] } & Paint)
  | { kind: "fill"; color: string; tolerance: number; points: number[] };

export interface SketchDoc {
  background: string;
  ops: SketchOp[];
}

export const emptyDoc = (): SketchDoc => ({ background: DEFAULT_BACKGROUND, ops: [] });

/** Nothing worth searching: no operation and the default background. */
export function isBlankDoc(doc: SketchDoc): boolean {
  return doc.ops.length === 0 && doc.background === DEFAULT_BACKGROUND;
}

export function addOp(doc: SketchDoc, op: SketchOp): SketchDoc {
  return { ...doc, ops: [...doc.ops, op] };
}

// ---- history -----------------------------------------------------------

export interface SketchHistory {
  past: SketchDoc[];
  present: SketchDoc;
  future: SketchDoc[];
}

/** Undo depth. States share their operation objects, so this costs references,
 *  not pixels. */
export const HISTORY_LIMIT = 200;

export const historyOf = (doc: SketchDoc): SketchHistory => ({ past: [], present: doc, future: [] });

/** History for a drawing restored from storage: every stored operation stays
 *  undoable one by one (up to the limit), as it was before the reload. */
export function restoreHistory(doc: SketchDoc | null): SketchHistory {
  if (!doc) return historyOf(emptyDoc());
  const first = Math.max(0, doc.ops.length - HISTORY_LIMIT);
  const past: SketchDoc[] = [];
  for (let count = first; count < doc.ops.length; count += 1) {
    past.push({ background: doc.background, ops: doc.ops.slice(0, count) });
  }
  return { past, present: doc, future: [] };
}

export function commitDoc(history: SketchHistory, doc: SketchDoc): SketchHistory {
  if (doc === history.present) return history;
  return {
    past: [...history.past, history.present].slice(-HISTORY_LIMIT),
    present: doc,
    future: [],
  };
}

export function undo(history: SketchHistory): SketchHistory {
  if (!history.past.length) return history;
  return {
    past: history.past.slice(0, -1),
    present: history.past[history.past.length - 1],
    future: [history.present, ...history.future],
  };
}

export function redo(history: SketchHistory): SketchHistory {
  if (!history.future.length) return history;
  return {
    past: [...history.past, history.present],
    present: history.future[0],
    future: history.future.slice(1),
  };
}

/** True when `next` is `prev` plus exactly one appended operation — the cached
 *  layer then only needs that operation drawn instead of a full replay. */
export function isAppend(prev: SketchDoc | null, next: SketchDoc): boolean {
  return (
    !!prev &&
    prev.background === next.background &&
    next.ops.length === prev.ops.length + 1 &&
    prev.ops.every((op, index) => op === next.ops[index])
  );
}

// ---- rendering ---------------------------------------------------------

/** Trace a freehand path, smoothed with quadratic curves through the midpoints
 *  of successive samples — mouse input reads as a polyline otherwise. */
function tracePath(ctx: CanvasRenderingContext2D, points: number[]) {
  const n = points.length >> 1;
  ctx.beginPath();
  if (!n) return;
  ctx.moveTo(points[0], points[1]);
  if (n === 1) {
    // A dab: a zero-length segment still gets the cap, so a click leaves a dot.
    ctx.lineTo(points[0] + 0.01, points[1]);
    return;
  }
  for (let i = 1; i < n - 1; i += 1) {
    const x = points[2 * i];
    const y = points[2 * i + 1];
    ctx.quadraticCurveTo(x, y, (x + points[2 * i + 2]) / 2, (y + points[2 * i + 3]) / 2);
  }
  ctx.lineTo(points[2 * n - 2], points[2 * n - 1]);
}

/** A soft tip is drawn as the blurred shadow of a copy of the path that sits far
 *  off the canvas. `ctx.filter` would be simpler but is not available in every
 *  browser; shadows are. */
const SOFT_SHADOW_OFFSET = 20_000;

function strokePath(
  ctx: CanvasRenderingContext2D,
  points: number[],
  { color, alpha, size, tip }: Paint & { size: number; tip: BrushTip },
) {
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.lineCap = tip === "square" ? "square" : "round";
  ctx.lineJoin = tip === "square" ? "miter" : "round";
  if (tip === "soft") {
    // Core width + blur chosen so the visible diameter stays ≈ `size` (what the
    // size ring shows) while the centre still reaches ~full coverage.
    ctx.lineWidth = Math.max(1, size * 0.6);
    ctx.shadowColor = color;
    ctx.shadowBlur = size * 0.25;
    ctx.shadowOffsetX = SOFT_SHADOW_OFFSET;
    ctx.translate(-SOFT_SHADOW_OFFSET, 0);
  } else {
    ctx.lineWidth = Math.max(1, size);
  }
  tracePath(ctx, points);
  ctx.stroke();
  ctx.restore();
}

function box(points: number[]) {
  const [x1, y1, x2, y2] = points;
  return { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1) };
}

/** Draw one operation onto a layer. `background` is only read by the bucket
 *  fill, which decides regions by what the operator actually sees. */
export function drawOp(ctx: CanvasRenderingContext2D, op: SketchOp, background: string) {
  switch (op.kind) {
    case "stroke":
    case "line":
      strokePath(ctx, op.points, op);
      return;
    case "erase":
      ctx.save();
      ctx.globalCompositeOperation = "destination-out";
      strokePath(ctx, op.points, { color: "#000000", alpha: 1, size: op.size, tip: "round" });
      ctx.restore();
      return;
    case "rect":
    case "ellipse": {
      const { x, y, w, h } = box(op.points);
      ctx.save();
      ctx.globalAlpha = op.alpha;
      ctx.fillStyle = op.color;
      ctx.strokeStyle = op.color;
      ctx.lineWidth = Math.max(1, op.size);
      ctx.lineJoin = "miter";
      ctx.beginPath();
      if (op.kind === "rect") ctx.rect(x, y, w, h);
      else ctx.ellipse(x + w / 2, y + h / 2, Math.max(0.5, w / 2), Math.max(0.5, h / 2), 0, 0, Math.PI * 2);
      if (op.filled) ctx.fill();
      else ctx.stroke();
      ctx.restore();
      return;
    }
    case "lasso":
      ctx.save();
      ctx.globalAlpha = op.alpha;
      ctx.fillStyle = op.color;
      tracePath(ctx, op.points);
      ctx.closePath();
      ctx.fill();
      ctx.restore();
      return;
    case "fill": {
      const { width, height } = ctx.canvas;
      const image = ctx.getImageData(0, 0, width, height);
      if (floodFill(image, hexToRgb(background), op.points[0], op.points[1], hexToRgb(op.color), op.tolerance)) {
        ctx.putImageData(image, 0, 0);
      }
      return;
    }
  }
}

/** Replay the whole document onto a (transparent) layer. */
export function renderLayer(ctx: CanvasRenderingContext2D, doc: SketchDoc) {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  for (const op of doc.ops) drawOp(ctx, op, doc.background);
}

/** Background + layer, opaque: what the operator sees and what PE is sent. */
export function composite(ctx: CanvasRenderingContext2D, layer: CanvasImageSource, background: string) {
  ctx.save();
  ctx.globalCompositeOperation = "source-over";
  ctx.globalAlpha = 1;
  ctx.fillStyle = background;
  ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.drawImage(layer, 0, 0);
  ctx.restore();
}

/** Rule-of-thirds guides: editor chrome only, never part of the export. */
export function drawGuides(ctx: CanvasRenderingContext2D) {
  const { width, height } = ctx.canvas;
  ctx.save();
  ctx.strokeStyle = "rgba(120, 130, 145, 0.45)";
  ctx.lineWidth = 1;
  ctx.setLineDash([6, 6]);
  ctx.beginPath();
  for (const ratio of [1 / 3, 2 / 3]) {
    ctx.moveTo(Math.round(width * ratio) + 0.5, 0);
    ctx.lineTo(Math.round(width * ratio) + 0.5, height);
    ctx.moveTo(0, Math.round(height * ratio) + 0.5);
    ctx.lineTo(width, Math.round(height * ratio) + 0.5);
  }
  ctx.stroke();
  ctx.restore();
}

/** What a layer pixel looks like over the background. */
function compositePixel(data: Uint8ClampedArray, offset: number, background: Rgb): Rgb {
  const a = data[offset + 3] / 255;
  return {
    r: Math.round(data[offset] * a + background.r * (1 - a)),
    g: Math.round(data[offset + 1] * a + background.g * (1 - a)),
    b: Math.round(data[offset + 2] * a + background.b * (1 - a)),
  };
}

/** Colour under (x, y) as the operator sees it (eyedropper). */
export function sampleColor(image: ImageData, background: string, x: number, y: number): string {
  const px = Math.min(image.width - 1, Math.max(0, Math.floor(x)));
  const py = Math.min(image.height - 1, Math.max(0, Math.floor(y)));
  return rgbToHex(compositePixel(image.data, (py * image.width + px) * 4, hexToRgb(background)));
}

/** Bucket fill of the contiguous region around (x, y), in place on `image`.
 *
 *  Regions are decided on the COMPOSITE (layer over background), so filling an
 *  empty area bounded by strokes works. `tolerance` is the largest per-channel
 *  difference still counted as the same region; the region is then grown by one
 *  pixel so the anti-aliased fringe of a boundary stroke does not leave a pale
 *  halo between the fill and the line. Returns false when nothing changes. */
export function floodFill(
  image: ImageData,
  background: Rgb,
  x: number,
  y: number,
  fill: Rgb,
  tolerance: number,
): boolean {
  const { data, width: w, height: h } = image;
  const px = Math.floor(x);
  const py = Math.floor(y);
  if (px < 0 || py < 0 || px >= w || py >= h) return false;
  const total = w * h;
  // Inlined `compositePixel`: this runs over every pixel of the canvas.
  const comp = new Uint8ClampedArray(total * 3);
  for (let i = 0; i < total; i += 1) {
    const o = i * 4;
    const a = data[o + 3] / 255;
    comp[i * 3] = data[o] * a + background.r * (1 - a);
    comp[i * 3 + 1] = data[o + 1] * a + background.g * (1 - a);
    comp[i * 3 + 2] = data[o + 2] * a + background.b * (1 - a);
  }
  const seed = py * w + px;
  const sr = comp[seed * 3];
  const sg = comp[seed * 3 + 1];
  const sb = comp[seed * 3 + 2];
  if (sr === fill.r && sg === fill.g && sb === fill.b) return false;

  const tol = Math.max(0, tolerance);
  const same = (i: number) =>
    Math.abs(comp[i * 3] - sr) <= tol &&
    Math.abs(comp[i * 3 + 1] - sg) <= tol &&
    Math.abs(comp[i * 3 + 2] - sb) <= tol;

  // Pixels are marked when pushed, so each enters the stack at most once.
  const mask = new Uint8Array(total);
  const stack = new Int32Array(total);
  let top = 0;
  mask[seed] = 1;
  stack[top++] = seed;
  while (top) {
    const i = stack[--top];
    const col = i % w;
    if (col > 0 && !mask[i - 1] && same(i - 1)) { mask[i - 1] = 1; stack[top++] = i - 1; }
    if (col < w - 1 && !mask[i + 1] && same(i + 1)) { mask[i + 1] = 1; stack[top++] = i + 1; }
    if (i >= w && !mask[i - w] && same(i - w)) { mask[i - w] = 1; stack[top++] = i - w; }
    if (i < total - w && !mask[i + w] && same(i + w)) { mask[i + w] = 1; stack[top++] = i + w; }
  }

  const paint = (i: number) => {
    const o = i * 4;
    data[o] = fill.r;
    data[o + 1] = fill.g;
    data[o + 2] = fill.b;
    data[o + 3] = 255;
  };
  for (let i = 0; i < total; i += 1) {
    if (mask[i] !== 1) continue;
    paint(i);
    const col = i % w;
    if (col > 0 && !mask[i - 1]) { mask[i - 1] = 2; paint(i - 1); }
    if (col < w - 1 && !mask[i + 1]) { mask[i + 1] = 2; paint(i + 1); }
    if (i >= w && !mask[i - w]) { mask[i - w] = 2; paint(i - w); }
    if (i < total - w && !mask[i + w]) { mask[i + w] = 2; paint(i + w); }
  }
  return true;
}

// ---- pointer input -----------------------------------------------------

/** Exponential stabilizer: 0 follows the pointer exactly, higher values lag and
 *  iron out mouse jitter into smoother curves. */
export function stabilize(prev: [number, number], next: [number, number], amount: number): [number, number] {
  const keep = Math.min(0.95, Math.max(0, amount));
  return [prev[0] + (next[0] - prev[0]) * (1 - keep), prev[1] + (next[1] - prev[1]) * (1 - keep)];
}

/** Samples closer than this to the previous one add nothing but bytes. */
export const MIN_POINT_GAP = 1.5;

export function farEnough(points: number[], x: number, y: number, gap = MIN_POINT_GAP): boolean {
  const n = points.length;
  return n < 2 || Math.hypot(x - points[n - 2], y - points[n - 1]) >= gap;
}

/** Shift-constrained end point: lines snap to 15° steps, boxes become squares
 *  and circles. */
export function constrainEnd(
  kind: "line" | "rect" | "ellipse",
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  shift: boolean,
): [number, number] {
  if (!shift) return [x2, y2];
  const dx = x2 - x1;
  const dy = y2 - y1;
  if (kind === "line") {
    const step = Math.PI / 12;
    const angle = Math.round(Math.atan2(dy, dx) / step) * step;
    const length = Math.hypot(dx, dy);
    return [x1 + Math.cos(angle) * length, y1 + Math.sin(angle) * length];
  }
  const side = Math.max(Math.abs(dx), Math.abs(dy));
  return [x1 + (dx < 0 ? -side : side), y1 + (dy < 0 ? -side : side)];
}

/** A drag worth keeping: a shape needs some extent, a lasso needs an area. */
export function isMeaningful(op: SketchOp): boolean {
  const p = op.points;
  switch (op.kind) {
    case "line":
      return Math.hypot(p[2] - p[0], p[3] - p[1]) >= 2;
    case "rect":
    case "ellipse":
      return Math.abs(p[2] - p[0]) >= 2 && Math.abs(p[3] - p[1]) >= 2;
    case "lasso":
      return p.length >= 6;
    default:
      return p.length >= 2;
  }
}

/** Keyboard size step: fine at small sizes, coarse at large ones. */
export function nextSize(size: number, direction: 1 | -1): number {
  const step = size < 10 ? 1 : size < 40 ? 4 : size < 100 ? 10 : 20;
  return Math.min(MAX_SIZE, Math.max(MIN_SIZE, size + direction * step));
}

// ---- keyboard ----------------------------------------------------------

export const TOOL_KEYS: Record<string, SketchTool> = {
  b: "brush", e: "eraser", l: "line", r: "rect", o: "ellipse", f: "lasso", g: "fill", i: "picker",
};

/** Marks the element whose keyboard focus hands the sketch shortcuts to it. */
export const SKETCH_SCOPE_ATTR = "data-sketch-scope";

/** Keys the sketch canvas handles while focus is inside it, which the console's
 *  global shortcuts must then leave alone. Enter is claimed as a whole: while
 *  drawing, a stray Enter must never open the submit guard (Ctrl+Enter searches
 *  the sketch instead). Arrows, v, k, t and the video keys still reach the
 *  console, so the result list stays drivable after a sketch search. */
export function ownsSketchKey(event: KeyboardEvent): boolean {
  const target = event.target;
  if (!(target instanceof Element) || !target.closest(`[${SKETCH_SCOPE_ATTR}]`)) return false;
  const key = event.key.toLowerCase();
  if (key === "enter" || key === "escape") return true;
  if (event.ctrlKey || event.metaKey) return key === "z" || key === "y";
  if (event.altKey) return false;
  return key in TOOL_KEYS || key === "[" || key === "]" || /^[0-9]$/.test(key);
}

// ---- autosave ----------------------------------------------------------

const STORAGE_PREFIX = "aic26.sketch.v1:";
const INDEX_KEY = "aic26.sketch.v1.index";
/** Drawings kept across reloads, least recently saved dropped first. */
export const MAX_SAVED_SKETCHES = 30;
const MAX_OPS = 5000;
const MAX_POINTS = 40_000;

const TIPS: BrushTip[] = ["round", "square", "soft"];
const finite = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value);
const clamp = (value: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, value));

function parsePoints(value: unknown, min: number, max: number): number[] | null {
  if (!Array.isArray(value) || value.length < min || value.length > max || value.length % 2) return null;
  return value.every(finite) ? (value as number[]) : null;
}

/** One stored operation, validated field by field; null drops it. Stored data
 *  can be stale or hand-edited, and a bad op must not break the whole drawing. */
function parseOp(raw: unknown): SketchOp | null {
  if (!raw || typeof raw !== "object") return null;
  const op = raw as Record<string, unknown>;
  const color = typeof op.color === "string" ? normalizeHex(op.color) : null;
  const alpha = finite(op.alpha) ? clamp(op.alpha, 0.01, 1) : 1;
  const size = finite(op.size) ? clamp(op.size, MIN_SIZE, MAX_SIZE) : null;
  const tip = TIPS.includes(op.tip as BrushTip) ? (op.tip as BrushTip) : "round";
  switch (op.kind) {
    case "stroke": {
      const points = parsePoints(op.points, 2, MAX_POINTS);
      return points && color && size ? { kind: "stroke", tip, color, alpha, size, points } : null;
    }
    case "erase": {
      const points = parsePoints(op.points, 2, MAX_POINTS);
      return points && size ? { kind: "erase", size, points } : null;
    }
    case "line": {
      const points = parsePoints(op.points, 4, 4);
      return points && color && size ? { kind: "line", tip, color, alpha, size, points } : null;
    }
    case "rect":
    case "ellipse": {
      const points = parsePoints(op.points, 4, 4);
      return points && color && size
        ? { kind: op.kind, filled: op.filled !== false, color, alpha, size, points }
        : null;
    }
    case "lasso": {
      const points = parsePoints(op.points, 6, MAX_POINTS);
      return points && color ? { kind: "lasso", color, alpha, points } : null;
    }
    case "fill": {
      const points = parsePoints(op.points, 2, 2);
      const tolerance = finite(op.tolerance) ? clamp(op.tolerance, 0, 255) : 32;
      return points && color ? { kind: "fill", color, tolerance, points } : null;
    }
    default:
      return null;
  }
}

export function parseDoc(raw: unknown): SketchDoc | null {
  if (!raw || typeof raw !== "object") return null;
  const value = raw as Record<string, unknown>;
  const background = typeof value.background === "string" ? normalizeHex(value.background) : null;
  if (!background || !Array.isArray(value.ops)) return null;
  const ops = value.ops.slice(0, MAX_OPS).map(parseOp).filter((op): op is SketchOp => op !== null);
  return { background, ops };
}

function readIndex(): string[] {
  try {
    const parsed = JSON.parse(localStorage.getItem(INDEX_KEY) ?? "[]");
    return Array.isArray(parsed) ? parsed.filter((item): item is string => typeof item === "string") : [];
  } catch {
    return [];
  }
}

function writeIndex(keys: string[]) {
  try {
    localStorage.setItem(INDEX_KEY, JSON.stringify(keys));
  } catch {
    // Storage is a convenience; the drawing itself lives in React state.
  }
}

export function loadSketch(key: string): SketchDoc | null {
  try {
    const text = localStorage.getItem(STORAGE_PREFIX + key);
    return text ? parseDoc(JSON.parse(text)) : null;
  } catch {
    return null;
  }
}

/** Persist the drawing of one question. A blank drawing removes the slot, so
 *  merely visiting a question never fills storage. Oldest drawings are evicted
 *  past `MAX_SAVED_SKETCHES`, and once more if the browser reports it full. */
export function saveSketch(key: string, doc: SketchDoc): boolean {
  const storageKey = STORAGE_PREFIX + key;
  let index = readIndex().filter((item) => item !== key);
  try {
    if (isBlankDoc(doc)) {
      localStorage.removeItem(storageKey);
      writeIndex(index);
      return true;
    }
  } catch {
    return false;
  }
  const text = JSON.stringify(doc);
  const evict = (keep: number) => {
    while (index.length > keep) {
      const oldest = index.shift();
      try {
        if (oldest !== undefined) localStorage.removeItem(STORAGE_PREFIX + oldest);
      } catch {
        // ignore
      }
    }
  };
  evict(MAX_SAVED_SKETCHES - 1);
  for (let attempt = 0; attempt < 2; attempt += 1) {
    try {
      localStorage.setItem(storageKey, text);
      writeIndex([...index, key]);
      return true;
    } catch {
      evict(Math.floor(index.length / 2));
    }
  }
  return false;
}
