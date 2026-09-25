import { useEffect, useRef, useState } from "react";
import { normalizeHex } from "../lib/color";
import {
  type BrushTip,
  type SketchDoc,
  type SketchHistory,
  type SketchOp,
  type SketchTool,
  MAX_SIZE,
  MIN_SIZE,
  SKETCH_HEIGHT,
  SKETCH_WIDTH,
  TOOL_KEYS,
  addOp,
  commitDoc,
  composite,
  constrainEnd,
  drawGuides,
  drawOp,
  emptyDoc,
  farEnough,
  restoreHistory,
  isAppend,
  isBlankDoc,
  isMeaningful,
  loadSketch,
  nextSize,
  redo,
  renderLayer,
  sampleColor,
  saveSketch,
  stabilize,
  undo,
} from "../lib/sketch";
import { ColorPicker } from "./ColorPicker";

interface Props {
  /** Called with the drawing flattened onto its background (PNG data URL).
   *  `suppressBlank` asks the backend to remove the direction PE gives blank
   *  frames from the query (app.canvas) — without it a mostly-flat sketch
   *  ranks title cards and fades first. */
  onSearch: (image: string, options: { suppressBlank: boolean }) => void;
  loading: boolean;
  /** `canvas_sketch_search` from /api/health; null while health is loading. */
  available: boolean | null;
  /** Autosave slot. The drawing belongs to the question this tab is on, so
   *  switching question swaps the drawing and a reload restores it. */
  storageKey: string;
  /** Which PE index answers, shown so the route of the query is never a guess. */
  indexLabel: string;
}

const TOOLS: { id: SketchTool; key: string; icon: string; label: string }[] = [
  { id: "brush", key: "B", icon: "✎", label: "Brush" },
  { id: "eraser", key: "E", icon: "⌫", label: "Eraser" },
  { id: "line", key: "L", icon: "╱", label: "Line — Shift snaps to 15°" },
  { id: "rect", key: "R", icon: "▭", label: "Rectangle — Shift for a square" },
  { id: "ellipse", key: "O", icon: "◯", label: "Ellipse — Shift for a circle" },
  { id: "lasso", key: "F", icon: "⬟", label: "Freeform fill — draw an outline, it is filled" },
  { id: "fill", key: "G", icon: "◧", label: "Bucket fill" },
  { id: "picker", key: "I", icon: "⊙", label: "Eyedropper — Alt+click works with every tool" },
];

const TIPS: { id: BrushTip; icon: string; label: string }[] = [
  { id: "round", icon: "●", label: "Round" },
  { id: "square", icon: "■", label: "Square" },
  { id: "soft", icon: "◌", label: "Soft (airbrush)" },
];

/** One-click colours in the toolbar; the full picker is one more click away. */
const QUICK_COLORS = [
  "#000000", "#ffffff", "#808080", "#e02020", "#ff8c1a", "#ffd83a", "#2e9e44",
  "#1e50dc", "#87c3ec", "#6aa84f", "#8b5a2b", "#555b63", "#f1c9a5", "#8a3ab9",
];

const PREFS_KEY = "aic26.sketch.v1.prefs";
const RECENT_KEY = "aic26.sketch.v1.recent";
const CUSTOM_KEY = "aic26.sketch.v1.swatches";
const AUTOSAVE_MS = 250;

interface Prefs {
  tip: BrushTip;
  color: string;
  alpha: number;
  size: number;
  eraserSize: number;
  filled: boolean;
  tolerance: number;
  smoothing: number;
  guides: boolean;
  ignoreBlank: boolean;
}

const DEFAULT_PREFS: Prefs = {
  tip: "round", color: "#1e50dc", alpha: 1, size: 18, eraserSize: 40,
  filled: true, tolerance: 32, smoothing: 0.35, guides: true, ignoreBlank: true,
};

function readJson(key: string): unknown {
  try {
    return JSON.parse(localStorage.getItem(key) ?? "null");
  } catch {
    return null;
  }
}

function writeJson(key: string, value: unknown) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Preferences are a convenience; a full or blocked storage changes nothing else.
  }
}

function loadPrefs(): Prefs {
  const raw = (readJson(PREFS_KEY) ?? {}) as Partial<Record<keyof Prefs, unknown>>;
  const num = (value: unknown, lo: number, hi: number, fallback: number) =>
    typeof value === "number" && Number.isFinite(value) ? Math.min(hi, Math.max(lo, value)) : fallback;
  return {
    tip: TIPS.some((tip) => tip.id === raw.tip) ? (raw.tip as BrushTip) : DEFAULT_PREFS.tip,
    color: (typeof raw.color === "string" && normalizeHex(raw.color)) || DEFAULT_PREFS.color,
    alpha: num(raw.alpha, 0.05, 1, DEFAULT_PREFS.alpha),
    size: num(raw.size, MIN_SIZE, MAX_SIZE, DEFAULT_PREFS.size),
    eraserSize: num(raw.eraserSize, MIN_SIZE, MAX_SIZE, DEFAULT_PREFS.eraserSize),
    filled: typeof raw.filled === "boolean" ? raw.filled : DEFAULT_PREFS.filled,
    tolerance: num(raw.tolerance, 0, 255, DEFAULT_PREFS.tolerance),
    smoothing: num(raw.smoothing, 0, 0.9, DEFAULT_PREFS.smoothing),
    guides: typeof raw.guides === "boolean" ? raw.guides : DEFAULT_PREFS.guides,
    ignoreBlank: typeof raw.ignoreBlank === "boolean" ? raw.ignoreBlank : DEFAULT_PREFS.ignoreBlank,
  };
}

function loadColors(key: string): string[] {
  const raw = readJson(key);
  if (!Array.isArray(raw)) return [];
  return raw
    .map((item) => (typeof item === "string" ? normalizeHex(item) : null))
    .filter((item): item is string => item !== null)
    .slice(0, 24);
}

/** Size slider on a log scale: 1–10 px needs fine control, 100–240 px does not. */
const sizeToSlider = (size: number) => Math.round((Math.log(size) / Math.log(MAX_SIZE)) * 1000);
const sliderToSize = (value: number) => Math.max(MIN_SIZE, Math.round(MAX_SIZE ** (value / 1000)));

interface Surface {
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
}

/** An offscreen surface at the backing size, or null when the browser cannot
 *  give a 2D context (context loss, or jsdom). Drawing is skipped then rather
 *  than crashing the console. */
function makeSurface(readback: boolean): Surface | null {
  try {
    const canvas = document.createElement("canvas");
    canvas.width = SKETCH_WIDTH;
    canvas.height = SKETCH_HEIGHT;
    const ctx = canvas.getContext("2d", readback ? { willReadFrequently: true } : undefined);
    return ctx ? { canvas, ctx } : null;
  } catch {
    return null;
  }
}

function context2d(canvas: HTMLCanvasElement | null): CanvasRenderingContext2D | null {
  try {
    return canvas?.getContext("2d") ?? null;
  } catch {
    return null;
  }
}

interface Live {
  pointerId: number;
  op: SketchOp;
  /** Stabilized pointer position, trailing the raw one. */
  smoothed: [number, number];
}

/** V-KIS sketch canvas: the operator redraws the clip and the drawing itself is
 *  the query — flattened to one PNG, encoded by PE's image tower and matched
 *  against the keyframe vectors. No object labels, no detector: PE compares
 *  layout and colour, so large colour areas matter more than fine detail. */
export function CanvasPanel({ onSearch, loading, available, storageKey, indexLabel }: Props) {
  const [history, setHistory] = useState<SketchHistory>(() => restoreHistory(loadSketch(storageKey)));
  const doc = history.present;
  const [prefs, setPrefs] = useState<Prefs>(loadPrefs);
  const patch = (update: Partial<Prefs>) => setPrefs((current) => ({ ...current, ...update }));
  const [tool, setToolState] = useState<SketchTool>("brush");
  const [picker, setPicker] = useState<null | "paint" | "background">(null);
  // Live background while its picker is open: committed once on close, so one
  // drag across the colour square is one undo step, not hundreds.
  const [backgroundPreview, setBackgroundPreview] = useState<string | null>(null);
  const [recent, setRecent] = useState<string[]>(() => loadColors(RECENT_KEY));
  const [custom, setCustom] = useState<string[]>(() => loadColors(CUSTOM_KEY));
  const [collapsed, setCollapsed] = useState(false);
  const [large, setLarge] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);

  const displayRef = useRef<HTMLCanvasElement>(null);
  const ringRef = useRef<HTMLDivElement>(null);
  const layerRef = useRef<Surface | null | undefined>(undefined);
  const scratchRef = useRef<Surface | null | undefined>(undefined);
  const renderedRef = useRef<SketchDoc | null>(null);
  const liveRef = useRef<Live | null>(null);
  const frameRef = useRef<number | null>(null);
  const paintToolRef = useRef<SketchTool>("brush");
  // The paint loop runs from animation frames, outside React's render, so it
  // reads the latest state through refs.
  const docRef = useRef(doc);
  docRef.current = doc;
  const backgroundRef = useRef(doc.background);
  backgroundRef.current = backgroundPreview ?? doc.background;
  const guidesRef = useRef(prefs.guides);
  guidesRef.current = prefs.guides;

  const blank = isBlankDoc(doc);

  // The eraser keeps its own size, as in any paint program: switching back to
  // the brush must not leave it at eraser width.
  const sizeKey = tool === "eraser" ? "eraserSize" : "size";
  const setSize = (value: number) => patch(tool === "eraser" ? { eraserSize: value } : { size: value });

  function setTool(next: SketchTool) {
    if (next !== "picker") paintToolRef.current = next;
    setToolState(next);
  }

  // ---- rendering ----
  function surface(ref: React.MutableRefObject<Surface | null | undefined>, readback: boolean): Surface | null {
    if (ref.current === undefined) ref.current = makeSurface(readback);
    return ref.current;
  }

  /** Bring the committed layer up to date with the document: one operation
   *  drawn when a stroke was just added, a full replay after undo/redo/clear. */
  function syncLayer(): Surface | null {
    const layer = surface(layerRef, true);
    if (!layer) return null;
    const current = docRef.current;
    if (renderedRef.current !== current) {
      if (isAppend(renderedRef.current, current)) {
        drawOp(layer.ctx, current.ops[current.ops.length - 1], current.background);
      } else {
        renderLayer(layer.ctx, current);
      }
      renderedRef.current = current;
    }
    return layer;
  }

  function paint() {
    const ctx = context2d(displayRef.current);
    const layer = syncLayer();
    if (!ctx || !layer) return;
    const background = backgroundRef.current;
    const op = liveRef.current?.op;
    if (op?.kind === "erase") {
      // An eraser cuts holes in the layer, so it is previewed on a copy of it.
      const scratch = surface(scratchRef, false);
      if (!scratch) return;
      scratch.ctx.clearRect(0, 0, SKETCH_WIDTH, SKETCH_HEIGHT);
      scratch.ctx.drawImage(layer.canvas, 0, 0);
      drawOp(scratch.ctx, op, background);
      composite(ctx, scratch.canvas, background);
    } else {
      composite(ctx, layer.canvas, background);
      if (op) drawOp(ctx, op, background);
      if (op?.kind === "lasso") {
        ctx.save();
        ctx.strokeStyle = "rgba(0, 0, 0, 0.6)";
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        for (let i = 0; i < op.points.length; i += 2) {
          if (i) ctx.lineTo(op.points[i], op.points[i + 1]);
          else ctx.moveTo(op.points[i], op.points[i + 1]);
        }
        ctx.closePath();
        ctx.stroke();
        ctx.restore();
      }
    }
    if (guidesRef.current) drawGuides(ctx);
  }

  function schedulePaint() {
    if (frameRef.current !== null) return;
    const run = () => {
      frameRef.current = null;
      paint();
    };
    frameRef.current =
      typeof window.requestAnimationFrame === "function"
        ? window.requestAnimationFrame(run)
        : window.setTimeout(run, 16);
  }

  useEffect(() => {
    paint();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [doc, prefs.guides, backgroundPreview, collapsed]);

  // ---- autosave ----
  const loadedKey = useRef(storageKey);
  const pendingSave = useRef<{ key: string; doc: SketchDoc } | null>(null);
  const flushSave = useRef(() => {
    const pending = pendingSave.current;
    pendingSave.current = null;
    if (pending) saveSketch(pending.key, pending.doc);
  }).current;

  useEffect(() => {
    // Until the reload below has run, `doc` still belongs to the previous key.
    if (loadedKey.current !== storageKey) return;
    pendingSave.current = { key: storageKey, doc };
    const timer = window.setTimeout(flushSave, AUTOSAVE_MS);
    return () => window.clearTimeout(timer);
  }, [doc, storageKey, flushSave]);

  useEffect(() => {
    if (loadedKey.current === storageKey) return;
    flushSave(); // the last edit of the question being left
    loadedKey.current = storageKey;
    liveRef.current = null;
    setBackgroundPreview(null);
    setHistory(restoreHistory(loadSketch(storageKey)));
  }, [storageKey, flushSave]);

  useEffect(() => {
    window.addEventListener("pagehide", flushSave);
    return () => {
      window.removeEventListener("pagehide", flushSave);
      flushSave();
    };
  }, [flushSave]);

  useEffect(() => writeJson(PREFS_KEY, prefs), [prefs]);
  useEffect(() => writeJson(RECENT_KEY, recent), [recent]);
  useEffect(() => writeJson(CUSTOM_KEY, custom), [custom]);

  // ---- document edits ----
  function rememberColor(hex: string) {
    setRecent((list) => [hex, ...list.filter((color) => color !== hex)].slice(0, 12));
  }

  function pushOp(op: SketchOp) {
    setHistory((current) => commitDoc(current, addOp(current.present, op)));
    if ("color" in op) rememberColor(op.color);
  }

  function clearAll() {
    liveRef.current = null;
    setHistory((current) => (isBlankDoc(current.present) ? current : commitDoc(current, emptyDoc())));
  }

  function closePicker(accepted = true) {
    if (accepted && picker === "background" && backgroundPreview && backgroundPreview !== doc.background) {
      const background = backgroundPreview;
      setHistory((current) => commitDoc(current, { ...current.present, background }));
    }
    setBackgroundPreview(null);
    setPicker(null);
  }

  // ---- pointer ----
  /** Client → backing-store coordinates, robust to letterboxing. */
  function toCanvas(clientX: number, clientY: number): [number, number] {
    const rect = displayRef.current?.getBoundingClientRect();
    if (!rect || !rect.width || !rect.height) return [SKETCH_WIDTH / 2, SKETCH_HEIGHT / 2];
    const scale = Math.min(rect.width / SKETCH_WIDTH, rect.height / SKETCH_HEIGHT);
    const offsetX = (rect.width - SKETCH_WIDTH * scale) / 2;
    const offsetY = (rect.height - SKETCH_HEIGHT * scale) / 2;
    return [(clientX - rect.left - offsetX) / scale, (clientY - rect.top - offsetY) / scale];
  }

  function moveRing(clientX: number, clientY: number) {
    const ring = ringRef.current;
    const rect = displayRef.current?.getBoundingClientRect();
    if (!ring || !rect || !rect.width) return;
    const sized = tool === "brush" || tool === "eraser" || tool === "line";
    const diameter = (tool === "eraser" ? prefs.eraserSize : prefs.size) * (rect.width / SKETCH_WIDTH);
    ring.hidden = !sized || diameter < 4;
    ring.classList.toggle("square", tool !== "eraser" && prefs.tip === "square");
    ring.style.left = `${clientX - rect.left}px`;
    ring.style.top = `${clientY - rect.top}px`;
    ring.style.width = `${diameter}px`;
    ring.style.height = `${diameter}px`;
  }

  /** The colour the operator sees at (x, y), or null outside the canvas. */
  function colorAt(x: number, y: number): string | null {
    if (x < 0 || y < 0 || x >= SKETCH_WIDTH || y >= SKETCH_HEIGHT) return null;
    const layer = syncLayer();
    if (!layer) return null;
    try {
      const image = layer.ctx.getImageData(Math.floor(x), Math.floor(y), 1, 1);
      return sampleColor(image, backgroundRef.current, 0, 0);
    } catch {
      return null;
    }
  }

  function pickAt(x: number, y: number) {
    const hex = colorAt(x, y);
    if (!hex) return;
    patch({ color: hex });
    rememberColor(hex);
    if (tool === "picker") setTool(paintToolRef.current);
  }

  function fillAt(x: number, y: number) {
    if (x < 0 || y < 0 || x >= SKETCH_WIDTH || y >= SKETCH_HEIGHT) return;
    if (colorAt(x, y) === prefs.color) return; // already that colour: no undo step for nothing
    pushOp({ kind: "fill", color: prefs.color, tolerance: prefs.tolerance, points: [Math.floor(x), Math.floor(y)] });
  }

  function onPointerDown(event: React.PointerEvent<HTMLCanvasElement>) {
    if (event.pointerType === "mouse" && event.button !== 0) return;
    event.preventDefault();
    event.currentTarget.focus({ preventScroll: true });
    setExportError(null);
    const [x, y] = toCanvas(event.clientX, event.clientY);
    const active: SketchTool = event.altKey ? "picker" : tool;
    if (active === "picker") return pickAt(x, y);
    if (active === "fill") return fillAt(x, y);
    event.currentTarget.setPointerCapture?.(event.pointerId);
    const px = Math.round(x);
    const py = Math.round(y);
    const ink = { color: prefs.color, alpha: prefs.alpha };
    let op: SketchOp;
    switch (active) {
      case "brush":
        op = { kind: "stroke", tip: prefs.tip, size: prefs.size, ...ink, points: [px, py] };
        break;
      case "eraser":
        op = { kind: "erase", size: prefs.eraserSize, points: [px, py] };
        break;
      case "lasso":
        op = { kind: "lasso", ...ink, points: [px, py] };
        break;
      case "line":
        op = { kind: "line", tip: prefs.tip, size: prefs.size, ...ink, points: [px, py, px, py] };
        break;
      default:
        op = { kind: active, filled: prefs.filled, size: prefs.size, ...ink, points: [px, py, px, py] };
    }
    liveRef.current = { pointerId: event.pointerId, op, smoothed: [x, y] };
    schedulePaint();
  }

  function extend(live: Live, clientX: number, clientY: number, shift: boolean) {
    const [x, y] = toCanvas(clientX, clientY);
    const op = live.op;
    if (op.kind === "stroke" || op.kind === "erase" || op.kind === "lasso") {
      const amount = op.kind === "erase" ? Math.min(prefs.smoothing, 0.2) : prefs.smoothing;
      live.smoothed = stabilize(live.smoothed, [x, y], amount);
      const px = Math.round(live.smoothed[0]);
      const py = Math.round(live.smoothed[1]);
      if (farEnough(op.points, px, py)) op.points.push(px, py);
    } else if (op.kind === "line" || op.kind === "rect" || op.kind === "ellipse") {
      const [ex, ey] = constrainEnd(op.kind, op.points[0], op.points[1], x, y, shift);
      op.points[2] = Math.round(ex);
      op.points[3] = Math.round(ey);
    }
  }

  function onPointerMove(event: React.PointerEvent<HTMLCanvasElement>) {
    moveRing(event.clientX, event.clientY);
    const live = liveRef.current;
    if (!live || live.pointerId !== event.pointerId) return;
    // High-rate mice and pens deliver several samples per frame; using all of
    // them is what keeps fast curves round instead of faceted.
    const native = event.nativeEvent;
    const coalesced = typeof native.getCoalescedEvents === "function" ? native.getCoalescedEvents() : [];
    for (const sample of coalesced.length ? coalesced : [native]) {
      extend(live, sample.clientX, sample.clientY, sample.shiftKey);
    }
    schedulePaint();
  }

  function finish(event: React.PointerEvent<HTMLCanvasElement>, keep: boolean) {
    const live = liveRef.current;
    if (!live || live.pointerId !== event.pointerId) return;
    liveRef.current = null;
    const op = live.op;
    if (keep && (op.kind === "stroke" || op.kind === "erase" || op.kind === "lasso")) {
      // The stabilizer trails the pointer; end where the pointer actually lifted.
      const [x, y] = toCanvas(event.clientX, event.clientY);
      const px = Math.round(x);
      const py = Math.round(y);
      if (farEnough(op.points, px, py, 1)) op.points.push(px, py);
    }
    if (keep && isMeaningful(op)) pushOp(op);
    else schedulePaint();
  }

  // ---- keyboard (only while focus is inside the panel) ----
  function onKeyDown(event: React.KeyboardEvent<HTMLDivElement>) {
    const target = event.target as HTMLElement;
    const typing = ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName);
    const mod = event.ctrlKey || event.metaKey;
    const key = event.key.toLowerCase();
    if (key === "enter" && mod) {
      event.preventDefault();
      search();
      return;
    }
    if (typing) return;
    const shift = event.shiftKey;
    if (mod && key === "z") {
      event.preventDefault();
      setHistory((current) => (shift ? redo(current) : undo(current)));
      return;
    }
    if (mod && key === "y") {
      event.preventDefault();
      setHistory(redo);
      return;
    }
    if (mod || event.altKey) return;
    if (key === "escape") {
      if (liveRef.current) {
        liveRef.current = null;
        schedulePaint();
      } else if (picker) {
        closePicker(false);
      }
      return;
    }
    if (key in TOOL_KEYS) {
      event.preventDefault();
      setTool(TOOL_KEYS[key]);
      return;
    }
    if (key === "[" || key === "]") {
      event.preventDefault();
      const direction = key === "]" ? 1 : -1;
      setSize(nextSize(prefs[sizeKey], direction));
      return;
    }
    if (/^[0-9]$/.test(key)) {
      event.preventDefault();
      patch({ alpha: key === "0" ? 1 : Number(key) / 10 });
    }
  }

  // ---- search ----
  /** The drawing flattened onto its background: opaque, chrome-free. */
  function exportPng(): string | null {
    const layer = syncLayer();
    const out = makeSurface(false);
    if (!layer || !out) return null;
    composite(out.ctx, layer.canvas, docRef.current.background);
    try {
      const url = out.canvas.toDataURL("image/png");
      return url.startsWith("data:image/png;base64,") ? url : null;
    } catch {
      return null;
    }
  }

  function search() {
    if (loading || isBlankDoc(docRef.current)) return;
    const image = exportPng();
    if (!image) {
      setExportError("This browser could not export the drawing.");
      return;
    }
    setExportError(null);
    onSearch(image, { suppressBlank: prefs.ignoreBlank });
  }

  // ---- view ----
  const shape = tool === "rect" || tool === "ellipse";
  const showSize = tool === "brush" || tool === "eraser" || tool === "line" || (shape && !prefs.filled);
  const showTip = tool === "brush" || tool === "line";
  const showAlpha = tool === "brush" || tool === "line" || shape || tool === "lasso";
  const extraRecent = recent.filter((color) => !QUICK_COLORS.includes(color)).slice(0, 6);

  const pickerFor = (target: "paint" | "background", placement: "below" | "above") => (
    <ColorPicker
      title={target === "paint" ? "Colour picker (brush)" : "Colour picker (background)"}
      value={target === "paint" ? prefs.color : doc.background}
      onChange={(hex) => (target === "paint" ? patch({ color: hex }) : setBackgroundPreview(hex))}
      onClose={closePicker}
      recent={recent}
      custom={custom}
      onAddCustom={(hex) => setCustom((list) => [...list.filter((color) => color !== hex), hex].slice(-24))}
      onRemoveCustom={(hex) => setCustom((list) => list.filter((color) => color !== hex))}
      placement={placement}
    />
  );

  const searchButton = (
    <button
      className="btn primary"
      onClick={search}
      disabled={loading || blank}
      title="Search with this sketch (Ctrl+Enter)"
      data-testid="canvas-search"
    >
      {loading ? "Searching…" : "Search sketch"} <span className="kbd">Ctrl ↵</span>
    </button>
  );

  return (
    <div className="panel canvas-panel" data-testid="canvas-panel" data-sketch-scope="" onKeyDown={onKeyDown}>
      <div className="row between sketch-head">
        <h3>
          <button
            className="canvas-collapse"
            onClick={() => setCollapsed((value) => !value)}
            data-testid="canvas-collapse"
            title={collapsed ? "Open the sketch canvas" : "Collapse the sketch canvas"}
          >
            {collapsed ? "▸" : "▾"}
          </button>{" "}
          Sketch · V-KIS
        </h3>
        <span
          className="sketch-route"
          title="The drawing is encoded by PE-Core-G14's image tower and matched against this profile's keyframe vectors"
          data-testid="canvas-route"
        >
          PE image → {indexLabel}
        </span>
        <div className="row" style={{ gap: 4 }}>
          {collapsed ? searchButton : (
            <>
              <button
                className={`btn sm ghost${prefs.guides ? " on" : ""}`}
                aria-pressed={prefs.guides}
                onClick={() => patch({ guides: !prefs.guides })}
                title="Rule-of-thirds guides (never sent to the search)"
                data-testid="canvas-guides"
              >
                ⋕
              </button>
              <button
                className={`btn sm ghost${large ? " on" : ""}`}
                aria-pressed={large}
                onClick={() => setLarge((value) => !value)}
                title={large ? "Smaller canvas" : "Larger canvas"}
                data-testid="canvas-large"
              >
                {large ? "⤡" : "⤢"}
              </button>
            </>
          )}
        </div>
      </div>

      {available === false && (
        <div className="canvas-warn" data-testid="canvas-unavailable">
          ⚠ PE image search is down on this profile (PE encoder or Milvus). You can keep drawing; the
          search will work again once it is back.
        </div>
      )}
      {exportError && <div className="canvas-warn">⚠ {exportError}</div>}

      {!collapsed && (
        <>
          <div className="sketch-toolbar" role="toolbar" aria-label="Sketch tools">
            <div className="seg sm sketch-tools">
              {TOOLS.map((item) => (
                <button
                  key={item.id}
                  className={tool === item.id ? "active" : ""}
                  aria-pressed={tool === item.id}
                  onClick={() => setTool(item.id)}
                  title={`${item.label} (${item.key})`}
                  data-testid={`canvas-tool-${item.id}`}
                >
                  {item.icon}
                </button>
              ))}
            </div>
            <div className="sketch-colors">
              <span className="sketch-anchor">
                <button
                  className="sketch-current"
                  style={{ background: prefs.color }}
                  onClick={() => setPicker(picker === "paint" ? null : "paint")}
                  title={`${prefs.color} — open the full RGB picker`}
                  aria-label="Brush colour"
                  data-testid="canvas-color-current"
                />
                {picker === "paint" && pickerFor("paint", "below")}
              </span>
              {QUICK_COLORS.map((color) => (
                <button
                  key={color}
                  className={`swatch${prefs.color === color ? " on" : ""}`}
                  style={{ background: color }}
                  title={color}
                  onClick={() => patch({ color })}
                  data-testid={`canvas-swatch-${color.slice(1)}`}
                />
              ))}
              {extraRecent.length > 0 && <span className="sketch-divider" />}
              {extraRecent.map((color) => (
                <button
                  key={color}
                  className={`swatch${prefs.color === color ? " on" : ""}`}
                  style={{ background: color }}
                  title={`${color} (recent)`}
                  onClick={() => patch({ color })}
                />
              ))}
            </div>
          </div>

          <div className="sketch-props" data-testid="canvas-props">
            {showTip && (
              <div className="seg sm" role="group" aria-label="Brush tip">
                {TIPS.map((tip) => (
                  <button
                    key={tip.id}
                    className={prefs.tip === tip.id ? "active" : ""}
                    onClick={() => patch({ tip: tip.id })}
                    title={tip.label}
                    data-testid={`canvas-tip-${tip.id}`}
                  >
                    {tip.icon} {tip.label.split(" ")[0]}
                  </button>
                ))}
              </div>
            )}
            {shape && (
              <div className="seg sm" role="group" aria-label="Shape style">
                <button className={!prefs.filled ? "active" : ""} onClick={() => patch({ filled: false })} data-testid="canvas-shape-outline">
                  ▭ Outline
                </button>
                <button className={prefs.filled ? "active" : ""} onClick={() => patch({ filled: true })} data-testid="canvas-shape-filled">
                  ▬ Filled
                </button>
              </div>
            )}
            {showSize && (
              <label className="sketch-prop" title="Size in canvas pixels ( [ and ] )">
                Size
                <input
                  type="range"
                  min={0}
                  max={1000}
                  value={sizeToSlider(prefs[sizeKey])}
                  onChange={(event) => setSize(sliderToSize(Number(event.target.value)))}
                  data-testid="canvas-size"
                />
                <input
                  className="sketch-num"
                  type="number"
                  min={MIN_SIZE}
                  max={MAX_SIZE}
                  value={prefs[sizeKey]}
                  onChange={(event) => {
                    const value = Number(event.target.value);
                    if (Number.isFinite(value) && value >= MIN_SIZE) setSize(Math.min(MAX_SIZE, Math.round(value)));
                  }}
                  data-testid="canvas-size-value"
                />
                px
              </label>
            )}
            {showAlpha && (
              <label className="sketch-prop" title="Opacity (keys 1–9, 0 = 100%)">
                Opacity
                <input
                  type="range"
                  min={5}
                  max={100}
                  value={Math.round(prefs.alpha * 100)}
                  onChange={(event) => patch({ alpha: Number(event.target.value) / 100 })}
                  data-testid="canvas-opacity"
                />
                <span className="sketch-val">{Math.round(prefs.alpha * 100)}%</span>
              </label>
            )}
            {(tool === "brush" || tool === "lasso") && (
              <label className="sketch-prop" title="Steadies a shaky mouse; higher values lag the pointer more">
                Smoothing
                <input
                  type="range"
                  min={0}
                  max={90}
                  value={Math.round(prefs.smoothing * 100)}
                  onChange={(event) => patch({ smoothing: Number(event.target.value) / 100 })}
                  data-testid="canvas-smoothing"
                />
                <span className="sketch-val">{Math.round(prefs.smoothing * 100)}%</span>
              </label>
            )}
            {tool === "fill" && (
              <label className="sketch-prop" title="How different a colour may be and still count as the same area">
                Tolerance
                <input
                  type="range"
                  min={0}
                  max={128}
                  value={prefs.tolerance}
                  onChange={(event) => patch({ tolerance: Number(event.target.value) })}
                  data-testid="canvas-tolerance"
                />
                <span className="sketch-val">{prefs.tolerance}</span>
              </label>
            )}
            {tool === "lasso" && <span className="hint">Draw an outline — it closes and fills on release.</span>}
            {tool === "fill" && <span className="hint">Click an area to fill it (e.g. sky above a horizon line).</span>}
            {tool === "picker" && <span className="hint">Click the drawing to take its colour.</span>}
          </div>

          <div className={`canvas-stage${large ? " large" : ""}`}>
            <canvas
              ref={displayRef}
              className={`canvas-surface tool-${tool}`}
              width={SKETCH_WIDTH}
              height={SKETCH_HEIGHT}
              tabIndex={0}
              aria-label="Sketch canvas"
              onPointerDown={onPointerDown}
              onPointerMove={onPointerMove}
              onPointerUp={(event) => finish(event, true)}
              onPointerCancel={(event) => finish(event, false)}
              onLostPointerCapture={(event) => finish(event, true)}
              onPointerLeave={() => {
                if (ringRef.current) ringRef.current.hidden = true;
              }}
              onContextMenu={(event) => event.preventDefault()}
              data-testid="canvas-board"
            />
            <div ref={ringRef} className="sketch-ring" hidden />
            {blank && (
              <div className="canvas-empty">
                Draw what you remember — big colour areas first (sky, road, walls), then the main shapes.
                <br />
                B brush · G fill · F freeform fill · R/O shapes · Ctrl+Z undo · Ctrl+Enter search
              </div>
            )}
          </div>

          <div className="row between sketch-foot">
            <div className="row" style={{ gap: 6 }}>
              <button
                className="btn sm"
                onClick={() => setHistory(undo)}
                disabled={!history.past.length}
                title="Undo (Ctrl+Z)"
                data-testid="canvas-undo"
              >
                ↶ Undo
              </button>
              <button
                className="btn sm"
                onClick={() => setHistory(redo)}
                disabled={!history.future.length}
                title="Redo (Ctrl+Shift+Z / Ctrl+Y)"
                data-testid="canvas-redo"
              >
                ↷ Redo
              </button>
              <button
                className="btn sm"
                onClick={clearAll}
                disabled={blank}
                title="Clear the drawing (undoable)"
                data-testid="canvas-clear"
              >
                Clear
              </button>
              <span className="sketch-anchor sketch-bg">
                Background
                <button
                  className="swatch"
                  style={{ background: backgroundPreview ?? doc.background }}
                  onClick={() => (picker === "background" ? closePicker() : setPicker("background"))}
                  title="Background colour (what the eraser reveals)"
                  data-testid="canvas-background"
                />
                {picker === "background" && pickerFor("background", "above")}
              </span>
            </div>
            <div className="row" style={{ gap: 6 }}>
              <label
                className="check-toggle"
                title="On: search ignores how 'blank' the sketch looks, so flat colour areas match frames of that colour instead of empty title cards and fades. Off: for a genuinely plain scene (e.g. a speaker in front of a white wall)."
              >
                <input
                  type="checkbox"
                  checked={prefs.ignoreBlank}
                  onChange={(event) => patch({ ignoreBlank: event.target.checked })}
                  data-testid="canvas-ignore-blank"
                />
                Ignore blank frames
              </label>
              {searchButton}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
