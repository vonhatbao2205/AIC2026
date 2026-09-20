import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api/client";
import type { CanvasMode, CanvasObjectSpec, CanvasPalette, CanvasSpec } from "../api/types";
import {
  CLAMP,
  FALLBACK_COLORS,
  FALLBACK_LABELS,
  makeObject,
  moveObject,
  outlineFor,
  resizeObject,
} from "../lib/canvas";
import { type Stroke, drawScene, eraseAt, hitTest } from "../lib/canvasDraw";

interface Props {
  onSearch: (canvas: CanvasSpec) => void;
  loading: boolean;
  /** Sentences the backend generated from the last canvas (explainability). */
  generatedQueries: string[];
  objectSearchAvailable: boolean | null;
}

type Tool = "select" | "brush" | "erase";

type Drag =
  | { kind: "move"; id: string; grabX: number; grabY: number }
  | { kind: "resize"; id: string }
  | { kind: "stroke" }
  | null;

// Backing-store size. 16:9 at a resolution PE (448²) can be downscaled from
// without losing the strokes, while staying small enough for a data URL.
const RENDER_WIDTH = 1024;
const RENDER_HEIGHT = 576;

/** A 2D context, or null when the surface cannot give one (context loss in a
 * real browser, an unimplemented canvas under jsdom). Drawing is skipped rather
 * than crashing the panel. */
function context2d(canvas: HTMLCanvasElement | null): CanvasRenderingContext2D | null {
  try {
    return canvas?.getContext("2d") ?? null;
  } catch {
    return null;
  }
}

/** V-KIS canvas: the operator redraws the clip.
 *
 * Everything is painted into one real <canvas>: object silhouettes and freehand
 * strokes. That is what lets the same surface serve two purposes — the objects
 * stay structured (label + bbox + colour) for the OD/text channels, and the
 * rendered pixels can be handed to PE's image encoder for the things that have
 * no OD label at all, like a rice field or an empty sky. */
export function CanvasPanel({ onSearch, loading, generatedQueries, objectSearchAvailable }: Props) {
  const [palette, setPalette] = useState<CanvasPalette | null>(null);
  const [objects, setObjects] = useState<CanvasObjectSpec[]>([]);
  const [strokes, setStrokes] = useState<Stroke[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [tool, setTool] = useState<Tool>("select");
  const [brushColor, setBrushColor] = useState("blue");
  const [brushWidth, setBrushWidth] = useState(0.035);
  const [mode, setMode] = useState<CanvasMode>("rough");
  const [actionText, setActionText] = useState("");
  const [filter, setFilter] = useState("");
  const [useRaster, setUseRaster] = useState(true);
  const [collapsed, setCollapsed] = useState(false);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const dragRef = useRef<Drag>(null);

  useEffect(() => {
    api.canvasPalette().then(setPalette).catch(() => setPalette(null));
  }, []);

  const colors = palette?.colors ?? FALLBACK_COLORS;
  const colorHex = useCallback(
    (name: string | null) => colors.find((color) => color.name === name)?.hex ?? null,
    [colors],
  );
  const labels = useMemo(() => {
    const all = palette?.labels ?? FALLBACK_LABELS.map((label) => ({ label, label_vi: null, colorable: true }));
    const needle = filter.trim().toLowerCase();
    if (!needle) return all.slice(0, 30);
    return all
      .filter((item) => item.label.includes(needle) || (item.label_vi ?? "").includes(needle))
      .slice(0, 30);
  }, [palette, filter]);
  const colorable = useMemo(
    () => new Set((palette?.labels ?? []).filter((item) => item.colorable).map((item) => item.label)),
    [palette],
  );
  const selected = objects.find((object) => object.id === selectedId) ?? null;
  const selectedColorable = selected ? !palette || colorable.has(selected.label) : false;

  // ---- rendering ----
  useEffect(() => {
    const context = context2d(canvasRef.current);
    if (!context) return;
    drawScene(context, RENDER_WIDTH, RENDER_HEIGHT, objects, strokes, {
      chrome: true,
      selectedId,
      outlineFor,
      colorHex,
    });
  }, [objects, strokes, selectedId, colorHex]);

  /** Re-render without the editor chrome so PE sees the drawing, not the tool. */
  function renderRaster(): string | null {
    const offscreen = document.createElement("canvas");
    offscreen.width = RENDER_WIDTH;
    offscreen.height = RENDER_HEIGHT;
    const context = context2d(offscreen);
    if (!context) return null;
    drawScene(context, RENDER_WIDTH, RENDER_HEIGHT, objects, strokes, { chrome: false, colorHex });
    return offscreen.toDataURL("image/png");
  }

  function pointOf(event: { clientX: number; clientY: number }): [number, number] {
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return [0.5, 0.5];
    return [CLAMP((event.clientX - rect.left) / rect.width), CLAMP((event.clientY - rect.top) / rect.height)];
  }

  function addObject(label: string, cx = 0.5, cy = 0.5) {
    const highest = objects.reduce((max, object) => Math.max(max, Number(object.id.slice(1)) || 0), 0);
    const object = { ...makeObject(label, cx, cy, objects.length), id: `q${highest + 1}` };
    setObjects((current) => [...current, object]);
    setSelectedId(object.id);
    setTool("select");
  }

  function updateObject(id: string, update: (object: CanvasObjectSpec) => CanvasObjectSpec) {
    setObjects((current) => current.map((object) => (object.id === id ? update(object) : object)));
  }

  function onPointerDown(event: React.PointerEvent<HTMLCanvasElement>) {
    event.preventDefault();
    canvasRef.current?.setPointerCapture(event.pointerId);
    const [x, y] = pointOf(event);

    if (tool === "brush") {
      dragRef.current = { kind: "stroke" };
      setStrokes((current) => [
        ...current,
        { color: colorHex(brushColor) ?? "#1e50dc", width: brushWidth, points: [[x, y]] },
      ]);
      return;
    }
    if (tool === "erase") {
      dragRef.current = { kind: "stroke" };
      setStrokes((current) => eraseAt(current, x, y));
      return;
    }

    const hit = hitTest(objects, x, y);
    if (!hit) {
      setSelectedId(null);
      dragRef.current = null;
      return;
    }
    setSelectedId(hit.object.id);
    dragRef.current = hit.onHandle
      ? { kind: "resize", id: hit.object.id }
      : { kind: "move", id: hit.object.id, grabX: x - hit.object.bbox[0], grabY: y - hit.object.bbox[1] };
  }

  function onPointerMove(event: React.PointerEvent<HTMLCanvasElement>) {
    const drag = dragRef.current;
    if (!drag) return;
    const [x, y] = pointOf(event);
    if (drag.kind === "stroke") {
      if (tool === "erase") {
        setStrokes((current) => eraseAt(current, x, y));
        return;
      }
      setStrokes((current) => {
        if (!current.length) return current;
        const last = current[current.length - 1];
        return [...current.slice(0, -1), { ...last, points: [...last.points, [x, y]] }];
      });
      return;
    }
    if (drag.kind === "resize") {
      updateObject(drag.id, (object) => resizeObject(object, x, y));
      return;
    }
    updateObject(drag.id, (object) =>
      moveObject(object, x - drag.grabX - object.bbox[0], y - drag.grabY - object.bbox[1]),
    );
  }

  function endDrag() {
    dragRef.current = null;
  }

  function search() {
    onSearch({
      objects,
      action_text: actionText.trim(),
      mode,
      exclude_labels: [],
      // Only pay the encode when there is something the structured channels
      // cannot express: a freehand stroke, or an explicit operator opt-in.
      image: useRaster && (strokes.length > 0 || objects.length > 0) ? renderRaster() : null,
    });
  }

  const canSearch = objects.length > 0 || strokes.length > 0 || actionText.trim().length > 0;

  return (
    <div className="panel canvas-panel" data-testid="canvas-panel">
      <div className="row between">
        <h3>
          <button
            className="canvas-collapse"
            onClick={() => setCollapsed((value) => !value)}
            data-testid="canvas-collapse"
            title={collapsed ? "Open drawing panel" : "Collapse drawing panel"}
          >
            {collapsed ? "▸" : "▾"}
          </button>{" "}
          Canvas · V-KIS
        </h3>
        <div className="row" style={{ gap: 6 }}>
          <div className="seg sm">
            {(["select", "brush", "erase"] as Tool[]).map((value) => (
              <button
                key={value}
                className={tool === value ? "active" : ""}
                onClick={() => setTool(value)}
                title={
                  value === "select" ? "Select / drag / resize an object"
                    : value === "brush" ? "Freehand drawing (sky, fields, water…)"
                    : "Erase strokes"
                }
                data-testid={`canvas-tool-${value}`}
              >
                {value === "select" ? "✥" : value === "brush" ? "✎" : "⌫"}
              </button>
            ))}
          </div>
          <div className="seg sm">
            {(["rough", "precise"] as CanvasMode[]).map((value) => (
              <button
                key={value}
                className={mode === value ? "active" : ""}
                onClick={() => setMode(value)}
                title={
                  value === "rough"
                    ? "Prioritize labels and relative position (drawing from memory)"
                    : "Increase IoU, size and color weights (when the clip is visible)"
                }
                data-testid={`canvas-mode-${value}`}
              >
                {value === "rough" ? "Rough" : "Precise"}
              </button>
            ))}
          </div>
        </div>
      </div>

      {objectSearchAvailable === false && (
        <div className="canvas-warn">
          ⚠ Object index unavailable — only PE text/drawing search is available; layout matching is disabled.
        </div>
      )}

      {!collapsed && (
        <>
          <div className="canvas-toolrow">
            <input
              className="input sm"
              placeholder="Filter objects… (person, motorcycle, fire…)"
              value={filter}
              onChange={(event) => setFilter(event.target.value)}
              data-testid="canvas-filter"
            />
            <div className="canvas-palette">
              {labels.map((item) => (
                <button
                  key={item.label}
                  className="chip draggable"
                  draggable
                  onDragStart={(event) => event.dataTransfer.setData("text/plain", item.label)}
                  onClick={() => addObject(item.label)}
                  title={item.label}
                  data-testid={`canvas-add-${item.label}`}
                >
                  {item.label}
                </button>
              ))}
            </div>
          </div>

          <div className="canvas-stage">
            <canvas
              ref={canvasRef}
              className={`canvas-surface tool-${tool}`}
              width={RENDER_WIDTH}
              height={RENDER_HEIGHT}
              onPointerDown={onPointerDown}
              onPointerMove={onPointerMove}
              onPointerUp={endDrag}
              onPointerLeave={endDrag}
              onDragOver={(event) => event.preventDefault()}
              onDrop={(event) => {
                event.preventDefault();
                const label = event.dataTransfer.getData("text/plain");
                if (!label) return;
                const [cx, cy] = pointOf(event);
                addObject(label, cx, cy);
              }}
              data-testid="canvas-board"
            />
            {objects.length === 0 && strokes.length === 0 && (
              <div className="canvas-empty">
                Drag or click an object above to place it · select ✎ to draw sky, fields and more
              </div>
            )}
          </div>

          {tool !== "select" && (
            <div className="canvas-editor" data-testid="canvas-brush-editor">
              <div className="row between">
                <strong>{tool === "brush" ? "Brush" : "Eraser"}</strong>
                <div className="row" style={{ gap: 6 }}>
                  <label className="canvas-required">
                    stroke
                    <input
                      type="range"
                      min={10}
                      max={90}
                      value={Math.round(brushWidth * 1000)}
                      onChange={(event) => setBrushWidth(Number(event.target.value) / 1000)}
                      data-testid="canvas-brush-width"
                    />
                  </label>
                  <button
                    className="btn sm"
                    onClick={() => setStrokes((current) => current.slice(0, -1))}
                    disabled={strokes.length === 0}
                    data-testid="canvas-undo-stroke"
                  >
                    Undo stroke
                  </button>
                </div>
              </div>
              {tool === "brush" && (
                <div className="canvas-colors">
                  {colors.map((color) => (
                    <button
                      key={color.name}
                      className={`swatch${brushColor === color.name ? " on" : ""}`}
                      style={{ background: color.hex }}
                      title={color.name}
                      onClick={() => setBrushColor(color.name)}
                      data-testid={`canvas-brush-${color.name}`}
                    />
                  ))}
                </div>
              )}
            </div>
          )}

          {selected && tool === "select" && (
            <div className="canvas-editor" data-testid="canvas-editor">
              <div className="row between">
                <strong>{selected.label}</strong>
                <div className="row" style={{ gap: 6 }}>
                  <label className="canvas-required">
                    <input
                      type="checkbox"
                      checked={selected.required}
                      onChange={(event) =>
                        updateObject(selected.id, (object) => ({ ...object, required: event.target.checked }))
                      }
                      data-testid="canvas-required"
                    />
                    required
                  </label>
                  <button
                    className="btn sm"
                    onClick={() => {
                      setObjects((current) => current.filter((object) => object.id !== selected.id));
                      setSelectedId(null);
                    }}
                    data-testid="canvas-delete"
                  >
                    Delete
                  </button>
                </div>
              </div>
              {selectedColorable ? (
                <div className="canvas-colors">
                  <button
                    className={`swatch none${selected.color === null ? " on" : ""}`}
                    onClick={() => updateObject(selected.id, (object) => ({ ...object, color: null }))}
                    title="no color specified"
                  >
                    ∅
                  </button>
                  {colors.map((color) => (
                    <button
                      key={color.name}
                      className={`swatch${selected.color === color.name ? " on" : ""}`}
                      style={{ background: color.hex }}
                      title={color.name}
                      onClick={() => updateObject(selected.id, (object) => ({ ...object, color: color.name }))}
                      data-testid={`canvas-color-${color.name}`}
                    />
                  ))}
                </div>
              ) : (
                <div className="hint">Object detection does not extract color for this class; color will be ignored.</div>
              )}
            </div>
          )}

          <input
            className="input sm"
            placeholder="Action (optional): a man walks past a car…"
            value={actionText}
            onChange={(event) => setActionText(event.target.value)}
            data-testid="canvas-action"
          />
        </>
      )}

      <div className="row between">
        <div className="row" style={{ gap: 6 }}>
          <button
            className="btn primary"
            onClick={search}
            disabled={loading || !canSearch}
            data-testid="canvas-search"
          >
            {loading ? "Searching…" : "Search canvas"}
          </button>
          <button
            className="btn sm"
            onClick={() => {
              setObjects([]);
              setStrokes([]);
              setSelectedId(null);
            }}
            disabled={objects.length === 0 && strokes.length === 0}
            data-testid="canvas-clear"
          >
            Clear all
          </button>
        </div>
        <label className="canvas-required" title="Encode the drawing with PE image and include it in fusion with a low weight">
          <input
            type="checkbox"
            checked={useRaster}
            onChange={(event) => setUseRaster(event.target.checked)}
            data-testid="canvas-use-raster"
          />
          PE image
        </label>
      </div>

      {generatedQueries.length > 0 && (
        <div className="canvas-queries" data-testid="canvas-queries">
          <div className="k">PE text generated from canvas</div>
          {generatedQueries.map((query) => (
            <div key={query} className="canvas-query">{query}</div>
          ))}
        </div>
      )}
    </div>
  );
}
