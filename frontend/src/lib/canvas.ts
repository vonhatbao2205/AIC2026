import type { CanvasLayoutEvidence, CanvasMatch, CanvasObjectSpec, FrameResult } from "../api/types";

/** Colour every drawn object gets on screen, cycled by creation order, so the
 * box on the canvas and its matched detection on the result share one colour. */
export const OBJECT_OUTLINES = [
  "#38bdf8", "#f472b6", "#facc15", "#4ade80", "#a78bfa", "#fb923c", "#22d3ee", "#f87171",
];

/** Fallback palette used until /api/canvas/palette answers — same 16 colours the
 * OD pipeline quantises into (aic16-lab-v1). */
export const FALLBACK_COLORS = [
  { name: "black", hex: "#000000" }, { name: "white", hex: "#ffffff" },
  { name: "gray", hex: "#808080" }, { name: "red", hex: "#dc1414" },
  { name: "orange", hex: "#ff8c00" }, { name: "yellow", hex: "#f0dc1e" },
  { name: "green", hex: "#1ea03c" }, { name: "blue", hex: "#1e50dc" },
  { name: "navy", hex: "#141e5a" }, { name: "purple", hex: "#8228a0" },
  { name: "pink", hex: "#f082b4" }, { name: "brown", hex: "#78461e" },
  { name: "beige", hex: "#e1c8a0" }, { name: "cyan", hex: "#28c8d2" },
  { name: "maroon", hex: "#6e141e" }, { name: "olive", hex: "#6e6e1e" },
];

/** Labels offered before the backend palette loads: the ones a news/culture
 * corpus actually yields, so the panel is usable immediately. */
export const FALLBACK_LABELS = [
  "person", "crowd", "car", "motorcycle", "bicycle", "bus", "truck", "boat",
  "microphone", "camera", "screen", "table", "chair", "document", "sign",
  "banner", "flag", "fire", "smoke", "flood", "building", "podium", "food",
];

export const CLAMP = (value: number) => Math.min(1, Math.max(0, value));

export function outlineFor(index: number): string {
  return OBJECT_OUTLINES[index % OBJECT_OUTLINES.length];
}

/** Nine-zone name, mirroring `zone_of` in backend/app/canvas.py. */
export function zoneOf(cx: number, cy: number): string {
  const horizontal = cx < 1 / 3 ? "left" : cx > 2 / 3 ? "right" : "center";
  const vertical = cy < 1 / 3 ? "top" : cy > 2 / 3 ? "bottom" : "middle";
  if (horizontal === "center" && vertical === "middle") return "center";
  return `${vertical}_${horizontal}`;
}

export function centerOf(bbox: [number, number, number, number]): [number, number] {
  return [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2];
}

/** A new object dropped at (cx, cy), sized as a sensible default box that stays
 * inside the canvas. */
export function makeObject(label: string, cx: number, cy: number, index: number): CanvasObjectSpec {
  const width = label === "crowd" || label === "building" ? 0.34 : 0.18;
  const height = label === "person" ? 0.42 : 0.26;
  const x1 = CLAMP(cx - width / 2);
  const y1 = CLAMP(cy - height / 2);
  return {
    id: `q${index + 1}`,
    label,
    bbox: [x1, y1, CLAMP(x1 + width), CLAMP(y1 + height)],
    color: null,
    required: true,
  };
}

export function moveObject(
  object: CanvasObjectSpec,
  dx: number,
  dy: number,
): CanvasObjectSpec {
  const [x1, y1, x2, y2] = object.bbox;
  // Shift by the amount that keeps the whole box on the canvas, so dragging into
  // an edge slides along it instead of shrinking the box.
  const shiftX = Math.min(Math.max(dx, -x1), 1 - x2);
  const shiftY = Math.min(Math.max(dy, -y1), 1 - y2);
  return { ...object, bbox: [x1 + shiftX, y1 + shiftY, x2 + shiftX, y2 + shiftY] };
}

export function resizeObject(
  object: CanvasObjectSpec,
  x: number,
  y: number,
): CanvasObjectSpec {
  const [x1, y1] = object.bbox;
  return {
    ...object,
    bbox: [x1, y1, CLAMP(Math.max(x, x1 + 0.03)), CLAMP(Math.max(y, y1 + 0.03))],
  };
}

/** The layout evidence attached to a frame by the canvas search, if any. */
export function layoutEvidenceOf(frame: FrameResult | null): CanvasLayoutEvidence | null {
  if (!frame) return null;
  const found = frame.evidence.find((item) => item.type === "object_layout");
  return (found as CanvasLayoutEvidence | undefined) ?? null;
}

/** Matches keyed by the drawn object id, for colouring the overlay. */
export function matchesByObject(evidence: CanvasLayoutEvidence | null): Map<string, CanvasMatch> {
  return new Map((evidence?.matches ?? []).map((match) => [match.object_id, match]));
}

/** Percent rect for absolutely positioning a normalized bbox over an image. */
export function boxStyle(bbox: { x1: number; y1: number; x2: number; y2: number }) {
  return {
    left: `${bbox.x1 * 100}%`,
    top: `${bbox.y1 * 100}%`,
    width: `${Math.max(0, bbox.x2 - bbox.x1) * 100}%`,
    height: `${Math.max(0, bbox.y2 - bbox.y1) * 100}%`,
  };
}
