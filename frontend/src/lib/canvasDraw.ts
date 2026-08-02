import type { CanvasObjectSpec } from "../api/types";

/** A freehand stroke in normalized canvas coordinates. */
export interface Stroke {
  color: string; // hex, already resolved from the palette
  width: number; // normalized to canvas height so it scales with the surface
  points: [number, number][];
}

/** Shape families the object icons are drawn from. A concrete silhouette is what
 * makes the rendered canvas readable to PE's image encoder — a labelled empty
 * rectangle carries no visual signal at all. */
export type Shape =
  | "person" | "crowd" | "car" | "bike" | "boat" | "plane" | "building"
  | "screen" | "mic" | "flag" | "card" | "tree" | "scenery" | "fire"
  | "furniture" | "bowl" | "generic";

const SHAPE_BY_LABEL: Record<string, Shape> = {
  person: "person", crowd: "crowd",
  car: "car", bus: "car", truck: "car", van: "car", ambulance: "car",
  "fire truck": "car", train: "car",
  motorcycle: "bike", bicycle: "bike",
  boat: "boat", sampan: "boat", "floating market": "boat",
  airplane: "plane", helicopter: "plane",
  building: "building", hospital: "building", factory: "building",
  temple: "building", pagoda: "building", "old house": "building",
  "traditional house": "building", village: "building", market: "building",
  screen: "screen", laptop: "screen", computer: "screen", whiteboard: "screen",
  blackboard: "screen", "tv logo": "screen", "text region": "screen",
  microphone: "mic",
  flag: "flag", banner: "flag",
  sign: "card", "license plate": "card", document: "card", book: "card",
  money: "card", medal: "card", "race number": "card",
  "coconut tree": "tree", "palm tree": "tree",
  "rice field": "scenery", "farm field": "scenery", wetland: "scenery",
  landscape: "scenery", river: "scenery", riverbank: "scenery",
  mountain: "scenery", sunset: "scenery", nature: "scenery",
  "city street": "scenery", road: "scenery", bridge: "scenery",
  fire: "fire", smoke: "fire", flood: "scenery", firecracker: "fire",
  table: "furniture", chair: "furniture", desk: "furniture",
  podium: "furniture", stage: "furniture", stove: "furniture",
  food: "bowl", dish: "bowl", plate: "bowl", bowl: "bowl", pan: "bowl",
  pot: "bowl", fruit: "bowl", rice: "bowl", noodle: "bowl",
  vegetable: "bowl", meat: "bowl", fish: "bowl",
};

export function shapeOf(label: string): Shape {
  return SHAPE_BY_LABEL[label] ?? "generic";
}

/** Default colour for an object whose colour the operator did not set. Neutral
 * grey keeps the raster honest: a made-up colour would be a claim they never made. */
const NEUTRAL = "#8a8f98";

function rounded(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  const radius = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.arcTo(x + w, y, x + w, y + h, radius);
  ctx.arcTo(x + w, y + h, x, y + h, radius);
  ctx.arcTo(x, y + h, x, y, radius);
  ctx.arcTo(x, y, x + w, y, radius);
  ctx.closePath();
  ctx.fill();
}

function circle(ctx: CanvasRenderingContext2D, cx: number, cy: number, r: number) {
  ctx.beginPath();
  ctx.arc(cx, cy, Math.max(1, r), 0, Math.PI * 2);
  ctx.fill();
}

function polygon(ctx: CanvasRenderingContext2D, points: [number, number][]) {
  ctx.beginPath();
  points.forEach(([x, y], index) => (index ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  ctx.closePath();
  ctx.fill();
}

/** Draw one object's silhouette inside the pixel rect (x, y, w, h). */
export function drawShape(
  ctx: CanvasRenderingContext2D,
  shape: Shape,
  x: number,
  y: number,
  w: number,
  h: number,
  color: string | null,
) {
  const fill = color ?? NEUTRAL;
  ctx.save();
  ctx.fillStyle = fill;
  switch (shape) {
    case "person": {
      const headR = Math.min(w, h) * 0.17;
      circle(ctx, x + w / 2, y + headR * 1.2, headR);
      rounded(ctx, x + w * 0.22, y + headR * 2.3, w * 0.56, h - headR * 2.3, w * 0.2);
      break;
    }
    case "crowd": {
      const headR = Math.min(w, h) * 0.11;
      [0.2, 0.5, 0.8].forEach((cx, index) => {
        const top = y + h * (index === 1 ? 0.06 : 0.16);
        circle(ctx, x + w * cx, top + headR, headR);
        rounded(ctx, x + w * cx - w * 0.13, top + headR * 2, w * 0.26, y + h - (top + headR * 2), w * 0.1);
      });
      break;
    }
    case "car": {
      rounded(ctx, x, y + h * 0.42, w, h * 0.42, h * 0.14);
      rounded(ctx, x + w * 0.2, y + h * 0.12, w * 0.6, h * 0.36, h * 0.12);
      ctx.fillStyle = "#2b2f36";
      circle(ctx, x + w * 0.25, y + h * 0.88, Math.min(w, h) * 0.12);
      circle(ctx, x + w * 0.75, y + h * 0.88, Math.min(w, h) * 0.12);
      break;
    }
    case "bike": {
      ctx.strokeStyle = fill;
      ctx.lineWidth = Math.max(2, h * 0.07);
      ctx.beginPath();
      ctx.arc(x + w * 0.22, y + h * 0.72, Math.min(w, h) * 0.24, 0, Math.PI * 2);
      ctx.moveTo(x + w * 0.98, y + h * 0.72);
      ctx.arc(x + w * 0.78, y + h * 0.72, Math.min(w, h) * 0.24, 0, Math.PI * 2);
      ctx.moveTo(x + w * 0.22, y + h * 0.72);
      ctx.lineTo(x + w * 0.45, y + h * 0.3);
      ctx.lineTo(x + w * 0.72, y + h * 0.3);
      ctx.lineTo(x + w * 0.78, y + h * 0.72);
      ctx.stroke();
      break;
    }
    case "boat": {
      polygon(ctx, [
        [x, y + h * 0.62], [x + w, y + h * 0.62],
        [x + w * 0.82, y + h], [x + w * 0.18, y + h],
      ]);
      polygon(ctx, [
        [x + w * 0.5, y + h * 0.06], [x + w * 0.5, y + h * 0.58], [x + w * 0.9, y + h * 0.58],
      ]);
      break;
    }
    case "plane": {
      polygon(ctx, [
        [x + w * 0.05, y + h * 0.55], [x + w * 0.95, y + h * 0.42],
        [x + w * 0.95, y + h * 0.58], [x + w * 0.05, y + h * 0.7],
      ]);
      polygon(ctx, [
        [x + w * 0.45, y + h * 0.5], [x + w * 0.62, y + h * 0.05], [x + w * 0.7, y + h * 0.5],
      ]);
      break;
    }
    case "building": {
      rounded(ctx, x + w * 0.08, y + h * 0.12, w * 0.84, h * 0.88, 2);
      ctx.fillStyle = "#ffffff";
      for (let row = 0; row < 3; row += 1) {
        for (let col = 0; col < 3; col += 1) {
          ctx.fillRect(x + w * (0.2 + col * 0.22), y + h * (0.26 + row * 0.22), w * 0.12, h * 0.12);
        }
      }
      break;
    }
    case "screen": {
      rounded(ctx, x, y, w, h * 0.74, 3);
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(x + w * 0.06, y + h * 0.08, w * 0.88, h * 0.58);
      ctx.fillStyle = fill;
      ctx.fillRect(x + w * 0.42, y + h * 0.74, w * 0.16, h * 0.14);
      ctx.fillRect(x + w * 0.25, y + h * 0.88, w * 0.5, h * 0.1);
      break;
    }
    case "mic": {
      rounded(ctx, x + w * 0.34, y, w * 0.32, h * 0.5, w * 0.16);
      ctx.strokeStyle = fill;
      ctx.lineWidth = Math.max(2, w * 0.07);
      ctx.beginPath();
      ctx.arc(x + w * 0.5, y + h * 0.45, w * 0.28, 0, Math.PI);
      ctx.moveTo(x + w * 0.5, y + h * 0.73);
      ctx.lineTo(x + w * 0.5, y + h);
      ctx.stroke();
      break;
    }
    case "flag": {
      ctx.fillRect(x + w * 0.08, y, w * 0.07, h);
      polygon(ctx, [
        [x + w * 0.15, y + h * 0.05], [x + w * 0.92, y + h * 0.22],
        [x + w * 0.15, y + h * 0.44],
      ]);
      break;
    }
    case "card": {
      rounded(ctx, x, y + h * 0.15, w, h * 0.7, 3);
      ctx.fillStyle = "#ffffff";
      [0.32, 0.5, 0.68].forEach((ratio) =>
        ctx.fillRect(x + w * 0.12, y + h * ratio, w * (ratio === 0.68 ? 0.5 : 0.76), h * 0.07),
      );
      break;
    }
    case "tree": {
      ctx.fillStyle = "#7a5230";
      ctx.fillRect(x + w * 0.44, y + h * 0.5, w * 0.12, h * 0.5);
      ctx.fillStyle = color ?? "#2f8f4e";
      circle(ctx, x + w * 0.5, y + h * 0.32, Math.min(w, h) * 0.3);
      circle(ctx, x + w * 0.28, y + h * 0.45, Math.min(w, h) * 0.2);
      circle(ctx, x + w * 0.72, y + h * 0.45, Math.min(w, h) * 0.2);
      break;
    }
    case "scenery": {
      // Horizontal bands read as ground/field/water to an image encoder, which is
      // exactly what a rice field or a riverbank looks like from a distance.
      ctx.fillStyle = color ?? "#8fbf5a";
      ctx.fillRect(x, y + h * 0.45, w, h * 0.55);
      ctx.fillStyle = "#ffffff";
      ctx.globalAlpha = 0.35;
      for (let row = 0; row < 4; row += 1) {
        ctx.fillRect(x, y + h * (0.55 + row * 0.11), w, h * 0.03);
      }
      ctx.globalAlpha = 1;
      ctx.fillStyle = color ?? "#8fbf5a";
      polygon(ctx, [
        [x, y + h * 0.45], [x + w * 0.3, y + h * 0.1],
        [x + w * 0.6, y + h * 0.45],
      ]);
      break;
    }
    case "fire": {
      polygon(ctx, [
        [x + w * 0.5, y], [x + w * 0.85, y + h * 0.45],
        [x + w * 0.72, y + h], [x + w * 0.28, y + h],
        [x + w * 0.15, y + h * 0.45],
      ]);
      ctx.fillStyle = "#ffd166";
      polygon(ctx, [
        [x + w * 0.5, y + h * 0.35], [x + w * 0.68, y + h * 0.68],
        [x + w * 0.5, y + h * 0.95], [x + w * 0.32, y + h * 0.68],
      ]);
      break;
    }
    case "furniture": {
      ctx.fillRect(x, y + h * 0.3, w, h * 0.16);
      ctx.fillRect(x + w * 0.08, y + h * 0.46, w * 0.1, h * 0.54);
      ctx.fillRect(x + w * 0.82, y + h * 0.46, w * 0.1, h * 0.54);
      break;
    }
    case "bowl": {
      ctx.beginPath();
      ctx.arc(x + w * 0.5, y + h * 0.45, Math.min(w, h) * 0.42, 0, Math.PI);
      ctx.closePath();
      ctx.fill();
      ctx.fillRect(x + w * 0.1, y + h * 0.38, w * 0.8, h * 0.08);
      break;
    }
    default:
      rounded(ctx, x, y, w, h, Math.min(w, h) * 0.12);
      ctx.fillStyle = "#ffffff";
      circle(ctx, x + w * 0.5, y + h * 0.5, Math.min(w, h) * 0.14);
      break;
  }
  ctx.restore();
}

export interface SceneOptions {
  /** UI-only decoration: thirds guide, selection handles, labels. Excluded from
   * the raster so PE sees a clean drawing rather than the editor. */
  chrome: boolean;
  selectedId?: string | null;
  outlineFor?: (index: number) => string;
  colorHex: (name: string | null) => string | null;
}

export function drawScene(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  objects: CanvasObjectSpec[],
  strokes: Stroke[],
  options: SceneOptions,
) {
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);

  if (options.chrome) {
    ctx.save();
    ctx.strokeStyle = "rgba(130,140,155,0.35)";
    ctx.lineWidth = 1;
    [1 / 3, 2 / 3].forEach((ratio) => {
      ctx.beginPath();
      ctx.moveTo(width * ratio, 0);
      ctx.lineTo(width * ratio, height);
      ctx.moveTo(0, height * ratio);
      ctx.lineTo(width, height * ratio);
      ctx.stroke();
    });
    ctx.restore();
  }

  // Strokes first: they are backdrop (sky, field, water) for the objects on top.
  ctx.save();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  strokes.forEach((stroke) => {
    if (stroke.points.length === 0) return;
    ctx.strokeStyle = stroke.color;
    ctx.lineWidth = Math.max(1, stroke.width * height);
    ctx.beginPath();
    stroke.points.forEach(([px, py], index) => {
      const x = px * width;
      const y = py * height;
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    if (stroke.points.length === 1) {
      const [px, py] = stroke.points[0];
      ctx.lineTo(px * width + 0.01, py * height);
    }
    ctx.stroke();
  });
  ctx.restore();

  objects.forEach((object, index) => {
    const [x1, y1, x2, y2] = object.bbox;
    const x = x1 * width;
    const y = y1 * height;
    const w = Math.max(2, (x2 - x1) * width);
    const h = Math.max(2, (y2 - y1) * height);
    drawShape(ctx, shapeOf(object.label), x, y, w, h, options.colorHex(object.color));

    if (!options.chrome) return;
    const outline = options.outlineFor?.(index) ?? "#38bdf8";
    ctx.save();
    ctx.strokeStyle = outline;
    ctx.lineWidth = object.id === options.selectedId ? 2.5 : 1.2;
    ctx.setLineDash(object.required ? [] : [5, 4]);
    ctx.strokeRect(x, y, w, h);
    ctx.setLineDash([]);

    const label = `${object.color ? `${object.color} ` : ""}${object.label}`;
    ctx.font = "600 12px system-ui, sans-serif";
    const textWidth = ctx.measureText(label).width;
    ctx.fillStyle = outline;
    ctx.fillRect(x, Math.max(0, y - 16), textWidth + 10, 16);
    ctx.fillStyle = "#08111c";
    ctx.fillText(label, x + 5, Math.max(11, y - 4));

    if (object.id === options.selectedId) {
      ctx.fillStyle = outline;
      ctx.fillRect(x + w - 7, y + h - 7, 12, 12);
    }
    ctx.restore();
  });
}

/** Topmost object containing the point, or null. Later objects are drawn on top,
 * so hit-testing walks the list backwards. */
export function hitTest(
  objects: CanvasObjectSpec[],
  x: number,
  y: number,
): { object: CanvasObjectSpec; onHandle: boolean } | null {
  for (let index = objects.length - 1; index >= 0; index -= 1) {
    const [x1, y1, x2, y2] = objects[index].bbox;
    if (x >= x1 && x <= x2 && y >= y1 && y <= y2) {
      const onHandle = Math.abs(x - x2) < 0.022 && Math.abs(y - y2) < 0.035;
      return { object: objects[index], onHandle };
    }
  }
  return null;
}

/** Strokes that survive an eraser dab at (x, y). */
export function eraseAt(strokes: Stroke[], x: number, y: number, radius = 0.035): Stroke[] {
  return strokes.filter(
    (stroke) => !stroke.points.some(([px, py]) => Math.hypot(px - x, py - y) <= radius),
  );
}
