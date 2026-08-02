import { describe, expect, it } from "vitest";
import type { CanvasLayoutEvidence, FrameResult } from "../api/types";
import { type Stroke, drawScene, eraseAt, hitTest, shapeOf } from "../lib/canvasDraw";
import {
  boxStyle,
  centerOf,
  layoutEvidenceOf,
  makeObject,
  matchesByObject,
  moveObject,
  outlineFor,
  resizeObject,
  zoneOf,
} from "../lib/canvas";

/** Minimal 2D-context stand-in: jsdom has no canvas, and the assertions are
 * about which drawing calls happen, not about pixels. */
function fakeContext() {
  const calls: Record<string, number> = {
    fillText: 0, stroke: 0, strokeRect: 0, fill: 0, fillRect: 0,
  };
  const bump = (name: string) => () => {
    calls[name] = (calls[name] ?? 0) + 1;
  };
  const ctx = {
    canvas: {} as HTMLCanvasElement,
    save: () => {}, restore: () => {}, beginPath: () => {}, closePath: () => {},
    moveTo: () => {}, lineTo: () => {}, arc: () => {}, arcTo: () => {},
    clearRect: () => {}, setLineDash: () => {},
    measureText: () => ({ width: 40 }) as TextMetrics,
    fill: bump("fill"), fillRect: bump("fillRect"), fillText: bump("fillText"),
    stroke: bump("stroke"), strokeRect: bump("strokeRect"),
    fillStyle: "", strokeStyle: "", lineWidth: 1, font: "", globalAlpha: 1,
    lineCap: "butt" as CanvasLineCap, lineJoin: "miter" as CanvasLineJoin,
  } as unknown as CanvasRenderingContext2D;
  return { ctx, calls };
}

function frameWith(evidence: FrameResult["evidence"]): FrameResult {
  return {
    image_id: "K01/K01_V001/002",
    submit_keyframe_id: "K01/K01_V001/002",
    video_id: "K01_V001",
    keyframe_n: 2,
    frame_idx: 100,
    fps: 25,
    pts_time: 4,
    score: 0.02,
    channels: ["object_layout"],
    per_channel_score: { object_layout: 0.8 },
    keyframe_url: "https://media.test/k.jpg",
    video_url: "https://media.test/v.mp4",
    evidence,
  };
}

const LAYOUT: CanvasLayoutEvidence = {
  type: "object_layout",
  score: 0.81,
  matches: [
    {
      object_id: "q1",
      label: "car",
      score: 0.86,
      detection_label: "car",
      conf: 0.91,
      bbox_norm: { x1: 0.05, y1: 0.5, x2: 0.35, y2: 0.86 },
      position: "bottom_left",
      dominant_color: "red",
      color_reliable: true,
      color_ok: true,
    },
  ],
  missing: ["q2"],
  coverage: 0.5,
  excluded_hits: [],
};

describe("canvas geometry", () => {
  it("mirrors the backend nine-zone vocabulary", () => {
    expect(zoneOf(0.1, 0.1)).toBe("top_left");
    expect(zoneOf(0.5, 0.5)).toBe("center");
    expect(zoneOf(0.9, 0.9)).toBe("bottom_right");
  });

  it("creates an object centred on the drop point", () => {
    const object = makeObject("car", 0.5, 0.5, 0);
    const [cx, cy] = centerOf(object.bbox);

    expect(cx).toBeCloseTo(0.5, 5);
    expect(cy).toBeCloseTo(0.5, 5);
    expect(object.required).toBe(true);
  });

  it("keeps a new object inside the canvas when dropped on the edge", () => {
    const object = makeObject("person", 0.01, 0.99, 0);

    expect(object.bbox[0]).toBeGreaterThanOrEqual(0);
    expect(object.bbox[3]).toBeLessThanOrEqual(1);
  });

  it("slides along an edge instead of shrinking when dragged past it", () => {
    const object = makeObject("car", 0.2, 0.5, 0);
    const width = object.bbox[2] - object.bbox[0];

    const moved = moveObject(object, -1, 0);

    expect(moved.bbox[0]).toBeCloseTo(0, 5);
    expect(moved.bbox[2] - moved.bbox[0]).toBeCloseTo(width, 5);
  });

  it("never resizes a box to nothing", () => {
    const object = makeObject("car", 0.5, 0.5, 0);

    const resized = resizeObject(object, 0, 0);

    expect(resized.bbox[2]).toBeGreaterThan(resized.bbox[0]);
    expect(resized.bbox[3]).toBeGreaterThan(resized.bbox[1]);
  });

  it("gives each drawn object a stable outline colour", () => {
    expect(outlineFor(0)).toBe(outlineFor(8));
    expect(outlineFor(0)).not.toBe(outlineFor(1));
  });
});

describe("canvas result overlay", () => {
  it("finds the layout evidence on a frame", () => {
    expect(layoutEvidenceOf(frameWith([LAYOUT]))?.coverage).toBe(0.5);
    expect(layoutEvidenceOf(frameWith([{ type: "ocr", score: 3 }]))).toBeNull();
    expect(layoutEvidenceOf(null)).toBeNull();
  });

  it("keys matches by the drawn object id", () => {
    const byObject = matchesByObject(LAYOUT);

    expect(byObject.get("q1")?.label).toBe("car");
    expect(byObject.has("q2")).toBe(false);
  });

  it("positions an overlay box as percentages of the image", () => {
    const style = boxStyle({ x1: 0.1, y1: 0.2, x2: 0.5, y2: 0.6 });

    expect(parseFloat(style.left)).toBeCloseTo(10, 6);
    expect(parseFloat(style.top)).toBeCloseTo(20, 6);
    expect(parseFloat(style.width)).toBeCloseTo(40, 6);
    expect(parseFloat(style.height)).toBeCloseTo(40, 6);
    expect(style.left.endsWith("%")).toBe(true);
  });

  it("never emits a negative-size box from an inverted bbox", () => {
    const style = boxStyle({ x1: 0.8, y1: 0.8, x2: 0.2, y2: 0.2 });

    expect(parseFloat(style.width)).toBe(0);
    expect(parseFloat(style.height)).toBe(0);
  });
});

describe("canvas drawing surface", () => {
  it("maps labels to a concrete silhouette, with a fallback", () => {
    expect(shapeOf("person")).toBe("person");
    expect(shapeOf("bus")).toBe("car");
    expect(shapeOf("rice field")).toBe("scenery");
    expect(shapeOf("something the detector never had")).toBe("generic");
  });

  it("hit-tests the topmost object and its resize handle", () => {
    const back = { ...makeObject("car", 0.5, 0.5, 0), id: "q1" };
    const front = { ...makeObject("person", 0.5, 0.5, 1), id: "q2" };

    expect(hitTest([back, front], 0.5, 0.5)?.object.id).toBe("q2");
    expect(hitTest([back, front], 0.02, 0.02)).toBeNull();

    const [, , x2, y2] = front.bbox;
    expect(hitTest([back, front], x2 - 0.005, y2 - 0.005)?.onHandle).toBe(true);
    expect(hitTest([back, front], 0.5, 0.5)?.onHandle).toBe(false);
  });

  it("erases only the strokes under the pointer", () => {
    const strokes = [
      { color: "#f00", width: 0.03, points: [[0.1, 0.1] as [number, number]] },
      { color: "#00f", width: 0.03, points: [[0.8, 0.8] as [number, number]] },
    ];

    expect(eraseAt(strokes, 0.1, 0.1)).toHaveLength(1);
    expect(eraseAt(strokes, 0.1, 0.1)[0].color).toBe("#00f");
    expect(eraseAt(strokes, 0.5, 0.5)).toHaveLength(2);
  });

  it("leaves the editor chrome out of the exported raster", () => {
    const chrome = fakeContext();
    const clean = fakeContext();
    const objects = [{ ...makeObject("car", 0.3, 0.6, 0), id: "q1" }];
    const strokes = [{ color: "#7ec8f0", width: 0.05, points: [[0.1, 0.2], [0.9, 0.2]] as [number, number][] }];
    const options = { colorHex: () => "#dc1414" };

    drawScene(chrome.ctx, 100, 100, objects, strokes, { ...options, chrome: true, outlineFor: () => "#fff" });
    drawScene(clean.ctx, 100, 100, objects, strokes, { ...options, chrome: false });

    // Guides, selection outline and the label chip are stroke/fillText calls the
    // clean render must not make; the drawing itself still paints.
    expect(clean.calls.fillText).toBe(0);
    expect(chrome.calls.fillText).toBeGreaterThan(0);
    expect(clean.calls.stroke).toBeGreaterThan(0); // the freehand stroke survives
    expect(chrome.calls.strokeRect).toBeGreaterThan(0);
    expect(clean.calls.strokeRect).toBe(0);
    expect(clean.calls.fill).toBeGreaterThan(0);
  });
});
