import type { Channel } from "../api/types";

// `image_qwen` is controlled by ImageModelSelector. `similar` is driven by
// operator feedback and `object_layout` by the V-KIS canvas, so none of those
// belongs to the general channel override switches.
export const CHANNELS: Channel[] = ["image_pe", "tara", "ocr", "speech", "audio"];

/** Evidence badges on a result frame: these name the index that FOUND it, so
 *  `image_pe` and `image_qwen` must stay distinguishable here.
 *
 *  `image_visual` is what a RERANKED visual search returns instead of those two:
 *  the reranker judges one merged PE+Qwen pool and produces one ranking, so the
 *  frame carries one visual badge. Which index actually retrieved it is not lost
 *  — it moves into the evidence card's `models` line. */
export const CHANNEL_LABEL: Record<Channel, string> = {
  image_pe: "PE Core",
  image_qwen: "Qwen3-VL",
  image_visual: "visual ⚖",
  tara: "TARA",
  ocr: "OCR",
  speech: "speech",
  audio: "audio",
  similar: "similar",
  object_layout: "layout",
  canvas_image: "sketch",
};

/** Labels for the manual channel switches, which are a different thing from the
 *  evidence badges above. The `image_pe` switch gates the whole visual channel:
 *  the backend reads `channels.image_pe.enabled` for the Qwen index too, so
 *  turning it off kills BOTH image models. Calling it "PE Core" collided with
 *  the ImageModelSelector's own "PE Core" box and read as "PE only", which made
 *  it look safe to switch off during a Qwen search — it is not. */
export const CHANNEL_TOGGLE_LABEL: Record<Channel, string> = {
  ...CHANNEL_LABEL,
  image_pe: "VISUAL",
};

export const EVENT_COLORS = ["var(--e1)", "var(--e2)", "var(--e3)", "var(--e4)", "var(--e5)"];

export function eventColor(idx: number): string {
  return EVENT_COLORS[idx % EVENT_COLORS.length];
}
