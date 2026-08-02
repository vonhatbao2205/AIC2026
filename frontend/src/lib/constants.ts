import type { Channel } from "../api/types";

// `similar` and `object_layout` are intentionally absent: the first is driven by
// operator feedback and the second by the V-KIS canvas, so neither has a manual
// on/off toggle in ChannelControls.
export const CHANNELS: Channel[] = ["image_pe", "ocr", "speech", "audio"];

export const CHANNEL_LABEL: Record<Channel, string> = {
  image_pe: "vector",
  ocr: "OCR",
  speech: "speech",
  audio: "audio",
  similar: "similar",
  object_layout: "layout",
  canvas_image: "sketch",
};

export const EVENT_COLORS = ["var(--e1)", "var(--e2)", "var(--e3)", "var(--e4)", "var(--e5)"];

export function eventColor(idx: number): string {
  return EVENT_COLORS[idx % EVENT_COLORS.length];
}
