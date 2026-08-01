import type { Channel } from "../api/types";

// `similar` is intentionally absent: it is driven by operator feedback, not by
// the query parser, so it has no manual on/off toggle in ChannelControls.
export const CHANNELS: Channel[] = ["image_pe", "ocr", "speech", "audio"];

export const CHANNEL_LABEL: Record<Channel, string> = {
  image_pe: "vector",
  ocr: "OCR",
  speech: "speech",
  audio: "audio",
  similar: "similar",
};

export const EVENT_COLORS = ["var(--e1)", "var(--e2)", "var(--e3)", "var(--e4)", "var(--e5)"];

export function eventColor(idx: number): string {
  return EVENT_COLORS[idx % EVENT_COLORS.length];
}
