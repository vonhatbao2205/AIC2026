import type { Channel } from "../api/types";

export const CHANNELS: Channel[] = ["image_pe", "ocr", "speech", "audio"];

export const CHANNEL_LABEL: Record<Channel, string> = {
  image_pe: "vector",
  ocr: "OCR",
  speech: "speech",
  audio: "audio",
};

export const EVENT_COLORS = ["var(--e1)", "var(--e2)", "var(--e3)", "var(--e4)", "var(--e5)"];

export function eventColor(idx: number): string {
  return EVENT_COLORS[idx % EVENT_COLORS.length];
}
