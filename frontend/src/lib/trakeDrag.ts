import type { TrakeHeatPeak } from "../api/types";

/** Drag payload for one TRAKE moment (a heat peak or an event representative).
 *
 *  Three components speak this: the heatmap and the event strip write it, the
 *  event slots and the event strip read it. It lives here so the wire format is
 *  one definition rather than three JSON.parse calls that can drift apart. */
export const TRAKE_PEAK_MIME = "text/x-trake-peak";

export interface TrakePeakDrag {
  videoId: string;
  peak: TrakeHeatPeak;
}

export function setPeakDrag(event: React.DragEvent, payload: TrakePeakDrag): void {
  event.dataTransfer.setData(TRAKE_PEAK_MIME, JSON.stringify(payload));
  event.dataTransfer.effectAllowed = "copy";
}

/** True while a TRAKE moment is being dragged over a target.
 *
 *  `dragover` is only allowed to see the TYPES, never the data — reading
 *  `getData` there returns "" in every browser — so a drop target that wants to
 *  light up has to ask this question, not parse the payload. */
export function hasPeakDrag(event: React.DragEvent): boolean {
  return Array.from(event.dataTransfer.types).includes(TRAKE_PEAK_MIME);
}

export function readPeakDrag(event: React.DragEvent): TrakePeakDrag | null {
  const raw = event.dataTransfer.getData(TRAKE_PEAK_MIME);
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw) as TrakePeakDrag;
    // A drop must not be able to inject a half-formed frame into a submission.
    if (!parsed?.videoId || !parsed?.peak?.submit_keyframe_id) return null;
    if (typeof parsed.peak.pts_time !== "number") return null;
    return parsed;
  } catch {
    return null;
  }
}
