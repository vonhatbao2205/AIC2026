import type { FrameOverlay } from "../api/types";

/** One line saying what a traffic-camera banner / race HUD printed on a frame:
 *  "Nguyễn Trãi – Cống Quỳnh (1) · 2026-06-15 · 19:05:08" or "Stage 6 · race 01:23:45". */
export function overlayText(overlay?: FrameOverlay | null): string {
  if (!overlay) return "";
  const parts: string[] = [];
  const camera = overlay.camera ?? overlay.banner_camera;
  if (camera) parts.push(camera);
  if (overlay.banner_date) parts.push(overlay.banner_date);
  if (overlay.clock) parts.push(overlay.clock);
  if (overlay.race_stage != null) parts.push(`Stage ${overlay.race_stage}`);
  if (overlay.race_time) parts.push(`race ${overlay.race_time}`);
  return parts.join(" · ");
}
