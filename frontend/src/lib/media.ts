// R2 media URL helpers. The backend already returns built URLs; these mirror the
// convention for any client-side need. Never rely on local/Kaggle paths.

export function groupFromVideoId(videoId: string): string {
  return videoId.split("_")[0];
}

export function keyframeUrl(baseUrl: string, videoId: string, keyframeN: number): string {
  const group = groupFromVideoId(videoId);
  const frame = String(keyframeN).padStart(3, "0");
  return `${baseUrl.replace(/\/$/, "")}/Keyframes/Keyframes_${group}/${videoId}/${frame}.jpg`;
}

export function keyframeUrlFromSubmitId(baseUrl: string, submitKeyframeId: string): string {
  const [, videoId, frame] = submitKeyframeId.split("/");
  const frameName = frame.endsWith(".jpg") ? frame : `${frame}.jpg`;
  const group = groupFromVideoId(videoId);
  return `${baseUrl.replace(/\/$/, "")}/Keyframes/Keyframes_${group}/${videoId}/${frameName}`;
}

export function videoUrl(baseUrl: string, videoId: string): string {
  const group = groupFromVideoId(videoId);
  return `${baseUrl.replace(/\/$/, "")}/Videos/Videos_${group}/${videoId}.mp4`;
}

export function formatTime(seconds: number | null | undefined): string {
  if (seconds == null || Number.isNaN(seconds)) return "--:--";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}
