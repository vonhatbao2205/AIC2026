import type { FrameResult, QaAnalyzeCandidate, VideoGroup } from "../api/types";

/** Select visual evidence in ranked-video blocks.
 *
 * The operator-selected frame is deliberately C01. After that, up to
 * `perVideo` diverse frames (including C01 when applicable) are packed from
 * the highest-scoring video before moving to the next ranked video.
 */
export function selectQaCandidateFrames(
  groups: VideoGroup[],
  selected: FrameResult | null,
  limit = 12,
  perVideo = 3,
): FrameResult[] {
  if (limit <= 0) return [];
  const rankedGroups = [...groups].sort((a, b) => b.video_score - a.video_score);
  const pools = rankedGroups.map((group) => {
    const ranked = [...group.frames].sort((a, b) => b.score - a.score);
    const diverse: FrameResult[] = [];
    const nearDuplicates: FrameResult[] = [];
    for (const frame of ranked) {
      const nearExisting = diverse.some(
        (other) => other.pts_time != null && frame.pts_time != null
          && Math.abs(other.pts_time - frame.pts_time) < 2,
      );
      (nearExisting ? nearDuplicates : diverse).push(frame);
    }
    return [...diverse, ...nearDuplicates].slice(0, perVideo);
  });

  const chosen: FrameResult[] = [];
  const seen = new Set<string>();
  const perVideoCount = new Map<string, number>();
  if (selected) {
    chosen.push(selected);
    seen.add(selected.submit_keyframe_id);
    perVideoCount.set(selected.video_id, 1);
  }

  for (const pool of pools) {
    for (const frame of pool) {
      if (!frame || seen.has(frame.submit_keyframe_id)) continue;
      const count = perVideoCount.get(frame.video_id) ?? 0;
      if (count >= perVideo) continue;
      chosen.push(frame);
      seen.add(frame.submit_keyframe_id);
      perVideoCount.set(frame.video_id, count + 1);
      if (chosen.length >= limit) break;
    }
    if (chosen.length >= limit) break;
  }
  return chosen;
}

export function buildQaCandidates(
  groups: VideoGroup[],
  selected: FrameResult | null,
  limit = 12,
  perVideo = 3,
): QaAnalyzeCandidate[] {
  return selectQaCandidateFrames(groups, selected, limit, perVideo).map((frame) => ({
    submit_keyframe_id: frame.submit_keyframe_id,
    frame_idx: frame.frame_idx,
    pts_time: frame.pts_time,
    retrieval_score: frame.score,
    evidence: frame.evidence,
  }));
}

export function findQaFrame(groups: VideoGroup[], submitKeyframeId: string): { video: number; frame: number } | null {
  for (let video = 0; video < groups.length; video += 1) {
    const frame = groups[video].frames.findIndex((item) => item.submit_keyframe_id === submitKeyframeId);
    if (frame >= 0) return { video, frame };
  }
  return null;
}
