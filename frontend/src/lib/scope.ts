import type { ResolvedScope, ScopeCatalogue, ScopeMode, SearchScope } from "../api/types";

/** Default: the whole profile, with the topic heuristic free to narrow it.
 *
 *  `auto` and `all` search the same thing on a query with no topic words, so the
 *  heuristic costs nothing to leave on; the difference only shows up once the
 *  operator types something that names a programme. */
export const DEFAULT_SCOPE_MODE: ScopeMode = "auto";

/** The request body for a scope selection. `manual` is the only mode that
 *  carries folders — sending them with `all`/`auto` would invite a stale list to
 *  be applied by a backend change later. */
export function scopeRequest(mode: ScopeMode, selected: string[]): SearchScope {
  return { mode, categories: mode === "manual" ? selected : [] };
}

/** Short label for the filter button. */
export function scopeButtonLabel(
  mode: ScopeMode,
  selected: string[],
  catalogue: ScopeCatalogue | null,
  applied: ResolvedScope | null,
): string {
  const total = catalogue?.categories.length ?? 0;
  if (mode === "manual") {
    // Compared by coverage, not count: a selection can hold single race stages.
    const picked = new Set(selected);
    const everything = total > 0 && catalogue!.categories.every((item) => picked.has(item.category));
    if (!selected.length || everything) return `All${total ? ` (${total})` : ""}`;
    if (selected.length <= 3) return selected.join(", ");
    return `${selected.length} folders`;
  }
  if (mode === "all") return `All${total ? ` (${total})` : ""}`;
  // Auto: the button reports what the LAST search actually ran on, because
  // that is the only scope the operator can act on; before then it is a promise.
  if (applied?.active) {
    const topics = applied.matched_topics.map((topic) => topic.label_en ?? topic.topic_id.replaceAll("_", " ")).join(", ");
    return `Auto · ${topics || `${applied.categories.length} folders`}`;
  }
  return "Auto";
}

/** Folders a group's "select all" checkbox covers, within this profile. */
export function groupCategories(catalogue: ScopeCatalogue | null, groupId: string): string[] {
  return catalogue?.groups.find((group) => group.id === groupId)?.categories ?? [];
}

/** Single videos a folder can be narrowed to (only S01's race stages have any). */
export function folderVideos(catalogue: ScopeCatalogue | null, category: string): string[] {
  return catalogue?.categories.find((item) => item.category === category)?.videos?.map((video) => video.video_id) ?? [];
}

/** Every item a selection may hold: each folder, then the single videos under it. */
export function scopeItems(catalogue: ScopeCatalogue | null): string[] {
  return (catalogue?.categories ?? []).flatMap((item) => [
    item.category,
    ...(item.videos ?? []).map((video) => video.video_id),
  ]);
}

/** Toggle one folder, keeping the catalogue's order so the label reads L→K.
 *  Either way its single videos go: the whole folder covers them, and taking
 *  the folder out takes them out with it. */
export function toggleCategory(
  selected: string[],
  category: string,
  catalogue: ScopeCatalogue | null,
): string[] {
  const videos = new Set(folderVideos(catalogue, category));
  const next = new Set(selected.filter((item) => !videos.has(item)));
  if (selected.includes(category)) next.delete(category);
  else next.add(category);
  return orderCategories([...next], catalogue);
}

/** Toggle one video of a folder. Unticking it from a whole folder keeps the
 *  folder's other videos; ticking the last missing one folds them back into
 *  the folder, so "all twelve stages" is sent as S01 rather than twelve ids. */
export function toggleVideo(
  selected: string[],
  category: string,
  video: string,
  catalogue: ScopeCatalogue | null,
): string[] {
  const videos = folderVideos(catalogue, category);
  const next = new Set(selected);
  if (next.has(category)) {
    next.delete(category);
    videos.filter((item) => item !== video).forEach((item) => next.add(item));
  } else if (next.has(video)) {
    next.delete(video);
  } else {
    next.add(video);
    if (videos.every((item) => next.has(item))) {
      videos.forEach((item) => next.delete(item));
      next.add(category);
    }
  }
  return orderCategories([...next], catalogue);
}

export function orderCategories(
  categories: string[],
  catalogue: ScopeCatalogue | null,
): string[] {
  const order = scopeItems(catalogue);
  const rank = new Map(order.map((category, index) => [category, index]));
  return [...new Set(categories)].sort(
    (a, b) => (rank.get(a) ?? Number.MAX_SAFE_INTEGER) - (rank.get(b) ?? Number.MAX_SAFE_INTEGER),
  );
}
