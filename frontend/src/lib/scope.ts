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
    if (!selected.length || selected.length === total) return `Tất cả${total ? ` (${total})` : ""}`;
    if (selected.length <= 3) return selected.join(", ");
    return `${selected.length} thư mục`;
  }
  if (mode === "all") return `Tất cả${total ? ` (${total})` : ""}`;
  // Auto: the button reports what the LAST search actually ran on, because
  // that is the only scope the operator can act on; before then it is a promise.
  if (applied?.active) {
    const topics = applied.matched_topics.map((topic) => topic.label_vi).join(", ");
    return `Tự động · ${topics || `${applied.categories.length} thư mục`}`;
  }
  return "Tự động";
}

/** Folders a group's "select all" checkbox covers, within this profile. */
export function groupCategories(catalogue: ScopeCatalogue | null, groupId: string): string[] {
  return catalogue?.groups.find((group) => group.id === groupId)?.categories ?? [];
}

/** Toggle one folder, keeping the catalogue's order so the label reads L→K. */
export function toggleCategory(
  selected: string[],
  category: string,
  catalogue: ScopeCatalogue | null,
): string[] {
  const next = new Set(selected);
  if (next.has(category)) next.delete(category);
  else next.add(category);
  return orderCategories([...next], catalogue);
}

export function orderCategories(
  categories: string[],
  catalogue: ScopeCatalogue | null,
): string[] {
  const order = catalogue?.categories.map((item) => item.category) ?? [];
  const rank = new Map(order.map((category, index) => [category, index]));
  return [...new Set(categories)].sort(
    (a, b) => (rank.get(a) ?? Number.MAX_SAFE_INTEGER) - (rank.get(b) ?? Number.MAX_SAFE_INTEGER),
  );
}
