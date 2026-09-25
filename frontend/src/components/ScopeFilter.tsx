import { Fragment, useEffect, useRef, useState } from "react";
import type { ResolvedScope, ScopeCatalogue, ScopeMode } from "../api/types";
import {
  folderVideos,
  groupCategories,
  orderCategories,
  scopeButtonLabel,
  toggleCategory,
  toggleVideo,
} from "../lib/scope";

interface Props {
  catalogue: ScopeCatalogue | null;
  mode: ScopeMode;
  onMode: (mode: ScopeMode) => void;
  /** Ticked folders. Only read in `manual` mode; kept across mode flips so
   *  going back to a hand-picked list does not mean re-ticking it. */
  selected: string[];
  onSelected: (categories: string[]) => void;
  /** The scope the last search really applied, echoed by the backend. */
  applied: ResolvedScope | null;
  error: string | null;
}

const MODES: { id: ScopeMode; label: string; title: string }[] = [
  {
    id: "auto",
    label: "Auto",
    title: "Automatically select folders by query topic; search all folders when no topic is detected",
  },
  { id: "all", label: "All", title: "Search all folders in this profile" },
  { id: "manual", label: "Custom", title: "Search only the selected folders" },
];

/** Ref that shows a checkbox half-ticked; React has no prop for it. */
function indeterminate(value: boolean) {
  return (input: HTMLInputElement | null) => {
    if (input) input.indeterminate = value;
  };
}

/** Folder filter: a button that drops down the checkbox list of dataset folders.
 *
 *  The list is the scope of the NEXT search, exactly like the retrieval-depth
 *  slider — nothing re-runs while it is open, because ticking six folders would
 *  otherwise fire six searches. */
export function ScopeFilter(props: Props) {
  const { catalogue, mode, onMode, selected, onSelected, applied, error } = props;
  const [open, setOpen] = useState(false);
  // Collapsed groups the operator unfolded to pick single folders from. Kept
  // while the menu closes, so reopening it does not fold the list back up.
  const [expanded, setExpanded] = useState<ReadonlySet<string>>(() => new Set());
  const rootRef = useRef<HTMLDivElement>(null);

  // Close on an outside click or Escape, like any other menu on the page.
  useEffect(() => {
    if (!open) return;
    function onDown(event: MouseEvent) {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false);
    }
    function onKey(event: KeyboardEvent) {
      if (event.key === "Escape") setOpen(false);
    }
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const categories = catalogue?.categories ?? [];
  const all = categories.map((item) => item.category);
  const label = scopeButtonLabel(mode, selected, catalogue, applied);
  // In auto/all the checkboxes mirror what a search would cover, read-only, so
  // the list always answers "what is being searched" rather than only "what did
  // I tick". Ticking one is how the operator takes over — it switches to manual.
  const effective =
    mode === "manual" ? selected : applied?.active && mode === "auto" ? applied.categories : all;
  const effectiveSet = new Set(effective);

  function pick(next: string[]) {
    onSelected(orderCategories(next, catalogue));
    if (mode !== "manual") onMode("manual");
  }

  function toggleExpanded(groupId: string) {
    setExpanded((current) => {
      const next = new Set(current);
      if (next.has(groupId)) next.delete(groupId);
      else next.add(groupId);
      return next;
    });
  }

  return (
    <div className="scope-filter" ref={rootRef}>
      <button
        className={`btn sm ${mode === "manual" || applied?.active ? "primary" : "ghost"}`}
        onClick={() => setOpen((value) => !value)}
        data-testid="scope-toggle"
        aria-expanded={open}
        aria-haspopup="true"
        title="Search scope: select which dataset folders can return results"
      >
        📂 {label} <span aria-hidden>▾</span>
      </button>

      {open && (
        <div className="scope-menu" data-testid="scope-menu" role="dialog" aria-label="Search scope">
          <div className="seg sm scope-modes" role="tablist" aria-label="Scope mode">
            {MODES.map((item) => (
              <button
                key={item.id}
                role="tab"
                aria-selected={mode === item.id}
                className={mode === item.id ? "active" : ""}
                title={item.title}
                data-testid={`scope-mode-${item.id}`}
                onClick={() => onMode(item.id)}
              >
                {item.label}
              </button>
            ))}
          </div>

          {applied && mode === "auto" && (
            <div className="scope-reason" data-testid="scope-reason">
              {applied.reason_en ?? (applied.active ? "Search restricted to the selected folders." : "Searching all folders.")}
              {applied.strict_categories.length > 0
                && applied.strict_categories.length < applied.categories.length && (
                <button
                  className="btn sm ghost"
                  data-testid="scope-strict"
                  title="Exclude news folders and L30; search only the topic-specific program"
                  onClick={() => pick(applied.strict_categories)}
                >
                  Only {applied.strict_categories.join(", ")}
                </button>
              )}
            </div>
          )}

          {mode === "manual" && selected.length === 0 && (
            <div className="scope-reason" data-testid="scope-empty">
              No folders selected. An empty filter searches all folders.
            </div>
          )}

          {error && <div className="scope-reason bad" data-testid="scope-error">{error}</div>}

          <div className="scope-actions">
            <button className="btn sm ghost" data-testid="scope-select-all" onClick={() => pick(all)}>
              Select all
            </button>
            <button className="btn sm ghost" data-testid="scope-clear" onClick={() => pick([])}>
              Deselect
            </button>
          </div>

          <div className="scope-list">
            {(catalogue?.groups ?? []).map((group) => {
              const groupCats = groupCategories(catalogue, group.id);
              // What the operator can tick one by one: a folder, or the single
              // videos of a folder that has them (S01's stages). A video is on
              // when it is picked or its whole folder is.
              const leaves = groupCats.flatMap((cat) => {
                const videos = folderVideos(catalogue, cat);
                return videos.length ? videos.map((video) => ({ cat, id: video })) : [{ cat, id: cat }];
              });
              const onCount = leaves.filter((leaf) => effectiveSet.has(leaf.cat) || effectiveSet.has(leaf.id)).length;
              const allOn = leaves.length > 0 && onCount === leaves.length;
              const groupItems = new Set([...groupCats, ...leaves.map((leaf) => leaf.id)]);
              const unfolded = !group.collapsed || expanded.has(group.id);
              return (
                <div key={group.id} className="scope-group">
                  <div className="scope-group-head-row">
                    <label className="scope-row scope-group-head">
                      <input
                        type="checkbox"
                        checked={allOn}
                        // Half-ticked when only part of it is in: a collapsed
                        // group would otherwise look empty with N042 picked inside.
                        ref={indeterminate(onCount > 0 && !allOn)}
                        data-testid={`scope-group-${group.id}`}
                        onChange={() => {
                          const rest = effective.filter((item) => !groupItems.has(item));
                          pick(allOn ? rest : [...rest, ...groupCats]);
                        }}
                      />
                      <span className="scope-code">{group.label_en ?? group.id}</span>
                    </label>
                    {group.collapsed && (
                      <button
                        type="button"
                        className="scope-expand"
                        data-testid={`scope-expand-${group.id}`}
                        aria-expanded={unfolded}
                        title={unfolded ? "Hide the list" : `Pick single items out of these ${leaves.length}`}
                        onClick={() => toggleExpanded(group.id)}
                      >
                        {onCount > 0 && !allOn ? `${onCount}/${leaves.length}` : leaves.length}{" "}
                        <span aria-hidden>{unfolded ? "▾" : "▸"}</span>
                      </button>
                    )}
                  </div>
                  {unfolded && groupCats.map((cat) => {
                    const item = categories.find((entry) => entry.category === cat);
                    const whole = effectiveSet.has(cat);
                    const someVideos = (item?.videos ?? []).some((video) => effectiveSet.has(video.video_id));
                    return (
                      <Fragment key={cat}>
                        <label className="scope-row" title={item?.label_en ?? cat}>
                          <input
                            type="checkbox"
                            checked={whole}
                            ref={indeterminate(!whole && someVideos)}
                            data-testid={`scope-cat-${cat}`}
                            onChange={() => pick(toggleCategory(effective, cat, catalogue))}
                          />
                          <span className="scope-code">{cat}</span>
                          <span className="scope-label">{item?.label_en ?? cat}</span>
                        </label>
                        {(item?.videos ?? []).map((video) => (
                          <label
                            key={video.video_id}
                            className="scope-row scope-sub"
                            title={video.summary ?? video.label_en ?? video.video_id}
                          >
                            <input
                              type="checkbox"
                              checked={whole || effectiveSet.has(video.video_id)}
                              data-testid={`scope-video-${video.video_id}`}
                              onChange={() => pick(toggleVideo(effective, cat, video.video_id, catalogue))}
                            />
                            <span className="scope-code">{video.video_id}</span>
                            <span className="scope-label">{video.label_en ?? video.label_vi}</span>
                          </label>
                        ))}
                      </Fragment>
                    );
                  })}
                </div>
              );
            })}
            {!catalogue && !error && <div className="scope-reason">Loading folders…</div>}
          </div>

          <div className="hint-text">
            Scope applies to the next <b>Search</b>; the displayed results stay the same.
          </div>
        </div>
      )}
    </div>
  );
}
