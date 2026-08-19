import { useEffect, useRef, useState } from "react";
import type { ResolvedScope, ScopeCatalogue, ScopeMode } from "../api/types";
import { groupCategories, orderCategories, scopeButtonLabel, toggleCategory } from "../lib/scope";

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
    label: "Tự động",
    title: "Heuristic chủ đề đọc query và tự chọn thư mục (không nhận ra chủ đề nào thì tìm toàn bộ)",
  },
  { id: "all", label: "Tất cả", title: "Tìm toàn bộ thư mục của profile" },
  { id: "manual", label: "Tùy chọn", title: "Chỉ tìm trong các thư mục bạn tick" },
];

/** Folder filter: a button that drops down the checkbox list of dataset folders.
 *
 *  The list is the scope of the NEXT search, exactly like the retrieval-depth
 *  slider — nothing re-runs while it is open, because ticking six folders would
 *  otherwise fire six searches. */
export function ScopeFilter(props: Props) {
  const { catalogue, mode, onMode, selected, onSelected, applied, error } = props;
  const [open, setOpen] = useState(false);
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

  return (
    <div className="scope-filter" ref={rootRef}>
      <button
        className={`btn sm ${mode === "manual" || applied?.active ? "primary" : "ghost"}`}
        onClick={() => setOpen((value) => !value)}
        data-testid="scope-toggle"
        aria-expanded={open}
        aria-haspopup="true"
        title="Phạm vi tìm kiếm: chọn thư mục dữ liệu được phép trả kết quả"
      >
        📂 {label} <span aria-hidden>▾</span>
      </button>

      {open && (
        <div className="scope-menu" data-testid="scope-menu" role="dialog" aria-label="Phạm vi tìm kiếm">
          <div className="seg sm scope-modes" role="tablist" aria-label="Chế độ phạm vi">
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
              {applied.reason_vi}
              {applied.strict_categories.length > 0
                && applied.strict_categories.length < applied.categories.length && (
                <button
                  className="btn sm ghost"
                  data-testid="scope-strict"
                  title="Bỏ luôn các thư mục thời sự / L30 và chỉ tìm đúng chương trình của chủ đề"
                  onClick={() => pick(applied.strict_categories)}
                >
                  Chỉ {applied.strict_categories.join(", ")}
                </button>
              )}
            </div>
          )}

          {mode === "manual" && selected.length === 0 && (
            <div className="scope-reason" data-testid="scope-empty">
              Chưa tick thư mục nào — lọc rỗng không loại được gì, nên vẫn tìm toàn bộ.
            </div>
          )}

          {error && <div className="scope-reason bad" data-testid="scope-error">{error}</div>}

          <div className="scope-actions">
            <button className="btn sm ghost" data-testid="scope-select-all" onClick={() => pick(all)}>
              Chọn tất cả
            </button>
            <button className="btn sm ghost" data-testid="scope-clear" onClick={() => pick([])}>
              Bỏ chọn
            </button>
          </div>

          <div className="scope-list">
            {(catalogue?.groups ?? []).map((group) => {
              const groupCats = groupCategories(catalogue, group.id);
              const allOn = groupCats.length > 0 && groupCats.every((cat) => effectiveSet.has(cat));
              return (
                <div key={group.id} className="scope-group">
                  <label className="scope-row scope-group-head">
                    <input
                      type="checkbox"
                      checked={allOn}
                      data-testid={`scope-group-${group.id}`}
                      onChange={() =>
                        pick(
                          allOn
                            ? effective.filter((cat) => !groupCats.includes(cat))
                            : [...effective, ...groupCats],
                        )
                      }
                    />
                    <span className="scope-code">{group.label_vi}</span>
                  </label>
                  {groupCats.map((cat) => {
                    const item = categories.find((entry) => entry.category === cat);
                    return (
                      <label key={cat} className="scope-row" title={item?.label_vi}>
                        <input
                          type="checkbox"
                          checked={effectiveSet.has(cat)}
                          data-testid={`scope-cat-${cat}`}
                          onChange={() => pick(toggleCategory(effective, cat, catalogue))}
                        />
                        <span className="scope-code">{cat}</span>
                        <span className="scope-label">{item?.label_vi ?? ""}</span>
                      </label>
                    );
                  })}
                </div>
              );
            })}
            {!catalogue && !error && <div className="scope-reason">Đang tải danh sách thư mục…</div>}
          </div>

          <div className="hint-text">
            Phạm vi áp dụng cho lần <b>Search</b> tiếp theo, không đổi kết quả đang hiển thị.
          </div>
        </div>
      )}
    </div>
  );
}
