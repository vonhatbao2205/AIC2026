import { useRef } from "react";
import type { ImportedQuestion } from "../lib/questions";

interface Props {
  questions: ImportedQuestion[];
  /** The question this tab answers; its statement prefills the query box. */
  selectedId: string | null;
  onSelect: (questionId: string | null) => void;
  onImport: (file: File) => void;
  importing: boolean;
  importError: string | null;
  /** Rows already collected for the selected question. */
  rowCount: number;
  stickyCount: number;
  stickyOpen: boolean;
  onOpenSticky: () => void;
  onOpenSubmission: () => void;
  /** Open one tab per question and search them all, the way an import does. */
  onSearchAll: () => void;
  searchingAll: boolean;
}

/** Replaces the DRES bar: the run/task selector became a query-pack selector.
 *
 *  DRES submission is switched off (see `DRES_ENABLED` in FullConsole), so there
 *  is no open task to route to any more — a submit now writes a row into the
 *  submission table for whichever imported question this tab is answering.
 */
export function QuestionBar(props: Props) {
  const { questions, selectedId, rowCount } = props;
  const fileRef = useRef<HTMLInputElement>(null);
  const selected = questions.find((question) => question.id === selectedId) ?? null;

  return (
    <div className="dres-bar" data-testid="question-bar">
      <span className={`health-dot ${selected ? "ok" : "warn"}`} />
      <span className="dres-label">QUERY PACK</span>

      <select
        className="dres-select"
        data-testid="question-select"
        value={selectedId ?? ""}
        onChange={(event) => props.onSelect(event.target.value || null)}
        title="Question assigned to this tab — select one to load its text into the query"
        disabled={questions.length === 0}
      >
        <option value="">
          {questions.length ? "— no question assigned —" : "no question pack imported"}
        </option>
        {questions.map((question) => (
          <option key={question.id} value={question.id}>
            {question.id} · {question.kind}
            {question.eventCount ? ` · ${question.eventCount} events` : ""}
          </option>
        ))}
      </select>

      {selected ? (
        <>
          <span className="dres-task" data-testid="question-statement" title={selected.text}>
            {selected.text.slice(0, 90)}
            {selected.text.length > 90 ? "…" : ""}
          </span>
          <span className="dres-dim">{rowCount} answers</span>
        </>
      ) : (
        <span className="dres-warn" data-testid="question-none">
          {questions.length
            ? "Assign a question to this tab before saving a submission"
            : "Import a question ZIP to get started"}
        </span>
      )}

      {props.importError && <span className="dres-warn">{props.importError}</span>}

      <div className="spacer" />
      <button
        className={`btn sm ghost sticky-toggle${props.stickyOpen ? " active" : ""}`}
        onClick={props.onOpenSticky}
        data-testid="sticky-toggle"
        aria-pressed={props.stickyOpen}
        title="Toggle sticky note (`) — drafts are stored on this device only"
      >
        ◆ Sticky{props.stickyCount ? ` (${props.stickyCount})` : ""} <span className="kbd">`</span>
      </button>
      <button className="btn sm ghost" onClick={props.onOpenSubmission} data-testid="open-submission">
        Submission
      </button>
      <button
        className="btn sm"
        onClick={props.onSearchAll}
        disabled={questions.length === 0 || props.searchingAll}
        data-testid="search-all"
        title={
          questions.length
            ? `Open ${questions.length} tabs and search all questions — replaces all open tabs`
            : "No question pack"
        }
      >
        {props.searchingAll ? "Searching…" : `⚡ Search all (${questions.length})`}
      </button>
      <input
        ref={fileRef}
        type="file"
        accept=".zip,application/zip"
        style={{ display: "none" }}
        data-testid="import-input"
        onChange={(event) => {
          const file = event.target.files?.[0];
          // Reset so re-importing the same file still fires a change event.
          event.target.value = "";
          if (file) props.onImport(file);
        }}
      />
      <button
        className="btn sm"
        onClick={() => fileRef.current?.click()}
        disabled={props.importing}
        data-testid="import-questions"
        title="Import a .zip file containing .txt questions"
      >
        {props.importing ? "Importing…" : "⭳ Import questions"}
      </button>
    </div>
  );
}
