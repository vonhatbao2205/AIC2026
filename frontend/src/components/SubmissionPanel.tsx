import { useEffect, useMemo, useRef, useState } from "react";
import type { ImportedQuestion } from "../lib/questions";
import {
  MAX_ANSWER_LENGTH,
  MAX_ROWS_PER_QUESTION,
  findDuplicate,
  questionCsv,
  validateRows,
  type SubmissionRow,
} from "../lib/submission";

interface Props {
  questions: ImportedQuestion[];
  rows: SubmissionRow[];
  onChangeRow: (rowId: string, patch: Partial<SubmissionRow>) => void;
  onDeleteRow: (rowId: string) => void;
  onAddRow: (questionId: string) => void;
  onClearQuestion: (questionId: string) => void;
  onExport: () => void;
  onJumpToQuestion: (questionId: string) => void;
  exportError: string | null;
  /** Back to the console tab the operator came from. */
  onBack: () => void;
  backLabel: string;
  /** Import lives here too: this view is reachable with no tab open behind it. */
  onImport: (file: File) => void;
  importing: boolean;
  importError: string | null;
}

function framesToText(frames: number[]): string {
  return frames.join(", ");
}

function parseFrames(text: string): number[] {
  return text
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean)
    .map((part) => Number(part))
    .filter((value) => Number.isFinite(value))
    .map((value) => Math.round(value));
}

/** Frame cell that keeps what was typed rather than what parses.
 *
 *  Driving the input straight off `number[]` swallows a separator the moment it
 *  is typed — "1200," re-renders as "1200" — which makes a multi-event TRAKE row
 *  impossible to enter. The raw text lives here; the parsed frames go up. */
function FramesInput(props: { value: number[]; onChange: (frames: number[]) => void; placeholder: string }) {
  const [text, setText] = useState(() => framesToText(props.value));
  const mine = useRef(text);

  useEffect(() => {
    // Re-sync only when the change came from somewhere other than this input.
    if (parseFrames(mine.current).join(",") !== props.value.join(",")) {
      const next = framesToText(props.value);
      mine.current = next;
      setText(next);
    }
  }, [props.value]);

  return (
    <input
      className="cell-input mono"
      value={text}
      placeholder={props.placeholder}
      onChange={(event) => {
        mine.current = event.target.value;
        setText(event.target.value);
        props.onChange(parseFrames(event.target.value));
      }}
    />
  );
}

export function SubmissionPanel(props: Props) {
  const { questions, rows } = props;
  const fileRef = useRef<HTMLInputElement>(null);

  const byQuestion = useMemo(() => {
    const map = new Map<string, SubmissionRow[]>();
    for (const row of rows) {
      const list = map.get(row.questionId);
      if (list) list.push(row);
      else map.set(row.questionId, [row]);
    }
    return map;
  }, [rows]);

  const answered = questions.filter((question) => (byQuestion.get(question.id) ?? []).length > 0);
  const empty = questions.filter((question) => (byQuestion.get(question.id) ?? []).length === 0);
  const totalRows = rows.length;
  const orphans = rows.filter((row) => !questions.some((question) => question.id === row.questionId));

  return (
    <div className="submission-view" data-testid="submission-panel">
      <div className="submission-head">
        <button className="btn sm ghost" onClick={props.onBack} data-testid="submission-back">
          ← {props.backLabel}
        </button>
        <div>
          <h2 style={{ margin: 0, fontSize: 15 }}>Submission</h2>
          <div className="hint-text" style={{ margin: 0 }}>
            {totalRows} dòng · {answered.length}/{questions.length} câu đã có đáp án · export ra{" "}
            <code>submission/&lt;tên câu hỏi&gt;.csv</code>
          </div>
        </div>
        <div className="spacer" />
        <input
          ref={fileRef}
          type="file"
          accept=".zip,application/zip"
          style={{ display: "none" }}
          data-testid="submission-import-input"
          onChange={(event) => {
            const file = event.target.files?.[0];
            event.target.value = "";
            if (file) props.onImport(file);
          }}
        />
        <button
          className="btn sm"
          onClick={() => fileRef.current?.click()}
          disabled={props.importing}
          data-testid="submission-import"
        >
          {props.importing ? "Đang import…" : "⭳ Import câu hỏi"}
        </button>
        <button
          className="btn primary"
          onClick={props.onExport}
          disabled={totalRows === 0}
          data-testid="export-submission"
        >
          ⭱ Export submission.zip
        </button>
      </div>
      {props.importError && <div className="dup-warn">⚠ {props.importError}</div>}
      {props.exportError && <div className="dup-warn">⚠ {props.exportError}</div>}
      {orphans.length > 0 && (
        <div className="dup-warn" data-testid="submission-orphans">
          ⚠ {orphans.length} dòng thuộc câu hỏi không còn trong gói đã import — chúng sẽ KHÔNG được export.
        </div>
      )}

      {questions.length === 0 && (
        <div className="empty">
          Chưa import gói câu hỏi nào — bấm “⭳ Import câu hỏi” ở trên để nạp file .zip.
        </div>
      )}

      {answered.map((question) => {
        const questionRows = byQuestion.get(question.id) ?? [];
        const problems = validateRows(questionRows, question);
        const problemsByRow = new Map<string, string[]>();
        for (const problem of problems) {
          problemsByRow.set(problem.rowId, [...(problemsByRow.get(problem.rowId) ?? []), problem.message]);
        }
        const csv = questionCsv(questionRows.slice(0, MAX_ROWS_PER_QUESTION), question.kind);
        return (
          <div className="submission-block" key={question.id} data-testid={`submission-${question.id}`}>
            <div className="submission-block-head">
              <button
                className="btn sm ghost"
                onClick={() => props.onJumpToQuestion(question.id)}
                title="Mở tab đang trả lời câu này"
              >
                {question.id}
              </button>
              <span className={`badge ${question.kind === "qa" ? "ocr" : question.kind === "trake" ? "speech" : "image_pe"}`}>
                {question.kind}
              </span>
              {question.eventCount && <span className="dres-dim">{question.eventCount} events</span>}
              <span className="dres-dim">
                {questionRows.length}/{MAX_ROWS_PER_QUESTION} dòng
              </span>
              {problems.length > 0 && (
                <span className="dres-warn" data-testid={`problems-${question.id}`}>
                  {problems.length} cảnh báo
                </span>
              )}
              <div className="spacer" />
              <button className="btn sm ghost" onClick={() => props.onAddRow(question.id)}>
                + dòng
              </button>
              <button className="btn sm ghost" onClick={() => props.onClearQuestion(question.id)}>
                xoá hết
              </button>
            </div>

            <table className="submission-table">
              <thead>
                <tr>
                  <th style={{ width: 28 }}>#</th>
                  <th style={{ width: 130 }}>video</th>
                  <th>{question.kind === "trake" ? `frames (${question.eventCount ?? "N"}, cách nhau dấu phẩy)` : "frame_idx"}</th>
                  {question.kind === "qa" && <th style={{ width: "38%" }}>answer</th>}
                  <th style={{ width: 34 }} />
                </tr>
              </thead>
              <tbody>
                {questionRows.map((row, index) => {
                  const rowProblems = problemsByRow.get(row.id) ?? [];
                  const duplicate = findDuplicate(questionRows, row, question.kind, row.id);
                  return (
                    <tr
                      key={row.id}
                      className={rowProblems.length || duplicate ? "bad-row" : ""}
                      data-testid={`row-${question.id}-${index}`}
                    >
                      <td className="mono dim">{index + 1}</td>
                      <td>
                        <input
                          className="cell-input mono"
                          value={row.videoId}
                          onChange={(event) => props.onChangeRow(row.id, { videoId: event.target.value })}
                          placeholder="L01_V028"
                        />
                      </td>
                      <td>
                        <FramesInput
                          value={row.frames}
                          onChange={(frames) => props.onChangeRow(row.id, { frames })}
                          placeholder={question.kind === "trake" ? "1200, 1850, 2100" : "25300"}
                        />
                      </td>
                      {question.kind === "qa" && (
                        <td>
                          <input
                            className="cell-input"
                            value={row.answer}
                            maxLength={MAX_ANSWER_LENGTH * 2}
                            onChange={(event) => props.onChangeRow(row.id, { answer: event.target.value })}
                            placeholder="đáp án (≤100 ký tự)"
                          />
                        </td>
                      )}
                      <td>
                        <button
                          className="rail-tab-close"
                          onClick={() => props.onDeleteRow(row.id)}
                          aria-label={`Xoá dòng ${index + 1}`}
                          data-testid={`delete-${question.id}-${index}`}
                        >
                          ×
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>

            {problems.length > 0 && (
              <ul className="submission-problems">
                {problems.map((problem, index) => (
                  <li key={`${problem.rowId}-${index}`}>{problem.message}</li>
                ))}
              </ul>
            )}
            <pre className="guard-json" data-testid={`csv-${question.id}`}>{csv}</pre>
          </div>
        );
      })}

      {empty.length > 0 && (
        <div className="submission-block">
          <div className="submission-block-head">
            <span className="dres-dim">Chưa có đáp án ({empty.length})</span>
          </div>
          <div className="submission-empty-list">
            {empty.map((question) => (
              <button
                key={question.id}
                className="btn sm ghost"
                onClick={() => props.onAddRow(question.id)}
                title={question.text.slice(0, 200)}
              >
                {question.id} +
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
