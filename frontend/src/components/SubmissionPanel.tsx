import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api/client";
import type { RetrievalDatabase } from "../api/types";
import type { ImportedQuestion } from "../lib/questions";
import { packSummary } from "../lib/questionPack";
import {
  MAX_ANSWER_LENGTH,
  MAX_ROWS_PER_QUESTION,
  auditSubmission,
  findDuplicate,
  qaAnswerMissingEverywhere,
  questionCsv,
  validateRows,
  type SubmissionRow,
} from "../lib/submission";
import type { AnswerGenProgress } from "../lib/answerGen";
import type { ImportMode, ImportedSubmission } from "../lib/submissionImport";
import { planImport } from "../lib/submissionImport";
import {
  hasRowDrag,
  planReorder,
  readRowDrag,
  setRowDrag,
  type OrderChange,
} from "../lib/submissionOrder";
import { SubmissionFrameEditor, type CommitMode, type FrameEdit } from "./SubmissionFrameEditor";

/** Rows shown per question before the operator asks for the rest. */
const COLLAPSED_ROWS = 10;

export interface PackInfo {
  shared: boolean;
  sessionName: string | null;
  packHash: string | null;
  publishedBy: string | null;
  origin: "server" | "cache" | "local" | "none";
  loading: boolean;
  error: string | null;
}

/** What the bulk answer generator asks for, and what it reports back. */
export interface AutoGenOptions {
  limit: number;
  /** Leave questions that already have answers alone. */
  onlyEmpty: boolean;
  /** Restrict to these questions; empty means the whole pack. */
  questionIds: string[];
}

export interface AutoGenInfo {
  running: boolean;
  progress: AnswerGenProgress | null;
  /** One entry per finished question, newest last. */
  log: AnswerGenProgress[];
  error: string | null;
  onStart: (options: AutoGenOptions) => void;
  onCancel: () => void;
}

/** Restoring answers from a previously exported `submission.zip`. */
export interface AnswerImportInfo {
  /** Parsed and waiting for the operator to choose how to apply it. */
  pending: ImportedSubmission | null;
  importing: boolean;
  error: string | null;
  onPick: (file: File) => void;
  onApply: (mode: ImportMode) => void;
  onCancel: () => void;
}

export interface SubmissionSyncInfo {
  shared: boolean;
  status: "offline" | "connecting" | "live" | "error";
  pending: number;
  error: string | null;
  room: string;
  user: string;
  conflicts: string[];
  onDismissConflict: (rowId: string) => void;
}

interface Props {
  questions: ImportedQuestion[];
  rows: SubmissionRow[];
  onChangeRow: (rowId: string, patch: Partial<SubmissionRow>) => void;
  /** Rewrite the rank of a question's answers — drag-and-drop and Alt+↑/↓.
   *  The plan is computed here because only this component knows the order the
   *  rows are actually shown in. An empty plan means nothing moved. */
  onReorderRows: (changes: OrderChange[], label: string) => void;
  onDeleteRow: (rowId: string) => void;
  onAddRow: (questionId: string) => void;
  /** Copy a row (with an edited frame) into a new answer for the same question. */
  onCloneRow: (row: SubmissionRow) => void;
  onClearQuestion: (questionId: string) => void;
  /** Write one Q&A answer into every row of a question, in a single write. */
  onFillAnswer: (questionId: string, answer: string) => void;
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
  sync: SubmissionSyncInfo;
  /** Parsed pack awaiting the publish decision. */
  pendingPack: ImportedQuestion[] | null;
  onApplyPack: () => void;
  onCancelPack: () => void;
  onSearchAll: () => void;
  searchingAll: boolean;
  /** Put the last action back; returns its label, or null when nothing is left. */
  onUndo: () => string | null;
  /** Label of the action Ctrl+Z would put back — null disables the button. */
  undoLabel: string | null;
  pack: PackInfo;
  /** Used for rows that predate the per-row profile field. */
  defaultRetrievalDatabase: RetrievalDatabase;
  autoGen: AutoGenInfo;
  answerImport: AnswerImportInfo;
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

/** What has been typed into a row but not sent yet.
 *
 *  Every cell used to write straight through to the shared table on each
 *  keystroke: typing "L30_V046" was nine writes, and the input was driven off
 *  the shared row, so anything arriving from Postgres mid-word — a teammate's
 *  edit, or this client's own echo — replaced what was on screen and ate the
 *  rest of the word. Held here instead, and sent once, as one write for the
 *  whole row. */
interface RowDraft {
  videoId?: string;
  /** Raw text, not `number[]`: a trailing separator must survive being typed. */
  frames?: string;
  answer?: string;
}

function draftDiffers(row: SubmissionRow, draft: RowDraft | undefined): boolean {
  if (!draft) return false;
  if (draft.videoId !== undefined && draft.videoId !== row.videoId) return true;
  if (draft.answer !== undefined && draft.answer !== row.answer) return true;
  if (draft.frames !== undefined && parseFrames(draft.frames).join(",") !== row.frames.join(",")) {
    return true;
  }
  return false;
}

/** Lazily resolved keyframe picture for exactly one row.
 *
 *  The table itself never loads media: a hundred answers would otherwise mean a
 *  hundred image requests on five machines. Only the frame being looked at is
 *  fetched, and only once the operator asks for it. */
function FramePreview(props: {
  videoId: string;
  frameIdx: number;
  keyframeId: string | null;
  retrievalDatabase: RetrievalDatabase;
  onClose: () => void;
  onOpenVideo: () => void;
}) {
  const { keyframeId, retrievalDatabase } = props;
  const [url, setUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setUrl(null);
    setError(null);
    if (!keyframeId) return;
    let cancelled = false;
    api
      .keyframe(keyframeId, retrievalDatabase)
      .then((info) => !cancelled && setUrl(info.keyframe_url))
      .catch(() => !cancelled && setError("Không resolve được ảnh keyframe."));
    return () => {
      cancelled = true;
    };
  }, [keyframeId, retrievalDatabase]);

  return (
    <div className="preview-overlay" data-testid="frame-preview">
      <div className="preview-head">
        <span className="mono">
          {props.videoId} · frame {props.frameIdx}
          {keyframeId ? ` · ${keyframeId}` : ""}
        </span>
        <div className="spacer" />
        <button className="btn sm ghost" onClick={props.onOpenVideo}>V · mở video</button>
        <button className="btn sm ghost" onClick={props.onClose}>P · đóng</button>
      </div>
      {keyframeId ? (
        error ? (
          <div className="dup-warn">⚠ {error}</div>
        ) : url ? (
          <img className="preview-img" src={url} alt={keyframeId} />
        ) : (
          <div className="empty">Đang tải ảnh…</div>
        )
      ) : (
        // A frame taken from a paused video need not be an extracted keyframe.
        // Showing the nearest one would quietly misrepresent what gets exported.
        <div className="empty" data-testid="preview-raw">
          RAW VIDEO FRAME — frame {props.frameIdx} không phải keyframe đã trích, nên không có
          ảnh tĩnh. Bấm <b>V</b> để mở video đúng tại frame này.
        </div>
      )}
    </div>
  );
}

/** The bulk-generation control: how many answers, which questions, and progress.
 *
 *  Deliberately explicit about what it will overwrite. Generation replaces a
 *  question's whole answer list, and an operator who spent ten minutes hand-picking
 *  three frames must not lose them to a mis-click — hence "chỉ câu chưa có đáp án"
 *  on by default and a count of exactly what is about to be rewritten. */
function AutoGenPanel(props: {
  questions: ImportedQuestion[];
  byQuestion: Map<string, SubmissionRow[]>;
  autoGen: AutoGenInfo;
  limit: number;
  onLimit: (limit: number) => void;
  onlyEmpty: boolean;
  onOnlyEmpty: (value: boolean) => void;
  onClose: () => void;
}) {
  const { autoGen } = props;
  const rowsOf = (questionId: string) => props.byQuestion.get(questionId) ?? [];
  const hasGenerated = (questionId: string) =>
    rowsOf(questionId).some((row) => row.source === "generated");
  const notGeneratedYet = props.questions.filter((question) => !hasGenerated(question.id)).length;
  const targets = props.onlyEmpty ? notGeneratedYet : props.questions.length;
  const regenerated = props.onlyEmpty ? 0 : props.questions.length - notGeneratedYet;
  // What the generator will leave alone, stated up front: this is the number the
  // operator is really asking about when they hesitate over the button.
  const kept = props.questions.reduce(
    (total, question) =>
      total + rowsOf(question.id).filter((row) => row.source !== "generated").length,
    0,
  );
  const done = autoGen.log.filter((entry) => entry.status !== "running").length;
  const failed = autoGen.log.filter((entry) => entry.status === "failed");

  return (
    <div className="pack-preview" data-testid="autogen-panel">
      <div>
        <b>✨ Tự sinh đáp án</b> — mỗi câu được truy xuất lại rồi xếp thành danh sách đáp án
        có thứ tự (thuật toán phân bổ ngân sách theo mốc R@1/5/20/50/100).
      </div>
      <div className="hint-text" style={{ margin: 0 }}>
        Đáp án bạn tự chấm <b>không bị đụng tới</b> và luôn đứng trước; máy chỉ điền phần còn
        thiếu cho đủ số dòng. Bấm lại chỉ thay các dòng do máy sinh lần trước.
      </div>
      <div className="row" style={{ gap: 12, flexWrap: "wrap", alignItems: "center" }}>
        <label className="row" style={{ gap: 6, alignItems: "center" }}>
          <span className="dres-dim">tổng số dòng / câu</span>
          <input
            className="cell-input mono"
            style={{ width: 64 }}
            type="number"
            min={1}
            max={MAX_ROWS_PER_QUESTION}
            value={props.limit}
            data-testid="autogen-limit"
            onChange={(event) =>
              props.onLimit(
                Math.max(1, Math.min(MAX_ROWS_PER_QUESTION, Math.round(Number(event.target.value) || 1))),
              )
            }
          />
        </label>
        <label className="row" style={{ gap: 6, alignItems: "center" }}>
          <input
            type="checkbox"
            checked={props.onlyEmpty}
            data-testid="autogen-only-empty"
            onChange={(event) => props.onOnlyEmpty(event.target.checked)}
          />
          <span className="dres-dim">chỉ câu máy chưa sinh ({notGeneratedYet})</span>
        </label>
        <span className="dres-dim">
          sẽ chạy {targets} câu
          {kept > 0 ? ` · giữ nguyên ${kept} dòng bạn đã chấm` : ""}
          {regenerated > 0 ? ` · sinh lại ${regenerated} câu` : ""}
        </span>
      </div>
      {regenerated > 0 && (
        <div className="hint-text" data-testid="autogen-regenerate-note">
          ↻ {regenerated} câu đã có dòng máy sinh — các dòng đó sẽ được thay bằng danh sách mới.
          Dòng bạn tự chấm vẫn giữ nguyên.
        </div>
      )}
      <div className="row" style={{ gap: 8, marginTop: 6 }}>
        <button
          className="btn primary"
          data-testid="autogen-start"
          disabled={autoGen.running || targets === 0}
          onClick={() =>
            autoGen.onStart({ limit: props.limit, onlyEmpty: props.onlyEmpty, questionIds: [] })
          }
        >
          {autoGen.running ? "Đang chạy…" : `⚙ Sinh đáp án cho ${targets} câu`}
        </button>
        {autoGen.running ? (
          <button className="btn ghost" data-testid="autogen-cancel" onClick={autoGen.onCancel}>
            Dừng
          </button>
        ) : (
          <button className="btn ghost" onClick={props.onClose}>
            Đóng
          </button>
        )}
      </div>
      {(autoGen.running || autoGen.log.length > 0) && (
        <div className="hint-text" data-testid="autogen-progress" style={{ marginTop: 6 }}>
          {autoGen.progress
            ? `[${autoGen.progress.index + 1}/${autoGen.progress.total}] ${autoGen.progress.questionId} — ${
                autoGen.progress.status === "running" ? "đang truy xuất…" : autoGen.progress.status
              }`
            : `xong ${done} câu`}
          {failed.length > 0 && (
            <ul className="submission-problems">
              {failed.slice(0, 8).map((entry) => (
                <li key={entry.questionId} className="error">
                  ⛔ {entry.questionId}: {entry.error}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
      {autoGen.error && <div className="dup-warn">⚠ {autoGen.error}</div>}
    </div>
  );
}

/** What an imported `submission.zip` would do, before it does it.
 *
 *  Answers are the one thing in this app that cannot be recomputed, so importing
 *  them is never a single click: the archive is parsed, counted against what is
 *  already on screen, and applied only once the operator picks how. */
function AnswerImportPreview(props: {
  parsed: ImportedSubmission;
  rows: SubmissionRow[];
  onApply: (mode: ImportMode) => void;
  onCancel: () => void;
}) {
  const [mode, setMode] = useState<ImportMode>("fill");
  const plan = useMemo(
    () => planImport(props.parsed, props.rows, mode),
    [props.parsed, props.rows, mode],
  );
  const problems = [
    ...props.parsed.problems,
    ...props.parsed.files.flatMap((file) => file.problems),
  ];
  const incoming = plan.write.reduce((total, file) => total + file.rows.length, 0);

  return (
    <div className="pack-preview" data-testid="answer-import-preview">
      <div>
        <b>⭳ Import đáp án</b> — {props.parsed.files.length} câu ·{" "}
        {props.parsed.totalRows} dòng trong file
      </div>
      <div className="row" style={{ gap: 14, flexWrap: "wrap", alignItems: "center" }}>
        <label className="row" style={{ gap: 6, alignItems: "center" }}>
          <input
            type="radio"
            checked={mode === "fill"}
            data-testid="answer-import-mode-fill"
            onChange={() => setMode("fill")}
          />
          <span className="dres-dim">chỉ điền câu đang trống</span>
        </label>
        <label className="row" style={{ gap: 6, alignItems: "center" }}>
          <input
            type="radio"
            checked={mode === "replace"}
            data-testid="answer-import-mode-replace"
            onChange={() => setMode("replace")}
          />
          <span className="dres-dim">thay thế đáp án của các câu có trong file</span>
        </label>
      </div>
      <div className="hint-text" style={{ margin: 0 }} data-testid="answer-import-plan">
        Sẽ ghi {incoming} dòng vào {plan.write.length} câu
        {plan.skipped.length > 0 ? ` · bỏ qua ${plan.skipped.length} câu` : ""}
        {plan.replacedRows > 0 ? ` · XOÁ ${plan.replacedRows} dòng đang có` : ""}
      </div>
      {plan.replacedRows > 0 && (
        <div className="dup-warn" data-testid="answer-import-replace-warning">
          ⚠ {plan.replacedRows} dòng đáp án hiện tại sẽ bị xoá và thay bằng nội dung trong file.
        </div>
      )}
      {props.parsed.orphanIds.length > 0 && (
        <div className="dup-warn" data-testid="answer-import-orphans">
          ⚠ {props.parsed.orphanIds.length} câu trong file không có trong gói câu hỏi đang mở
          ({props.parsed.orphanIds.slice(0, 4).join(", ")}
          {props.parsed.orphanIds.length > 4 ? "…" : ""}) — nạp vào vẫn được nhưng sẽ KHÔNG
          được export. Có thể bạn đang mở nhầm gói câu hỏi.
        </div>
      )}
      {problems.length > 0 && (
        <ul className="submission-problems" data-testid="answer-import-problems">
          {problems.slice(0, 10).map((problem, index) => (
            <li key={index} className="error">⛔ {problem}</li>
          ))}
          {problems.length > 10 && <li>… và {problems.length - 10} dòng lỗi nữa</li>}
        </ul>
      )}
      <div className="row" style={{ gap: 8, marginTop: 6 }}>
        <button
          className="btn primary"
          data-testid="answer-import-apply"
          disabled={plan.write.length === 0}
          onClick={() => props.onApply(mode)}
        >
          ⇩ Nạp {incoming} dòng
        </button>
        <button className="btn ghost" data-testid="answer-import-cancel" onClick={props.onCancel}>
          Huỷ
        </button>
      </div>
    </div>
  );
}

/** One answer, typed once, written into every row of a Q&A question.
 *
 *  The generator ranks WHERE to look; the text is a human judgement, so it is not
 *  something the generated rows can carry. Without this the normal state after
 *  generating is a hundred rows sharing one blank — a hundred export-blocking
 *  errors for a single missing decision.
 *
 *  A button rather than live binding: every keystroke would otherwise rewrite a
 *  hundred rows, and each rewrite is a write to the shared store. */
function QaAnswerFill(props: {
  rows: SubmissionRow[];
  onFill: (answer: string) => void;
}) {
  const { rows } = props;
  // Prefill with the answer already shared by every row, so editing an existing
  // one is a correction rather than retyping. Rows deliberately disagreeing (a
  // second guess on the same frame is a legitimate Q&A tactic) start blank.
  const common = useMemo(() => {
    const answers = new Set(rows.map((row) => row.answer.trim()).filter(Boolean));
    return answers.size === 1 && rows.every((row) => row.answer.trim()) ? [...answers][0] : "";
  }, [rows]);
  const [text, setText] = useState(common);
  const lastCommon = useRef(common);
  useEffect(() => {
    // Re-sync only when the rows changed underneath, never over live typing.
    if (lastCommon.current !== common) {
      lastCommon.current = common;
      setText(common);
    }
  }, [common]);

  const missing = rows.filter((row) => !row.answer.trim()).length;
  const tooLong = text.trim().length > MAX_ANSWER_LENGTH;
  const apply = () => {
    if (!text.trim() || tooLong) return;
    props.onFill(text.trim());
  };

  return (
    <div className="qa-answer-fill" data-testid="qa-answer-fill">
      <span className="dres-dim">answer cho cả câu</span>
      <input
        className="cell-input"
        value={text}
        maxLength={MAX_ANSWER_LENGTH * 2}
        placeholder="ví dụ: 5 · màu xanh · Giang Ly"
        data-testid="qa-answer-input"
        onChange={(event) => setText(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter") {
            event.preventDefault();
            apply();
          }
        }}
      />
      <button
        className="btn sm"
        disabled={!text.trim() || tooLong || rows.length === 0}
        onClick={apply}
        data-testid="qa-answer-apply"
        title="Ghi đáp án này vào tất cả các dòng của câu"
      >
        Điền vào {rows.length} dòng
      </button>
      {tooLong && (
        <span className="dres-warn" data-testid="qa-answer-too-long">
          ⛔ {text.trim().length}/{MAX_ANSWER_LENGTH} ký tự
        </span>
      )}
      {!tooLong && missing > 0 && (
        <span className="dres-warn" data-testid="qa-answer-missing">
          ⚠ {missing}/{rows.length} dòng chưa có answer — sẽ bị chặn khi export
        </span>
      )}
    </div>
  );
}

export function SubmissionPanel(props: Props) {
  const { questions, rows, sync, autoGen } = props;
  const fileRef = useRef<HTMLInputElement>(null);
  const answerFileRef = useRef<HTMLInputElement>(null);
  const [selected, setSelected] = useState<{ rowId: string; slot: number } | null>(null);
  // Collapsed by default: 25 questions × 100 rows is one scroll with no shape to
  // it, and comparing two questions means paging past everything between them.
  const [expanded, setExpanded] = useState<ReadonlySet<string>>(() => new Set());
  const [previewOpen, setPreviewOpen] = useState(false);
  const [editorOpen, setEditorOpen] = useState(false);
  const [genOpen, setGenOpen] = useState(false);
  const [genLimit, setGenLimit] = useState(MAX_ROWS_PER_QUESTION);
  const [genOnlyEmpty, setGenOnlyEmpty] = useState(true);
  // What the last Ctrl+Z put back. Undo is a write the whole team sees, so
  // saying which action it reversed matters more here than in a solo editor.
  const [undone, setUndone] = useState<string | null>(null);
  const [drafts, setDrafts] = useState<Record<string, RowDraft>>({});

  const editDraft = useCallback((rowId: string, patch: RowDraft) => {
    setDrafts((current) => ({ ...current, [rowId]: { ...current[rowId], ...patch } }));
  }, []);

  const discardDraft = useCallback((rowId: string) => {
    setDrafts((current) => {
      if (!(rowId in current)) return current;
      const { [rowId]: _dropped, ...rest } = current;
      return rest;
    });
  }, []);

  /** Send a whole row at once. One write per row rather than per field keeps a
   *  hand-typed video id and its frames from racing each other into the table. */
  const commitDraft = useCallback(
    (row: SubmissionRow) => {
      const draft = draftsRef.current[row.id];
      discardDraft(row.id);
      if (!draft) return;
      const patch: Partial<SubmissionRow> = {};
      if (draft.videoId !== undefined && draft.videoId !== row.videoId) patch.videoId = draft.videoId;
      if (draft.answer !== undefined && draft.answer !== row.answer) patch.answer = draft.answer;
      if (draft.frames !== undefined) {
        const frames = parseFrames(draft.frames);
        if (frames.join(",") !== row.frames.join(",")) patch.frames = frames;
      }
      if (Object.keys(patch).length) props.onChangeRow(row.id, patch);
    },
    [discardDraft, props],
  );

  const draftsRef = useRef(drafts);
  draftsRef.current = drafts;

  const onCellKey = useCallback(
    (event: React.KeyboardEvent<HTMLInputElement>, row: SubmissionRow) => {
      if (event.key === "Enter") {
        event.preventDefault();
        commitDraft(row);
      } else if (event.key === "Escape") {
        // The panel closes the preview on Escape; inside a cell it must mean
        // "throw away what I typed" instead.
        event.stopPropagation();
        discardDraft(row.id);
      }
    },
    [commitDraft, discardDraft],
  );

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
  // Typed but never sent. The operator has to be able to see this before
  // pressing export, or the CSV silently ships the values from before the edit.
  const unsent = rows.filter((row) => draftDiffers(row, drafts[row.id]));

  const commitAll = useCallback(() => {
    for (const row of rows.filter((item) => draftDiffers(item, draftsRef.current[item.id]))) {
      commitDraft(row);
    }
  }, [commitDraft, rows]);
  const orphans = rows.filter((row) => !questions.some((question) => question.id === row.questionId));
  // Recomputed here so the operator sees what blocks the export before pressing
  // it, not as an error message afterwards.
  const audit = useMemo(() => auditSubmission(questions, rows), [questions, rows]);

  const visibleRowsOf = useCallback(
    (questionId: string) => {
      const list = byQuestion.get(questionId) ?? [];
      return expanded.has(questionId) ? list : list.slice(0, COLLAPSED_ROWS);
    },
    [byQuestion, expanded],
  );

  /** Rows in the order they are rendered, for ↑/↓ navigation across questions.
   *  Collapsed rows are left out on purpose, so the arrows walk exactly what is
   *  on screen instead of stopping on rows nobody can see. */
  const flatRows = useMemo(
    () => answered.flatMap((question) => visibleRowsOf(question.id)),
    [answered, visibleRowsOf],
  );
  const selectedRow = selected ? rows.find((row) => row.id === selected.rowId) ?? null : null;

  // Collapsing hides rows, and P / V / Delete all act on the selection —
  // Delete would remove a row the operator cannot see. Drop the selection
  // instead of leaving it pointing off screen.
  useEffect(() => {
    if (selected && !flatRows.some((row) => row.id === selected.rowId)) setSelected(null);
  }, [flatRows, selected]);
  const slot = Math.min(selected?.slot ?? 0, Math.max(0, (selectedRow?.frames.length ?? 1) - 1));

  const move = useCallback(
    (delta: number) => {
      if (!flatRows.length) return;
      const index = flatRows.findIndex((row) => row.id === selected?.rowId);
      const next = Math.min(Math.max((index < 0 ? 0 : index) + delta, 0), flatRows.length - 1);
      setSelected({ rowId: flatRows[next].id, slot: 0 });
    },
    [flatRows, selected?.rowId],
  );

  /** Send one ranking change for the whole team.
   *
   *  Rank is the score — `R@k` is a max over the first k answers — so moving an
   *  answer up is the same class of edit as changing its frame, and it is
   *  written, undoable and visible on every other screen. The index is taken
   *  from the QUESTION's full list, never from the visible slice: the two only
   *  coincide while a question is expanded. */
  const reorder = useCallback(
    (questionId: string, rowId: string, targetRowId: string) => {
      if (rowId === targetRowId) return;
      const list = byQuestion.get(questionId) ?? [];
      const from = list.findIndex((row) => row.id === rowId);
      const to = list.findIndex((row) => row.id === targetRowId);
      if (from < 0 || to < 0) return;
      const plan = planReorder(list, from, to);
      if (!plan.length) return;
      // Past the fold the row would land out of sight, and the button would read
      // as a no-op on a question showing only its first ten answers.
      if (to >= COLLAPSED_ROWS) setExpanded((current) => new Set(current).add(questionId));
      props.onReorderRows(plan, `${questionId}: hạng ${from + 1} → ${to + 1}`);
      // Follow the row, so ↑/↓ and Alt+↑/↓ keep acting on what was just moved.
      setSelected({ rowId, slot: 0 });
    },
    [byQuestion, props],
  );

  /** Alt+↑/↓: move the selected answer one rank. The keyboard path exists
   *  because a hundred-row question does not fit on screen, and dragging rank 60
   *  to rank 1 through an auto-scrolling table is not a thing anyone can do
   *  under a clock. */
  const nudge = useCallback(
    (delta: number) => {
      if (!selectedRow) return;
      const list = byQuestion.get(selectedRow.questionId) ?? [];
      const from = list.findIndex((row) => row.id === selectedRow.id);
      const to = from + delta;
      if (from < 0 || to < 0 || to >= list.length) return;
      reorder(selectedRow.questionId, selectedRow.id, list[to].id);
    },
    [byQuestion, reorder, selectedRow],
  );

  /** Which row the pointer is currently over, and from which side.
   *
   *  `dragover` fires continuously, so the state is only written when the target
   *  actually changes — otherwise every mouse move re-renders the table. */
  const [dropTarget, setDropTarget] = useState<{ rowId: string; below: boolean } | null>(null);
  /** The row being dragged, so it can be dimmed and its own line hidden. */
  const [dragging, setDragging] = useState<string | null>(null);
  /** Rows are only draggable while the grip is held. Making every `<tr>`
   *  permanently draggable is what breaks selecting text inside its inputs. */
  const [gripped, setGripped] = useState<string | null>(null);

  const endDrag = useCallback(() => {
    setDropTarget(null);
    setDragging(null);
    setGripped(null);
  }, []);

  // A grip pressed and then released anywhere else — a click that never became a
  // drag — would otherwise leave that row permanently draggable, and a
  // draggable `<tr>` is exactly what stops the mouse selecting text in its own
  // cells.
  useEffect(() => {
    if (!gripped) return;
    const release = () => setGripped(null);
    window.addEventListener("mouseup", release);
    return () => window.removeEventListener("mouseup", release);
  }, [gripped]);

  useEffect(() => {
    function isTyping() {
      const element = document.activeElement;
      return element && (element.tagName === "INPUT" || element.tagName === "TEXTAREA");
    }
    function onKey(event: KeyboardEvent) {
      if (event.key === "Escape") {
        if (editorOpen) setEditorOpen(false);
        else if (previewOpen) setPreviewOpen(false);
        return;
      }
      if ((event.key === "e" || event.key === "E") && (event.ctrlKey || event.metaKey)) {
        event.preventDefault();
        props.onExport();
        return;
      }
      // Deliberately after the typing guard below is duplicated here: inside a
      // cell Ctrl+Z must stay the browser's own text undo, so click out of the
      // cell to undo the action itself.
      if ((event.key === "z" || event.key === "Z") && (event.ctrlKey || event.metaKey)) {
        if (isTyping() || editorOpen) return;
        event.preventDefault();
        setUndone(props.onUndo());
        return;
      }
      if (isTyping() || editorOpen) return;
      switch (event.key) {
        // Alt moves the ANSWER, not the cursor: same key, one modifier, the way
        // every list editor does it.
        case "ArrowDown":
          event.preventDefault();
          if (event.altKey) nudge(1);
          else move(1);
          break;
        case "ArrowUp":
          event.preventDefault();
          if (event.altKey) nudge(-1);
          else move(-1);
          break;
        case "ArrowRight":
          if (selectedRow && selectedRow.frames.length > 1) {
            event.preventDefault();
            setSelected({ rowId: selectedRow.id, slot: Math.min(slot + 1, selectedRow.frames.length - 1) });
          }
          break;
        case "ArrowLeft":
          if (selectedRow && selectedRow.frames.length > 1) {
            event.preventDefault();
            setSelected({ rowId: selectedRow.id, slot: Math.max(slot - 1, 0) });
          }
          break;
        case "p":
        case "P":
          if (selectedRow) setPreviewOpen((open) => !open);
          break;
        case "v":
        case "V":
          if (selectedRow) setEditorOpen(true);
          break;
        case "Delete":
          if (selectedRow) props.onDeleteRow(selectedRow.id);
          break;
        default:
          break;
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [move, nudge, selectedRow, slot, previewOpen, editorOpen, props]);

  /** Write a re-picked frame into the slot that was edited — either over the
   *  selected row, or as a second answer to the same question. */
  const commitFrameEdit = useCallback(
    (edit: FrameEdit, mode: CommitMode) => {
      if (!selectedRow) return;
      const frames = [...selectedRow.frames];
      const keyframeIds = [...(selectedRow.keyframeIds ?? [])];
      const ptsTimes = [...(selectedRow.ptsTimes ?? [])];
      while (keyframeIds.length < frames.length) keyframeIds.push(null);
      while (ptsTimes.length < frames.length) ptsTimes.push(null);
      frames[slot] = edit.frameIdx;
      keyframeIds[slot] = edit.keyframeId;
      ptsTimes[slot] = edit.ptsTime;
      if (mode === "new") {
        // A TRAKE alternative keeps the other events, so the whole row is
        // copied and only the edited slot differs.
        props.onCloneRow({ ...selectedRow, frames, keyframeIds, ptsTimes });
      } else {
        props.onChangeRow(selectedRow.id, { frames, keyframeIds, ptsTimes });
      }
      setEditorOpen(false);
    },
    [selectedRow, slot, props],
  );

  const syncLabel =
    !sync.shared
      ? "● local (chưa cấu hình Supabase)"
      : sync.pending > 0
        ? `● ${sync.pending} chờ đồng bộ`
        : sync.status === "live"
          ? "● Live"
          : sync.status === "error"
            ? "● mất kết nối"
            : "● đang kết nối…";

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
        {props.pack.sessionName && (
          <span className="sync-pill" data-testid="pack-session" title={`pack ${props.pack.packHash}`}>
            ⛭ {props.pack.sessionName} · {props.pack.packHash}
          </span>
        )}
        <span
          className={`sync-pill ${sync.shared ? sync.status : "offline"}${sync.pending ? " pending" : ""}`}
          data-testid="sync-status"
          title={sync.shared ? `room ${sync.room} · ${sync.user}` : "Đặt VITE_SUPABASE_URL để bật sync"}
        >
          {syncLabel}
        </span>
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
          onClick={() => setGenOpen((open) => !open)}
          disabled={questions.length === 0}
          data-testid="autogen-toggle"
          aria-expanded={genOpen}
          title="Sinh sẵn danh sách đáp án có thứ tự cho từng câu hỏi"
        >
          {autoGen.running ? "Đang sinh đáp án…" : "✨ Tự sinh đáp án"}
        </button>
        <button
          className="btn sm ghost"
          onClick={() => setUndone(props.onUndo())}
          disabled={!props.undoLabel}
          data-testid="submission-undo"
          title={
            props.undoLabel
              ? `Hoàn tác: ${props.undoLabel} (Ctrl+Z) — cả đội sẽ thấy thay đổi này`
              : "Không có thao tác nào để hoàn tác"
          }
        >
          ↩ Hoàn tác
        </button>
        <button
          className="btn sm ghost"
          onClick={props.onSearchAll}
          disabled={questions.length === 0 || props.searchingAll}
          data-testid="submission-search-all"
          title={`Mở ${questions.length} tab và search tất cả — thay toàn bộ tab đang mở`}
        >
          {props.searchingAll ? "Đang search…" : `⚡ Search tất cả (${questions.length})`}
        </button>
        <button
          className="btn sm"
          onClick={() => fileRef.current?.click()}
          disabled={props.importing}
          data-testid="submission-import"
        >
          {props.importing ? "Đang import…" : "⭳ Import câu hỏi"}
        </button>
        <input
          ref={answerFileRef}
          type="file"
          accept=".zip,application/zip"
          style={{ display: "none" }}
          data-testid="answer-import-input"
          onChange={(event) => {
            const file = event.target.files?.[0];
            event.target.value = "";
            if (file) props.answerImport.onPick(file);
          }}
        />
        <button
          className="btn sm"
          onClick={() => answerFileRef.current?.click()}
          disabled={props.answerImport.importing}
          data-testid="answer-import"
          title="Nạp lại submission.zip đã export trước đó"
        >
          {props.answerImport.importing ? "Đang đọc…" : "⭳ Import đáp án"}
        </button>
        <button
          className="btn primary"
          onClick={props.onExport}
          disabled={totalRows === 0 || audit.errors.length > 0}
          data-testid="export-submission"
          title={
            audit.errors.length
              ? `${audit.errors.length} lỗi định dạng phải sửa trước khi export`
              : "Tạo submission.zip"
          }
        >
          ⭱ Export submission.zip
        </button>
      </div>

      <div className="hint-text" style={{ marginTop: 0 }}>
        ↑↓ chọn dòng · ←→ chọn sự kiện (TRAKE) · kéo <b>⠿</b> hoặc <b>Alt+↑↓</b> đổi thứ hạng ·{" "}
        <b>P</b> xem ảnh keyframe · <b>V</b> mở video để
        đổi frame · <b>Delete</b> xoá dòng · <b>Ctrl+Z</b> hoàn tác · <b>Ctrl+E</b> export
      </div>
      {unsent.length > 0 && (
        <div className="submission-unsent" data-testid="unsent-banner">
          <span>
            ✎ {unsent.length} dòng đã gõ nhưng <b>chưa đồng bộ</b> — export bây giờ sẽ lấy giá trị
            cũ.
          </span>
          <div className="spacer" />
          <button className="btn sm" onClick={commitAll} data-testid="commit-all">
            ✓ Đồng bộ {unsent.length} dòng
          </button>
        </div>
      )}

      {undone && (
        <div className="hint-text" data-testid="undo-toast" style={{ marginTop: 0 }}>
          ↩ đã hoàn tác: {undone}
        </div>
      )}

      {genOpen && (
        <AutoGenPanel
          questions={questions}
          byQuestion={byQuestion}
          autoGen={autoGen}
          limit={genLimit}
          onLimit={setGenLimit}
          onlyEmpty={genOnlyEmpty}
          onOnlyEmpty={setGenOnlyEmpty}
          onClose={() => setGenOpen(false)}
        />
      )}

      {props.answerImport.pending && (
        <AnswerImportPreview
          parsed={props.answerImport.pending}
          rows={rows}
          onApply={props.answerImport.onApply}
          onCancel={props.answerImport.onCancel}
        />
      )}
      {props.answerImport.error && (
        <div className="dup-warn" data-testid="answer-import-error">
          ⚠ {props.answerImport.error}
        </div>
      )}

      {props.pendingPack && (() => {
        const summary = packSummary(props.pendingPack);
        return (
          <div className="pack-preview" data-testid="pack-preview">
            <div>
              <b>{summary.total} câu hỏi</b> · KIS {summary.byKind.kis} · QA {summary.byKind.qa} ·
              TRAKE {summary.byKind.trake}
            </div>
            <div className="hint-text" style={{ margin: 0 }}>
              {props.pack.shared
                ? "Publish sẽ thay gói câu hỏi của cả team và mở lại tab theo gói mới."
                : "Chưa cấu hình Supabase — gói này sẽ chỉ áp dụng trên máy bạn."}
            </div>
            <div className="row" style={{ gap: 8, marginTop: 6 }}>
              <button className="btn primary" data-testid="pack-publish" onClick={props.onApplyPack}>
                ⇪ Publish cho cả team
              </button>
              <button className="btn ghost" data-testid="pack-cancel" onClick={props.onCancelPack}>
                Huỷ
              </button>
            </div>
          </div>
        );
      })()}

      {props.pack.origin === "cache" && (
        <div className="dup-warn" data-testid="pack-cached">
          ⚠ Đang dùng gói câu hỏi từ cache — chưa xác nhận được với server. Nếu người khác vừa
          publish gói mới thì máy này có thể đang hiểu `question_id` khác cả team.
        </div>
      )}
      {props.pack.origin === "local" && props.pack.shared && (
        <div className="dup-warn" data-testid="pack-local-only">
          ⚠ Gói câu hỏi này chỉ nằm trên máy bạn (chưa publish) — đáp án vẫn ghi chung với team.
        </div>
      )}
      {audit.errors.length > 0 && (
        <div className="dup-warn" data-testid="export-blocked">
          ⛔ {audit.errors.length} lỗi định dạng — export bị chặn. Nộp sai định dạng vẫn tính là
          một lần nộp (mỗi gói chỉ có 3 lần), nên phải sửa hết trước khi tạo file:
          <ul className="submission-problems">
            {audit.errors.slice(0, 12).map((problem, index) => (
              <li key={`${problem.questionId}-${index}`}>
                <b>{problem.questionId}</b> {problem.message}
              </li>
            ))}
            {audit.errors.length > 12 && <li>… và {audit.errors.length - 12} lỗi nữa</li>}
          </ul>
        </div>
      )}
      {audit.unanswered.length > 0 && (
        <div className="hint-text" data-testid="unanswered-note">
          {audit.unanswered.length}/{questions.length} câu chưa có đáp án — vẫn được export dưới
          dạng file CSV rỗng theo đúng cấu trúc BTC yêu cầu.
        </div>
      )}
      {props.pack.error && <div className="dup-warn">⚠ Pack: {props.pack.error}</div>}
      {sync.error && <div className="dup-warn" data-testid="sync-error">⚠ Sync: {sync.error}</div>}
      {sync.conflicts.map((rowId) => (
        <div className="dup-warn" key={rowId} data-testid="sync-conflict">
          ⚠ Dòng này vừa được người khác sửa nên thay đổi của bạn chưa được ghi. Bảng đang hiển thị
          bản mới nhất từ server.{" "}
          <button className="btn sm ghost" onClick={() => sync.onDismissConflict(rowId)}>OK</button>
        </div>
      ))}
      {props.importError && <div className="dup-warn">⚠ {props.importError}</div>}
      {props.exportError && <div className="dup-warn">⚠ {props.exportError}</div>}
      {orphans.length > 0 && (
        <div className="dup-warn" data-testid="submission-orphans">
          ⚠ {orphans.length} dòng thuộc câu hỏi không còn trong gói đã import — chúng sẽ KHÔNG được export.
        </div>
      )}

      {previewOpen && selectedRow && (
        <FramePreview
          videoId={selectedRow.videoId}
          frameIdx={selectedRow.frames[slot] ?? 0}
          keyframeId={selectedRow.keyframeIds?.[slot] ?? null}
          retrievalDatabase={selectedRow.retrievalDatabase ?? props.defaultRetrievalDatabase}
          onClose={() => setPreviewOpen(false)}
          onOpenVideo={() => setEditorOpen(true)}
        />
      )}

      {selectedRow && (
        <SubmissionFrameEditor
          open={editorOpen}
          videoId={selectedRow.videoId}
          frameIdx={selectedRow.frames[slot] ?? 0}
          keyframeId={selectedRow.keyframeIds?.[slot] ?? null}
          slot={slot}
          slotCount={selectedRow.frames.length}
          retrievalDatabase={selectedRow.retrievalDatabase ?? props.defaultRetrievalDatabase}
          onCommit={commitFrameEdit}
          onCancel={() => setEditorOpen(false)}
        />
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
        const isExpanded = expanded.has(question.id);
        const shownRows = isExpanded ? questionRows : questionRows.slice(0, COLLAPSED_ROWS);
        const hiddenCount = questionRows.length - shownRows.length;
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
              {qaAnswerMissingEverywhere(questionRows, question) && (
                <span className="dres-warn" data-testid={`qa-needs-answer-${question.id}`}>
                  ⚠ chưa có answer
                </span>
              )}
              {problems.some((problem) => problem.severity === "error") && (
                <span className="dres-warn" data-testid={`errors-${question.id}`}>
                  ⛔ {problems.filter((problem) => problem.severity === "error").length} lỗi
                </span>
              )}
              {problems.some((problem) => problem.severity === "warning") && (
                <span className="dres-dim" data-testid={`problems-${question.id}`}>
                  {problems.filter((problem) => problem.severity === "warning").length} cảnh báo
                </span>
              )}
              <div className="spacer" />
              <button
                className="btn sm ghost"
                disabled={autoGen.running}
                onClick={() =>
                  autoGen.onStart({ limit: genLimit, onlyEmpty: false, questionIds: [question.id] })
                }
                title="Xoá đáp án hiện có của câu này rồi sinh lại"
                data-testid={`autogen-one-${question.id}`}
              >
                ✨ sinh lại
              </button>
              <button
                className="btn sm ghost"
                onClick={() => {
                  // A new row lands at the end, which is out of sight while the
                  // question is collapsed — the button would look like a no-op.
                  setExpanded((current) => new Set(current).add(question.id));
                  props.onAddRow(question.id);
                }}
              >
                + dòng
              </button>
              <button className="btn sm ghost" onClick={() => props.onClearQuestion(question.id)}>
                xoá hết
              </button>
            </div>

            {question.kind === "qa" && questionRows.length > 0 && (
              <QaAnswerFill
                rows={questionRows}
                onFill={(answer) => props.onFillAnswer(question.id, answer)}
              />
            )}

            <table className="submission-table">
              <thead>
                <tr>
                  <th style={{ width: 44 }} title="Thứ hạng — kéo ⠿ hoặc Alt+↑/↓ để đổi">#</th>
                  <th style={{ width: 70 }}>ai</th>
                  <th style={{ width: 120 }}>video</th>
                  <th>{question.kind === "trake" ? `frames (${question.eventCount ?? "N"})` : "frame_idx"}</th>
                  <th style={{ width: 150 }}>keyframe</th>
                  {question.kind === "qa" && <th style={{ width: "26%" }}>answer</th>}
                  <th style={{ width: 34 }} />
                </tr>
              </thead>
              <tbody>
                {shownRows.map((row, index) => {
                  const rowProblems = problemsByRow.get(row.id) ?? [];
                  const duplicate = findDuplicate(questionRows, row, question.kind, row.id);
                  const draft = drafts[row.id];
                  const dirty = draftDiffers(row, draft);
                  const isSelected = selected?.rowId === row.id;
                  const activeSlot = isSelected ? slot : 0;
                  const isDragging = dragging === row.id;
                  const target = dropTarget?.rowId === row.id && !isDragging ? dropTarget : null;
                  return (
                    <tr
                      key={row.id}
                      className={[
                        rowProblems.length || duplicate ? "bad-row" : "",
                        isSelected ? "selected-row" : "",
                        dirty ? "draft-row" : "",
                        isDragging ? "dragging-row" : "",
                        target ? (target.below ? "drop-below" : "drop-above") : "",
                      ].join(" ").trim()}
                      onClick={() => setSelected({ rowId: row.id, slot: 0 })}
                      // Armed by the grip only, so text inside the cells stays
                      // selectable — a permanently draggable `<tr>` swallows
                      // click-and-drag inside its own inputs.
                      draggable={gripped === row.id}
                      onDragStart={(event) => {
                        setRowDrag(event, { questionId: question.id, rowId: row.id });
                        setDragging(row.id);
                      }}
                      onDragEnd={endDrag}
                      onDragOver={(event) => {
                        if (!hasRowDrag(event)) return;
                        // Without this the browser refuses the drop outright.
                        event.preventDefault();
                        event.dataTransfer.dropEffect = "move";
                        if (isDragging) return;
                        // Which side the line is drawn on follows the direction
                        // of travel: a row moving down lands after its target,
                        // one moving up lands before it. That is exactly what
                        // `planReorder` does, so the hint cannot lie.
                        const source = dragging
                          ? questionRows.findIndex((item) => item.id === dragging)
                          : -1;
                        const below = source >= 0 && source < index;
                        if (dropTarget?.rowId !== row.id || dropTarget.below !== below) {
                          setDropTarget({ rowId: row.id, below });
                        }
                      }}
                      onDrop={(event) => {
                        event.preventDefault();
                        const payload = readRowDrag(event);
                        endDrag();
                        // A drop from another question is not a reorder: it would
                        // silently re-attribute an answer. Refuse it.
                        if (!payload || payload.questionId !== question.id) return;
                        reorder(question.id, payload.rowId, row.id);
                      }}
                      data-testid={`row-${question.id}-${index}`}
                    >
                      <td className="mono dim">
                        <span
                          className={`sync-dot ${row.syncState ?? "local"}`}
                          title={row.syncState ?? "local"}
                        />
                        <span
                          className="row-grip"
                          role="button"
                          tabIndex={-1}
                          title="Kéo để đổi thứ hạng · Alt+↑/↓"
                          aria-label={`Kéo dòng ${index + 1} để đổi thứ hạng`}
                          data-testid={`grip-${question.id}-${index}`}
                          // Pointer down arms the row. Releasing disarms it —
                          // here, or through the window listener when the button
                          // comes up somewhere else — so a click that never
                          // became a drag leaves no draggable row behind.
                          onMouseDown={() => setGripped(row.id)}
                          onMouseUp={() => setGripped(null)}
                        >
                          ⠿
                        </span>
                        {index + 1}
                      </td>
                      <td className="mono dim" data-testid={`row-user-${question.id}-${index}`}>
                        {row.submittedBy ?? "—"}
                      </td>
                      <td>
                        <input
                          className="cell-input mono"
                          value={draft?.videoId ?? row.videoId}
                          onChange={(event) => editDraft(row.id, { videoId: event.target.value })}
                          onKeyDown={(event) => onCellKey(event, row)}
                          placeholder="L01_V028"
                        />
                      </td>
                      <td>
                        <input
                          className="cell-input mono"
                          value={draft?.frames ?? framesToText(row.frames)}
                          onChange={(event) => editDraft(row.id, { frames: event.target.value })}
                          onKeyDown={(event) => onCellKey(event, row)}
                          placeholder={question.kind === "trake" ? "1200, 1850, 2100" : "25300"}
                        />
                      </td>
                      <td className="mono dim keyframe-cell">
                        {(row.keyframeIds?.[activeSlot] ?? null) ?? (
                          <span title="frame lấy trực tiếp từ video, không phải keyframe đã trích">raw</span>
                        )}
                      </td>
                      {question.kind === "qa" && (
                        <td>
                          <input
                            className="cell-input"
                            value={draft?.answer ?? row.answer}
                            maxLength={MAX_ANSWER_LENGTH * 2}
                            onChange={(event) => editDraft(row.id, { answer: event.target.value })}
                            onKeyDown={(event) => onCellKey(event, row)}
                            placeholder="đáp án (≤100 ký tự)"
                          />
                        </td>
                      )}
                      <td>
                        {dirty && (
                          <button
                            className="cell-commit"
                            onClick={(event) => {
                              event.stopPropagation();
                              commitDraft(row);
                            }}
                            title="Đồng bộ dòng này cho cả đội (Enter). Esc để huỷ."
                            aria-label={`Đồng bộ dòng ${index + 1}`}
                            data-testid={`commit-${question.id}-${index}`}
                          >
                            ✓
                          </button>
                        )}
                        <button
                          className="rail-tab-close"
                          onClick={(event) => {
                            event.stopPropagation();
                            props.onDeleteRow(row.id);
                          }}
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

            {questionRows.length > COLLAPSED_ROWS && (
              <button
                className="submission-more"
                onClick={() =>
                  setExpanded((current) => {
                    const next = new Set(current);
                    if (!next.delete(question.id)) next.add(question.id);
                    return next;
                  })
                }
                data-testid={`expand-${question.id}`}
              >
                {isExpanded
                  ? `▴ thu gọn còn ${COLLAPSED_ROWS} dòng`
                  : `▾ hiện tất cả ${questionRows.length} dòng — đang ẩn ${hiddenCount}`}
              </button>
            )}

            {problems.length > 0 && (
              <ul className="submission-problems">
                {problems.map((problem, index) => (
                  <li key={`${problem.rowId}-${index}`} className={problem.severity}>
                    {problem.severity === "error" ? "⛔ " : "⚠ "}
                    {problem.message}
                  </li>
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
              <span key={question.id} className="empty-question">
                <button
                  className="btn sm ghost"
                  onClick={() => props.onAddRow(question.id)}
                  title={question.text.slice(0, 200)}
                >
                  {question.id} +
                </button>
                <button
                  className="btn sm ghost"
                  disabled={autoGen.running}
                  onClick={() =>
                    autoGen.onStart({ limit: genLimit, onlyEmpty: true, questionIds: [question.id] })
                  }
                  title="Tự sinh đáp án cho riêng câu này"
                >
                  ✨
                </button>
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
