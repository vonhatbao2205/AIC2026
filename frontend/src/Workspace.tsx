import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import FullConsole, { type SubmissionDraft } from "./FullConsole";
import { SubmissionPanel, type AutoGenOptions } from "./components/SubmissionPanel";
import { SUBMISSION_VIEW, TabRail, type ActiveView, type ConsoleTab } from "./components/TabRail";
import type { QueryType, RetrievalDatabase } from "./api/types";
import { parseQuestionPack, type ImportedQuestion } from "./lib/questions";
import {
  buildSubmissionZip,
  MAX_ROWS_PER_QUESTION,
  SubmissionFormatError,
  type SubmissionRow,
} from "./lib/submission";
import { useSharedSubmission } from "./hooks/useSharedSubmission";
import { useSharedSession } from "./hooks/useSharedSession";
import { defaultSessionName, packSummary } from "./lib/questionPack";
import { generateAll, type AnswerGenProgress } from "./lib/answerGen";
import {
  parseSubmissionPack,
  planImport,
  type ImportMode,
  type ImportedSubmission,
} from "./lib/submissionImport";
import { newRowId } from "./lib/sharedSubmission";
import { readZipTextFiles, ZipError } from "./lib/zip";

interface Props {
  onSimpleMode: () => void;
  onShowSettings: () => void;
  configVersion?: number;
  retrievalDatabase: RetrievalDatabase;
  onRetrievalDatabase: (database: RetrievalDatabase) => void;
}

let tabSeq = 0;
function newTab(queryType: QueryType = "T-KIS", questionId: string | null = null): ConsoleTab {
  tabSeq += 1;
  return { id: `tab-${tabSeq}`, queryType, questionId, autoRunToken: 0 };
}

/** Owns everything shared across search tabs: the imported query pack, the tab
 *  list, and the submission rows the export is built from.
 *
 *  Every tab stays mounted (hidden with `display:none` when parked) because an
 *  import has to run all of their searches at once. Result thumbnails are
 *  `loading="lazy"`, so a hidden tab costs DOM but no network. */
export default function Workspace(props: Props) {
  // The pack is shared state now: Supabase decides which questions the team is
  // answering, localStorage only caches them for an offline stretch.
  const pack = useSharedSession();
  const questions = pack.questions;
  const [tabs, setTabs] = useState<ConsoleTab[]>(() => [newTab()]);
  // Derived, never hardcoded: tab ids come from a counter, so a literal "tab-1"
  // would leave the workspace pointing at nothing on any later mount.
  const [active, setActive] = useState<ActiveView>(() => tabs[0].id);
  // Where "back" goes from the Submission view: the console tab last looked at,
  // falling back to the first one if that tab has since been closed.
  const lastConsoleTab = useRef<string>(tabs[0].id);
  useEffect(() => {
    if (active !== SUBMISSION_VIEW) lastConsoleTab.current = active;
  }, [active]);
  // Rows are owned by the shared store now: five operators search on their own
  // machines but write into one Submission table. With no Supabase configured it
  // degrades to the previous localStorage-only behaviour.
  const shared = useSharedSubmission(pack.session?.id ?? null);
  const rows = shared.rows;
  const [importing, setImporting] = useState(false);
  /** Parsed but not yet applied — waiting for the publish decision. */
  const [pendingPack, setPendingPack] = useState<ImportedQuestion[] | null>(null);
  const [importError, setImportError] = useState<string | null>(null);
  const [exportError, setExportError] = useState<string | null>(null);
  const [busyTabs, setBusyTabs] = useState<Set<string>>(new Set());
  const [genRunning, setGenRunning] = useState(false);
  const [genProgress, setGenProgress] = useState<AnswerGenProgress | null>(null);
  const [genLog, setGenLog] = useState<AnswerGenProgress[]>([]);
  const [genError, setGenError] = useState<string | null>(null);
  const genAbort = useRef<AbortController | null>(null);
  /** A parsed `submission.zip` waiting for the operator to choose how to apply it. */
  const [pendingAnswers, setPendingAnswers] = useState<ImportedSubmission | null>(null);
  const [answerImporting, setAnswerImporting] = useState(false);
  const [answerImportError, setAnswerImportError] = useState<string | null>(null);

  const questionById = useMemo(
    () => new Map(questions.map((question) => [question.id, question])),
    [questions],
  );
  const rowCounts = useMemo(() => {
    const counts = new Map<string, number>();
    for (const row of rows) counts.set(row.questionId, (counts.get(row.questionId) ?? 0) + 1);
    return counts;
  }, [rows]);

  // ---- tabs ----
  const patchTab = useCallback((id: string, patch: Partial<ConsoleTab>) => {
    setTabs((current) => current.map((tab) => (tab.id === id ? { ...tab, ...patch } : tab)));
  }, []);

  const addTab = useCallback(() => {
    const tab = newTab();
    setTabs((current) => [...current, tab]);
    setActive(tab.id);
  }, []);

  const closeTab = useCallback((id: string) => {
    setTabs((current) => {
      if (current.length <= 1) return current;
      const next = current.filter((tab) => tab.id !== id);
      setActive((view) => (view === id ? next[next.length - 1].id : view));
      return next;
    });
    setBusyTabs((current) => {
      if (!current.has(id)) return current;
      const next = new Set(current);
      next.delete(id);
      return next;
    });
  }, []);

  const setTabBusy = useCallback((id: string, busy: boolean) => {
    setBusyTabs((current) => {
      if (current.has(id) === busy) return current;
      const next = new Set(current);
      if (busy) next.add(id);
      else next.delete(id);
      return next;
    });
  }, []);

  /** Assigning a question also switches the tab to that question's query type. */
  const selectQuestion = useCallback(
    (tabId: string, questionId: string | null) => {
      const question = questionId ? questionById.get(questionId) ?? null : null;
      patchTab(tabId, {
        questionId,
        ...(question ? { queryType: question.queryType } : {}),
      });
    },
    [patchTab, questionById],
  );

  // ---- import ----
  /** Open one tab per question, each on the right type and already searching. */
  const openTabsFor = useCallback((list: ImportedQuestion[]) => {
    const opened = list.map((question) => {
      const tab = newTab(question.queryType, question.id);
      tab.autoRunToken = 1;
      return tab;
    });
    if (!opened.length) return;
    setTabs(opened);
    setActive(opened[0].id);
    setBusyTabs(new Set());
  }, []);

  /** Parsing no longer applies the pack. Publishing it replaces what five
   *  machines are answering, so it is a decision the operator states. */
  const importPack = useCallback(async (file: File) => {
    setImporting(true);
    setImportError(null);
    try {
      const entries = await readZipTextFiles(file);
      const parsed = parseQuestionPack(entries);
      if (!parsed.length) {
        throw new ZipError(
          "Không tìm thấy file câu hỏi nào. Tên file phải dạng <tên>-<số>-<kis|qa|trake>.txt",
        );
      }
      setPendingPack(parsed);
      // The publish decision lives in the Submission view; go where it is shown.
      setActive(SUBMISSION_VIEW);
    } catch (error) {
      setImportError(error instanceof Error ? error.message : "Import thất bại");
    } finally {
      setImporting(false);
    }
  }, []);

  /** Bring this machine to the same state the importing machine reached.
   *
   *  Importing a pack opens a tab per question and searches them all, but a
   *  machine that received the pack over realtime never ran that step — it just
   *  had its question list replaced. This is that step, on demand. */
  const searchAllQuestions = useCallback(() => {
    if (!questions.length) return;
    openTabsFor(questions);
  }, [questions, openTabsFor]);

  const applyPack = useCallback(async () => {
    if (!pendingPack) return;
    setImportError(null);
    try {
      await pack.publish(pendingPack, defaultSessionName(pendingPack));
      openTabsFor(pendingPack);
      setPendingPack(null);
    } catch (error) {
      setImportError(error instanceof Error ? error.message : "Không publish được gói câu hỏi");
    }
  }, [pendingPack, pack, openTabsFor]);

  // ---- submission rows (delegated to the shared store) ----
  const addRow = useCallback(
    (draft: SubmissionDraft) => {
      // Duplicates are allowed through deliberately — the guard already warned,
      // and the operator confirming past it is a decision, not a slip.
      if (rows.filter((row) => row.questionId === draft.questionId).length >= MAX_ROWS_PER_QUESTION) {
        return;
      }
      shared.addRow({ id: newRowId(), source: "submit", ...draft });
    },
    [rows, shared],
  );

  const addBlankRow = useCallback(
    (questionId: string) => {
      shared.addRow({
        id: newRowId(),
        questionId,
        videoId: "",
        frames: [],
        answer: "",
        source: "manual",
        retrievalDatabase: props.retrievalDatabase,
      });
    },
    [shared, props.retrievalDatabase],
  );

  const changeRow = useCallback(
    (rowId: string, patch: Partial<SubmissionRow>) => shared.updateRow(rowId, patch),
    [shared],
  );

  const deleteRow = useCallback((rowId: string) => shared.deleteRow(rowId), [shared]);

  /** Same answer shape, new identity: a second guess for the same question. */
  const cloneRow = useCallback(
    (row: SubmissionRow) => {
      if (rows.filter((item) => item.questionId === row.questionId).length >= MAX_ROWS_PER_QUESTION) {
        return;
      }
      shared.addRow({ ...row, id: newRowId(), revision: 1, createdAt: undefined, syncState: undefined });
    },
    [rows, shared],
  );

  const clearQuestion = useCallback(
    (questionId: string) => shared.clearQuestion(questionId),
    [shared],
  );

  const fillAnswer = useCallback(
    (questionId: string, answer: string) => shared.setAnswerForQuestion(questionId, answer),
    [shared],
  );

  // ---- bulk answer generation ----
  /** Fill the submission table from the answer generator.
   *
   *  The generator replaces exactly its OWN rows and nothing else. A frame a
   *  person verified by eye keeps the head of the list, because `R@k` is a max
   *  over the first k answers: that frame at rank 1 scores the whole query, and
   *  the same frame at rank 100 scores 0.2 of it. It also costs a position, so
   *  the generated block shrinks to keep the question at 100 rows. */
  const startAutoGen = useCallback(
    async (options: AutoGenOptions) => {
      const targets = options.questionIds.length
        ? questions.filter((question) => options.questionIds.includes(question.id))
        : questions;
      if (!targets.length || genRunning) return;
      const controller = new AbortController();
      genAbort.current = controller;
      setGenRunning(true);
      setGenError(null);
      setGenLog([]);
      setGenProgress(null);
      const rowsFor = (questionId: string) => rows.filter((row) => row.questionId === questionId);
      const isGenerated = (row: SubmissionRow) => row.source === "generated";
      try {
        const outcomes = await generateAll(
          {
            questions: targets,
            retrievalDatabase: props.retrievalDatabase,
            limit: options.limit,
            keptRows: (questionId) => rowsFor(questionId).filter((row) => !isGenerated(row)),
            // Only the generator's own work counts as "already done": a question
            // holding nothing but hand-picked rows still wants its insurance tail.
            skip: (questionId) =>
              options.onlyEmpty && rowsFor(questionId).some(isGenerated),
            // A Q&A row still needs its answer text; the generator only ranks
            // where to look. Carry over the text the operator already wrote for
            // this question rather than emitting a hundred blank answers.
            answerText: (questionId) =>
              rowsFor(questionId).find((row) => row.answer.trim())?.answer.trim() ?? "",
          },
          (progress) => {
            setGenProgress(progress);
            if (progress.status !== "running") {
              setGenLog((current) => [...current, progress]);
            }
          },
          controller.signal,
        );
        if (controller.signal.aborted) return;
        const fresh = outcomes.filter((outcome) => outcome.rows.length > 0);
        // Only the previous generated block goes; the hand-picked rows stay put,
        // and their older `createdAt` is what keeps them ahead of the new ones.
        shared.deleteRows(
          fresh.flatMap((outcome) =>
            rowsFor(outcome.questionId).filter(isGenerated).map((row) => row.id),
          ),
        );
        shared.addRows(
          fresh.flatMap((outcome) =>
            outcome.rows.slice(0, MAX_ROWS_PER_QUESTION).map((row) => ({ ...row, id: newRowId() })),
          ),
        );
        const failures = outcomes.filter((outcome) => outcome.error);
        if (failures.length) {
          setGenError(
            `${failures.length}/${targets.length} câu không sinh được đáp án: ` +
              failures.map((failure) => `${failure.questionId} (${failure.error})`).join("; "),
          );
        }
      } catch (error) {
        setGenError(error instanceof Error ? error.message : "Sinh đáp án thất bại");
      } finally {
        genAbort.current = null;
        setGenRunning(false);
        setGenProgress(null);
      }
    },
    [questions, rows, shared, genRunning, props.retrievalDatabase],
  );

  const cancelAutoGen = useCallback(() => {
    genAbort.current?.abort();
    genAbort.current = null;
    setGenRunning(false);
    setGenProgress(null);
  }, []);

  // ---- import a previously exported submission.zip ----
  const pickAnswerPack = useCallback(
    async (file: File) => {
      setAnswerImporting(true);
      setAnswerImportError(null);
      setPendingAnswers(null);
      try {
        setPendingAnswers(parseSubmissionPack(await readZipTextFiles(file), questions));
      } catch (error) {
        setAnswerImportError(error instanceof Error ? error.message : "Đọc file thất bại");
      } finally {
        setAnswerImporting(false);
      }
    },
    [questions],
  );

  /** Answers cannot be recomputed, so the write only happens on this call — after
   *  the operator has seen how many rows it replaces. */
  const applyAnswerPack = useCallback(
    (mode: ImportMode) => {
      if (!pendingAnswers) return;
      const plan = planImport(pendingAnswers, rows, mode);
      const targets = new Set(plan.write.map((file) => file.questionId));
      shared.deleteRows(rows.filter((row) => targets.has(row.questionId)).map((row) => row.id));
      shared.addRows(
        plan.write.flatMap((file) =>
          file.rows.slice(0, MAX_ROWS_PER_QUESTION).map((row) => ({
            ...row,
            id: newRowId(),
            retrievalDatabase: props.retrievalDatabase,
          })),
        ),
      );
      setPendingAnswers(null);
      setAnswerImportError(null);
    },
    [pendingAnswers, rows, shared, props.retrievalDatabase],
  );

  const exportZip = useCallback(() => {
    setExportError(null);
    try {
      const blob = buildSubmissionZip(questions, rows);
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = "submission.zip";
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      setTimeout(() => URL.revokeObjectURL(url), 10_000);
    } catch (error) {
      // The builder refuses rather than writing a file the organiser's parser
      // would reject; surface exactly which rows are in the way.
      if (error instanceof SubmissionFormatError) {
        setExportError(error.message);
        return;
      }
      setExportError(error instanceof Error ? error.message : "Export thất bại");
    }
  }, [questions, rows]);

  /** Jump to the tab answering a question (opening one if none is bound). */
  const jumpToQuestion = useCallback(
    (questionId: string) => {
      const existing = tabs.find((tab) => tab.questionId === questionId);
      if (existing) {
        setActive(existing.id);
        return;
      }
      const question = questionById.get(questionId);
      const tab = newTab(question?.queryType ?? "T-KIS", questionId);
      setTabs((current) => [...current, tab]);
      setActive(tab.id);
    },
    [tabs, questionById],
  );

  const busyRef = useRef(setTabBusy);
  busyRef.current = setTabBusy;

  return (
    <div className="workspace-shell">
      <TabRail
        tabs={tabs}
        active={active}
        questions={questions}
        rowCounts={rowCounts}
        submissionTotal={rows.length}
        busyTabs={busyTabs}
        onSelect={setActive}
        onAdd={addTab}
        onClose={closeTab}
      />
      <div className="workspace-body">
        {tabs.map((tab) => {
          const question = tab.questionId ? questionById.get(tab.questionId) ?? null : null;
          const isActive = active === tab.id;
          return (
            <div key={tab.id} className="tab-pane" data-testid={`tab-pane-${tab.id}`}>
              <FullConsole
                onSimpleMode={props.onSimpleMode}
                onShowSettings={props.onShowSettings}
                configVersion={props.configVersion}
                retrievalDatabase={props.retrievalDatabase}
                onRetrievalDatabase={props.onRetrievalDatabase}
                active={isActive}
                queryType={tab.queryType}
                onQueryType={(queryType) => patchTab(tab.id, { queryType })}
                questions={questions}
                question={question}
                onSelectQuestion={(questionId) => selectQuestion(tab.id, questionId)}
                onImportQuestions={importPack}
                importing={importing}
                importError={importError}
                onOpenSubmission={() => setActive(SUBMISSION_VIEW)}
                onSearchAll={searchAllQuestions}
                searchingAll={busyTabs.size > 0}
                questionRows={question ? rows.filter((row) => row.questionId === question.id) : []}
                onSubmitRow={addRow}
                autoRunToken={tab.autoRunToken}
                onBusyChange={(busy) => busyRef.current(tab.id, busy)}
              />
            </div>
          );
        })}
        {active === SUBMISSION_VIEW && (
          <div className="tab-pane" style={{ display: "flex" }}>
            <SubmissionPanel
              questions={questions}
              rows={rows}
              onChangeRow={changeRow}
              onDeleteRow={deleteRow}
              onAddRow={addBlankRow}
              onCloneRow={cloneRow}
              onClearQuestion={clearQuestion}
              onFillAnswer={fillAnswer}
              onExport={exportZip}
              onJumpToQuestion={jumpToQuestion}
              exportError={exportError}
              onBack={() => {
                const target = tabs.find((tab) => tab.id === lastConsoleTab.current) ?? tabs[0];
                if (target) setActive(target.id);
              }}
              backLabel={(() => {
                const target = tabs.find((tab) => tab.id === lastConsoleTab.current) ?? tabs[0];
                const question = target?.questionId ? questionById.get(target.questionId) : null;
                return question ? `${target.queryType} · #${question.order}` : target?.queryType ?? "Tìm kiếm";
              })()}
              onImport={importPack}
              importing={importing}
              importError={importError}
              defaultRetrievalDatabase={props.retrievalDatabase}
              pendingPack={pendingPack}
              onApplyPack={applyPack}
              onCancelPack={() => setPendingPack(null)}
              onSearchAll={searchAllQuestions}
              searchingAll={busyTabs.size > 0}
              pack={{
                shared: pack.shared,
                sessionName: pack.session?.name ?? null,
                packHash: pack.session?.packHash ?? null,
                publishedBy: pack.session?.publishedBy ?? null,
                origin: pack.origin,
                loading: pack.loading,
                error: pack.error,
              }}
              answerImport={{
                pending: pendingAnswers,
                importing: answerImporting,
                error: answerImportError,
                onPick: pickAnswerPack,
                onApply: applyAnswerPack,
                onCancel: () => {
                  setPendingAnswers(null);
                  setAnswerImportError(null);
                },
              }}
              autoGen={{
                running: genRunning,
                progress: genProgress,
                log: genLog,
                error: genError,
                onStart: startAutoGen,
                onCancel: cancelAutoGen,
              }}
              sync={{
                shared: shared.shared,
                status: shared.status,
                pending: shared.pending,
                error: shared.error,
                room: shared.room,
                user: shared.user,
                conflicts: shared.conflicts,
                onDismissConflict: shared.dismissConflict,
              }}
            />
          </div>
        )}
      </div>
    </div>
  );
}
