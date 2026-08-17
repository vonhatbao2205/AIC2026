import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import FullConsole, { type SubmissionDraft } from "./FullConsole";
import { SubmissionPanel } from "./components/SubmissionPanel";
import { SUBMISSION_VIEW, TabRail, type ActiveView, type ConsoleTab } from "./components/TabRail";
import type { QueryType, RetrievalDatabase } from "./api/types";
import { parseQuestionPack, type ImportedQuestion } from "./lib/questions";
import { buildSubmissionZip, MAX_ROWS_PER_QUESTION, findDuplicate, type SubmissionRow } from "./lib/submission";
import { readZipTextFiles, ZipError } from "./lib/zip";

interface Props {
  onSimpleMode: () => void;
  onShowSettings: () => void;
  configVersion?: number;
  retrievalDatabase: RetrievalDatabase;
  onRetrievalDatabase: (database: RetrievalDatabase) => void;
}

const ROWS_KEY = "aic26_submission_rows";
const QUESTIONS_KEY = "aic26_question_pack";

let tabSeq = 0;
function newTab(queryType: QueryType = "T-KIS", questionId: string | null = null): ConsoleTab {
  tabSeq += 1;
  return { id: `tab-${tabSeq}`, queryType, questionId, autoRunToken: 0 };
}

let rowSeq = 0;
function newRowId(): string {
  rowSeq += 1;
  return `row-${Date.now().toString(36)}-${rowSeq}`;
}

function loadJson<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

/** Owns everything shared across search tabs: the imported query pack, the tab
 *  list, and the submission rows the export is built from.
 *
 *  Every tab stays mounted (hidden with `display:none` when parked) because an
 *  import has to run all of their searches at once. Result thumbnails are
 *  `loading="lazy"`, so a hidden tab costs DOM but no network. */
export default function Workspace(props: Props) {
  const [questions, setQuestions] = useState<ImportedQuestion[]>(() =>
    loadJson<ImportedQuestion[]>(QUESTIONS_KEY, []),
  );
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
  const [rows, setRows] = useState<SubmissionRow[]>(() => loadJson<SubmissionRow[]>(ROWS_KEY, []));
  const [importing, setImporting] = useState(false);
  const [importError, setImportError] = useState<string | null>(null);
  const [exportError, setExportError] = useState<string | null>(null);
  const [busyTabs, setBusyTabs] = useState<Set<string>>(new Set());

  // Answers are the one thing in this app that cannot be recomputed, so they
  // survive a reload. The pack rides along so the rows keep their CSV names.
  useEffect(() => {
    try {
      localStorage.setItem(ROWS_KEY, JSON.stringify(rows));
    } catch {
      /* quota — the table is still authoritative in memory */
    }
  }, [rows]);
  useEffect(() => {
    try {
      localStorage.setItem(QUESTIONS_KEY, JSON.stringify(questions));
    } catch {
      /* ignore */
    }
  }, [questions]);

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
  const importPack = useCallback(async (file: File) => {
    setImporting(true);
    setImportError(null);
    try {
      const entries = await readZipTextFiles(file);
      const pack = parseQuestionPack(entries);
      if (!pack.length) {
        throw new ZipError(
          "Không tìm thấy file câu hỏi nào. Tên file phải dạng <tên>-<số>-<kis|qa|trake>.txt",
        );
      }
      setQuestions(pack);
      // One tab per question, each already on the right query type, each told to
      // run its own search — that is the whole point of importing a pack.
      const opened = pack.map((question) => {
        const tab = newTab(question.queryType, question.id);
        tab.autoRunToken = 1;
        return tab;
      });
      setTabs(opened);
      setActive(opened[0].id);
      setBusyTabs(new Set());
    } catch (error) {
      setImportError(error instanceof Error ? error.message : "Import thất bại");
    } finally {
      setImporting(false);
    }
  }, []);

  // ---- submission rows ----
  const addRow = useCallback(
    (draft: SubmissionDraft) => {
      const question = questionById.get(draft.questionId);
      setRows((current) => {
        if (current.filter((row) => row.questionId === draft.questionId).length >= MAX_ROWS_PER_QUESTION) {
          return current;
        }
        // Duplicates are allowed through deliberately — the guard already warned,
        // and the operator confirming past it is a decision, not a slip.
        void (question && findDuplicate(current, draft, question.kind));
        return [...current, { id: newRowId(), source: "submit", ...draft }];
      });
    },
    [questionById],
  );

  const addBlankRow = useCallback((questionId: string) => {
    setRows((current) => [
      ...current,
      { id: newRowId(), questionId, videoId: "", frames: [], answer: "", source: "manual" },
    ]);
  }, []);

  const changeRow = useCallback((rowId: string, patch: Partial<SubmissionRow>) => {
    setRows((current) => current.map((row) => (row.id === rowId ? { ...row, ...patch } : row)));
  }, []);

  const deleteRow = useCallback((rowId: string) => {
    setRows((current) => current.filter((row) => row.id !== rowId));
  }, []);

  const clearQuestion = useCallback((questionId: string) => {
    setRows((current) => current.filter((row) => row.questionId !== questionId));
  }, []);

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
              onClearQuestion={clearQuestion}
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
            />
          </div>
        )}
      </div>
    </div>
  );
}
