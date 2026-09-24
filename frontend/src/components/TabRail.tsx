import type { QueryType } from "../api/types";
import type { ImportedQuestion } from "../lib/questions";

export interface ConsoleTab {
  id: string;
  /** Drives the tab label, and follows the query-type switch inside the tab. */
  queryType: QueryType;
  questionId: string | null;
  /** Bumped to make the tab run its search without the operator pressing search. */
  autoRunToken: number;
}

export const SUBMISSION_VIEW = "submission" as const;
export type ActiveView = string | typeof SUBMISSION_VIEW;

interface Props {
  tabs: ConsoleTab[];
  active: ActiveView;
  questions: ImportedQuestion[];
  /** Rows already collected, per question id — shown as a count on the tab. */
  rowCounts: Map<string, number>;
  submissionTotal: number;
  busyTabs: Set<string>;
  onSelect: (view: ActiveView) => void;
  onAdd: () => void;
  onClose: (id: string) => void;
}

const TYPE_CLASS: Record<QueryType, string> = {
  AVS: "tkis",
  "T-KIS": "tkis",
  QA: "qa",
  "V-KIS": "vkis",
  TRAKE: "trake",
};

/** Two-letter stand-ins shown while the rail is collapsed. */
const TYPE_CODE: Record<QueryType, string> = {
  AVS: "AV",
  "T-KIS": "TK",
  QA: "QA",
  "V-KIS": "VK",
  TRAKE: "TR",
};

export function TabRail(props: Props) {
  const { tabs, active, questions, rowCounts, busyTabs } = props;
  const questionById = new Map(questions.map((question) => [question.id, question]));

  return (
    // The inner panel is what widens on hover, so it can overlay the console
    // instead of pushing it — re-laying out a three-column console on every
    // pointer entry would be both jarring and expensive.
    <div className="tab-rail" data-testid="tab-rail">
      <div className="tab-rail-inner">
      <div className="tab-rail-list">
        {tabs.map((tab, index) => {
          const question = tab.questionId ? questionById.get(tab.questionId) ?? null : null;
          const rows = question ? rowCounts.get(question.id) ?? 0 : 0;
          return (
            <div
              key={tab.id}
              className={`rail-tab ${TYPE_CLASS[tab.queryType]} ${active === tab.id ? "active" : ""}`}
              onClick={() => props.onSelect(tab.id)}
              role="tab"
              aria-selected={active === tab.id}
              // Collapsed, the only visible text is an aria-hidden two-letter
              // code, so the name has to be stated rather than read off the DOM.
              aria-label={`Tab ${index + 1} · ${tab.queryType}${question ? ` · question ${question.order}` : ""}`}
              data-testid={`rail-tab-${index}`}
              title={question ? `${question.id}\n${question.text.slice(0, 160)}` : "No question assigned"}
            >
              <span className="rail-tab-code" aria-hidden="true">{TYPE_CODE[tab.queryType]}</span>
              <span className="rail-tab-type">{tab.queryType}</span>
              <span className="rail-tab-sub">
                {question ? `#${question.order}` : `tab ${index + 1}`}
                {busyTabs.has(tab.id) && <span className="rail-spin" aria-label="searching" />}
              </span>
              {rows > 0 && <span className="rail-tab-rows">{rows}</span>}
              {tabs.length > 1 && (
                <button
                  className="rail-tab-close"
                  aria-label={`Close tab ${index + 1}`}
                  data-testid={`rail-close-${index}`}
                  onClick={(event) => {
                    event.stopPropagation();
                    props.onClose(tab.id);
                  }}
                >
                  ×
                </button>
              )}
            </div>
          );
        })}
        <button className="rail-add" onClick={props.onAdd} data-testid="rail-add" title="Add search tab">
          +
        </button>
      </div>

      <div
        className={`rail-tab submission ${active === SUBMISSION_VIEW ? "active" : ""}`}
        onClick={() => props.onSelect(SUBMISSION_VIEW)}
        role="tab"
        aria-selected={active === SUBMISSION_VIEW}
        aria-label={`Submission · ${props.submissionTotal} rows`}
        data-testid="rail-tab-submission"
        title="Export the answer table as submission.zip"
      >
        <span className="rail-tab-code" aria-hidden="true">CSV</span>
        <span className="rail-tab-type">Submission</span>
        <span className="rail-tab-sub">{props.submissionTotal} rows</span>
      </div>
      </div>
    </div>
  );
}
