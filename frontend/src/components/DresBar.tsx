import type { DresEvaluation, DresStatus } from "../api/types";

interface Props {
  status: DresStatus | null;
  evaluations: DresEvaluation[];
  /** The run the next submit goes to (auto-picked from the query type, or chosen). */
  evaluationId: string | null;
  /** Set when the operator picked the run by hand instead of letting it auto-route. */
  pinned: boolean;
  /** Local countdown seconds, ticked between polls; null for open-ended tasks. */
  timeLeft: number | null;
  error: string | null;
  busy: boolean;
  onSelect: (id: string | null) => void;
  onRefresh: () => void;
  onReconnect: () => void;
}

function clock(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  const m = Math.floor(s / 60);
  return `${m}:${String(s % 60).padStart(2, "0")}`;
}

export function DresBar(props: Props) {
  const { status, evaluations, evaluationId, pinned, timeLeft, error, busy } = props;
  const active = evaluations.find((e) => e.id === evaluationId) ?? null;
  const task = active?.current_task ?? null;
  const state = active?.state ?? null;
  const running = state?.taskStatus === "RUNNING";

  const dot = !status?.configured ? "bad" : error || !status.logged_in ? "warn" : running ? "ok" : "warn";

  return (
    <div className="dres-bar" data-testid="dres-bar">
      <span className={`health-dot ${dot}`} />
      <span className="dres-label">DRES</span>

      {!status?.configured ? (
        <span className="dres-warn" data-testid="dres-offline">
          not configured — set DRES_USERNAME / DRES_PASSWORD in backend/.env
        </span>
      ) : (
        <>
          <span className="dres-user" title={status.base_url}>
            {status.username ?? "?"}
          </span>

          <select
            className="dres-select"
            data-testid="dres-evaluation"
            value={pinned ? evaluationId ?? "" : ""}
            onChange={(e) => props.onSelect(e.target.value || null)}
            title="Evaluation run — leave empty to select by query type"
          >
            <option value="">auto{active ? ` · ${active.name}` : ""}</option>
            {evaluations.map((e) => (
              <option key={e.id} value={e.id}>
                {e.name} {e.current_task ? `· ${e.current_task.name}` : "· no task"}
              </option>
            ))}
          </select>

          {task ? (
            <>
              <span className="dres-task" data-testid="dres-task">
                {task.name}
              </span>
              <span className="dres-dim">{task.taskType}</span>
              <span className={`dres-status ${running ? "ok" : "warn"}`} data-testid="dres-task-status">
                {state?.taskStatus ?? "?"}
              </span>
              <span className="mono dres-clock" data-testid="dres-clock">
                {timeLeft != null
                  ? `⏳ ${clock(timeLeft)} left`
                  : state
                  ? `⏱ ${clock(state.timeElapsed)} elapsed`
                  : ""}
              </span>
            </>
          ) : (
            <span className="dres-warn" data-testid="dres-no-task">
              {error ?? "no task is currently open"}
            </span>
          )}
          {task && error && <span className="dres-warn">{error}</span>}
        </>
      )}

      <div className="spacer" />
      <button className="btn sm ghost" onClick={props.onRefresh} disabled={busy} data-testid="dres-refresh">
        ⟳ refresh
      </button>
      {status?.configured && !status.logged_in && (
        <button className="btn sm" onClick={props.onReconnect} disabled={busy} data-testid="dres-reconnect">
          Reconnect
        </button>
      )}
    </div>
  );
}
