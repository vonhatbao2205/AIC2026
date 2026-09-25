import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError } from "../api/client";
import type {
  AgentName,
  AgentRunSnapshot,
  ImageEmbeddingModel,
  QueryType,
  RetrievalDatabase,
  SearchScope,
} from "../api/types";

/** The agents take tens of seconds per step, so a poll this slow loses nothing. */
const POLL_MS = 1500;

export interface AgentLaunch {
  retrieval_database: RetrievalDatabase;
  image_models: ImageEmbeddingModel[];
  query: string;
  query_type: QueryType;
  scope?: SearchScope;
  previous_hints: string[];
  agents: AgentName[];
}

/** One tab's Codex/Claude run: start, poll, cancel.
 *
 *  Only ever one run per tab. Starting a new one hands the old id to the backend
 *  as `replaces`, which cancels it before the new processes spawn, so a quick
 *  re-search never leaves agents working on a query the operator has abandoned.
 *  Every response is checked against the current id, so a late answer about the
 *  old run can never overwrite the new one. */
export function useAgentRun() {
  const [run, setRun] = useState<AgentRunSnapshot | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  // Bumped to retry a poll that failed for a reason other than "run gone".
  const [retry, setRetry] = useState(0);
  const runId = useRef<string | null>(null);
  const generation = useRef(0);

  const start = useCallback(async (launch: AgentLaunch) => {
    const gen = ++generation.current;
    const replaces = runId.current;
    runId.current = null;
    setRun(null);
    setError(null);
    setStarting(true);
    try {
      const snapshot = await api.startAgentRun({ ...launch, replaces });
      if (gen !== generation.current) {
        // Superseded while starting: nobody will ever look at this run.
        void api.cancelAgentRun(snapshot.run_id).catch(() => {});
        return;
      }
      runId.current = snapshot.run_id;
      setRun(snapshot);
    } catch (e) {
      if (gen !== generation.current) return;
      setError(
        e instanceof ApiError && typeof e.detail === "string"
          ? e.detail
          : "Could not start the agents — is the backend running?",
      );
    } finally {
      if (gen === generation.current) setStarting(false);
    }
  }, []);

  /** Stop the agents; their candidates stay on screen. */
  const cancel = useCallback(async () => {
    generation.current++;
    setStarting(false);
    const id = runId.current;
    if (!id) return;
    try {
      const snapshot = await api.cancelAgentRun(id);
      if (runId.current === id) setRun(snapshot);
    } catch {
      /* already gone: nothing left to stop */
    }
  }, []);

  /** Stop the agents and forget the run (new question, other dataset). */
  const clear = useCallback(() => {
    void cancel();
    runId.current = null;
    setRun(null);
    setError(null);
  }, [cancel]);

  useEffect(() => {
    if (!run || run.finished) return;
    const id = run.run_id;
    const controller = new AbortController();
    const timer = setTimeout(async () => {
      try {
        const snapshot = await api.agentRun(id, controller.signal);
        if (runId.current === id) setRun(snapshot);
      } catch (e) {
        if (controller.signal.aborted || runId.current !== id) return;
        if (e instanceof ApiError && e.status === 404) {
          // The backend restarted and took the run with it.
          setRun((current) => (current && current.run_id === id ? { ...current, finished: true } : current));
          setError("The agent run was lost (backend restarted).");
          return;
        }
        setRetry((n) => n + 1);
      }
    }, POLL_MS);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [run, retry]);

  // A closed tab must not leave two CLIs searching for nobody.
  useEffect(() => () => {
    const id = runId.current;
    if (id) void api.cancelAgentRun(id).catch(() => {});
  }, []);

  return { run, error, starting, start, cancel, clear };
}
