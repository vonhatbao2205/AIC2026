/** Ownership of the Submission rows: shared across clients when Supabase is
 *  configured, local-only when it is not.
 *
 *  The Workspace used to hold `rows` in state and mirror them into localStorage.
 *  That contract is unchanged from its point of view — it still gets `rows` plus
 *  mutators — but the rows now round-trip through Postgres and arrive back on
 *  every teammate's screen.
 *
 *  Writes are optimistic: the row appears immediately and is queued for the
 *  server. A contest network drops for a few seconds at a time, and an operator
 *  must never lose an answer to that, so a failed write stays in an outbox and
 *  is retried instead of being surfaced as a lost submit.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  SUBMISSIONS_TABLE,
  SUBMISSION_ROOM,
  getDisplayName,
  getSupabase,
  isSupabaseConfigured,
} from "../lib/supabase";
import {
  MISSING_TABLE_HINT,
  describeSupabaseError,
  isMissingTableError,
  mergeRow,
  newRowId,
  recordToRow,
  rowToRecord,
  sortRows,
  type SubmissionRecord,
} from "../lib/sharedSubmission";
import type { SubmissionRow } from "../lib/submission";

const CACHE_KEY = "aic26_submission_rows";
const RETRY_MS = 4000;

export type SyncStatus = "offline" | "connecting" | "live" | "error";

export interface SharedSubmission {
  rows: SubmissionRow[];
  status: SyncStatus;
  /** Writes still waiting for the server. */
  pending: number;
  error: string | null;
  /** Row ids another client changed while this client had a stale revision. */
  conflicts: string[];
  shared: boolean;
  room: string;
  user: string;
  addRow: (row: SubmissionRow) => void;
  updateRow: (id: string, patch: Partial<SubmissionRow>) => void;
  deleteRow: (id: string) => void;
  clearQuestion: (questionId: string) => void;
  dismissConflict: (id: string) => void;
}

type Op =
  | { kind: "insert"; row: SubmissionRow }
  // The edited row travels WITH the op. Reading it back from a ref at flush
  // time raced React: `flush()` runs in the same tick as the optimistic
  // `setRows`, so the ref still held the pre-edit row, the old values were sent,
  // and the server's reply — carrying a higher revision — overwrote the
  // operator's edit. The change simply vanished from the table.
  | { kind: "update"; id: string; row: SubmissionRow; expectedRevision: number }
  | { kind: "delete"; id: string };

function loadCache(): SubmissionRow[] {
  try {
    const raw = localStorage.getItem(CACHE_KEY);
    return raw ? (JSON.parse(raw) as SubmissionRow[]) : [];
  } catch {
    return [];
  }
}

export function useSharedSubmission(sessionId: string | null = null): SharedSubmission {
  const shared = isSupabaseConfigured();
  const [rows, setRows] = useState<SubmissionRow[]>(() => loadCache());
  const [status, setStatus] = useState<SyncStatus>(shared ? "connecting" : "offline");
  const [error, setError] = useState<string | null>(null);
  const [conflicts, setConflicts] = useState<string[]>([]);
  const [pending, setPending] = useState(0);
  const user = useMemo(() => getDisplayName() || "unknown", []);
  // Answers are scoped to the pack that defined their questions, so a rehearsal
  // and the real round never land in the same export.
  const sessionRef = useRef(sessionId);
  sessionRef.current = sessionId;

  // Rows are the one thing here that cannot be recomputed, so the local mirror
  // is kept even when sharing works — it is what survives a reload mid-contest.
  useEffect(() => {
    try {
      localStorage.setItem(CACHE_KEY, JSON.stringify(rows));
    } catch {
      /* quota — memory stays authoritative */
    }
  }, [rows]);

  const outbox = useRef<Op[]>([]);
  const rowsRef = useRef(rows);
  rowsRef.current = rows;
  const flushing = useRef(false);

  const flush = useCallback(async () => {
    const supabase = getSupabase();
    if (!supabase || flushing.current) return;
    flushing.current = true;
    try {
      while (outbox.current.length) {
        const op = outbox.current[0];
        try {
          if (op.kind === "insert") {
            const record = rowToRecord(op.row, SUBMISSION_ROOM, user, sessionRef.current);
            const { error: insertError } = await supabase.from(SUBMISSIONS_TABLE).insert(record);
            if (insertError) throw insertError;
            setRows((current) =>
              current.map((row) => (row.id === op.row.id ? { ...row, syncState: "synced" } : row)),
            );
          } else if (op.kind === "update") {
            {
              const record = rowToRecord(op.row, SUBMISSION_ROOM, user, sessionRef.current);
              const { data, error: updateError } = await supabase
                .from(SUBMISSIONS_TABLE)
                .update({
                  video_id: record.video_id,
                  frames: record.frames,
                  keyframe_ids: record.keyframe_ids,
                  pts_times: record.pts_times,
                  answer: record.answer,
                  source: record.source,
                })
                .eq("id", op.id)
                // Optimistic concurrency: zero rows back means somebody else
                // wrote this row after we read it.
                .eq("revision", op.expectedRevision)
                .select();
              if (updateError) throw updateError;
              if (!data || data.length === 0) {
                setConflicts((current) =>
                  current.includes(op.id) ? current : [...current, op.id],
                );
              } else {
                const fresh = recordToRow(data[0] as SubmissionRecord);
                setRows((current) => sortRows(mergeRow(current, fresh)));
              }
            }
          } else {
            const { error: deleteError } = await supabase
              .from(SUBMISSIONS_TABLE)
              .delete()
              .eq("id", op.id);
            if (deleteError) throw deleteError;
          }
          outbox.current.shift();
          setPending(outbox.current.length);
          setError(null);
        } catch (opError) {
          // Keep the op queued and try again; do not drop an operator's answer.
          setError(
            isMissingTableError(opError) ? MISSING_TABLE_HINT : describeSupabaseError(opError),
          );
          setStatus("error");
          break;
        }
      }
    } finally {
      flushing.current = false;
    }
  }, [user]);

  const enqueue = useCallback(
    (op: Op) => {
      if (!shared) return;
      outbox.current.push(op);
      setPending(outbox.current.length);
      void flush();
    },
    [shared, flush],
  );

  useEffect(() => {
    if (!shared) return;
    const id = setInterval(() => {
      if (outbox.current.length) void flush();
    }, RETRY_MS);
    return () => clearInterval(id);
  }, [shared, flush]);

  // Initial load + realtime subscription.
  useEffect(() => {
    const supabase = getSupabase();
    if (!supabase) return;
    let cancelled = false;

    void (async () => {
      const query = supabase.from(SUBMISSIONS_TABLE).select("*").eq("room", SUBMISSION_ROOM);
      if (sessionId) query.eq("session_id", sessionId);
      const { data, error: loadError } = await query.order("created_at", { ascending: true });
      if (cancelled) return;
      if (loadError) {
        setError(
          isMissingTableError(loadError) ? MISSING_TABLE_HINT : describeSupabaseError(loadError),
        );
        setStatus("error");
        return;
      }
      // The server is the truth for shared rows; anything still in the outbox is
      // re-applied on top so an unsynced local answer is not wiped by the fetch.
      const remote = ((data ?? []) as SubmissionRecord[]).map(recordToRow);
      setRows((current) => {
        const unsynced = current.filter((row) => row.syncState !== "synced");
        return sortRows(unsynced.reduce((acc, row) => mergeRow(acc, row), remote));
      });
      setStatus("live");
      void flush();
    })();

    const channel = supabase
      .channel(`submissions:${sessionId ?? SUBMISSION_ROOM}`)
      .on(
        "postgres_changes",
        {
          event: "*",
          schema: "public",
          table: SUBMISSIONS_TABLE,
          filter: sessionId ? `session_id=eq.${sessionId}` : `room=eq.${SUBMISSION_ROOM}`,
        },
        (payload: { eventType: string; new: unknown; old: unknown }) => {
          if (payload.eventType === "DELETE") {
            const removed = payload.old as { id?: string };
            if (removed?.id) setRows((current) => current.filter((row) => row.id !== removed.id));
            return;
          }
          const record = payload.new as SubmissionRecord;
          if (!record?.id) return;
          setRows((current) => sortRows(mergeRow(current, recordToRow(record))));
        },
      )
      .subscribe((state: string) => {
        if (cancelled) return;
        if (state === "SUBSCRIBED") setStatus("live");
        else if (state === "CHANNEL_ERROR" || state === "TIMED_OUT") setStatus("error");
      });

    return () => {
      cancelled = true;
      void supabase.removeChannel(channel);
    };
  }, [flush, sessionId]);

  const addRow = useCallback(
    (row: SubmissionRow) => {
      const complete: SubmissionRow = {
        ...row,
        sessionId: row.sessionId ?? sessionRef.current,
        submittedBy: row.submittedBy || user,
        revision: 1,
        createdAt: row.createdAt ?? new Date().toISOString(),
        syncState: shared ? "pending" : "local",
      };
      setRows((current) => sortRows([...current, complete]));
      enqueue({ kind: "insert", row: complete });
    },
    [enqueue, shared, user],
  );

  const updateRow = useCallback(
    (id: string, patch: Partial<SubmissionRow>) => {
      const current = rowsRef.current.find((row) => row.id === id);
      if (!current) return;
      const next: SubmissionRow = {
        ...current,
        ...patch,
        syncState: shared ? "pending" : "local",
      };
      rowsRef.current = rowsRef.current.map((row) => (row.id === id ? next : row));
      setRows((rows) => rows.map((row) => (row.id === id ? next : row)));
      if (!shared) return;
      // Typing in a cell fires one update per keystroke. Collapsing the queued
      // ones into the latest keeps that from becoming a request per character —
      // and from every one of them but the first failing the revision check.
      const queued = outbox.current.findIndex(
        (op, index) => index > 0 && op.kind === "update" && op.id === id,
      );
      if (queued > 0) {
        const existing = outbox.current[queued] as Extract<Op, { kind: "update" }>;
        outbox.current[queued] = { ...existing, row: next };
        setPending(outbox.current.length);
        void flush();
        return;
      }
      enqueue({ kind: "update", id, row: next, expectedRevision: current.revision ?? 1 });
    },
    [enqueue, flush, shared],
  );

  const deleteRow = useCallback(
    (id: string) => {
      setRows((current) => current.filter((row) => row.id !== id));
      enqueue({ kind: "delete", id });
    },
    [enqueue],
  );

  const clearQuestion = useCallback(
    (questionId: string) => {
      const doomed = rowsRef.current.filter((row) => row.questionId === questionId);
      setRows((current) => current.filter((row) => row.questionId !== questionId));
      for (const row of doomed) enqueue({ kind: "delete", id: row.id });
    },
    [enqueue],
  );

  const dismissConflict = useCallback((id: string) => {
    setConflicts((current) => current.filter((item) => item !== id));
  }, []);

  return {
    rows,
    status: shared ? status : "offline",
    pending,
    error,
    conflicts,
    shared,
    room: SUBMISSION_ROOM,
    user,
    addRow,
    updateRow,
    deleteRow,
    clearQuestion,
    dismissConflict,
  };
}

export { newRowId };
