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
  getSupabase,
  isSupabaseConfigured,
} from "../lib/supabase";
import {
  MISSING_TABLE_HINT,
  describeSupabaseError,
  isMissingTableError,
  mergeRow,
  newRowId,
  nextCreatedAt,
  recordToRow,
  rowToRecord,
  sortRows,
  type SubmissionRecord,
} from "../lib/sharedSubmission";
import type { SubmissionRow } from "../lib/submission";
import { useDisplayName } from "./useDisplayName";

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
  /** Insert many rows as one write. The answer generator produces up to 100 rows
   *  per question; queueing them one by one would put thousands of round trips
   *  in the outbox and take minutes to drain over a contest network. */
  addRows: (rows: SubmissionRow[]) => void;
  updateRow: (id: string, patch: Partial<SubmissionRow>) => void;
  /** Write one Q&A answer into every row of a question, as a single write.
   *  The generator produces the frames; the text is a human judgement typed
   *  once, so per-row edits would be 100 round trips for one decision. */
  setAnswerForQuestion: (questionId: string, answer: string) => void;
  deleteRow: (id: string) => void;
  /** Remove many rows as one write — regenerating a question replaces up to a
   *  hundred of them at once. */
  deleteRows: (ids: string[]) => void;
  clearQuestion: (questionId: string) => void;
  dismissConflict: (id: string) => void;
}

type Op =
  | { kind: "insert"; row: SubmissionRow }
  | { kind: "insert_many"; rows: SubmissionRow[] }
  | { kind: "answer_many"; ids: string[]; answer: string }
  | { kind: "delete_many"; ids: string[] }
  // Clearing a question deletes BY QUESTION, not by the ids this client happens
  // to hold: a teammate may have added a row a second ago that this client has
  // not seen yet, and "xoá hết" means the question, not "the rows I know about".
  | { kind: "delete_question"; questionId: string; sessionId: string | null }
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

/** Rows that belong to the pack currently being answered.
 *
 *  The local cache spans every session this browser has seen. A row that never
 *  reached the server — written offline, or before Supabase was configured —
 *  would otherwise be merged into whatever session is active now, and if the two
 *  packs share a question id (re-importing the same pack does) it would be
 *  attributed to a question it was never an answer to. */
function belongsToSession(row: SubmissionRow, sessionId: string | null): boolean {
  if (!sessionId) return true;
  return !row.sessionId || row.sessionId === sessionId;
}

export function useSharedSubmission(sessionId: string | null = null): SharedSubmission {
  const shared = isSupabaseConfigured();
  const [rows, setRows] = useState<SubmissionRow[]>(() => loadCache());
  const [status, setStatus] = useState<SyncStatus>(shared ? "connecting" : "offline");
  const [error, setError] = useState<string | null>(null);
  const [conflicts, setConflicts] = useState<string[]>([]);
  const [pending, setPending] = useState(0);
  const user = useDisplayName();
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
          } else if (op.kind === "insert_many") {
            const records = op.rows.map((row) =>
              rowToRecord(row, SUBMISSION_ROOM, user, sessionRef.current),
            );
            const { error: insertError } = await supabase.from(SUBMISSIONS_TABLE).insert(records);
            if (insertError) throw insertError;
            const ids = new Set(op.rows.map((row) => row.id));
            setRows((current) =>
              current.map((row) => (ids.has(row.id) ? { ...row, syncState: "synced" } : row)),
            );
          } else if (op.kind === "answer_many") {
            const { data, error: answerError } = await supabase
              .from(SUBMISSIONS_TABLE)
              .update({ answer: op.answer })
              .in("id", op.ids)
              .select();
            if (answerError) throw answerError;
            // Deliberately no per-row `revision` guard: the operator is asserting
            // one answer for the whole question, and checking a hundred revisions
            // would mean a hundred round trips for one decision. The server rows
            // are merged straight back so every revision stays current — without
            // that, the next single-row edit would send a stale revision and be
            // reported as a conflict nobody caused.
            const fresh = ((data ?? []) as SubmissionRecord[]).map(recordToRow);
            if (fresh.length) {
              setRows((current) => sortRows(fresh.reduce(mergeRow, current)));
            }
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
          } else if (op.kind === "delete_question") {
            const request = supabase
              .from(SUBMISSIONS_TABLE)
              .delete()
              .eq("question_id", op.questionId);
            if (op.sessionId) request.eq("session_id", op.sessionId);
            else request.eq("room", SUBMISSION_ROOM);
            const { error: deleteError } = await request;
            if (deleteError) throw deleteError;
          } else if (op.kind === "delete_many") {
            const { error: deleteError } = await supabase
              .from(SUBMISSIONS_TABLE)
              .delete()
              .in("id", op.ids);
            if (deleteError) throw deleteError;
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
      // Paged on purpose. PostgREST caps a response at the project's "Max rows"
      // setting (1000 by default) and says so only in `content-range` — an
      // unpaged read would silently stop there, and 24 questions x 100 answers
      // is 2400. Missing rows would then be missing from the export.
      const page = 1000;
      const collected: SubmissionRecord[] = [];
      let loadError: unknown = null;
      for (let from = 0; ; from += page) {
        const query = supabase.from(SUBMISSIONS_TABLE).select("*").eq("room", SUBMISSION_ROOM);
        if (sessionId) query.eq("session_id", sessionId);
        const { data: chunk, error } = await query
          .order("created_at", { ascending: true })
          .range(from, from + page - 1);
        if (error) {
          loadError = error;
          break;
        }
        const rows = (chunk ?? []) as SubmissionRecord[];
        collected.push(...rows);
        if (rows.length < page) break;
      }
      const data = collected;
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
        const unsynced = current.filter(
          (row) => row.syncState !== "synced" && belongsToSession(row, sessionId),
        );
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
        createdAt: row.createdAt ?? nextCreatedAt(),
        syncState: shared ? "pending" : "local",
      };
      setRows((current) => sortRows([...current, complete]));
      enqueue({ kind: "insert", row: complete });
    },
    [enqueue, shared, user],
  );

  const addRows = useCallback(
    (incoming: SubmissionRow[]) => {
      if (!incoming.length) return;
      // One stamp per row, in the order they arrive: this array IS the ranking
      // the answer generator produced, and `created_at` is what preserves it.
      const complete = incoming.map((row) => ({
        ...row,
        sessionId: row.sessionId ?? sessionRef.current,
        submittedBy: row.submittedBy || user,
        revision: 1,
        createdAt: row.createdAt ?? nextCreatedAt(),
        syncState: (shared ? "pending" : "local") as SubmissionRow["syncState"],
      }));
      setRows((current) => sortRows([...current, ...complete]));
      enqueue({ kind: "insert_many", rows: complete });
    },
    [enqueue, shared, user],
  );

  const setAnswerForQuestion = useCallback(
    (questionId: string, answer: string) => {
      const targets = rowsRef.current.filter((row) => row.questionId === questionId);
      if (!targets.length) return;
      const text = answer.trim();
      setRows((current) =>
        current.map((row) =>
          row.questionId === questionId
            ? { ...row, answer: text, syncState: shared ? "pending" : "local" }
            : row,
        ),
      );
      enqueue({ kind: "answer_many", ids: targets.map((row) => row.id), answer: text });
    },
    [enqueue, shared],
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

  const deleteRows = useCallback(
    (ids: string[]) => {
      if (!ids.length) return;
      const doomed = new Set(ids);
      setRows((current) => current.filter((row) => !doomed.has(row.id)));
      enqueue({ kind: "delete_many", ids: [...doomed] });
    },
    [enqueue],
  );

  const clearQuestion = useCallback(
    (questionId: string) => {
      // One request, not one per row. A hundred queued deletes are a hundred
      // sequential round trips: the table looks cleared instantly while the
      // outbox is still draining, and closing the tab in that window leaves the
      // rest on the server to reappear on the next load.
      setRows((current) => current.filter((row) => row.questionId !== questionId));
      enqueue({ kind: "delete_question", questionId, sessionId: sessionRef.current });
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
    addRows,
    setAnswerForQuestion,
    updateRow,
    deleteRow,
    deleteRows,
    clearQuestion,
    dismissConflict,
  };
}

export { newRowId };
