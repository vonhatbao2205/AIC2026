/** Ownership of the sticky notes: this browser, and only this browser.
 *
 *  The mirror image of `useSharedSubmission`. That hook exists to get five
 *  operators to agree; this one exists so one operator can think without the
 *  other four watching. There is no outbox, no revision, no realtime channel —
 *  a draft that needed conflict resolution would not be a draft.
 *
 *  localStorage is the source of truth, not a cache: nothing else holds these
 *  candidates, so a reload mid-contest has to bring them all back.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  DEFAULT_NOTE_WINDOW,
  applyOrder,
  clampWindow,
  keepOnlyPack,
  loadNotes,
  loadWindow,
  noteKey,
  pushPlan,
  saveNotes,
  saveWindow,
  sortCandidates,
  type NoteCandidate,
  type NoteStore,
  type NoteWindow,
} from "../lib/stickyNotes";
import type { OrderChange } from "../lib/submissionOrder";
import { newRowId } from "../lib/sharedSubmission";

/** Everything a candidate needs except the fields the store owns. */
export type NoteDraft = Omit<NoteCandidate, "id" | "createdAt" | "pushState" | "submissionRowId"> &
  Partial<Pick<NoteCandidate, "createdAt">>;

export interface NoteQueueAssignment {
  candidateId: string;
  submissionRowId: string;
}

export interface StickyNotes {
  /** Candidates of one question, in rank order. Empty for an unassigned tab. */
  candidatesFor: (questionId: string | null) => NoteCandidate[];
  add: (questionId: string, draft: NoteDraft) => NoteCandidate;
  update: (questionId: string, id: string, patch: Partial<NoteCandidate>) => void;
  remove: (questionId: string, id: string) => void;
  /** Apply a `planReorder` plan — the same ranking maths the Submission table uses. */
  reorder: (questionId: string, changes: OrderChange[]) => void;
  clear: (questionId: string) => void;
  /** The candidates a push would send: unpushed only, in note order, restamped
   *  so no two of them tie. Empty means the button has nothing to do. */
  pending: (questionId: string | null) => NoteCandidate[];
  /** Reserve Submission ids and persist `queued` before the optimistic write. */
  queue: (questionId: string, assignments: NoteQueueAssignment[]) => NoteCandidate[];
  /** Record server confirmation for the matching Submission row ids. */
  markSynced: (questionId: string, submissionRowIds: string[]) => void;
  window: NoteWindow;
  setWindow: (patch: Partial<NoteWindow>) => void;
  toggle: () => void;
}

/** Instants handed out for new candidates, strictly increasing.
 *
 *  `createdAt` IS the rank, so two frames noted inside the same millisecond
 *  would arrive tied and be ordered by their uuid rather than by the order they
 *  were clicked. Same rule, and same reason, as `nextCreatedAt` for the
 *  Submission table — kept separate so a note never consumes a stamp the shared
 *  table was about to use. */
let lastNoteStampMs = 0;

export function nextNoteCreatedAt(): string {
  const now = Date.now();
  lastNoteStampMs = now > lastNoteStampMs ? now : lastNoteStampMs + 1;
  return new Date(lastNoteStampMs).toISOString();
}

export function useStickyNotes(packId: string | null = null): StickyNotes {
  const [store, setStore] = useState<NoteStore>(() => loadNotes());
  const [win, setWin] = useState<NoteWindow>(() => loadWindow());
  const storeRef = useRef(store);
  storeRef.current = store;

  useEffect(() => {
    saveNotes(store);
  }, [store]);

  useEffect(() => {
    saveWindow(win);
  }, [win]);

  // A new import changes the pack fingerprint. Old question ids are commonly
  // reused between rounds, so their drafts must leave this browser before they
  // can be mistaken for candidates of the new statements.
  useEffect(() => {
    if (!packId) return;
    setStore((current) => {
      const next = keepOnlyPack(current, packId);
      if (Object.keys(next).length === Object.keys(current).length) return current;
      // Draft ownership is localStorage, so pack cleanup is persisted in the
      // same turn rather than waiting for React's post-render effect.
      storeRef.current = next;
      saveNotes(next);
      return next;
    });
  }, [packId]);

  // A window remembered from a bigger screen would open off-canvas with no way
  // to drag it back, so the stored position is re-checked against the viewport
  // on mount and whenever the window is resized.
  useEffect(() => {
    const fit = () =>
      setWin((current) => {
        const next = clampWindow(current, {
          width: window.innerWidth,
          height: window.innerHeight,
        });
        return next.x === current.x && next.y === current.y && next.w === current.w && next.h === current.h
          ? current
          : next;
      });
    fit();
    window.addEventListener("resize", fit);
    return () => window.removeEventListener("resize", fit);
  }, []);

  const keyOf = useCallback(
    (questionId: string) => noteKey(packId, questionId),
    [packId],
  );

  const candidatesFor = useCallback(
    (questionId: string | null) =>
      questionId ? sortCandidates(store[keyOf(questionId)] ?? []) : [],
    [keyOf, store],
  );

  const write = useCallback(
    (questionId: string, change: (current: NoteCandidate[]) => NoteCandidate[]) => {
      const key = keyOf(questionId);
      const current = storeRef.current;
      const changed = change(current[key] ?? []);
      // An emptied note is deleted rather than stored as `[]` — the store is
      // keyed by every question this browser has ever opened, and dead keys
      // accumulate for the life of the localStorage entry.
      let next: NoteStore;
      if (!changed.length) {
        if (!(key in current)) return;
        const { [key]: _dropped, ...rest } = current;
        next = rest;
      } else {
        next = { ...current, [key]: changed };
      }
      // This synchronous mirror is important for queueing: `queued` and its
      // Submission row id must survive a reload before addRows starts.
      storeRef.current = next;
      saveNotes(next);
      setStore(next);
    },
    [keyOf],
  );

  const add = useCallback(
    (questionId: string, draft: NoteDraft): NoteCandidate => {
      const candidate: NoteCandidate = {
        ...draft,
        id: newRowId(),
        createdAt: draft.createdAt ?? nextNoteCreatedAt(),
        pushState: "draft",
      };
      write(questionId, (current) => sortCandidates([...current, candidate]));
      return candidate;
    },
    [write],
  );

  const update = useCallback(
    (questionId: string, id: string, patch: Partial<NoteCandidate>) => {
      const changesDraft = [
        "videoId",
        "frames",
        "answer",
        "keyframeIds",
        "ptsTimes",
        "retrievalDatabase",
      ].some((field) => field in patch);
      write(questionId, (current) =>
        sortCandidates(
          current.map((candidate) => {
            if (candidate.id !== id) return candidate;
            const next = { ...candidate, ...patch };
            if (changesDraft) {
              // Editing a queued/synced candidate makes a new version. The old
              // Submission row keeps its immutable id; the edited note gets a
              // fresh id only when the operator pushes it again.
              if (candidate.pushState !== "draft") next.createdAt = nextNoteCreatedAt();
              next.pushState = "draft";
              delete next.submissionRowId;
            }
            return next;
          }),
        ),
      );
    },
    [write],
  );

  const remove = useCallback(
    (questionId: string, id: string) => {
      write(questionId, (current) => current.filter((candidate) => candidate.id !== id));
    },
    [write],
  );

  const reorder = useCallback(
    (questionId: string, changes: OrderChange[]) => {
      if (!changes.length) return;
      write(questionId, (current) => applyOrder(sortCandidates(current), changes));
    },
    [write],
  );

  const clear = useCallback(
    (questionId: string) => {
      write(questionId, () => []);
    },
    [write],
  );

  const pending = useCallback(
    (questionId: string | null) => (questionId ? pushPlan(candidatesFor(questionId)) : []),
    [candidatesFor],
  );

  const queue = useCallback(
    (questionId: string, assignments: NoteQueueAssignment[]): NoteCandidate[] => {
      if (!assignments.length) return [];
      const assigned = new Map(assignments.map((item) => [item.candidateId, item.submissionRowId]));
      const planned = new Map(pushPlan(candidatesFor(questionId)).map((item) => [item.id, item]));
      const queued: NoteCandidate[] = [];
      write(questionId, (current) =>
        current.map((candidate) => {
          const submissionRowId = assigned.get(candidate.id);
          const ordered = planned.get(candidate.id);
          if (!submissionRowId || !ordered || candidate.pushState !== "draft") return candidate;
          const next: NoteCandidate = {
            ...candidate,
            createdAt: ordered.createdAt,
            submissionRowId,
            pushState: "queued",
          };
          queued.push(next);
          return next;
        }),
      );
      return queued;
    },
    [candidatesFor, write],
  );

  const markSynced = useCallback(
    (questionId: string, submissionRowIds: string[]) => {
      if (!submissionRowIds.length) return;
      const done = new Set(submissionRowIds);
      write(questionId, (current) =>
        current.map((candidate) =>
          candidate.pushState === "queued" &&
          candidate.submissionRowId &&
          done.has(candidate.submissionRowId)
            ? { ...candidate, pushState: "synced" }
            : candidate,
        ),
      );
    },
    [write],
  );

  const setWindow = useCallback((patch: Partial<NoteWindow>) => {
    setWin((current) =>
      clampWindow({ ...current, ...patch }, { width: window.innerWidth, height: window.innerHeight }),
    );
  }, []);

  const toggle = useCallback(() => {
    setWin((current) => {
      const next = { ...current, open: !current.open };
      // Opening is also the moment to re-check the position: the note may have
      // been closed on one machine and reopened on another.
      return next.open
        ? clampWindow(next, { width: window.innerWidth, height: window.innerHeight })
        : next;
    });
  }, []);

  return useMemo(
    () => ({
      candidatesFor,
      add,
      update,
      remove,
      reorder,
      clear,
      pending,
      queue,
      markSynced,
      window: win,
      setWindow,
      toggle,
    }),
    [add, candidatesFor, clear, markSynced, pending, queue, remove, reorder, setWindow, toggle, win],
  );
}

export { DEFAULT_NOTE_WINDOW };
