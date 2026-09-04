/** Sticky notes: the scratchpad between "I found something" and "this is my answer".
 *
 *  A search turns up frames an operator is not ready to commit — a maybe, a
 *  second reading of the statement, a frame to compare against the next video.
 *  Today those either go straight into the Submission table (where they are a
 *  shared write five people see) or nowhere at all, so the usual move is to
 *  remember them and lose them.
 *
 *  A note is DELIBERATELY LOCAL. Nothing here touches Supabase: a draft is one
 *  operator thinking out loud, and syncing it would put five half-formed lists
 *  in front of everybody. It becomes shared state at exactly one moment — the
 *  push — and that moment is a button, not a side effect.
 *
 *  The rank key is `createdAt`, the same field and the same maths the Submission
 *  table ranks by (`lib/submissionOrder`). That is not incidental: a draft list
 *  ordered by different rules than the table it feeds would silently re-rank
 *  itself on the way across, and rank is score.
 */
import type { RetrievalDatabase } from "../api/types";
import { orderSlots, type OrderChange } from "./submissionOrder";

/** One frame (or one TRAKE sequence) parked for later.
 *
 *  Shaped like the `SubmissionDraft` the console already builds, so the push is
 *  a field-for-field hand-over with nothing to reinterpret. */
export interface NoteCandidate {
  id: string;
  videoId: string;
  /** One frame for KIS/QA; one per event for TRAKE — same rule as the CSV. */
  frames: number[];
  answer: string;
  /** Preview/provenance, positionally parallel to `frames`. Null where the frame
   *  came off a paused video and is not an extracted keyframe. */
  keyframeIds: (string | null)[];
  ptsTimes: (number | null)[];
  retrievalDatabase: RetrievalDatabase;
  /** When the frame was noted — and therefore its rank inside the note. Carried
   *  into the Submission row verbatim on push, so a pushed candidate sits
   *  exactly where a frame submitted the moment it was noted would have sat. */
  createdAt: string;
  /** Stable id reserved before the optimistic Submission insert starts. It is
   *  the recovery handle after a reload: the client can ask the server whether
   *  this exact row landed and enqueue it again when it did not. */
  submissionRowId?: string;
  /** `queued` means handed to the optimistic Submission store, not delivered.
   *  Only an observed server-backed row may advance this to `synced`. */
  pushState: "draft" | "queued" | "synced";
}

/** Every note this browser holds, keyed by `noteKey`. */
export type NoteStore = Record<string, NoteCandidate[]>;

export interface NoteWindow {
  x: number;
  y: number;
  w: number;
  h: number;
  open: boolean;
}

const NOTES_KEY = "aic26_sticky_notes";
const WINDOW_KEY = "aic26_sticky_window";

export const MIN_NOTE_W = 320;
export const MIN_NOTE_H = 200;

export const DEFAULT_NOTE_WINDOW: NoteWindow = {
  x: 120,
  y: 120,
  w: 560,
  h: 380,
  open: false,
};

/** Notes are scoped to the question AND the pack that defined it.
 *
 *  Re-importing a pack reuses the same question ids, so a note keyed on the id
 *  alone would show yesterday's candidates under today's question — the exact
 *  confusion migration 002 removed from the Submission table. A machine with no
 *  session yet keeps its drafts under one local bucket rather than losing them. */
export function noteKey(packId: string | null, questionId: string): string {
  return `${packId ?? "local"}::${questionId}`;
}

/** Drop drafts belonging to every pack except the active one.
 *
 *  The pack fingerprint is stable across reloads and across the shared/local
 *  modes. Keeping only its prefix both prevents an old candidate appearing
 *  under a re-used question id and bounds the lifetime of the localStorage
 *  entry. An unknown pack deliberately does nothing: startup briefly has no
 *  server session, and clearing in that gap would destroy the current drafts. */
export function keepOnlyPack(store: NoteStore, packId: string | null): NoteStore {
  if (!packId) return store;
  const prefix = `${packId}::`;
  return Object.fromEntries(Object.entries(store).filter(([key]) => key.startsWith(prefix)));
}

export function loadNotes(): NoteStore {
  try {
    const raw = localStorage.getItem(NOTES_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw) as Record<string, unknown>;
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return {};
    const store: NoteStore = {};
    for (const [key, value] of Object.entries(parsed)) {
      if (!Array.isArray(value)) continue;
      const candidates = value.flatMap((item): NoteCandidate[] => {
        if (!item || typeof item !== "object") return [];
        const rawCandidate = item as Partial<NoteCandidate>;
        if (
          typeof rawCandidate.id !== "string" ||
          typeof rawCandidate.videoId !== "string" ||
          !Array.isArray(rawCandidate.frames)
        ) {
          return [];
        }
        const frames = rawCandidate.frames.map(Number).filter(Number.isFinite).map(Math.round);
        const created = Date.parse(rawCandidate.createdAt ?? "");
        const submissionRowId = typeof rawCandidate.submissionRowId === "string"
          ? rawCandidate.submissionRowId
          : undefined;
        const rawState = rawCandidate.pushState;
        // `pushedAt` is the pre-state-machine format. There is no row id with
        // which to reconcile those old entries, so preserve their old visible
        // meaning while every new push uses the recoverable protocol.
        const legacyPushed = typeof (rawCandidate as Partial<NoteCandidate> & { pushedAt?: unknown }).pushedAt === "string";
        const pushState = rawState === "synced"
          ? "synced"
          : rawState === "queued" && submissionRowId
            ? "queued"
            : legacyPushed
              ? "synced"
              : "draft";
        return [{
          id: rawCandidate.id,
          videoId: rawCandidate.videoId,
          frames,
          answer: typeof rawCandidate.answer === "string" ? rawCandidate.answer : "",
          keyframeIds: frames.map((_, index) =>
            typeof rawCandidate.keyframeIds?.[index] === "string"
              ? rawCandidate.keyframeIds[index] as string
              : null,
          ),
          ptsTimes: frames.map((_, index) => {
            const pts = rawCandidate.ptsTimes?.[index];
            return typeof pts === "number" && Number.isFinite(pts) ? pts : null;
          }),
          retrievalDatabase: rawCandidate.retrievalDatabase === "infoshotpp" ? "infoshotpp" : "btc",
          createdAt: Number.isFinite(created)
            ? new Date(created).toISOString()
            : new Date(0).toISOString(),
          ...(submissionRowId ? { submissionRowId } : {}),
          pushState,
        }];
      });
      if (candidates.length) store[key] = candidates;
    }
    return store;
  } catch {
    // A corrupt draft must never take the console down with it.
    return {};
  }
}

export function saveNotes(store: NoteStore): void {
  try {
    localStorage.setItem(NOTES_KEY, JSON.stringify(store));
  } catch {
    /* quota — memory stays authoritative for this session */
  }
}

export function loadWindow(): NoteWindow {
  try {
    const raw = localStorage.getItem(WINDOW_KEY);
    if (!raw) return { ...DEFAULT_NOTE_WINDOW };
    const parsed = JSON.parse(raw) as Partial<NoteWindow>;
    return {
      x: Number.isFinite(parsed.x) ? Number(parsed.x) : DEFAULT_NOTE_WINDOW.x,
      y: Number.isFinite(parsed.y) ? Number(parsed.y) : DEFAULT_NOTE_WINDOW.y,
      w: Number.isFinite(parsed.w) ? Number(parsed.w) : DEFAULT_NOTE_WINDOW.w,
      h: Number.isFinite(parsed.h) ? Number(parsed.h) : DEFAULT_NOTE_WINDOW.h,
      open: parsed.open === true,
    };
  } catch {
    return { ...DEFAULT_NOTE_WINDOW };
  }
}

export function saveWindow(win: NoteWindow): void {
  try {
    localStorage.setItem(WINDOW_KEY, JSON.stringify(win));
  } catch {
    /* quota */
  }
}

/** Keep the window reachable.
 *
 *  A remembered position is a liability across machines: the contest laptop may
 *  have a smaller screen than the one the note was dragged on, and a window
 *  restored at x=1900 on a 1366-wide display is gone with no way to get it back.
 *  The title bar is what has to stay grabbable, so the clamp keeps a strip of it
 *  on screen rather than the whole frame. */
export function clampWindow(
  win: NoteWindow,
  viewport: { width: number; height: number },
  grabStrip = 80,
): NoteWindow {
  const w = Math.max(MIN_NOTE_W, Math.min(win.w, Math.max(MIN_NOTE_W, viewport.width)));
  const h = Math.max(MIN_NOTE_H, Math.min(win.h, Math.max(MIN_NOTE_H, viewport.height)));
  return {
    ...win,
    w,
    h,
    // Left edge may go slightly negative, but never so far that the title bar
    // has no grabbable strip left inside the viewport.
    x: Math.max(grabStrip - w, Math.min(win.x, viewport.width - grabStrip)),
    y: Math.max(0, Math.min(win.y, Math.max(0, viewport.height - 28))),
  };
}

/** Candidates that have not been handed to the Submission store yet. Queued
 *  candidates are deliberately excluded so a second click cannot duplicate an
 *  insert that is merely waiting for Supabase. */
export function unpushed(candidates: readonly NoteCandidate[]): NoteCandidate[] {
  return candidates.filter((candidate) => candidate.pushState === "draft");
}

/** What one push hands to the Submission store.
 *
 *  Only the candidates that never went, in note order, each with a strictly
 *  increasing instant. The instants are the note's own — a pushed candidate must
 *  land where a frame submitted the moment it was noted would have landed, which
 *  is what keeps hand-picked answers ahead of a generated block written later.
 *  `orderSlots` deduplicates them first: two frames noted in the same
 *  millisecond would otherwise reach the table tied, and a tie is decided by the
 *  uuid tiebreak rather than by the order the operator arranged.
 */
export function pushPlan(candidates: readonly NoteCandidate[]): NoteCandidate[] {
  const pending = unpushed(candidates);
  if (!pending.length) return [];
  const slots = orderSlots(pending);
  return pending.map((candidate, index) => ({
    ...candidate,
    createdAt: new Date(slots[index]).toISOString(),
  }));
}

/** Apply a ranking plan from `planReorder` to a note's candidates. */
export function applyOrder(
  candidates: readonly NoteCandidate[],
  changes: readonly OrderChange[],
): NoteCandidate[] {
  if (!changes.length) return [...candidates];
  const wanted = new Map(changes.map((change) => [change.id, change.createdAt]));
  return sortCandidates(
    candidates.map((candidate) =>
      wanted.has(candidate.id)
        ? { ...candidate, createdAt: wanted.get(candidate.id) as string }
        : candidate,
    ),
  );
}

/** Note order: oldest first, uuid as the last-resort tiebreak.
 *
 *  Deliberately the same comparison as `sortRows` in `sharedSubmission`, by
 *  instant rather than by text, so a note and the table it feeds never disagree
 *  about which of two candidates is first. */
export function sortCandidates(candidates: readonly NoteCandidate[]): NoteCandidate[] {
  const ms = (candidate: NoteCandidate) => {
    const parsed = Date.parse(candidate.createdAt ?? "");
    return Number.isNaN(parsed) ? 0 : parsed;
  };
  return [...candidates].sort((a, b) => ms(a) - ms(b) || a.id.localeCompare(b.id));
}

/** True when the note already holds this exact prediction.
 *
 *  Parking the same frame twice is pure noise in a list whose whole job is to be
 *  read at a glance — and after a push it would be a duplicate CSV row. */
export function findNoteDuplicate(
  candidates: readonly NoteCandidate[],
  candidate: Pick<NoteCandidate, "videoId" | "frames">,
  ignoreId?: string,
): NoteCandidate | null {
  const key = `${candidate.videoId.trim()}|${candidate.frames.join(",")}`;
  return (
    candidates.find(
      (item) =>
        item.id !== ignoreId &&
        `${item.videoId.trim()}|${item.frames.join(",")}` === key,
    ) ?? null
  );
}
