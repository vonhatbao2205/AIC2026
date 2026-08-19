/** Supabase client for the shared Submission tab.
 *
 * Everything else in this app stays local: search hits Elastic/Milvus/PE on the
 * operator's own machine, and no media is ever synced. Only submission rows —
 * a handful of strings and integers each — travel through here.
 *
 * The client is created lazily and may be null. When the two env vars are
 * absent the Submission tab keeps working exactly as before, backed by
 * localStorage alone, so a checkout with no Supabase project still runs.
 */
import { createClient, type SupabaseClient } from "@supabase/supabase-js";

const env = (import.meta as unknown as { env?: Record<string, string | undefined> }).env ?? {};

export const SUPABASE_URL = (env.VITE_SUPABASE_URL ?? "").trim();
export const SUPABASE_KEY = (env.VITE_SUPABASE_PUBLISHABLE_KEY ?? "").trim();

/** Logical room, so a rehearsal and the real contest never mix rows. */
export const SUBMISSION_ROOM = (env.VITE_SUBMISSION_ROOM ?? "aic26").trim() || "aic26";

export const SUBMISSIONS_TABLE = "submissions";

/** Vite loads `.env.local` for the test run too, so a suite would otherwise
 *  build a real client and read and write the team's live submission table.
 *  Sharing is therefore refused outright in test mode; the wire format is
 *  covered by the pure record tests instead. */
const IS_TEST = (env.MODE ?? "") === "test";

export function isSupabaseConfigured(): boolean {
  if (IS_TEST) return false;
  return Boolean(SUPABASE_URL && SUPABASE_KEY);
}

let client: SupabaseClient | null = null;

export function getSupabase(): SupabaseClient | null {
  if (!isSupabaseConfigured()) return null;
  if (!client) {
    client = createClient(SUPABASE_URL, SUPABASE_KEY, {
      auth: { persistSession: false },
      // The rows are tiny; the default 10/s is plenty and keeps a burst of
      // submits from being throttled into a visible lag on the other clients.
      realtime: { params: { eventsPerSecond: 10 } },
    });
  }
  return client;
}

const USER_KEY = "aic26_submission_user";

/** Display name shown next to a row. Asked for once, then remembered.
 *
 * localStorage outranks `VITE_SUBMISSION_USER` because the env var is baked at
 * build time: one packaged image serves the whole team, so it cannot carry a
 * per-person name. It stays as the default for a source checkout that sets it
 * in `.env.local`, but a name typed into the settings screen has to win — a
 * control that silently loses to a build constant is worse than no control.
 */
export function getDisplayName(): string {
  try {
    const stored = (localStorage.getItem(USER_KEY) ?? "").trim();
    if (stored) return stored;
  } catch {
    /* private mode / disabled storage — fall through to the build default */
  }
  // Vite loads `.env.local` for the test run too, and `test.env` in the Vitest
  // config does not override a variable an env file already set. So without
  // this the name tests would assert against whatever the developer running
  // them happens to have configured, and pass on one machine while failing on
  // the next. Same guard the Supabase credentials above use, for the same reason.
  if (IS_TEST) return "";
  return (env.VITE_SUBMISSION_USER ?? "").trim();
}

/** Notified whenever the name changes, so open views relabel themselves.
 *
 * The name is read once per hook instance, at mount. Without this the operator
 * would set their name, keep working, and still submit rows tagged `unknown`
 * until they thought to reload — during a timed contest, with no hint anything
 * was wrong.
 */
const nameListeners = new Set<() => void>();

export function subscribeDisplayName(listener: () => void): () => void {
  nameListeners.add(listener);
  return () => {
    nameListeners.delete(listener);
  };
}

export function setDisplayName(name: string): void {
  try {
    localStorage.setItem(USER_KEY, name.trim());
  } catch {
    /* ignore */
  }
  for (const listener of nameListeners) listener();
}
