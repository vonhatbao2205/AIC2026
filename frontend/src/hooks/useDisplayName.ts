/** The operator's display name, kept in sync across every view that shows it.
 *
 * One packaged image serves the whole team, so the name cannot be baked in at
 * build time — it lives in localStorage and is set from the settings screen.
 * That makes it the one piece of identity that can change mid-session, which is
 * why it is a subscription rather than a value read once at mount.
 */
import { useSyncExternalStore } from "react";
import { getDisplayName, subscribeDisplayName } from "../lib/supabase";

/** Falls back to `"unknown"`, the value the submission table stores for an
 *  operator who never named themselves. */
export function useDisplayName(): string {
  return useSyncExternalStore(
    subscribeDisplayName,
    () => getDisplayName() || "unknown",
    // Server snapshot: no localStorage during SSR/prerender. Unused today, but
    // useSyncExternalStore requires it to be a pure constant, not a read.
    () => "unknown",
  );
}
