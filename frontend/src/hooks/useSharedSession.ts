/** The active question pack, shared across the team.
 *
 *  Supabase is the source of truth; localStorage is a cache that keeps the app
 *  usable when the network drops mid-contest. That direction matters: with the
 *  cache as the source of truth, two machines could hold different packs while
 *  both wrote into the same shared submissions table, and a `question_id` would
 *  silently mean different things on each screen.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import {
  SUBMISSION_ROOM,
  getSupabase,
  isSupabaseConfigured,
} from "../lib/supabase";
import { describeSupabaseError } from "../lib/sharedSubmission";
import {
  canonicalQuestions,
  normalizeQuestions,
  packHash,
  recordsToQuestions,
  type QuestionRecord,
  type SessionInfo,
} from "../lib/questionPack";
import type { ImportedQuestion } from "../lib/questions";
import { useDisplayName } from "./useDisplayName";

const CACHE_KEY = "aic26_question_pack";

export type PackOrigin = "server" | "cache" | "local" | "none";

export interface SharedSession {
  session: SessionInfo | null;
  questions: ImportedQuestion[];
  /** Where the questions on screen came from — surfaced so nobody assumes a
   *  cached pack is the one the team is answering. */
  origin: PackOrigin;
  loading: boolean;
  error: string | null;
  shared: boolean;
  /** Push a parsed pack to the whole team and make it the active session.
   *
   *  With no Supabase project configured there is no team to publish to, so it
   *  applies the pack to this machine instead of failing — the console has to
   *  stay usable on a lone laptop. */
  publish: (questions: ImportedQuestion[], name: string) => Promise<void>;
}

interface CachedPack {
  sessionId: string | null;
  sessionName: string | null;
  packHash: string;
  questions: ImportedQuestion[];
  origin: PackOrigin;
}

function loadCache(): CachedPack | null {
  try {
    const raw = localStorage.getItem(CACHE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    // The pre-session cache was a bare array of questions.
    if (Array.isArray(parsed)) {
      const questions = normalizeQuestions(parsed as ImportedQuestion[]);
      return {
        sessionId: null,
        sessionName: null,
        packHash: packHash(questions),
        questions,
        origin: "local",
      };
    }
    const pack = parsed as CachedPack;
    // `packHash` stays whatever the server said: normalising a derived field
    // locally must not make this client believe it holds a different pack from
    // the rest of the team.
    return { ...pack, questions: normalizeQuestions(pack.questions ?? []) };
  } catch {
    return null;
  }
}

function saveCache(pack: CachedPack): void {
  try {
    localStorage.setItem(CACHE_KEY, JSON.stringify(pack));
  } catch {
    /* quota — memory stays authoritative for this run */
  }
}

export function useSharedSession(): SharedSession {
  const shared = isSupabaseConfigured();
  const cached = useRef<CachedPack | null>(loadCache());
  const [session, setSession] = useState<SessionInfo | null>(
    cached.current?.sessionId
      ? {
          id: cached.current.sessionId,
          room: SUBMISSION_ROOM,
          name: cached.current.sessionName ?? "cached",
          packHash: cached.current.packHash,
          questionCount: cached.current.questions.length,
          publishedBy: "?",
          createdAt: "",
        }
      : null,
  );
  const [questions, setQuestions] = useState<ImportedQuestion[]>(cached.current?.questions ?? []);
  const [origin, setOrigin] = useState<PackOrigin>(
    cached.current ? (shared ? "cache" : cached.current.origin) : "none",
  );
  const [loading, setLoading] = useState(shared);
  const [error, setError] = useState<string | null>(null);
  const user = useDisplayName();

  const adopt = useCallback((info: SessionInfo, list: ImportedQuestion[]) => {
    setSession(info);
    setQuestions(list);
    setOrigin("server");
    saveCache({
      sessionId: info.id,
      sessionName: info.name,
      packHash: info.packHash,
      questions: list,
      origin: "server",
    });
  }, []);

  /** Fetch the active session and, when its fingerprint differs from what is
   *  cached, its questions too. */
  const refresh = useCallback(async () => {
    const supabase = getSupabase();
    if (!supabase) return;
    setLoading(true);
    try {
      const { data, error: sessionError } = await supabase
        .from("submission_sessions")
        .select("*")
        .eq("room", SUBMISSION_ROOM)
        .eq("active", true)
        .limit(1);
      if (sessionError) throw sessionError;
      const row = (data ?? [])[0] as
        | {
            id: string;
            room: string;
            name: string;
            pack_hash: string;
            question_count: number;
            published_by: string;
            created_at: string;
          }
        | undefined;
      if (!row) {
        setError(null);
        // Nobody has published yet; a locally imported pack stays usable.
        setOrigin((current) => (current === "server" ? "cache" : current));
        return;
      }
      const info: SessionInfo = {
        id: row.id,
        room: row.room,
        name: row.name,
        packHash: row.pack_hash,
        questionCount: row.question_count,
        publishedBy: row.published_by,
        createdAt: row.created_at,
      };
      if (cached.current?.sessionId === info.id && cached.current.packHash === info.packHash) {
        // Same pack we already hold: skip downloading every question again.
        setSession(info);
        setOrigin("server");
        return;
      }
      const { data: questionRows, error: questionError } = await supabase
        .from("session_questions")
        .select("*")
        .eq("session_id", info.id)
        .order("ordinal", { ascending: true });
      if (questionError) throw questionError;
      const list = recordsToQuestions((questionRows ?? []) as QuestionRecord[]);
      cached.current = {
        sessionId: info.id,
        sessionName: info.name,
        packHash: info.packHash,
        questions: list,
        origin: "server",
      };
      adopt(info, list);
      setError(null);
    } catch (fetchError) {
      // Offline or misconfigured: the cached pack keeps the console working.
      setError(describeSupabaseError(fetchError));
      setOrigin((current) => (current === "none" ? "none" : "cache"));
    } finally {
      setLoading(false);
    }
  }, [adopt]);

  useEffect(() => {
    if (!shared) {
      setLoading(false);
      return;
    }
    void refresh();
    const supabase = getSupabase();
    if (!supabase) return;
    // A teammate publishing a new pack must land on every screen without
    // anyone re-importing the zip by hand.
    const channel = supabase
      .channel(`sessions:${SUBMISSION_ROOM}`)
      .on(
        "postgres_changes",
        { event: "*", schema: "public", table: "submission_sessions" },
        () => void refresh(),
      )
      .subscribe();
    return () => {
      void supabase.removeChannel(channel);
    };
  }, [shared, refresh]);

  const publish = useCallback(
    async (list: ImportedQuestion[], name: string) => {
      const supabase = getSupabase();
      const hash = packHash(list);
      if (!supabase) {
        // No project configured: apply here rather than fail, and let the panel
        // say the pack never left this machine.
        cached.current = { sessionId: null, sessionName: name, packHash: hash, questions: list, origin: "local" };
        saveCache(cached.current);
        setQuestions(list);
        setOrigin("local");
        return;
      }
      const { data, error: publishError } = await supabase.rpc("publish_question_pack", {
        p_room: SUBMISSION_ROOM,
        p_name: name,
        p_pack_hash: hash,
        p_published_by: user,
        p_questions: canonicalQuestions(list),
      });
      if (publishError) throw publishError;
      const row = (Array.isArray(data) ? data[0] : data) as {
        id: string;
        room: string;
        name: string;
        pack_hash: string;
        question_count: number;
        published_by: string;
        created_at: string;
      };
      const info: SessionInfo = {
        id: row.id,
        room: row.room,
        name: row.name,
        packHash: row.pack_hash,
        questionCount: row.question_count,
        publishedBy: row.published_by,
        createdAt: row.created_at,
      };
      cached.current = {
        sessionId: info.id,
        sessionName: info.name,
        packHash: info.packHash,
        questions: list,
        origin: "server",
      };
      adopt(info, list);
      setError(null);
    },
    [adopt],
  );

  return { session, questions, origin, loading, error, shared, publish };
}
