-- Publishing the same pack twice must not orphan the team's answers.
--
-- `publish_question_pack` always inserted a new session, so re-importing the
-- identical ZIP produced a second session with the same `pack_hash`. Answers
-- carry `session_id`, and the client reads only the active session — so every
-- answer written before the re-import silently vanished from the table (still in
-- the database, attached to the previous session id). This project already has
-- two `8f8c4e85` sessions from exactly that.
--
-- Re-publishing identical content is now a no-op: the active session is returned
-- unchanged, answers included. A pack whose content differs still creates a new
-- session, which is the behaviour that keeps a rehearsal out of the real round.
--
-- Run after 002_shared_question_pack.sql.

create or replace function public.publish_question_pack(
  p_room text,
  p_name text,
  p_pack_hash text,
  p_published_by text,
  p_questions jsonb
)
returns public.submission_sessions
language plpgsql
as $$
declare
  session public.submission_sessions;
begin
  -- Same content as what is already live: keep the session, keep the answers.
  select * into session
  from public.submission_sessions
  where room = p_room and active and pack_hash = p_pack_hash
  limit 1;

  if found then
    return session;
  end if;

  update public.submission_sessions set active = false where room = p_room and active;

  insert into public.submission_sessions (room, name, pack_hash, question_count, active, published_by)
  values (p_room, p_name, p_pack_hash, jsonb_array_length(p_questions), true, p_published_by)
  returning * into session;

  insert into public.session_questions
    (session_id, question_id, ordinal, kind, query_type, text, event_count)
  select
    session.id,
    item ->> 'question_id',
    (item ->> 'ordinal')::integer,
    item ->> 'kind',
    item ->> 'query_type',
    coalesce(item ->> 'text', ''),
    nullif(item ->> 'event_count', '')::integer
  from jsonb_array_elements(p_questions) as item;

  return session;
end;
$$;

grant execute on function public.publish_question_pack(text, text, text, text, jsonb) to anon;
