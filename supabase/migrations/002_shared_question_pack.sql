-- Question pack as shared state, so `question_id` means one thing team-wide.
--
-- Submissions were already shared while the pack stayed in each browser's
-- localStorage. That combination is unsafe during a contest: machine B could
-- hold yesterday's pack, so a row saying `question_id = query-p1-4-trake` would
-- be read against a different question than the one machine A answered. The
-- pack therefore moves to the same place the answers live, and localStorage is
-- demoted from source of truth to offline cache.
--
-- Run after 001_shared_submission.sql.

create table if not exists public.submission_sessions (
  id uuid primary key default gen_random_uuid(),
  room text not null default 'aic26',
  name text not null,
  -- Content fingerprint of the parsed pack; a client compares it against its
  -- cache to know whether to re-download, without diffing every question.
  pack_hash text not null,
  question_count integer not null default 0,
  active boolean not null default true,
  published_by text not null default 'unknown',
  created_at timestamptz not null default now()
);

-- One live pack per room: practice and the real round can coexist as rows but
-- never both be current, which is what kept them from mixing.
create unique index if not exists submission_sessions_one_active
  on public.submission_sessions (room) where active;

create table if not exists public.session_questions (
  session_id uuid not null references public.submission_sessions (id) on delete cascade,
  question_id text not null,
  ordinal integer not null,
  kind text not null,
  query_type text not null,
  text text not null default '',
  event_count integer,
  primary key (session_id, question_id)
);

create index if not exists session_questions_session_idx
  on public.session_questions (session_id, ordinal);

-- Answers belong to the session that defined their questions, so an export can
-- never pair a row with a question from another pack.
alter table public.submissions
  add column if not exists session_id uuid references public.submission_sessions (id) on delete set null;
create index if not exists submissions_session_idx
  on public.submissions (session_id, question_id);

-- Publishing must be atomic: a half-written pack would leave four machines
-- reading a session whose questions do not all exist yet.
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

alter table public.submission_sessions replica identity full;
alter table public.session_questions replica identity full;
do $$
begin
  alter publication supabase_realtime add table public.submission_sessions;
exception when duplicate_object then null;
end;
$$;
do $$
begin
  alter publication supabase_realtime add table public.session_questions;
exception when duplicate_object then null;
end;
$$;

alter table public.submission_sessions enable row level security;
alter table public.session_questions enable row level security;

drop policy if exists sessions_all on public.submission_sessions;
drop policy if exists session_questions_all on public.session_questions;

-- Same trade as 001: the publishable key is in the browser, so `anon` is the
-- role the app runs as. Swap to `authenticated` after enabling Supabase Auth.
create policy sessions_all on public.submission_sessions
  for all to anon using (true) with check (true);
create policy session_questions_all on public.session_questions
  for all to anon using (true) with check (true);

grant select, insert, update, delete on public.submission_sessions to anon;
grant select, insert, update, delete on public.session_questions to anon;
grant execute on function public.publish_question_pack(text, text, text, text, jsonb) to anon;
