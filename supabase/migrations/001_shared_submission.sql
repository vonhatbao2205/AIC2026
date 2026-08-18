-- Shared submission table for the AIC26 retrieval console.
--
-- Five operators run the retrieval UI locally and search independently; only the
-- Submission tab is shared. Nothing here stores media: a row is the CSV line
-- plus enough metadata to preview it (`keyframe_ids`, `pts_times`), so the sync
-- payload stays a few strings and integers per answer.
--
-- Run once against the project:
--   Supabase Dashboard -> SQL Editor -> paste -> Run
-- or, with the CLI:  supabase db push
--
-- SECURITY MODEL — read this before using it outside a private team.
-- The frontend talks to PostgREST with the publishable (anon) key, which ships
-- inside the browser bundle by design. The policies below therefore grant the
-- `anon` role full CRUD on this one table: anybody holding that key can read and
-- write submissions. That is an accepted trade for a 5-person contest on a
-- private repo, NOT a general-purpose security posture. To tighten it later,
-- enable Supabase Auth, give each member an account, and replace `to anon` with
-- `to authenticated` in the four policies at the bottom (the app already stores a
-- display name per client, so no schema change is needed).

create extension if not exists pgcrypto;

create table if not exists public.submissions (
  id uuid primary key default gen_random_uuid(),

  -- Logical room, so two runs (or a rehearsal and the real contest) never mix.
  room text not null default 'aic26',
  question_id text not null,
  retrieval_database text not null default 'btc',

  video_id text not null default '',

  -- Authoritative for the CSV export: one frame for KIS/QA, one per event for
  -- TRAKE. A frame may be a raw paused video frame with no extracted keyframe,
  -- which is why the keyframe id below is metadata and may be null.
  frames integer[] not null default '{}',

  -- Preview/provenance only, positionally parallel to `frames`.
  keyframe_ids text[] not null default '{}',
  pts_times double precision[] not null default '{}',

  answer text not null default '',

  submitted_by text not null default 'unknown',
  source text not null default 'search',

  -- Optimistic concurrency: an update must state the revision it read.
  revision integer not null default 1,

  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists submissions_room_question_idx
  on public.submissions (room, question_id, created_at);
create index if not exists submissions_room_updated_idx
  on public.submissions (room, updated_at desc);

-- Every write bumps the revision, so a client that read revision N and writes
-- with `.eq('revision', N)` updates zero rows once someone else got there first.
create or replace function public.touch_submission()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  new.revision = old.revision + 1;
  return new;
end;
$$;

drop trigger if exists submissions_touch on public.submissions;
create trigger submissions_touch
  before update on public.submissions
  for each row execute function public.touch_submission();

-- Realtime: clients subscribe to postgres_changes filtered by room.
alter table public.submissions replica identity full;
do $$
begin
  alter publication supabase_realtime add table public.submissions;
exception
  when duplicate_object then null;
end;
$$;

alter table public.submissions enable row level security;

drop policy if exists submissions_read on public.submissions;
drop policy if exists submissions_insert on public.submissions;
drop policy if exists submissions_update on public.submissions;
drop policy if exists submissions_delete on public.submissions;

create policy submissions_read   on public.submissions for select to anon using (true);
create policy submissions_insert on public.submissions for insert to anon with check (true);
create policy submissions_update on public.submissions for update to anon using (true) with check (true);
create policy submissions_delete on public.submissions for delete to anon using (true);

grant select, insert, update, delete on public.submissions to anon;
