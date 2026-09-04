-- Reordering a question's answers, as one atomic write.
--
-- Rank is scored: the preliminary round takes
--   Final = (R@1 + R@5 + R@20 + R@50 + R@100) / 5
-- over the answer list, and `created_at` is the column that decides that order
-- (see `sortRows` / `nextCreatedAt` in the frontend). Dragging an answer from
-- rank 7 to rank 1 therefore rewrites `created_at` on the rows it passes.
--
-- Doing that one PATCH per row would be N sequential round trips over contest
-- wifi, and a client that died halfway would leave the list in an order nobody
-- chose. This function does the whole permutation in a single statement, and
-- touches NO other column — so an ordering write can never clobber the frames
-- or the answer text a teammate is editing at the same moment.
--
-- The `submissions_touch` trigger still fires, so every moved row's `revision`
-- is bumped and every client's realtime subscription sees the new order. The
-- updated rows are returned for exactly that reason: the caller merges them back
-- to refresh the revisions it holds, otherwise its next content edit would state
-- a revision the server has already moved past and be reported as a conflict
-- nobody caused.
--
-- Run after 001_shared_submission.sql. The app falls back to per-row updates
-- when this function is absent, so an un-migrated project keeps working — just
-- one request per moved row.

create or replace function public.reorder_submissions(
  p_ids uuid[],
  p_created_at timestamptz[]
)
returns setof public.submissions
language plpgsql
as $$
declare
  n integer := coalesce(array_length(p_ids, 1), 0);
begin
  if n = 0 then
    return;
  end if;

  if n is distinct from coalesce(array_length(p_created_at, 1), 0) then
    raise exception 'reorder_submissions: % ids but % timestamps', n,
      coalesce(array_length(p_created_at, 1), 0);
  end if;

  -- A repeated id would make the update non-deterministic (Postgres picks one
  -- of the matching source rows), which is precisely the ambiguity this whole
  -- feature exists to remove. Refuse instead of guessing.
  if (select count(distinct t.id) from unnest(p_ids) as t(id)) is distinct from n then
    raise exception 'reorder_submissions: duplicate id in the plan';
  end if;

  -- Wrapped in a CTE because `return query` takes a query: a bare
  -- `return query update … returning` is not accepted by plpgsql.
  return query
  with moved as (
    update public.submissions as s
       set created_at = plan.created_at
      from unnest(p_ids, p_created_at) as plan(id, created_at)
     where s.id = plan.id
    returning s.*
  )
  select * from moved;
end;
$$;

-- Same trade as 001: the publishable key is in the browser, so `anon` is the
-- role the app runs as. The function is SECURITY INVOKER, so the RLS policies on
-- `submissions` still apply — this grants no reach the table did not already.
grant execute on function public.reorder_submissions(uuid[], timestamptz[]) to anon;
