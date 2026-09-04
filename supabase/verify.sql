-- Paste into the Supabase SQL Editor after running 001, 002, 003 and 004.
-- Every row must come back ✅. This checks what the app actually depends on at
-- runtime, not just that the migrations parsed.
with checks as (
  select 'bảng submissions' as item,
         to_regclass('public.submissions') is not null as ok,
         coalesce(to_regclass('public.submissions')::text, 'THIẾU — chạy 001') as detail
  union all
  select 'bảng submission_sessions',
         to_regclass('public.submission_sessions') is not null,
         coalesce(to_regclass('public.submission_sessions')::text, 'THIẾU — chạy 002')
  union all
  select 'bảng session_questions',
         to_regclass('public.session_questions') is not null,
         coalesce(to_regclass('public.session_questions')::text, 'THIẾU — chạy 002')
  union all
  select 'cột submissions.session_id',
         exists (select 1 from information_schema.columns
                 where table_schema = 'public' and table_name = 'submissions'
                   and column_name = 'session_id'),
         'gắn đáp án vào đúng gói câu hỏi'
  union all
  select 'index 1 pack active / room',
         exists (select 1 from pg_indexes
                 where schemaname = 'public' and indexname = 'submission_sessions_one_active'),
         'chặn practice và vòng thi cùng active'
  union all
  select 'function publish_question_pack',
         exists (select 1 from pg_proc p join pg_namespace n on n.oid = p.pronamespace
                 where n.nspname = 'public' and p.proname = 'publish_question_pack'),
         'publish atomic'
  union all
  select 'function reorder_submissions',
         exists (select 1 from pg_proc p join pg_namespace n on n.oid = p.pronamespace
                 where n.nspname = 'public' and p.proname = 'reorder_submissions'),
         'kéo-thả đổi thứ hạng trong 1 câu lệnh (thiếu thì app tự ghi từng dòng)'
  union all
  select 'trigger revision',
         exists (select 1 from pg_trigger where tgname = 'submissions_touch' and not tgisinternal),
         'optimistic concurrency'
  union all
  select 'realtime publication (cần 3 bảng)',
         (select count(*) from pg_publication_tables
          where pubname = 'supabase_realtime' and schemaname = 'public'
            and tablename in ('submissions', 'submission_sessions', 'session_questions')) = 3,
         coalesce((select string_agg(tablename, ', ' order by tablename)
                   from pg_publication_tables
                   where pubname = 'supabase_realtime' and schemaname = 'public'
                     and tablename in ('submissions', 'submission_sessions', 'session_questions')),
                  'chưa bảng nào')
  union all
  select 'RLS bật đủ 3 bảng',
         (select count(*) from pg_class c join pg_namespace n on n.oid = c.relnamespace
          where n.nspname = 'public' and c.relrowsecurity
            and c.relname in ('submissions', 'submission_sessions', 'session_questions')) = 3,
         'row level security'
  union all
  select 'policy cho anon (cần ≥ 6)',
         (select count(*) from pg_policies where schemaname = 'public'
          and tablename in ('submissions', 'submission_sessions', 'session_questions')) >= 6,
         (select count(*)::text || ' policy'
          from pg_policies where schemaname = 'public'
          and tablename in ('submissions', 'submission_sessions', 'session_questions'))
)
select case when ok then '✅' else '❌' end as status, item, detail
from checks
order by ok, item;
