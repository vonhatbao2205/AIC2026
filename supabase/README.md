# Shared Submission tab

Five operators run the retrieval console on their own machines. Search stays
local — Elastic, Milvus, PE, OCR, speech and every image or video are untouched.
Only the Submission tab is shared, and only as text: one row is the CSV line plus
the ids needed to preview it.

```
local UI ×5  ──submit──►  submissions (Postgres)  ──realtime──►  Submission tab ×5
```

## One-time setup (once for the whole team)

1. Supabase Dashboard → **SQL Editor** → run, in order:
   - `supabase/migrations/001_shared_submission.sql` — the answers table,
     indexes, realtime publication, `revision` trigger and RLS policies.
   - `supabase/migrations/002_shared_question_pack.sql` — sessions, the shared
     question pack and the atomic `publish_question_pack` function.
   - `supabase/migrations/003_idempotent_publish.sql` — re-publishing the same
     pack keeps the session, and with it every answer already written to it.
   - `supabase/migrations/004_reorder_submissions.sql` — `reorder_submissions`,
     which rewrites a question's ranking in one statement. **Optional**: without
     it the console falls back to one write per moved row, which works but is
     slower and shows the reorder arriving piecemeal on the other screens.
2. Check **Database → Replication** lists `submissions`, `submission_sessions`
   and `session_questions` in `supabase_realtime`.
3. Paste `supabase/verify.sql` into the SQL Editor — every row must come back ✅.

## Per machine

```bash
cp frontend/.env.local.example frontend/.env.local
```

```env
VITE_SUPABASE_URL=https://<project-ref>.supabase.co
VITE_SUPABASE_PUBLISHABLE_KEY=sb_publishable_...
VITE_SUBMISSION_ROOM=aic26-p1   # identical on all five machines
VITE_SUBMISSION_USER=Bao        # different on each machine
```

Restart `npm run dev`. The Submission tab header shows `● Live` when the
subscription is up, `● N chờ đồng bộ` while writes are queued, and
`● local (chưa cấu hình Supabase)` when the two keys are missing — in which case
everything still works, just privately, exactly as before.

`VITE_SUBMISSION_ROOM` is what separates a rehearsal from the real round. Two
machines on different rooms will not see each other and will not error.

## The question pack is shared too

Answers were shared while the pack sat in each browser's localStorage — an unsafe
pair. Machine B could hold yesterday's pack, so a row saying
`question_id = query-p1-4-trake` would be read against a different question than
the one machine A answered. Both now live in the same session.

One person imports the ZIP and presses **⇪ Publish cho cả team**. Everyone else's
console picks the pack up over realtime; nobody re-imports by hand. There is
deliberately no per-machine import: it was the one way two clients could end up
answering different questions under the same `question_id`. Rehearse by
publishing into a different `VITE_SUBMISSION_ROOM` instead.

The Submission header shows the live session and its pack fingerprint:

```
⛭ query-p1 · 2026-08-18 · 3f9a1c04     ● Live
```

If a client cannot reach the server it keeps working from its cached pack and
says so — `⚠ Đang dùng gói câu hỏi từ cache`. With no Supabase configured at all,
publishing applies the pack to that machine and the panel says so, because the
console still has to run on a lone laptop.

Answers carry the `session_id` of the pack that defined their questions, so a
practice run and the real round can never be exported together.

## Using it

The table is deliberately textual: a hundred answers must not mean a hundred
image requests on five machines.

| key | action |
|---|---|
| `↑` `↓` | move between rows |
| `Alt`+`↑` `↓` | move the selected ANSWER one rank |
| `←` `→` | move between events of a TRAKE row |
| `P` | load and show the keyframe picture of the selected frame |
| `V` | open the video at that frame to re-pick it |
| `Delete` | remove the row |
| `Ctrl+Z` | undo the last action, reorders included |
| `Ctrl+E` | export `submission.zip` |

In the video editor, pausing only produces a *draft*. Nothing is written until
**Dùng frame chính xác** (keeps the raw paused frame) or **Snap keyframe gần
nhất** (moves to the nearest extracted keyframe, so a picture exists). Pushing
every pause would make the row jump on four other screens.

`frames` is the only value the CSV is built from. `keyframe_ids` is provenance:
it is null for a frame taken straight off the video, and the preview says
`RAW VIDEO FRAME` rather than showing a neighbouring keyframe that is not what
would be exported.

## Rank is an edit, so reordering is a write

The preliminary round scores `Final = (R@1 + R@5 + R@20 + R@50 + R@100) / 5` — a
max over the first *k* answers — so the same frame at rank 1 scores the whole
query and at rank 100 scores a fifth of it. Dragging the `⠿` grip of rank 7 onto
rank 1 (or pressing `Alt`+`↑`) is therefore the same class of change as
re-picking a frame: it is written to Postgres, it appears on every teammate's
screen, and `Ctrl+Z` puts it back.

There is no `rank` column. `created_at` **is** the order key — `sortRows` sorts by
it and the CSV is written in that order — so a reorder rewrites `created_at` on
the rows the move passes, and nothing else. Three consequences worth knowing:

- **Only the rows that move are written.** Rank 7 → rank 1 is seven timestamps,
  not a hundred.
- **The instants are permuted, not replaced.** The answers keep the same slice of
  the table's timeline instead of jumping past every other question's rows.
- **Ties are broken on the way through.** Two rows sharing a millisecond left the
  uuid tiebreak deciding rank 1; a reorder normalises the whole question, so the
  order it produces is the order that comes back on the next load.

The write touches the ordering column alone, so it can never clobber the frames
or the answer text a teammate is editing on a row it moves past. It does bump
`revision` on those rows — the server's copies are merged straight back, which is
what keeps a content edit queued behind the drag from being reported as a
conflict nobody caused.

One thing it does not survive: pressing **✨ sinh lại** on a question rebuilds the
generated block with fresh timestamps, so hand-picked answers return to the head
of that question. That is the documented rule ("dòng bạn tự chấm luôn đứng
trước"), not the reorder failing — reorder after regenerating, not before.

## Concurrency

Each row carries a `revision`. An update states the revision it read; if someone
else wrote first the update matches zero rows, the panel keeps the server's
version and warns instead of silently overwriting a teammate.

Writes are optimistic and queued: a row appears immediately and is retried until
it lands, so a few seconds of bad contest wifi cannot lose an answer.

## Security, stated plainly

The publishable key ships inside the browser bundle by design, and the policies
in the migration grant the `anon` role full CRUD on `submissions`. **Anyone
holding that key can read and write this table.** That is an accepted trade for a
private, five-person contest — not a general posture.

To tighten it: enable Supabase Auth, create five accounts, and change `to anon`
to `to authenticated` in the four policies. No schema or app change is needed.
After the contest, rotate the key in the Supabase dashboard.

Tests never talk to this project: `isSupabaseConfigured()` refuses to build a
client in test mode, because Vite loads `.env.local` for the test run too.
