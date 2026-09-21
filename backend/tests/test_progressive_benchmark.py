import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[2] / 'benchmarks/progressive/run_benchmark.py'
spec = importlib.util.spec_from_file_location('progressive_benchmark', path)
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


def test_split_keeps_transitive_multitarget_groups_together():
    rows = [{'query_id': q, 'targets': [{'video_id': v} for v in vs]}
            for q, vs in [('a', ['01']), ('b', ['02']), ('c', ['01', '02']),
                          ('d', ['03']), ('e', ['04'])]]
    split = b.split_groups(rows, 12)
    assert split['a'] == split['b'] == split['c']
    assert set(v['split'] for v in split.values()) == {'dev', 'eval'}
    assert split == b.split_groups(list(reversed(rows)), 12)


def test_moment_uses_preferred_frame_and_half_open_milliseconds():
    row = {'targets': [{'video_id': '01', 'target_intervals': [{'start_ms': 1000, 'end_ms': 2000}]}]}
    snap = {'history': [{'ranking': ['01']}], 'ranking_before_rescue': [],
            'groups': [{'video_id': '01', 'frames': [{'pts_time': 2}, {'pts_time': 1.5}]}]}
    assert b.score_snapshot(snap, row)['moment_hit@1'] == 0
    snap['groups'][0]['frames'][0]['pts_time'] = 1
    assert b.score_snapshot(snap, row)['moment_hit@1'] == 1


def test_metrics_average_queries_before_prefixes_and_keep_failures():
    records = []
    for q, ranks in [('a', [1]), ('b', [None, None, None])]:
        for turn, rank in enumerate(ranks, 1):
            snap = {'history': [{'ranking': ['target'] if rank else []}],
                    'ranking_before_rescue': [], 'groups': []}
            records.append({'method': 'cumulative', 'query_id': q, 'turn': turn,
                            'status': 'ok' if rank else 'TimeoutError', 'wall_ms': 10,
                            'metrics': b.score_snapshot(snap, {'targets': [{'video_id': 'target'}]})})
    summary = b.summarize(records)['cumulative']
    assert summary['mean_prefix_video_mrr'] == .5
    assert summary['timeout_rate'] == .75
    assert summary['queries'][1]['stable_correct_turn'] is None


def test_real_gt_split_and_prefix_isolation():
    rows = b.load_gt(b.GT)
    split = b.split_groups(rows, 20260921)
    videos = {s: {t['video_id'] for r in rows if split[r['query_id']]['split'] == s
                  for t in r['targets']} for s in ('dev', 'eval')}
    assert not videos['dev'] & videos['eval']
    for row in rows:
        ledger = b.ledger_for(row)
        assert len(ledger) == len(row['hints'])
        assert [h['cumulative_text'] for h in ledger] == [h['cumulative_text'] for h in row['hints']]


def test_paired_bootstrap_preserves_query_weight_and_group_unit():
    records = []
    for q, group in [('a', 'shared'), ('b', 'shared'), ('c', 'separate')]:
        for method, rr in [('cumulative', .25), ('phm', .75)]:
            records.append({'query_id': q, 'group': group, 'method': method, 'metrics': {'rr': rr}})
    result = b.paired_bootstrap(records, 12, samples=100)['phm-cumulative']
    assert result['delta'] == .5
    assert result['ci95'] == [.5, .5]


@pytest.mark.asyncio
async def test_run_rejects_modified_lock_before_creating_output(tmp_path):
    from types import SimpleNamespace
    lock = tmp_path / 'lock.json'
    b.write_new(lock, {'sha256': 'invalid'})
    with pytest.raises(ValueError, match='Lock or ground truth changed'):
        await b.run(SimpleNamespace(lock=lock))


def test_output_never_overwrites_previous_artifacts(tmp_path):
    path = tmp_path / 'artifact.json'
    b.write_new(path, {'first': True})
    with pytest.raises(FileExistsError):
        b.write_new(path, {'first': False})
