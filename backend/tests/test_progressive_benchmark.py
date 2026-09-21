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


def options(**kwargs):
    from types import SimpleNamespace
    return SimpleNamespace(**{'profile': None, 'suite': None, 'model_revision': None,
        'dataset_revision': None, 'gt': b.GT, 'seed': 20260921, 'deep_top_k': 800,
        'reuse_plans': None, 'split': 'dev', 'dev_run': None, 'limit': 1, 'timeout': 30, **kwargs})


def test_profiles_revisions_and_stateless_control():
    _, cfg = b.profile_config(options(profile='pe-qwen', model_revision=['pe=rev1', 'qwen3_vl=rev2']))
    assert cfg.image_models == ['pe', 'qwen3_vl']
    assert cfg.model_revisions == {'pe': 'rev1', 'qwen3_vl': 'rev2'}
    assert cfg.hybrid is False
    assert 'dual_view' in b.SUITES['main'] and 'dual_view' in b.SUITES['extended']
    assert b.SUITES['replication'] == ('cumulative', 'hint_rrf', 'phm_no_rescue', 'phm')
    assert b.profile_config(options(model_revision=['legacy-pe-rev']))[1].model_revisions == {'pe': 'legacy-pe-rev'}
    for revisions in [['pe=a', 'pe=b'], ['qwen3_vl=q'], ['pe= '], ['unknown=x']]:
        with pytest.raises(ValueError, match='Model revisions'):
            b.profile_config(options(model_revision=revisions))


@pytest.mark.asyncio
async def test_preflight_requires_every_selected_encoder_and_index():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    service = SimpleNamespace(s=SimpleNamespace(mock_mode=False),
        pe=SimpleNamespace(health=AsyncMock(return_value={'ok': True})),
        qwen3_vl=SimpleNamespace(health=AsyncMock(return_value={'ok': False, 'error': 'SECRET_URL'})),
        milvus=SimpleNamespace(health=AsyncMock(return_value={'ok': True}),
                              health_qwen_image=AsyncMock(return_value={'ok': True})))
    pe = await b.preflight(service, ['pe'], 1)
    assert all(v['ok'] for v in pe.values())
    service.qwen3_vl.health.assert_not_called()
    both = await b.preflight(service, ['pe', 'qwen3_vl'], 1)
    assert not both['qwen_encoder']['ok'] and both['qwen_index']['ok']
    assert 'SECRET' not in str(both)
    service.pe.health.return_value = {'ok': True, 'mode': 'mock'}
    assert not (await b.preflight(service, ['pe'], 1))['pe_encoder']['ok']


@pytest.mark.asyncio
@pytest.mark.parametrize('profile,suite,count', [('pe', 'extended', 8), ('pe-qwen', None, 4)])
async def test_profile_end_to_end_frozen_replay(settings, tmp_path, profile, suite, count):
    import json
    args = options(profile=profile, suite=suite, lock=tmp_path / 'lock.json', out=tmp_path / 'run')
    await b.prepare(args)
    await b.run(args)
    report = json.loads((args.out / 'summary.json').read_text())
    assert report['healthy'] and report['profile'] == profile
    assert not report['complete']  # A one-query pilot cannot unlock evaluation.
    records = [json.loads(s) for s in (args.out / 'records.jsonl').read_text().splitlines()]
    assert len(records) == count * 3
    for record in records:
        assert set(record['channel_status']['delta']) == {b.CHANNELS[m] for m in b.PROFILES[profile]}
    if profile == 'pe':
        dual = next(r for r in records if r['method'] == 'dual_view' and r['turn'] == 3)
        trace = json.loads((args.out / dual['trace']).read_text())
        assert dual['budget']['local_index_calls'] == 0
        for group in trace['snapshot']['groups']:
            assert all(e['status'] == 'not_used' for e in group['progressive']['hint_evidence'][:-1])
    # Depth sweep reuses the exact parser plans, even if parser behavior later differs.
    next_args = options(profile=profile, suite=suite, lock=tmp_path / 'depth400.json',
                        reuse_plans=args.lock, deep_top_k=400)
    await b.prepare(next_args)
    first, second = json.loads(args.lock.read_text()), json.loads(next_args.lock.read_text())
    assert first['plans'] == second['plans'] and first['split'] == second['split']


def test_depth_selection_uses_only_comparable_live_dev_latencies(tmp_path):
    import json
    directories = []
    for depth, latency in [(200, 100), (400, 300), (800, 700), (1000, 1000)]:
        directory = tmp_path / str(depth)
        lock = {'deep_top_k': depth, 'plans': {'identical': True}}
        checksum = b.digest(lock)
        b.write_new(directory / 'manifest.json', {'lock': lock, 'lock_sha256': checksum, 'split': 'dev'})
        b.write_new(directory / 'summary.json', {'split': 'dev', 'complete': True, 'healthy': True,
            'mock': False, 'lock_sha256': checksum, 'metrics': {
                'phm': {'latency_p50_ms': 650}, 'cumulative_deep': {'latency_p50_ms': latency}}})
        directories.append(directory)
    selected = b.select_depth(directories)
    assert selected['selected_deep_top_k'] == 800 and selected['absolute_gap_ms'] == 50
    report_path = directories[0] / 'summary.json'
    report = json.loads(report_path.read_text())
    report['split'] = 'eval'
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='live dev'):
        b.select_depth(directories)


@pytest.mark.asyncio
async def test_run_stops_before_output_when_selected_model_preflight_fails(settings, tmp_path, monkeypatch):
    from unittest.mock import AsyncMock
    args = options(profile='pe-qwen', lock=tmp_path / 'lock.json', out=tmp_path / 'run')
    await b.prepare(args)
    monkeypatch.setattr(b, 'preflight', AsyncMock(return_value={'qwen_encoder': {'ok': False}}))
    with pytest.raises(ValueError, match='qwen_encoder'):
        await b.run(args)
    assert not args.out.exists()


@pytest.mark.asyncio
async def test_eval_requires_qwen_revision_not_just_pe(settings, tmp_path, monkeypatch):
    import json
    from dataclasses import replace
    args = options(profile='pe-qwen', model_revision=['pe=pe-rev'], dataset_revision='index-rev',
                   lock=tmp_path / 'lock.json', out=tmp_path / 'eval')
    await b.prepare(args)
    lock = json.loads(args.lock.read_text())
    lock.pop('sha256')
    lock['mock'] = False
    lock['sha256'] = b.digest(lock)
    args.lock.write_text(json.dumps(lock))
    args.split, args.limit, args.dev_run = 'eval', None, tmp_path / 'dev'
    b.write_new(args.dev_run / 'summary.json', {'lock_sha256': lock['sha256'], 'split': 'dev',
                                              'complete': True, 'healthy': True})
    monkeypatch.setattr(b, 'get_settings', lambda: replace(settings, mock_mode=False))
    with pytest.raises(ValueError, match='every selected model revision'):
        await b.run(args)
    assert not args.out.exists()


def test_channel_health_catches_missing_selected_channel():
    from types import SimpleNamespace
    observation = SimpleNamespace(trace=SimpleNamespace(channel_status={'image_pe': 'ok'}))
    state = SimpleNamespace(observations=[observation], cumulative=observation)
    status, healthy = b.channel_health(state, ['pe', 'qwen3_vl'])
    assert not healthy and status['delta']['image_qwen'] == 'missing'
