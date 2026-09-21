"""Locked progressive replay. Run --help from the repository root."""
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import math
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'backend'))
from app.config import get_settings
from app.progressive.core import derive_ledger
from app.progressive.engine import ProgressiveEngine
from app.progressive.models import HintInput, ProgressiveConfig
from app.services.search_service import SearchService

GT = Path(__file__).parent / 'annotations/ground_truth.jsonl'
METHODS = ('cumulative', 'latest', 'hint_rrf', 'dual_view', 'phm_no_rescue', 'phm', 'phm_arithmetic', 'cumulative_deep')
PROFILES = {'pe': ['pe'], 'pe-qwen': ['pe', 'qwen3_vl']}
SUITES = {
    'main': ('cumulative', 'hint_rrf', 'dual_view', 'phm_no_rescue', 'phm', 'cumulative_deep'),
    'replication': ('cumulative', 'hint_rrf', 'phm_no_rescue', 'phm'),
    'extended': METHODS,
}
COLLECTIONS = {'pe': 'milvus_image_collection', 'qwen3_vl': 'milvus_qwen3_vl_image_collection'}
CHANNELS = {'pe': 'image_pe', 'qwen3_vl': 'image_qwen'}


def profile_config(args):
    profile = args.profile or 'pe'
    revisions = {}
    for entry in args.model_revision or []:
        model, sep, revision = entry.partition('=')
        if not sep:  # Preserve the original PE-only CLI shorthand.
            model, revision = 'pe', entry
        if model not in PROFILES[profile] or not revision.strip() or model in revisions:
            raise ValueError('Model revisions must be unique MODEL=REV entries for selected models')
        revisions[model] = revision.strip()
    return profile, ProgressiveConfig(retrieval_database='infoshotpp', image_models=PROFILES[profile],
        translate=True, dataset_revision=args.dataset_revision, model_revisions=revisions)


async def preflight(service, models, timeout):
    """Check each selected encoder AND index; never export raw health error/URL data."""
    checks = {'pe_encoder': service.pe.health, 'pe_index': service.milvus.health}
    if 'qwen3_vl' in models:
        checks.update(qwen_encoder=service.qwen3_vl.health, qwen_index=service.milvus.health_qwen_image)

    async def check(name, health):
        try:
            result = await asyncio.wait_for(health(), timeout)
            mode = result.get('mode', 'live')
            return name, {'ok': bool(result.get('ok')) and (service.s.mock_mode or mode != 'mock'),
                          'mode': mode}
        except Exception as exc:
            return name, {'ok': False, 'error_type': type(exc).__name__}

    return dict(await asyncio.gather(*(check(name, health) for name, health in checks.items())))


def channel_health(state, models):
    observations = [state.observations[-1], state.cumulative]
    statuses = {view: {CHANNELS[m]: obs.trace.channel_status.get(CHANNELS[m], 'missing') for m in models}
                for view, obs in zip(('delta', 'cumulative'), observations)}
    return statuses, all(s == 'ok' for view in statuses.values() for s in view.values())


def source_digest():
    files = sorted((ROOT / 'backend/app').rglob('*.py')) + [Path(__file__).resolve()]
    return digest({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as out:
        json.dump(value, out, ensure_ascii=False, indent=2, allow_nan=False)


def load_gt(path):
    rows = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    selected = [r for r in rows if r.get('eligible_for_benchmark') is True]
    if not selected or len({r['query_id'] for r in selected}) != len(selected):
        raise ValueError('Empty dataset or duplicate query IDs')
    for r in selected:
        if r['review_status'] != 'approved' or r['task_type'] != 'TKIS' or r['time_unit'] != 'ms':
            raise ValueError('Only approved TKIS millisecond annotations are allowed')
        if not r['hints'] or not any(t['target_intervals'] for t in r['targets']):
            raise ValueError('Missing hints or accepted intervals')
    return selected


def split_groups(rows, seed):
    # Connected components prevent leakage even for multi-target queries.
    components = []
    for r in sorted(rows, key=lambda r: r['query_id']):
        videos = {t['video_id'] for t in r['targets']}
        ids = {r['query_id']}
        remaining = []
        for old_videos, old_ids in components:
            if videos & old_videos:
                videos |= old_videos
                ids |= old_ids
            else:
                remaining.append((old_videos, old_ids))
        components = remaining + [(videos, ids)]
    components.sort(key=lambda c: sorted(c[1]))
    random.Random(seed).shuffle(components)
    ndev = max(1, round(len(components) * .2))
    if ndev == len(components):
        raise ValueError('Need at least two disjoint target-video groups')
    return {qid: {'split': 'dev' if i < ndev else 'eval', 'group': min(ids)}
            for i, (_, ids) in enumerate(components) for qid in ids}


def ledger_for(row):
    return derive_ledger([HintInput(hint_id=h['hint_id'], raw_text=h['delta_text'], source=h['source'])
                          for h in row['hints']])


def rank_of(ranking, targets):
    return next((i for i, v in enumerate(ranking, 1) if v in targets), None)


def score_snapshot(snapshot, row):
    targets = {t['video_id'] for t in row['targets']}
    rank = rank_of(snapshot['history'][-1]['ranking'], targets)
    before = rank_of(snapshot['ranking_before_rescue'], targets)
    # One preferred moment per displayed video; never scan historical frames for an oracle hit.
    moment = None
    for i, g in enumerate(snapshot['groups'], 1):
        frames = g.get('frames', [])
        pts = frames[0].get('pts_time') if frames else None
        if pts is not None and any(t['video_id'] == g['video_id'] and any(
            iv['start_ms'] <= pts * 1000 < iv['end_ms'] for iv in t['target_intervals']) for t in row['targets']):
            moment = i
            break
    return {'video_rank': rank, 'rr': 1 / rank if rank else 0,
            **{f'video_hit@{k}': int(rank is not None and rank <= k) for k in (1, 5, 10)},
            **{f'moment_hit@{k}': int(moment is not None and moment <= k) for k in (1, 5, 10)},
            'rescue_recovery': int(before is None and rank is not None),
            'harmful_rescue': int(before is not None and (rank is None or rank > before))}


def summarize(records):
    output = {}
    for method in sorted({r['method'] for r in records}):
        subset = [r for r in records if r['method'] == method]
        query_scores, queries = {}, []
        for qid in sorted({r['query_id'] for r in subset}):
            turns = sorted((r for r in subset if r['query_id'] == qid), key=lambda r: r['turn'])
            query_scores[qid] = statistics.mean(r['metrics']['rr'] for r in turns)
            correct = [r['metrics']['video_rank'] == 1 for r in turns]
            first = next((i + 1 for i, ok in enumerate(correct) if ok), None)
            stable = next((i + 1 for i in range(len(correct)) if all(correct[i:])), None)
            tops = [r.get('top1') for r in turns]
            queries.append({'query_id': qid, 'mean_prefix_mrr': query_scores[qid],
                            'first_correct_turn': first, 'stable_correct_turn': stable,
                            'top1_churn': sum(a != b for a, b in zip(tops, tops[1:])) / max(1, len(tops) - 1)})
        latencies = sorted(r['wall_ms'] for r in subset)
        output[method] = {'queries': queries, 'mean_prefix_video_mrr': statistics.mean(query_scores.values()),
            'prefixes': {str(t): {key: statistics.mean(r['metrics'][key] for r in subset if r['turn'] == t)
                                 for key in ('rr', 'video_hit@1', 'video_hit@5', 'video_hit@10',
                                             'moment_hit@1', 'moment_hit@5', 'moment_hit@10')}
                         for t in sorted({r['turn'] for r in subset})},
            'latency_p50_ms': statistics.median(latencies),
            'latency_p95_ms': latencies[math.ceil(.95 * len(latencies)) - 1],
            'failure_rate': sum(r['status'] != 'ok' for r in subset) / len(subset),
            'timeout_rate': sum(r['status'] == 'TimeoutError' for r in subset) / len(subset),
            'rescue_recovery': sum(r['metrics']['rescue_recovery'] for r in subset),
            'harmful_rescue': sum(r['metrics']['harmful_rescue'] for r in subset),
            'work_totals': {k: sum(r.get('budget', {}).get(k, 0) for r in subset)
                            for k in ('global_index_calls', 'local_index_calls', 'query_vectors', 'frames_returned')}}
    return output


def paired_bootstrap(records, seed, samples=10000):
    methods = sorted({r['method'] for r in records} - {'cumulative'})
    scores = {}
    for r in records:
        scores.setdefault((r['method'], r['query_id']), []).append(r['metrics']['rr'])
    groups = {}
    for r in records:
        groups.setdefault(r['group'], set()).add(r['query_id'])
    output = {}
    for method in methods:
        blocks = [[statistics.mean(scores[method, q]) - statistics.mean(scores['cumulative', q])
                   for q in sorted(qids)] for qids in groups.values()]
        rng = random.Random(seed)
        draws = []
        for _ in range(samples):
            values = [v for block in rng.choices(blocks, k=len(blocks)) for v in block]
            draws.append(statistics.mean(values))
        draws.sort()
        output[method + '-cumulative'] = {'delta': statistics.mean(v for b in blocks for v in b),
            'ci95': [draws[int(.025 * samples)], draws[int(.975 * samples)]],
            'unit': 'target-video connected component', 'samples': samples}
    return output


class FrozenParser:
    def __init__(self, plans):
        self.plans = plans

    async def parse(self, text, *args, **kwargs):
        return copy.deepcopy(self.plans[text])


async def prepare(args):
    rows = load_gt(args.gt)
    profile, config = profile_config(args)
    suite = args.suite or ('main' if profile == 'pe' else 'replication')
    service = SearchService(get_settings().for_retrieval_database(config.retrieval_database))
    plans = {}
    if args.reuse_plans:
        previous = json.loads(args.reuse_plans.read_text())
        checksum = previous.pop('sha256')
        if (digest(previous) != checksum or previous['gt_sha256'] != hashlib.sha256(args.gt.read_bytes()).hexdigest()
                or previous['source_sha256'] != source_digest() or previous['mock'] != service.s.mock_mode):
            raise ValueError('Reused parser plans require an intact lock with the same GT, source and mode')
        plans = copy.deepcopy(previous['plans'])
    for row in rows:
        for hint in ledger_for(row):
            for text in (hint['delta_text'], hint['cumulative_text']):
                if text not in plans:
                    parsed = await service.parser.parse(text, 'T-KIS', [], use_llm=False, translate=True)
                    if parsed.get('translation_failed'):
                        raise ValueError('Translation failed; refusing to lock fallback text')
                    plans[text] = parsed
    lock = {'version': 2, 'profile': profile, 'suite': suite,
            'gt_sha256': hashlib.sha256(args.gt.read_bytes()).hexdigest(),
            'source_sha256': source_digest(), 'seed': args.seed, 'split': split_groups(rows, args.seed), 'config': config.model_dump(),
            'plans': plans, 'deep_top_k': args.deep_top_k, 'methods': list(SUITES[suite]),
            'dataset_revision': config.dataset_revision, 'model_revisions': config.model_revisions,
            'mock': service.s.mock_mode,
            'collections': {COLLECTIONS[m]: getattr(service.s, COLLECTIONS[m]) for m in config.image_models}}
    lock['sha256'] = digest(lock)
    write_new(args.lock, lock)
    print(json.dumps({'lock': str(args.lock), 'queries': len(rows), 'split': {
        s: sum(v['split'] == s for v in lock['split'].values()) for s in ('dev', 'eval')}}))


def select_depth(run_dirs):
    """Use latency only, from four comparable complete dev runs; never inspect eval quality."""
    reference = None
    measurements, phm_latencies = {}, []
    for directory in run_dirs:
        report = json.loads((directory / 'summary.json').read_text())
        manifest = json.loads((directory / 'manifest.json').read_text())
        lock = copy.deepcopy(manifest['lock'])
        if (report['split'] != 'dev' or manifest['split'] != 'dev' or not report['complete']
                or not report['healthy'] or report['mock'] or report['lock_sha256'] != digest(lock)
                or manifest['lock_sha256'] != report['lock_sha256']):
            raise ValueError('Depth selection requires complete healthy live dev runs with valid manifests')
        depth = lock.pop('deep_top_k')
        if depth not in (200, 400, 800, 1000) or depth in measurements:
            raise ValueError('Supply one run for each depth: 200, 400, 800, 1000')
        if reference is not None and lock != reference:
            raise ValueError('Depth pilots must differ only in deep_top_k; reuse frozen parser plans')
        reference = lock
        measurements[depth] = report['metrics']['cumulative_deep']['latency_p50_ms']
        phm_latencies.append(report['metrics']['phm']['latency_p50_ms'])
    if set(measurements) != {200, 400, 800, 1000}:
        raise ValueError('Supply one run for each depth: 200, 400, 800, 1000')
    target = statistics.median(phm_latencies)
    selected = min(measurements, key=lambda d: (abs(measurements[d] - target), d))
    return {'selected_deep_top_k': selected, 'phm_reference_p50_ms': target,
            'cumulative_deep_p50_ms': measurements, 'absolute_gap_ms': abs(measurements[selected] - target),
            'rule': 'nearest median latency; median of four PHM p50s; shallower depth wins ties',
            'runs': [str(p) for p in run_dirs], 'matched_compute_claim': False}


async def run(args):
    lock = json.loads(args.lock.read_text())
    checksum = lock.pop('sha256')
    if digest(lock) != checksum or hashlib.sha256(args.gt.read_bytes()).hexdigest() != lock['gt_sha256']:
        raise ValueError('Lock or ground truth changed')
    if lock.get('version') != 2:
        raise ValueError('Prepare a version 2 lock with visual profile and model revisions')
    if (args.profile and args.profile != lock['profile']) or (args.suite and args.suite != lock['suite']):
        raise ValueError('Profile/suite must match the lock; prepare a separate lock for replication')
    if lock['source_sha256'] != source_digest():
        raise ValueError('Source changed; create a new lock and repeat development')
    if args.split == 'eval' and (not args.dev_run or args.limit):
        raise ValueError('Evaluation requires a completed --dev-run and disallows --limit')
    if args.dev_run:
        dev = json.loads((args.dev_run / 'summary.json').read_text())
        if dev['lock_sha256'] != checksum or dev['split'] != 'dev' or not dev['complete'] or not dev['healthy']:
            raise ValueError('Development must be complete, healthy and use the same lock')
    settings = get_settings().for_retrieval_database(lock['config']['retrieval_database'])
    if settings.mock_mode != lock['mock'] or (args.split == 'eval' and settings.mock_mode):
        raise ValueError('Mock/live mismatch or attempted mock evaluation')
    for key, value in lock['collections'].items():
        if getattr(settings, key) != value:
            raise ValueError('Collection configuration changed')
    models = lock['config']['image_models']
    if args.split == 'eval' and (not lock['dataset_revision'] or any(
            not lock['config']['model_revisions'].get(m, '').strip() for m in models)):
        raise ValueError('Evaluation requires attested dataset and every selected model revision in the lock')
    health = await preflight(SearchService(settings), models, min(args.timeout, 30))
    if not all(v['ok'] for v in health.values()):
        raise ValueError('Retrieval preflight failed: ' + ', '.join(k for k, v in health.items() if not v['ok']))
    rows = [r for r in load_gt(args.gt) if lock['split'][r['query_id']]['split'] == args.split]
    random.Random(lock['seed']).shuffle(rows)
    total = len(rows)
    if args.limit:
        rows = rows[:args.limit]
    args.out.mkdir(parents=True, exist_ok=False)
    write_new(args.out / 'manifest.json', {'lock_sha256': checksum, 'split': args.split,
        'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'working_tree_diff_sha256': hashlib.sha256(subprocess.check_output(['git', 'diff', 'HEAD'], cwd=ROOT)).hexdigest(),
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'lock': lock, 'preflight': health})
    records = []
    rng = random.Random(lock['seed'])
    for row in rows:
        methods = list(lock['methods'])
        rng.shuffle(methods)
        for method in methods:
            cfg = {**lock['config'], 'method': 'cumulative' if method == 'cumulative_deep' else method}
            if method == 'cumulative_deep':
                cfg['top_k'] = lock['deep_top_k']
            service = SearchService(settings)
            service.parser = FrozenParser(lock['plans'])
            engine = ProgressiveEngine(service, ProgressiveConfig(**cfg))
            state = None
            ledger = ledger_for(row)
            for turn in range(1, len(ledger) + 1):
                started = time.perf_counter()
                record = {'query_id': row['query_id'], 'group': lock['split'][row['query_id']]['group'],
                          'method': method, 'turn': turn, 'status': 'ok'}
                try:
                    state = await asyncio.wait_for(engine.run(ledger[:turn], state), args.timeout)
                    snapshot = state.snapshot
                    record['wall_ms'] = (time.perf_counter() - started) * 1000
                    record.update(metrics=score_snapshot(snapshot, row), budget=snapshot['revision_budget'],
                                  top1=state.rankings[-1][0] if state.rankings[-1] else None)
                    record['channel_status'], channels_ok = channel_health(state, models)
                    if snapshot['degraded'] or not channels_ok:
                        record['status'] = 'degraded'
                    trace_path = args.out / 'traces' / f'{rows.index(row):03d}-{method}-{turn}.json'
                    write_new(trace_path, engine.trace_export(state))
                    record['trace'] = str(trace_path.relative_to(args.out))
                except Exception as exc:
                    # Failed prefixes remain in every denominator; no stale snapshot scored as fresh.
                    record['status'] = type(exc).__name__
                    record['metrics'] = {'video_rank': None, 'rr': 0, 'rescue_recovery': 0, 'harmful_rescue': 0,
                        **{f'{kind}_hit@{k}': 0 for kind in ('video', 'moment') for k in (1, 5, 10)}}
                record.setdefault('wall_ms', (time.perf_counter() - started) * 1000)
                records.append(record)
                with (args.out / 'records.jsonl').open('a') as out:
                    out.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                print(f"{row['query_id']} {method} H{turn}: {record['status']}", flush=True)
    report = {'lock_sha256': checksum, 'profile': lock['profile'], 'suite': lock['suite'],
              'split': args.split, 'complete': len(rows) == total,
              'healthy': all(r['status'] == 'ok' for r in records), 'mock': settings.mock_mode,
              'moment_policy': 'first frame per displayed video; half-open accepted interval',
              'metrics': summarize(records), 'paired_bootstrap': paired_bootstrap(records, lock['seed'])}
    write_new(args.out / 'summary.json', report)


async def smoke(args):
    rows = load_gt(args.gt)
    split = split_groups(rows, args.seed)
    row = next(r for r in rows if split[r['query_id']]['split'] == 'dev')
    service = SearchService(get_settings().for_retrieval_database('infoshotpp'))
    if service.s.mock_mode:
        raise ValueError('Live smoke requires mock mode disabled')
    profile, config = profile_config(args)
    args.out.mkdir(parents=True, exist_ok=False)
    health = await preflight(service, config.image_models, min(args.timeout, 30))
    write_new(args.out / 'preflight.json', health)
    if not all(v['ok'] for v in health.values()):
        write_new(args.out / 'smoke.json', {'profile': profile, 'healthy': False, 'preflight': health,
            'results': [], 'benchmark': False, 'source_sha256': source_digest()})
        raise ValueError('Live smoke preflight failed: ' + ', '.join(k for k, v in health.items() if not v['ok']))
    engine = ProgressiveEngine(service, config)
    state = None
    ledger = ledger_for(row)
    results = []
    for turn in range(1, len(ledger) + 1):
        try:
            state = await asyncio.wait_for(engine.run(ledger[:turn], state), args.timeout)
            snap = state.snapshot
            write_new(args.out / f'trace-h{turn}.json', engine.trace_export(state))
            channels, channels_ok = channel_health(state, config.image_models)
            result = {'turn': turn, 'status': 'degraded' if snap['degraded'] or not channels_ok else 'ok',
                      'channel_status': channels,
                      'latency_ms': snap['revision_latency_ms'], 'budget': snap['revision_budget']}
        except Exception as exc:
            result = {'turn': turn, 'status': type(exc).__name__}
        results.append(result)
        print(json.dumps(result), flush=True)
    write_new(args.out / 'smoke.json', {'query_id': row['query_id'], 'split': 'dev',
        'profile': profile, 'config': config.model_dump(), 'preflight': health,
        'healthy': all(r['status'] == 'ok' for r in results),
        'source_sha256': source_digest(), 'results': results, 'benchmark': False})
    if any(r['status'] != 'ok' for r in results):
        raise ValueError('Live smoke failed or degraded; inspect smoke.json before benchmarking')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['prepare', 'run', 'smoke', 'select-depth'])
    p.add_argument('--gt', type=Path, default=GT)
    p.add_argument('--lock', type=Path)
    p.add_argument('--seed', type=int, default=20260921)
    p.add_argument('--deep-top-k', type=int, choices=[200, 400, 800, 1000], default=800)
    p.add_argument('--dataset-revision')
    p.add_argument('--profile', choices=PROFILES, help='prepare/smoke default: pe; run uses locked profile')
    p.add_argument('--suite', choices=SUITES, help='prepare default: main for pe, replication for pe-qwen')
    p.add_argument('--model-revision', action='append', metavar='MODEL=REV',
                   help='Repeat for pe and qwen3_vl; a bare revision is a PE shorthand')
    p.add_argument('--reuse-plans', type=Path, help='prepare: reuse frozen translations from an existing lock')
    p.add_argument('--depth-runs', type=Path, nargs=4, help='select-depth: dev directories for depths 200/400/800/1000')
    p.add_argument('--split', choices=['dev', 'eval'], default='dev')
    p.add_argument('--dev-run', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--limit', type=int)
    p.add_argument('--timeout', type=float, default=120)
    args = p.parse_args()
    if args.timeout <= 0 or (args.limit is not None and args.limit <= 0):
        p.error('timeout and limit must be positive')
    if args.command == 'select-depth':
        if not args.depth_runs or not args.out:
            p.error('select-depth requires --depth-runs and --out (JSON file)')
        selected = select_depth(args.depth_runs)
        write_new(args.out, selected)
        print(json.dumps(selected))
        return
    if args.command != 'smoke' and not args.lock:
        p.error('prepare/run require --lock')
    if args.command in ('run', 'smoke') and not args.out:
        p.error('run requires --out')
    asyncio.run({'prepare': prepare, 'run': run, 'smoke': smoke}[args.command](args))


if __name__ == '__main__':
    main()
