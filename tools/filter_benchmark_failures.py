"""Offline exclusion of whole base queries with execution errors.

All variants and hint levels are removed together; original outputs are kept.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.agent.hints import build_plan, run_key
from benchmarks.agent.paper_report import export_paper_report
from benchmarks.agent.run_benchmark import load_queries, quota_interrupted
from tools.benchmark_video_report import export as export_video_report, plot_group_curves

ERROR_FIELDS = ('agent_failed', 'failed', 'fallback', 'retrieval_failed', 'retrieval_degraded')


def exclusion_reasons(row):
    reasons = [field for field in ERROR_FIELDS if row.get('metrics', {}).get(field)]
    if quota_interrupted(row) or row.get('legacy_quota_interrupted'):
        reasons.append('quota_interrupted')
    return reasons


def filter_run(source, output):
    source, output = Path(source), Path(output)
    if output.exists():
        raise ValueError('Use a new output directory; original reports will not be overwritten')
    manifest = json.loads((source/'manifest.json').read_text())
    original_plan = json.loads((source/'run_plan.json').read_text())
    dataset = source/'evaluation-dataset.jsonl'
    queries = load_queries(dataset)
    assert hashlib.sha256(dataset.read_bytes()).hexdigest() == manifest['config']['dataset_sha256']
    original_ids = {s['query_id'] for s in original_plan['stages']}
    variants = manifest['config']['variants']
    source_path = source/'results.jsonl'
    snapshot_size = source_path.stat().st_size
    digest = hashlib.sha256()
    excluded, seen = {}, set()
    consumed = 0
    with source_path.open('rb') as file:
        while consumed < snapshot_size:
            line = file.readline(snapshot_size-consumed)
            consumed += len(line)
            digest.update(line)
            if not line.endswith(b'\n'):
                raise ValueError('Source contains an incomplete row')
            if not line.strip():
                continue
            row = json.loads(line)
            key = run_key(row)
            if key in seen:
                raise ValueError('Duplicate result key')
            seen.add(key)
            reasons = exclusion_reasons(row)
            if reasons:
                excluded.setdefault(row['query_id'], []).append({'variant': row['variant'], 'hint_stage': key[1], 'reasons': reasons})
    selected = [q for q in queries if q['query_id'] in original_ids and q['query_id'] not in excluded]
    stages, plan = build_plan(selected, manifest['config']['hint_mode'], variants)
    expected = {(q['query_id'], q['hint_stage'], v) for q in stages for v in variants}
    retained = {key for key in seen if key[0] not in excluded}
    if retained != expected:
        raise ValueError('Remaining query/level/variant cohort is not complete')
    output.mkdir(parents=True)
    (output/'evaluation-dataset.jsonl').write_text(''.join(json.dumps(q,ensure_ascii=False,allow_nan=False)+'\n' for q in selected))
    config = copy.deepcopy(manifest['config'])
    config.update(query_ids=sorted(q['query_id'] for q in selected),
                  dataset_sha256=hashlib.sha256((output/'evaluation-dataset.jsonl').read_bytes()).hexdigest(),
                  run_plan_sha256=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest())
    timestamp = datetime.now(timezone.utc).isoformat()
    note = (f'Execution-clean cumulative cohort: {len(selected)} base queries retained from {len(original_ids)}; '
            f'{len(excluded)} whole base queries excluded due to execution errors. Every hint level and all variants '
            'of an excluded query are removed together. This is a filtered cohort, not the complete original dataset. '
            'No query is excluded based on retrieval accuracy. Original outputs and all excluded attempts are archived. '
            'No model calls or retries were performed.')
    merged = {**manifest, 'config': config, 'created_at': timestamp, 'report_only': True,
              'fingerprint': hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),
              'evaluation_note': note, 'filtered_from': str(source.resolve()),
              'source_fingerprint': manifest['fingerprint'], 'dataset_path': str((output/'evaluation-dataset.jsonl').resolve())}
    (output/'manifest.json').write_text(json.dumps(merged,indent=2)+'\n')
    (output/'run_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    counts = Counter()
    with source_path.open('rb') as file, (output/'results.jsonl').open('w') as dest, (output/'excluded_attempts.jsonl').open('w') as archive:
        consumed = 0
        while consumed < snapshot_size:
            line = file.readline(snapshot_size-consumed)
            consumed += len(line)
            if not line.strip():
                continue
            row = json.loads(line)
            if row['query_id'] in excluded:
                archive.write(line.decode())
            else:
                assert not exclusion_reasons(row)
                dest.write(line.decode())
                counts[row['variant']] += 1
    audit = {'created_at': timestamp, 'source': str(source.resolve()), 'source_results_snapshot_sha256': digest.hexdigest(),
             'source_snapshot_bytes': snapshot_size, 'selection': 'exclude whole base query with any execution error in any variant/hint level',
             'error_fields': list(ERROR_FIELDS)+['quota_interrupted'], 'excluded': excluded,
             'original_base_queries': len(original_ids), 'retained_base_queries': len(selected),
             'original_runs': len(seen), 'retained_runs': sum(counts.values()), 'excluded_runs': len(seen)-sum(counts.values()),
             'rows_by_variant': dict(counts), 'retained_source_splits': dict(Counter(q.get('original_split',q['split']) for q in selected)),
             'model_calls': 0, 'remaining_retries': 0, 'evaluation_note': note}
    (output/'EXCLUSIONS.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    export_paper_report(output, plots=config['hint_mode']=='cumulative')
    export_video_report(output, output/'evaluation-dataset.jsonl')
    if config['hint_mode']=='cumulative':
        plot_group_curves(output)
    for filename in ['REPORT.md','PAPER_REPORT.md','CUMULATIVE_HINT_REPORT.md']:
        path = output/filename
        if path.exists():
            first,rest = path.read_text().split('\n',1)
            path.write_text(first+'\n\n'+note+'\n'+rest)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(filter_run(args.source,args.output),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
