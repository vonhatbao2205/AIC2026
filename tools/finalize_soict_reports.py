"""Publish offline DEV, TEST and pooled SOICT reports from completed retry runs.

Original query splits determine DEV/TEST membership, including fresh retries
whose execution dataset had labelled every query TEST. No model calls.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.agent.hints import build_plan, run_key
from benchmarks.agent.metrics import measure
from benchmarks.agent.paper_report import export_paper_report
from benchmarks.agent.run_benchmark import load_queries
from tools.benchmark_video_report import export as export_video_report, plot_group_curves
from tools.filter_benchmark_failures import exclusion_reasons


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def original_split(row, queries):
    """Restore lineage from the reviewed dataset, not the retry request split."""
    query = queries[row['query_id']]
    expected = query['split']
    if row.get('original_split', expected) != expected:
        raise ValueError(f"Split provenance conflicts for {row['query_id']}")
    return expected


def finalize(main_run, cumulative_run, dataset, output, *, samples=2000):
    sources = {'main': Path(main_run), 'cumulative': Path(cumulative_run)}
    dataset, output = Path(dataset), Path(output)
    if output.exists():
        raise ValueError('Choose a new output directory; existing results will not be overwritten')
    queries = load_queries(dataset)
    by_id = {q['query_id']: q for q in queries}
    manifests = {name: json.loads((source/'manifest.json').read_text()) for name, source in sources.items()}
    if manifests['main']['config']['status'] != manifests['cumulative']['config']['status']:
        raise ValueError('Main/cumulative model/controller settings differ')
    if {m['config']['tolerance_s'] for m in manifests.values()} != {1.0}:
        raise ValueError('Final SOICT suite requires consistent 1s evaluation tolerance')
    stage_maps, expected = {}, {}
    for experiment, manifest in manifests.items():
        mode = 'full' if experiment == 'main' else 'cumulative'
        config = manifest['config']
        if config['hint_mode'] != mode:
            raise ValueError('Source experiment mode mismatch')
        variants = list('ABCDEF') if experiment == 'main' else ['A','C','F']
        if config['variants'] != variants:
            raise ValueError('Source variants differ from the planned SOICT suite')
        stages, _ = build_plan(queries, mode, variants)
        stage_maps[experiment] = {(q['query_id'],q['hint_stage']):q for q in stages}
        expected[experiment] = {(q['query_id'],q['hint_stage'],v) for q in stages for v in variants}

    # Check completeness and execution health before creating any outputs.
    audit_sources = {}
    for experiment, source in sources.items():
        size = (source/'results.jsonl').stat().st_size
        consumed, seen = 0, set()
        digest = hashlib.sha256()
        with (source/'results.jsonl').open('rb') as file:
            while consumed < size:
                line = file.readline(size-consumed)
                consumed += len(line)
                digest.update(line)
                if not line.endswith(b'\n'):
                    raise ValueError('Source contains an incomplete result row')
                if not line.strip():
                    continue
                row = json.loads(line)
                key = run_key(row)
                if key not in expected[experiment] or key in seen:
                    raise ValueError(f'Duplicate/unexpected result: {key}')
                seen.add(key)
                original_split(row,by_id)
                if exclusion_reasons(row):
                    raise ValueError(f'Retry still has an execution error: {key}')
                if row.get('snapshot',{}).get('controller',{}).get('calibrated'):
                    raise ValueError('Final suite requires the same uncalibrated raw policy')
        if seen != expected[experiment]:
            raise ValueError(f'{experiment} source is incomplete')
        audit_sources[experiment] = {'directory':str(source.resolve()),'rows':len(seen),'snapshot_bytes':size,
                                     'results_sha256':digest.hexdigest(),'fingerprint':manifests[experiment]['fingerprint']}
    output.mkdir(parents=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    metadata = {}
    for cohort in ['dev','test','pooled']:
        selected = queries if cohort == 'pooled' else [q for q in queries if q['split'] == cohort]
        for experiment, source in sources.items():
            directory = output/cohort/experiment
            directory.mkdir(parents=True)
            data_path = directory/'evaluation-dataset.jsonl'
            data_path.write_text(''.join(json.dumps(q,ensure_ascii=False,allow_nan=False)+'\n' for q in selected))
            config = copy.deepcopy(manifests[experiment]['config'])
            stages, plan = build_plan(selected, config['hint_mode'], config['variants'])
            config.update(split=cohort,query_ids=sorted({q['query_id'] for q in stages}),
                          dataset_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
                          run_plan_sha256=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest())
            # Reporting fingerprints are separate from live runner fingerprints.
            config.pop('code_sha256',None)
            note = (f'Final {cohort.upper()} evaluation, original reviewed split membership. Completed successful retries replace interrupted attempts. '
                    'Raw policy, STOP 0.9/margin 0.1, calibration disabled, tolerance 1s. '
                    'Pooled combines DEV+TEST; split-specific results are reported separately. '
                    'All planned query/level/variant results are present; no base query exclusions remain. '
                    'This directory is an offline report export, not an agent runner resume directory.')
            manifest = {'config':config,'created_at':timestamp,'report_only':True,'evaluation_note':note,
                        'fingerprint':hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),
                        'dataset_path':str(data_path.resolve()),'source':audit_sources[experiment],
                        'source_original_manifest':manifests[experiment],
                        'source_dataset_sha256':hashlib.sha256(dataset.read_bytes()).hexdigest()}
            dump(directory/'manifest.json',manifest)
            dump(directory/'run_plan.json',plan)
            metadata[cohort,experiment]={'directory':str(directory.relative_to(output)),
                                        'base_queries':plan['base_queries'],'stage_queries':plan['stage_queries'],
                                        'planned_runs':plan['planned_runs']}
    for experiment, source in sources.items():
        handles = {cohort:(output/cohort/experiment/'results.jsonl').open('w') for cohort in ['dev','test','pooled']}
        counts = Counter()
        try:
            consumed, line_number = 0, 0
            with (source/'results.jsonl').open('rb') as file:
                while consumed < audit_sources[experiment]['snapshot_bytes']:
                    line = file.readline(audit_sources[experiment]['snapshot_bytes']-consumed)
                    consumed += len(line)
                    line_number += 1
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    split = original_split(row,by_id)
                    key = run_key(row)
                    stage = stage_maps[experiment][key[:2]]
                    if row.get('evaluated_query') != stage['query']:
                        raise ValueError(f'Saved query text differs from reviewed stage: {key}')
                    measured = measure(stage,row['snapshot'],row.get('trace',{}).get('trace',[]),
                                       wall_s=row['metrics']['latency_s'],tolerance_s=1)
                    for field in ['rank','baseline_rank','video_rank','r1','r5','mrr','moment_r1','moment_r5','calibration']:
                        if json.dumps(measured[field]) != json.dumps(row['metrics'][field]):
                            raise ValueError(f'Cannot reproduce source metric: {key}, {field}')
                    row.update(split=split,original_split=split,
                               finalized_from={'directory':str(source.resolve()),'line':line_number,
                                               'execution_split':json.loads(line)['split'],
                                               'run_id':row['snapshot'].get('run_id')})
                    for cohort in [split,'pooled']:
                        handles[cohort].write(json.dumps({**row,'report_cohort':cohort},ensure_ascii=False,allow_nan=False)+'\n')
                        counts[cohort] += 1
        finally:
            for handle in handles.values():
                handle.close()
        for cohort in handles:
            assert counts[cohort] == metadata[cohort,experiment]['planned_runs']
        print(f'{experiment}: restored split lineage and verified all source metrics',flush=True)

    # Copy interruption archives for provenance; they do not enter scored results.
    archive = output/'provenance'
    archive.mkdir()
    for experiment,source in sources.items():
        for name in ['manifest.json','RETRY_PENDING.json','quota_attempts.jsonl','retry_attempts.jsonl']:
            if (source/name).exists():
                shutil.copyfile(source/name,archive/f'{experiment}-{name}')
    for cohort in ['dev','test','pooled']:
        for experiment in ['main','cumulative']:
            directory = output/cohort/experiment
            export_paper_report(directory,samples=samples,plots=experiment=='cumulative')
            report = export_video_report(directory,directory/'evaluation-dataset.jsonl')
            if experiment == 'cumulative':
                plot_group_curves(directory)
                assert report['solve']['complete_base_queries'] == metadata[cohort,experiment]['base_queries']
            assert report['scored_rows'] == metadata[cohort,experiment]['planned_runs']
            manifest = json.loads((directory/'manifest.json').read_text())
            note = manifest['evaluation_note']
            for name in ['REPORT.md','PAPER_REPORT.md','CUMULATIVE_HINT_REPORT.md']:
                path = directory/name
                if path.exists():
                    first,rest = path.read_text().split('\n',1)
                    path.write_text(first+'\n\n'+note+'\n'+rest)
            dump(directory/'VALIDATION.json',{'expected_runs':metadata[cohort,experiment]['planned_runs'],
                                              'actual_runs':report['scored_rows'],'complete':True,'execution_errors':0,
                                              'excluded_queries':0,'metrics_reproduced':True,'split_restored_from_dataset':True})
            print(f'{cohort}/{experiment}: reports, CSV, intervals and video metrics exported',flush=True)
    overview = {'created_at':timestamp,'status':'complete','tolerance_s':1,'model_calls_during_export':0,
                'sources':audit_sources,'dataset':str(dataset.resolve()),'cohorts':{cohort:{experiment:metadata[cohort,experiment]
                   for experiment in ['main','cumulative']} for cohort in ['dev','test','pooled']},
                'execution_errors':0,'excluded_base_queries':0,'note':'All original queries restored after successful retries. Pooled metrics combine both original splits.'}
    dump(output/'SUITE_SUMMARY.json',overview)
    table_rows=[]
    for cohort in ['dev','test','pooled']:
        for experiment in ['main','cumulative']:
            directory=output/cohort/experiment
            with (directory/'paper_tables.csv').open() as file:
                for row in csv.DictReader(file):
                    table_rows.append({'cohort':cohort,'experiment':experiment,**row})
    with (output/'ALL_PAPER_TABLES.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(table_rows[0]))
        writer.writeheader();writer.writerows(table_rows)
    lines=['# Final SOICT benchmark: DEV, TEST and pooled','',
           'Tolerance 1s; raw policy; calibration disabled. All planned results are present with zero execution errors or fallback. No queries remain excluded. Reports are exported offline with no model calls.','',
           '| Cohort | Main queries | Main runs | Cumulative queries | Cumulative hint stages | Cumulative runs |',
           '|---|---:|---:|---:|---:|---:|']
    for cohort in ['dev','test','pooled']:
        m,c=metadata[cohort,'main'],metadata[cohort,'cumulative']
        lines.append(f"| {cohort.upper()} | {m['base_queries']} | {m['planned_runs']} | {c['base_queries']} | {c['stage_queries']} | {c['planned_runs']} |")
    lines+=['','DEV and TEST retain the membership of the original reviewed dataset, including retries originally executed using the combined TEST dataset. Pooled is their descriptive combined result, not a relabelling of the source split.','',
            '## Reports','',
            'Each cohort contains main/ and cumulative/. Open PAPER_REPORT.md for strict paper tables and bootstrap intervals; video-report/REPORT.md for Video / Group moment / Group strict R@1/R@5/MRR.',
            'Cumulative includes CUMULATIVE_HINT_REPORT.md, summary_by_hint.json, hint_results.csv and fixed-cohort PDF/PNG hint curves. All hint levels are independent fresh runs; no cross-hint memory is used.','',
            '## Main strict R@1','', '| Variant | DEV (25) | TEST (86) | Pooled (111) |','|---|---:|---:|---:|']
    summaries={cohort:json.loads((output/cohort/'main/summary.json').read_text()) for cohort in ['dev','test','pooled']}
    for variant in 'ABCDEF':
        lines.append('| '+variant+' | '+' | '.join(f"{summaries[cohort][variant]['r1']:.2%}" for cohort in ['dev','test','pooled'])+' |')
    lines+=['','## Cumulative Full strict R@1','', '| Variant | DEV (3) | TEST (24) | Pooled (27) |','|---|---:|---:|---:|']
    summaries={cohort:json.loads((output/cohort/'cumulative/summary.json').read_text()) for cohort in ['dev','test','pooled']}
    for variant in ['A','C','F']:
        lines.append('| '+variant+' | '+' | '.join(f"{summaries[cohort][variant]['r1']:.2%}" for cohort in ['dev','test','pooled'])+' |')
    lines+=['','Original source outputs are preserved. Interrupted attempts are copied under provenance/ and excluded from the final accuracy denominator; completed successful retries are included. Source hashes and complete-run checks are in SUITE_SUMMARY.json and each VALIDATION.json.',
            'DEV cumulative has only three base queries; bootstrap intervals there are exploratory. Combined CSV: ALL_PAPER_TABLES.csv.']
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    return overview


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main-run',required=True)
    parser.add_argument('--cumulative-run',required=True)
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--bootstrap-samples',type=int,default=2000)
    args=parser.parse_args()
    if args.bootstrap_samples<=0:
        parser.error('bootstrap-samples must be positive')
    finalize(args.main_run,args.cumulative_run,args.dataset,args.output,samples=args.bootstrap_samples)
    print(f'Final suite: {args.output}',flush=True)


if __name__=='__main__':
    main()
