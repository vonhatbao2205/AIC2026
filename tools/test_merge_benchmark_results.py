import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.agent.metrics import measure
from benchmarks.agent.hints import hint_stages
from tools.merge_benchmark_results import merge


class MergeResultsTests(unittest.TestCase):
    def prepare(self, root):
        queries = [{"query_id": split, "query": f"query {split}", "split": split, "query_type": "T-KIS",
                    "targets": [{"video_id": "right", "start_s": 10, "end_s": 10}]} for split in ['dev', 'test']]
        dataset = root / 'dataset.jsonl'
        dataset.write_text(''.join(json.dumps(q) + '\n' for q in queries))
        source_files = {}
        for query in queries:
            source = root / query['split']
            source.mkdir()
            config = {'split': query['split'], 'dataset_sha256': hashlib.sha256(dataset.read_bytes()).hexdigest(),
                      'variants': ['A', 'F'], 'variant_parameters': {'A': {'policy': 'retrieval'}, 'F': {'policy': 'full'}},
                      'hint_mode': 'full', 'tolerance_s': 1.0, 'status': {'calibration': None}, 'seed': 2026, 'mode': 'live'}
            (source / 'manifest.json').write_text(json.dumps({'fingerprint': query['split'], 'config': config}))
            rows = []
            for variant in ['A', 'F']:
                snapshot = {'ranking': [{'video_id': 'right', 'time': 10 if query['split'] == 'test' else 20}],
                            'controller': {}, 'agents': {}, 'metrics': {}}
                if query['split'] == 'dev' and variant == 'F':
                    snapshot['agents'] = {'codex': {'status': 'failed', 'error_kind': 'quota'}}
                    snapshot['metrics'] = {'agent_calls': {'codex': 1}}
                rows.append({'query_id': query['query_id'], 'query_type': 'T-KIS', 'split': query['split'],
                             'hint_stage': 'full', 'variant': variant, 'snapshot': snapshot, 'trace': {'trace': []},
                             'metrics': measure(query, snapshot, [], wall_s=1, tolerance_s=1)})
            (source / 'results.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
            source_files[query['split']] = (source / 'results.jsonl').read_bytes()
        return dataset, source_files

    def test_combined_test_preserves_sources_and_keeps_failure_denominator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, original = self.prepare(root)
            result = merge(root/'dev', root/'test', dataset, root/'combined', treat_dev_as_test=True)
            self.assertEqual(result['rows'], 4)
            self.assertEqual(result['summary']['F']['queries'], 2)
            self.assertEqual(result['summary']['F']['r1'], .5)
            self.assertEqual(result['summary']['F']['agent_failed'], .5)
            rows = [json.loads(l) for l in (root/'combined/results.jsonl').read_text().splitlines()]
            self.assertEqual({r['split'] for r in rows}, {'test'})
            self.assertEqual({r['original_split'] for r in rows}, {'dev', 'test'})
            video = json.loads((root/'combined/video-report/summary.json').read_text())
            self.assertEqual(video['summary']['F']['queries'], 2)
            self.assertEqual(len(video['legacy_failures_retained']), 1)
            for split in original:
                self.assertEqual((root/split/'results.jsonl').read_bytes(), original[split])
            copied = [json.loads(l) for l in (root/'combined/evaluation-dataset.jsonl').read_text().splitlines()]
            self.assertEqual({q['split'] for q in copied}, {'test'})

    def test_mismatched_tolerances_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, _ = self.prepare(root)
            path = root/'test/manifest.json'
            manifest = json.loads(path.read_text())
            manifest['config']['tolerance_s'] = 5
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'tolerance_s'):
                merge(root/'dev', root/'test', dataset, root/'combined')
            self.assertFalse((root/'combined').exists())

    def test_cumulative_merge_preserves_levels_and_full_summary_denominator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, _ = self.prepare(root)
            queries = [json.loads(line) for line in dataset.read_text().splitlines()]
            for q in queries:
                q['hints_vi'] = q['query'].split()
            dataset.write_text(''.join(json.dumps(q)+'\n' for q in queries))
            for q in queries:
                source = root/q['split']
                manifest = json.loads((source/'manifest.json').read_text())
                manifest['config'].update(hint_mode='cumulative', dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest())
                (source/'manifest.json').write_text(json.dumps(manifest))
                original = [json.loads(line) for line in (source/'results.jsonl').read_text().splitlines()]
                rows = []
                for stage in hint_stages(q, 'cumulative'):
                    for row in original:
                        rows.append({**row, **{k:stage[k] for k in ['hint_stage','hint_level','hint_count','hint_fraction','is_full_hint']},
                                     'evaluated_query': stage['query']})
                (source/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
                from benchmarks.agent.hints import build_plan
                _, plan = build_plan([q], 'cumulative', ['A','F'])
                (source/'run_plan.json').write_text(json.dumps(plan))
            result = merge(root/'dev', root/'test', dataset, root/'combined', treat_dev_as_test=True)
            self.assertEqual(result['rows'], 8)
            self.assertEqual(result['queries'], 2)
            self.assertEqual(result['summary']['F']['queries'], 2)
            hints = json.loads((root/'combined/summary_by_hint.json').read_text())
            self.assertEqual(hints['solve']['complete_base_queries'], 2)
            video = json.loads((root/'combined/video-report/summary.json').read_text())
            self.assertEqual(video['legacy_quota_rows_retained'], 2)
            self.assertEqual(video['solve']['complete_base_queries'], 2)
            self.assertEqual(set(video['levels']), {'h1','full'})


if __name__ == '__main__':
    unittest.main()
