"""Read-only audit of the completed campaign artifacts."""
import hashlib, importlib.util, json
from pathlib import Path
ROOT=Path.cwd();BASE=ROOT/'benchmarks/progressive/runs/paper-20260921'
spec=importlib.util.spec_from_file_location('benchmark',ROOT/'benchmarks/progressive/run_benchmark.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
rows={r['query_id']:r for r in b.load_gt(b.GT)}
report={}
for name in ['dev-200','dev-400','dev-800','dev-1000','pe-dev-final','pe-eval-final','pe-qwen-dev-final','pe-qwen-eval-final']:
 directory=BASE/name
 manifest=json.loads((directory/'manifest.json').read_text());summary=json.loads((directory/'summary.json').read_text())
 lock=manifest['lock'];split=summary['split']
 assert b.digest(lock)==summary['lock_sha256']==manifest['lock_sha256']
 assert lock['source_sha256']==b.source_digest()
 assert lock['gt_sha256']==hashlib.sha256(b.GT.read_bytes()).hexdigest()
 records=[json.loads(s) for s in (directory/'records.jsonl').read_text().splitlines()]
 expected={(qid,method,t) for qid,r in rows.items() if lock['split'][qid]['split']==split
           for method in lock['methods'] for t in range(1,len(b.ledger_for(r))+1)}
 actual=[(r['query_id'],r['method'],r['turn']) for r in records]
 assert len(actual)==len(set(actual)) and set(actual)==expected
 for r in records:
  assert r['status']=='ok'
  trace=json.loads((directory/r['trace']).read_text())
  prefix=b.ledger_for(rows[r['query_id']])[:r['turn']]
  assert trace['ledger']==prefix
  assert len(trace['observations'])==r['turn']
  assert b.score_snapshot(trace['snapshot'],rows[r['query_id']])==r['metrics']
  assert trace['snapshot']['revision_budget']==r['budget']
  assert trace['snapshot']['mode']=='live' and not trace['snapshot']['degraded']
  effective=trace['snapshot']['effective_config']
  assert not effective['hybrid'] and not effective['tara'] and not effective['rerank'] and not effective['expand']
  assert effective['image_models']==lock['config']['image_models']
 assert b.summarize(records)==summary['metrics']
 report[name]={'prefixes':len(records),'queries':len({r['query_id'] for r in records}),
               'target_video_groups':len({r['group'] for r in records}),'all_checks_passed':True}
b.write_new(BASE/'audit.json',report)
print(json.dumps(report,indent=2))
