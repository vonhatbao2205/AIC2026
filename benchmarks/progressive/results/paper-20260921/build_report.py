"""Export paper tables from completed locked evaluations, without retrieval or tuning."""
import csv, hashlib, importlib.util, json, random, shutil, statistics
from pathlib import Path
ROOT=Path.cwd(); BASE=ROOT/'benchmarks/progressive/runs/paper-20260921'
OUT=ROOT/'benchmarks/progressive/results/paper-20260921'
NAMES={'cumulative':'Cumulative','hint_rrf':'Hint-RRF','dual_view':'Dual-view','phm_no_rescue':'PHM without rescue','phm':'PHM','cumulative_deep':'Deeper cumulative'}
spec=importlib.util.spec_from_file_location('benchmark',ROOT/'benchmarks/progressive/run_benchmark.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)

def write(path,data):
 path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def contrasts(records, pairs):
 values={}; groups={}
 for r in records:
  values.setdefault((r['method'],r['query_id']),[]).append(r['metrics']['rr'])
  groups.setdefault(r['group'],set()).add(r['query_id'])
 out={}
 for a,c in pairs:
  blocks=[[statistics.mean(values[a,q])-statistics.mean(values[c,q]) for q in sorted(qs)] for qs in groups.values()]
  rng=random.Random(20260921); draws=[]
  for _ in range(10000):
   selected=rng.choices(blocks,k=len(blocks));draws.append(statistics.mean(v for block in selected for v in block))
  draws.sort();out[a+' - '+c]={'delta_mrr':statistics.mean(v for block in blocks for v in block),'ci95':[draws[250],draws[9750]],'bootstrap_samples':10000,'unit':'target-video connected component'}
 return out

def main():
 if OUT.exists(): raise RuntimeError('Report destination exists; do not overwrite')
 reports={}; records={}; locks={}
 for profile in ['pe','pe-qwen']:
  run=BASE/(profile+'-eval-final')
  reports[profile]=json.loads((run/'summary.json').read_text())
  records[profile]=[json.loads(x) for x in (run/'records.jsonl').read_text().splitlines()]
  manifest=json.loads((run/'manifest.json').read_text());locks[profile]=manifest['lock']
  assert reports[profile]['complete'] and reports[profile]['healthy'] and not reports[profile]['mock']
  assert len({r['query_id'] for r in records[profile]})==60
  assert reports[profile]['lock_sha256']==b.digest(locks[profile])
 assert locks['pe']['split']==locks['pe-qwen']['split'] and locks['pe']['plans']==locks['pe-qwen']['plans']
 OUT.mkdir(parents=True)
 selection=json.loads((BASE/'depth-selection.json').read_text())
 lines=['# PHM locked benchmark — 2026-09-21','',
 'Code: `b012b9bd40a93faaeb51f9252fc7087ca95a04b9`. This report describes controlled visual PHM evaluation, not the full multimodal AIC2026 system.', '',
 '75 eligible owner-approved synthetic progressive queries; fixed target-video-group split: 15 development / 60 evaluation. Three prefixes per query. Dataset: InfoShot++ L21–L30, 1,339,055 indexed frames/model; corpus scope is all indexed videos. GT: 90 accepted answer intervals across the approved annotation set; no claim of independently verified continuous playback.', '',
 'Mean prefix video MRR averages three prefixes within each query before averaging 60 queries. Ranking cap: 250 videos; missing targets score zero. Moment Hit@K tests the first preferred frame of each of the first K displayed video groups against half-open GT intervals, with no search over historical support frames.', '',
 f"Depth selected on dev by latency only: **{selection['selected_deep_top_k']}**. PHM reference p50: {selection['phm_reference_p50_ms']:.1f} ms; nearest deep-cumulative absolute gap: {selection['absolute_gap_ms']:.1f} ms. This is the nearest tested depth, not a claim of equal compute. See `depth-selection.json` and archived dev summaries.", '',
 'Each evaluation profile was run once after a complete healthy dev run using its exact lock. All translations/parser plans and splits are identical between profiles. No tuning on evaluation. TARA/OCR/ASR/audio/reranker/query expansion disabled.', '']
 all_csv=[]; all_contrasts={}
 for profile,report in reports.items():
  lines += ['## '+('A — PE controlled experiment' if profile=='pe' else 'B — PE + Qwen replication'), '',
   '| Method | Mean prefix MRR | H1 Hit@1 | H2 Hit@1 | H3 Hit@1 | H3 Hit@5 | H3 Hit@10 | p50 ms | p95 ms |',
   '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
  tex=['\\begin{tabular}{lrrrrr}','\\hline','Method & MRR & H1 H@1 & H2 H@1 & H3 H@1 & p50 (ms) \\\\','\\hline']
  for method in locks[profile]['methods']:
   m=report['metrics'][method];p=m['prefixes'];q=m['queries'];v=m['mean_prefix_video_mrr']
   lines.append(f"| {NAMES[method]} | {v:.4f} | {p['1']['video_hit@1']:.3f} | {p['2']['video_hit@1']:.3f} | {p['3']['video_hit@1']:.3f} | {p['3']['video_hit@5']:.3f} | {p['3']['video_hit@10']:.3f} | {m['latency_p50_ms']:.1f} | {m['latency_p95_ms']:.1f} |")
   tex.append(f"{NAMES[method]} & {v:.4f} & {p['1']['video_hit@1']:.3f} & {p['2']['video_hit@1']:.3f} & {p['3']['video_hit@1']:.3f} & {m['latency_p50_ms']:.1f} \\\\")
   all_csv.append({'profile':profile,'method':method,'mean_prefix_mrr':v,'p50_ms':m['latency_p50_ms'],'p95_ms':m['latency_p95_ms'],**{f'h{t}_{k}':val for t,metrics in p.items() for k,val in metrics.items()}})
  tex+=['\\hline','\\end{tabular}'];(OUT/(profile+'-table.tex')).write_text('\n'.join(tex)+'\n')
  lines+=['','### Localization, stability and rescue','',
   '| Method | H3 Moment H@1/5/10 | First correct H1/H2/H3/unsolved | Stable correct H1/H2/H3/unsolved | Mean top-1 churn | Rescue recovery/harm |',
   '|---|---|---|---|---:|---:|']
  for method in locks[profile]['methods']:
   m=report['metrics'][method];q=m['queries'];p=m['prefixes']['3']
   first='/'.join(str(sum(x['first_correct_turn']==t for x in q)) for t in [1,2,3,None]);stable='/'.join(str(sum(x['stable_correct_turn']==t for x in q)) for t in [1,2,3,None])
   moment='/'.join(f"{p[f'moment_hit@{k}']:.3f}" for k in [1,5,10])
   lines.append(f"| {NAMES[method]} | {moment} | {first} | {stable} | {statistics.mean(x['top1_churn'] for x in q):.3f} | {m['rescue_recovery']}/{m['harmful_rescue']} |")
  lines+=['','Stable-correct turn is retrospective, not an online stopping policy. Unsolved queries remain explicitly counted. Rescue counts are prefix-level before/after rescue comparisons; recovery means absent-before/present-after within the rank cap, and harm means worse target rank or disappearance.', '',
    '| Method | Global calls | Local calls | Query vectors | Frames returned | Failure/timeout rate |',
    '|---|---:|---:|---:|---:|---:|']
  for method in locks[profile]['methods']:
   m=report['metrics'][method];w=m['work_totals']
   lines.append(f"| {NAMES[method]} | {w['global_index_calls']} | {w['local_index_calls']} | {w['query_vectors']} | {w['frames_returned']} | {m['failure_rate']:.3f}/{m['timeout_rate']:.3f} |")
  pairs=[('phm','cumulative'),('phm','hint_rrf'),('phm','phm_no_rescue')]
  if profile=='pe': pairs += [('phm_no_rescue','dual_view'),('phm','cumulative_deep')]
  all_contrasts[profile]=contrasts(records[profile],pairs)
  lines+=['','### Paired differences in mean prefix MRR','', '| Contrast | Delta | 95% group bootstrap CI |','|---|---:|---|']
  for label,stats in all_contrasts[profile].items():
   lo,hi=stats['ci95'];lines.append(f"| {label} | {stats['delta_mrr']:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
  lines+=['','10,000 paired bootstrap resamples of target-video connected components, seed 20260921. Intervals are unadjusted for multiple comparisons and describe this small synthetic-query benchmark.','']
  for f in ['summary.json','manifest.json','records.jsonl']:
   shutil.copy2(BASE/(profile+'-eval-final')/f,OUT/(profile+'-'+f))
  shutil.copy2(BASE/'locks'/(profile+'-final.json'),OUT/(profile+'-lock.json'))
 for directory in ['dev-200','dev-400','dev-800','dev-1000','pe-dev-final','pe-qwen-dev-final']:
  shutil.copy2(BASE/directory/'summary.json',OUT/(directory+'-summary.json'))
 shutil.copy2(BASE/'depth-selection.json',OUT/'depth-selection.json')
 shutil.copytree(BASE/'provenance',OUT/'provenance')
 write(OUT/'paired-contrasts.json',all_contrasts)
 with (OUT/'metrics.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(all_csv[0]));writer.writeheader();writer.writerows(all_csv)
 lines+=['## Reproducibility and limits','',
 'Full raw retrieval traces, runner logs and frozen locks are retained locally under `benchmarks/progressive/runs/paper-20260921/`. The compact export contains evaluation records, summaries, locks, manifests, depth-selection evidence, CSV and LaTeX tables. `artifact-hashes.json` binds every raw campaign file by SHA-256; raw files must be transferred separately for independent trace replay.', '',
 'PE checkpoint provenance comes from the embedding-export manifest. The live PE health endpoint reports model identity but does not attest the loaded checkpoint hash. Qwen live health reports the pinned model revision. Remote index cardinalities match manifests; complete remote index bytes were not rehashed. This limitation must remain in any reproducibility claim.', '',
 'The initial translation-preparation attempt stopped on a provider failure before retrieval began. Preparation was retried with a checkpoint/retry wrapper that preserves successful production-parser outputs verbatim. All benchmark methods subsequently consume the same frozen plans; this operational retry does not change scoring or count as an evaluation run.', '',
 'Measured latency includes the production engine path with frozen parser plans and its session-local caches, but excludes trace serialization and initial translation preparation. Methods run sequentially in seeded order; live service load can affect latency. The selected deeper cumulative depth may remain far from PHM latency; report that gap rather than calling the runs compute-matched.', '',
 'No user study, real reveal-time savings, calibrated submission confidence, or generalization to VBS/V3C is established by this experiment.']
 (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
 hashes={str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(BASE.rglob('*')) if p.is_file()}
 write(OUT/'artifact-hashes.json',hashes)
 print(OUT)
if __name__=='__main__': main()
