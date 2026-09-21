from __future__ import annotations
import asyncio, contextlib, fcntl, hashlib, importlib.util, json, subprocess, sys, traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
ROOT = Path.cwd()
BASE = ROOT / 'benchmarks/progressive/runs/paper-20260921'
spec=importlib.util.spec_from_file_location('benchmark', ROOT/'benchmarks/progressive/run_benchmark.py')
b=importlib.util.module_from_spec(spec); spec.loader.exec_module(b)

def save(path, value):
 if path.exists():
  existing=json.loads(path.read_text())
  if existing != value: raise RuntimeError('Existing provenance changed: '+path.name)
 else: b.write_new(path,value)

def event(phase, status, **extra):
 data={'utc':datetime.now(timezone.utc).isoformat(),'phase':phase,'status':status,**extra}
 with (BASE/'progress.jsonl').open('a') as f: f.write(json.dumps(data)+'\n')
 print(json.dumps(data),flush=True)

def args(**extra):
 return SimpleNamespace(gt=b.GT, seed=20260921, timeout=120, limit=None, split='dev',dev_run=None,
   profile='pe',suite='main',reuse_plans=None,deep_top_k=800,**extra)

async def main():
 lease=(BASE/'campaign.lck').open('a')
 fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
 # Preparation-only reliability wrapper. It preserves successful production
 # parser outputs verbatim; retrieval uses the runner's FrozenParser unchanged.
 from app.query_parser import QueryParser
 original_parse=QueryParser.parse
 cache_path=BASE/'translation-checkpoint.json'
 plans=json.loads(cache_path.read_text()) if cache_path.exists() else {}
 async def reliable_parse(self,text,*args,**kwargs):
  if text in plans: return json.loads(json.dumps(plans[text]))
  for attempt in range(1,6):
   parsed=await original_parse(self,text,*args,**kwargs)
   if not parsed.get('translation_failed'):
    plans[text]=parsed
    temporary=cache_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(plans,ensure_ascii=False))
    temporary.replace(cache_path)
    with (BASE/'translation-progress.jsonl').open('a') as out:
     out.write(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'completed':len(plans),'attempt':attempt})+'\n')
    return parsed
   with (BASE/'translation-progress.jsonl').open('a') as out:
    out.write(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'retry':attempt,'text_sha256':hashlib.sha256(text.encode()).hexdigest()})+'\n')
   await asyncio.sleep(min(5*attempt,20))
  raise RuntimeError('Translation failed after five attempts')
 QueryParser.parse=reliable_parse

 manifests={}
 for model,name in [('pe','pe-core-g14-448-v1'),('qwen3_vl','qwen3-vl-embedding-8b-4096-v3')]:
  p=Path('/home/bao/Projects/EncoderModel')/name/'embedding_dataset_manifest.json'
  manifests[model]=json.loads(p.read_text())
  save(BASE/'provenance'/f'{model}-embedding-manifest.json',manifests[model])
 s=b.SearchService(b.get_settings().for_retrieval_database('infoshotpp'))
 health=await b.preflight(s,['pe','qwen3_vl'],30)
 if not all(v['ok'] for v in health.values()): raise RuntimeError('Preflight failed')
 encoder_health={}
 for name,client in [('pe',s.pe),('qwen3_vl',s.qwen3_vl)]:
  h=await client.health()
  encoder_health[name]={k:v for k,v in h.items() if k in ('ok','model','revision','dim','device','dtype')}
 client=s.milvus._connect()
 collections={}
 for model,attr in b.COLLECTIONS.items():
  name=getattr(s.s,attr)
  stats=client.get_collection_stats(collection_name=name)
  desc=client.describe_collection(collection_name=name)
  collections[model]={'name':name,'stats':stats,'schema':desc}
  if int(stats['row_count']) != manifests[model]['total_committed_rows']:
   raise RuntimeError('Index count differs from embedding manifest')
 dataset='infoshootpp-v1:sha256:'+manifests['pe']['semantic_config']['source']['dataset_manifest_sha256']
 pe='checkpoint-sha256:'+manifests['pe']['semantic_config']['model']['checkpoint_sha256']
 qwen=manifests['qwen3_vl']['semantic_config']['model']['revision']
 save(BASE/'provenance'/'freeze.json',{'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
  'source_sha256':b.source_digest(),'dataset_revision':dataset,'model_revisions':{'pe':pe,'qwen3_vl':qwen},
  'encoder_health':encoder_health,'collections':collections,'preflight':health,
  'limitation':'PE checkpoint SHA is attested by the embedding export manifest; live PE health reports model identity but does not expose loaded weight hash. Remote index bytes are not fully rehashed.',
  'test_verification':'488 backend tests passed on frozen commit; full annotation validation passed (2749 local artifacts).',
  'protocol':'Four complete PE dev depth pilots; latency-only choice; final PE dev then one eval; PE+Qwen dev then one eval. No scoring tuning.'})
 event('freeze','ok',dataset=dataset,models={'pe':pe,'qwen3_vl':qwen})

 async def prepare(name,depth,profile='pe',reuse=None):
  a=args(lock=BASE/'locks'/f'{name}.json',dataset_revision=dataset,model_revision=['pe='+pe]+(['qwen3_vl='+qwen] if profile=='pe-qwen' else []))
  a.profile=profile; a.suite='main' if profile=='pe' else 'replication'; a.deep_top_k=depth; a.reuse_plans=reuse
  if a.lock.exists():
   saved=json.loads(a.lock.read_text()); checksum=saved.pop('sha256')
   if b.digest(saved)!=checksum or saved['source_sha256']!=b.source_digest():
    raise RuntimeError('Existing lock changed')
   event('prepare-'+name,'already_complete')
   return a.lock
  event('prepare-'+name,'running')
  with (BASE/f'prepare-{name}-resume.log').open('a') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
   await b.prepare(a)
  event('prepare-'+name,'complete')
  return a.lock

 async def run(name,lock,split='dev',dev=None):
  a=args(lock=lock,out=BASE/name)
  a.profile=None;a.suite=None;a.split=split;a.dev_run=dev
  if a.out.exists():
   summary=a.out/'summary.json'
   if not summary.exists(): raise RuntimeError('Partial run requires inspection: '+name)
   report=json.loads(summary.read_text())
   checksum=json.loads(lock.read_text())['sha256']
   if not report['healthy'] or not report['complete'] or report['lock_sha256']!=checksum:
    raise RuntimeError('Existing run unhealthy or mismatched: '+name)
   event(name,'already_complete')
   return a.out
  event(name,'running')
  with (BASE/f'{name}.log').open('x') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
   await b.run(a)
  report=json.loads((a.out/'summary.json').read_text())
  event(name,'complete',healthy=report['healthy'],complete=report['complete'])
  if not report['healthy'] or not report['complete']: raise RuntimeError(name+' unhealthy/incomplete')
  return a.out

 first=None; pilots=[]
 for depth in [200,400,800,1000]:
  lock=await prepare(f'depth-{depth}',depth,reuse=first)
  first=first or lock
  pilots.append(await run(f'dev-{depth}',lock))
 selection=b.select_depth(pilots); save(BASE/'depth-selection.json',selection)
 event('select-depth','complete',depth=selection['selected_deep_top_k'],gap_ms=selection['absolute_gap_ms'])
 final=await prepare('pe-final',selection['selected_deep_top_k'],reuse=first)
 dev=await run('pe-dev-final',final)
 await run('pe-eval-final',final,'eval',dev)
 replication=await prepare('pe-qwen-final',selection['selected_deep_top_k'],'pe-qwen',final)
 qdev=await run('pe-qwen-dev-final',replication)
 await run('pe-qwen-eval-final',replication,'eval',qdev)
 event('campaign','complete')

try: asyncio.run(main())
except BaseException as exc:
 event('campaign','stopped',error_type=type(exc).__name__)
 with (BASE/'failure-resume.log').open('a') as f: traceback.print_exc(file=f)
 sys.exit(1)
