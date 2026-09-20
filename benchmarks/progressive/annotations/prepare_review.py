"""Build a scoped, unverified annotation inventory and a local review page."""
import hashlib, json, re, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
import openpyxl
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'backend'))
from app.config import get_settings
from app.media import MediaUrlBuilder
OUT = Path(__file__).resolve().parent
now = datetime.now(timezone.utc).isoformat()
commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
def read(name, sheet, required):
    path = ROOT / name
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    wb = openpyxl.load_workbook(path,read_only=True,data_only=True)
    headers = None
    records = []
    for n,row in enumerate(wb[sheet].iter_rows(values_only=True),1):
        if headers is None:
            if required.issubset(set(row)):
                headers = row
            continue
        values = {str(k):v for k,v in zip(headers,row) if k is not None}
        if not any(v is not None for v in values.values()): continue
        records.append(dict(workbook=name,sha256=sha,sheet=sheet,row=n,values=values))
    wb.close()
    if headers is None: raise ValueError('Missing headers: '+sheet)
    return records
base=read('TKIS_queries.xlsx','TKIS tổng hợp',{'Query ID','Nội dung TKIS','Video ID'})
frames=read('TKIS_queries.xlsx','GT từng frame',{'Query ID','Frame ID','Video ID'})
phm=read('TKIS_72_queries_PHM_3_hints.xlsx','72 queries',{'PHM Query ID','Hint 1 (delta)'})
def key(r): return (r['values'].get('Đợt'),r['values'].get('Query ID'))
def vids(r): return re.findall(r'L\d{2}_V\d+',str(r['values'].get('Video ID') or ''))
def scoped(v): return 21 <= int(v[1:3]) <= 30
s=get_settings()
builder=MediaUrlBuilder(s.keyframe_media_base_url_2,s.video_media_base_url_2)
rows=[]; inventory=[]; linked=set()
for r in phm:
    d=r['values']; ids=vids(r)
    candidates=[b for b in base if (key(b)==key(r) if d.get('Query ID') else key(b)[0]==key(r)[0] and b['values'].get('Nội dung TKIS')==d.get('Nội dung TKIS') and set(vids(b))==set(ids))]
    matches=[b for b in candidates if b['values'].get('Nội dung TKIS')==d.get('Nội dung TKIS') and set(vids(b))==set(ids)]
    status='confirmed' if len(matches)==1 else 'ambiguous' if candidates else 'unmatched'
    sources=[r]
    if status=='confirmed':
        sources+=matches+[f for f in frames if key(f)==key(matches[0]) and key(f)[1]]
        linked.add(matches[0]['row'])
    selected=[v for v in dict.fromkeys(ids) if scoped(v)]
    qid=str(d.get('PHM Query ID') or f'phm-row-{r["row"]}')
    rec=dict(query_id=qid,phm_query_id=qid,source_query_id=str(matches[0]['values']['Query ID']) if status=='confirmed' and matches[0]['values'].get('Query ID') else None,source_match_status=status,source_records=sources,disposition='pending_review' if selected else 'outside_L21_L30_scope',annotation_status='unverified',created_at=now,code_commit=commit,tool_version='prepare_review_v1',openpyxl_version=openpyxl.__version__)
    inventory.append(rec)
    if selected:
        rows.append(dict(**rec,original_query=d['Nội dung TKIS'],video_ids=selected,media_urls=[builder.video_url(v) for v in selected],original_hints={k:v for k,v in d.items() if 'Hint' in k or 'cumulative' in k},hints=[d.get(f'Hint {i} (delta)') or '' for i in (1,2,3)],seeds={k:d.get(k) for k in ['Frame ID','start,end','ms']},time_unit='ms',interval_convention='start_inclusive_end_exclusive',review_status='pending'))
for b in base:
    if b['row'] not in linked:
        inventory.append(dict(source_records=[b]+[f for f in frames if key(f)==key(b) and key(b)[1]],disposition='source_only_pending_review' if any(scoped(v) for v in vids(b)) else 'outside_scope_or_missing_video',annotation_status='unverified'))
(OUT/'source_inventory.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in inventory))
(OUT/'review_queue.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
(OUT/'audit_report.md').write_text(f'# L21–L30 annotation preparation\n\n- Source workbook: {len(base)} query rows; {len(frames)} frame rows.\n- PHM workbook: {len(phm)} rows; selected: {len(rows)} queries, {len(set(v for r in rows for v in r["video_ids"]))} videos.\n- Video verified: 0; approved: 0. All selected queries remain unverified.\n- Source matches: '+str({k:sum(r['source_match_status']==k for r in rows) for k in ['confirmed','ambiguous','unmatched']})+'\n- Media probe successful for L21_V015 only; other media availability pending.\n- Frame coordinate/index base and boundary inspection pending. Source ranges are not accepted GT.\n- Duplicate/paraphrase audit pending. No split or retrieval evaluation performed.\n- Browser exports are annotation drafts, not final ground_truth.jsonl.\n')
print(json.dumps({'selected_queries':len(rows),'selected_videos':len(set(v for r in rows for v in r['video_ids'])),'first':rows[0]['query_id']},ensure_ascii=False))
