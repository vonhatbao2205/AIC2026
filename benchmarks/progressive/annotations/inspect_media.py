"""Decode media seeds into reviewable clips, timestamped contact sheets, and ASR excerpts.
Contact sheets and ASR are navigation aids, not automatic verification.
"""
import concurrent.futures,csv,hashlib,json,math,re,subprocess,sys
from pathlib import Path
from PIL import Image,ImageDraw
P=Path(__file__).resolve().parent; ROOT=P.parents[2]
QUEUE=json.loads((P/'review_queue.json').read_text())
MAP={p.stem:p for p in (ROOT/'map-keyframes-s1-s2').rglob('*.csv')}

def run(cmd,timeout=180):
 r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=timeout)
 if r.returncode:raise RuntimeError('media command failed: '+r.stderr[-300:].split('https:')[0])
 return r

def capture(i):
 q=QUEUE[i-1];v=q['video_ids'][0]; out=P/'evidence'/f'{i:03d}';out.mkdir(parents=True,exist_ok=True)
 if (out/'inspection.json').exists():return i,'cached'
 probe=json.loads(run(['ffprobe','-v','error','-show_entries','format=duration:stream=index,codec_type,time_base,start_time,avg_frame_rate,r_frame_rate','-of','json',q['media_urls'][0]],60).stdout)
 (out/'probe.json').write_text(json.dumps(probe,indent=2));duration=float(probe['format']['duration'])
 with MAP[v].open() as f:m=list(csv.DictReader(f))
 def seed_time(frame):
  nearest=min(m,key=lambda r:abs(int(r['frame_idx'])-frame))
  # Only a navigation estimate. Final annotation uses decoded playback time.
  return float(nearest['pts_time'])+(frame-int(nearest['frame_idx']))/float(nearest['fps'])
 frame_values=[int(n) for n in re.findall(r'\d+',str(q['seeds']['Frame ID'] or ''))]
 ranges=[]
 for a,b in re.findall(r'\[\s*(\d+)\s*,\s*(\d+)\s*\]',str(q['seeds']['start,end'] or '')):
  ranges.append((seed_time(int(a)),seed_time(int(b))))
 for a,b in re.findall(r'\[\s*(\d+)\s*,\s*(\d+)\s*\]',str(q['seeds']['ms'] or '')):ranges.append((int(a)/1000,int(b)/1000))
 ranges.extend((seed_time(f),seed_time(f)) for f in frame_values)
 ranges=sorted((max(0,a-15),min(duration,b+15)) for a,b in ranges if 0<=a<duration and b>=a)
 merged=[]
 for a,b in ranges:
  if merged and a<=merged[-1][1]+5:merged[-1][1]=max(b,merged[-1][1])
  else:merged.append([a,b])
 info=dict(query_number=i,query_id=q['query_id'],video_id=v,duration_ms=round(duration*1000),mapping_path=str(MAP[v].relative_to(ROOT)),mapping_sha256=hashlib.sha256(MAP[v].read_bytes()).hexdigest(),frame_id_kind='absolute_frame_index_in_source_mapping',seed_mapping='nearest registry frame pts_time plus frame difference / registry fps; navigation estimate only',ranges=[],limitations=['Contact sheets are sampled; continuous video and exact boundary verification remain separate.','ASR excerpts are unverified machine transcripts.'])
 speech=ROOT/'speech_out'/f'{v}.speech.json';segments=json.loads(speech.read_text()) if speech.exists() else []
 excerpts=[{k:s.get(k) for k in ['start','end','text']} for s in segments if any(s.get('start',0)<b and s.get('end',0)>a for a,b in merged)]
 (out/'asr_navigation.json').write_text(json.dumps(excerpts,ensure_ascii=False,indent=2))
 for j,(a,b) in enumerate(merged):
  clip=out/f'context_{j}.mp4'
  if not clip.exists():run(['ffmpeg','-hide_banner','-loglevel','error','-threads','1','-ss',str(a),'-i',q['media_urls'][0],'-t',str(b-a),'-map','0:v:0','-map','0:a:0?','-vf','scale=640:-2','-c:v','libx264','-threads','1','-preset','ultrafast','-crf','25','-c:a','aac','-b:a','64k','-y',str(clip)],240)
  files=out/f'sampled_{j}';files.mkdir(exist_ok=True)
  step=max(1,math.ceil((b-a)/36))
  res=run(['ffmpeg','-hide_banner','-i',str(clip),'-vf',f"select='isnan(prev_selected_t)+gte(t-prev_selected_t,{step})',showinfo",'-fps_mode','vfr','-q:v','3','-threads','1','-y',str(files/'%04d.jpg')])
  ts=[float(t) for t in re.findall(r'pts_time:([\d.]+)',res.stderr)]
  imgs=sorted(files.glob('*.jpg'));items=[]
  for n,(path,t) in enumerate(zip(imgs,ts)):items.append(dict(path=str(path.relative_to(ROOT)),time_ms=round((a+t)*1000)))
  for page in range(math.ceil(len(items)/24)):
   group=items[page*24:(page+1)*24];sheet=Image.new('RGB',(1280,math.ceil(len(group)/4)*202),(20,25,35));d=ImageDraw.Draw(sheet)
   for k,it in enumerate(group):
    im=Image.open(ROOT/it['path']);im.thumbnail((320,180));x=k%4*320;y=k//4*202;sheet.paste(im,(x,y));d.text((x+3,y+181),f'Q{i:02d}  {it["time_ms"]/1000:.3f}s',fill='white')
   sheet.save(out/f'overview_{j}_{page}.jpg',quality=90)
  info['ranges'].append(dict(start_ms=round(a*1000),end_ms=round(b*1000),clip_path=str(clip.relative_to(ROOT)),clip_sha256=hashlib.sha256(clip.read_bytes()).hexdigest(),sample_step_seconds=step,samples=items))
 (out/'inspection.json').write_text(json.dumps(info,ensure_ascii=False,indent=2))
 return i,'ok'
if __name__=='__main__':
 ids=[int(x) for x in sys.argv[1:]] or list(range(1,len(QUEUE)+1))
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  jobs={pool.submit(capture,i):i for i in ids}
  for f in concurrent.futures.as_completed(jobs):
   try:print(f.result(),flush=True)
   except Exception as e:
    i=jobs[f];(P/'evidence'/f'{i:03d}'/'error.txt').write_text(type(e).__name__+': '+str(e));print(i,type(e).__name__,str(e),flush=True)
