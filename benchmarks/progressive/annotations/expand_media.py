import concurrent.futures,hashlib,json,math,re,sys
from pathlib import Path
from PIL import Image,ImageDraw
from inspect_media import P,ROOT,QUEUE,run

def expand(n,a,b,tag='extra',step=None):
 out=P/'evidence'/f'{n:03d}';q=QUEUE[n-1];info=json.loads((out/'inspection.json').read_text());b=min(b,info['duration_ms']/1000);a=max(0,a)
 dest=out/f'{tag}_{int(a*1000)}_{int(b*1000)}';dest.mkdir(exist_ok=True)
 if (dest/'samples.json').exists():return n,str(dest.relative_to(P))
 local=next((r for r in info['ranges'] if r['start_ms']/1000<=a and r['end_ms']/1000>=b),None)
 source=str(ROOT/local['clip_path']) if local else q['media_urls'][0];offset=a-local['start_ms']/1000 if local else a
 clip=dest/'clip.mp4'
 run(['ffmpeg','-hide_banner','-loglevel','error','-ss',str(offset),'-i',source,'-t',str(b-a),'-vf','scale=640:-2','-c:v','libx264','-threads','1','-preset','ultrafast','-crf','23','-c:a','aac','-y',str(clip)],240)
 step=step or max(.2,math.ceil((b-a)/36))
 res=run(['ffmpeg','-hide_banner','-i',str(clip),'-vf',f"select='isnan(prev_selected_t)+gte(t-prev_selected_t,{step})',showinfo",'-fps_mode','vfr','-q:v','2','-threads','1','-y',str(dest/'%04d.jpg')])
 ts=[float(t) for t in re.findall(r'pts_time:([\d.]+)',res.stderr)];files=sorted(dest.glob('[0-9]*.jpg'))
 items=[dict(path=str(p.relative_to(ROOT)),time_ms=round((a+t)*1000)) for p,t in zip(files,ts)]
 for page in range(math.ceil(len(items)/24)):
  group=items[page*24:(page+1)*24];sheet=Image.new('RGB',(1280,math.ceil(len(group)/4)*202),(20,25,35));d=ImageDraw.Draw(sheet)
  for k,it in enumerate(group):
   im=Image.open(ROOT/it['path']);im.thumbnail((320,180));x=k%4*320;y=k//4*202;sheet.paste(im,(x,y));d.text((x+3,y+181),f'Q{n:02d} {it["time_ms"]/1000:.3f}s',fill='white')
  sheet.save(dest/f'sheet_{page}.jpg',quality=90)
 (dest/'samples.json').write_text(json.dumps(dict(start_ms=round(a*1000),end_ms=round(b*1000),sample_step_seconds=step,clip_path=str(clip.relative_to(ROOT)),clip_sha256=hashlib.sha256(clip.read_bytes()).hexdigest(),samples=items),indent=2))
 return n,str(dest.relative_to(P))
if __name__=='__main__':
 tasks=json.loads(sys.argv[1])
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  fs=[pool.submit(expand,*x) for x in tasks]
  for f in concurrent.futures.as_completed(fs):
   try:print(f.result(),flush=True)
   except Exception as e:print(type(e).__name__,str(e),flush=True)
