import json
from collections import Counter
from pathlib import Path
base=Path(__file__).parent
progress=[json.loads(x) for x in (base/'progress.jsonl').read_text().splitlines()]
print(json.dumps(progress[-1],ensure_ascii=False))
for p in sorted(base.glob('*/records.jsonl')):
 rows=[json.loads(x) for x in p.read_text().splitlines()]
 print(p.parent.name,'prefixes',len(rows),'queries',len({r['query_id'] for r in rows}),'status',dict(Counter(r['status'] for r in rows)))
