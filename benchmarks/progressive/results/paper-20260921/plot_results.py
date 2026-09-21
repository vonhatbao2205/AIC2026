"""Publication plots from completed evaluation summaries (no model calls)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path.cwd(); OUT=ROOT/'benchmarks/progressive/results/paper-20260921'
colors={'cumulative':'#444444','hint_rrf':'#0072B2','dual_view':'#009E73','phm_no_rescue':'#CC79A7','phm':'#D55E00','cumulative_deep':'#E69F00'}
labels={'cumulative':'Cumulative','hint_rrf':'Hint-RRF','dual_view':'Dual-view','phm_no_rescue':'PHM w/o rescue','phm':'PHM','cumulative_deep':'Cumulative depth 400'}
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
reports={p:json.loads((OUT/(p+'-summary.json')).read_text()) for p in ['pe','pe-qwen']}
fig,axes=plt.subplots(1,2,figsize=(7.2,2.8),sharey=True,layout='constrained')
for ax,(profile,report) in zip(axes,reports.items()):
 for method,data in report['metrics'].items():
  ax.plot([1,2,3],[data['prefixes'][str(t)]['rr'] for t in [1,2,3]],label=labels[method],color=colors[method],marker='o',markersize=4,linewidth=1.4)
 ax.set(title='PE' if profile=='pe' else 'PE + Qwen',xlabel='Revealed hint',xticks=[1,2,3],ylim=(0,1))
 ax.grid(axis='y',alpha=.2);ax.legend(fontsize=7,loc='lower right')
axes[0].set_ylabel('Video MRR (60 held-out queries)')
for ext in ['pdf','svg','png']:fig.savefig(OUT/('prefix-mrr.'+ext),dpi=220)
plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(7.2,2.8),sharey=True,layout='constrained')
for ax,(profile,report) in zip(axes,reports.items()):
 for method,data in report['metrics'].items():
  ax.scatter(data['latency_p50_ms']/1000,data['mean_prefix_video_mrr'],color=colors[method],label=labels[method],s=30,marker='D' if method=='phm' else 'o')
 ax.set(title='PE' if profile=='pe' else 'PE + Qwen',xlabel='Median retrieval latency (s)',ylim=(0,1),xlim=(0,None))
 ax.grid(alpha=.2);ax.legend(fontsize=7,loc='lower right')
axes[0].set_ylabel('Mean prefix video MRR')
for ext in ['pdf','svg','png']:fig.savefig(OUT/('quality-latency.'+ext),dpi=220)
plt.close(fig)
print('Wrote plots to',OUT)
