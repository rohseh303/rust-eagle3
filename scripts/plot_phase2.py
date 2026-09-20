"""Render the measured replication comparisons and conditional uncertainty."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 p=argparse.ArgumentParser();p.add_argument('report',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 data=json.loads(a.report.read_text());rows=[r for r in data['comparisons'] if r['baseline']=='general-draft']
 cohorts=['rust_fresh','rust_seen','python','prose','reasoning','tool_calls']
 labels=['Fresh Rust (32)','Prior Rust (32)','Python (32)','Prose (8)','Reasoning (8)','JSON calls (8)']
 arms=list(dict.fromkeys(r['candidate'] for r in rows));colors=['#247b9e','#d26a33','#59832d']
 fig,ax=plt.subplots(figsize=(9,5.8),layout='constrained')
 titles={'seed2':'Rust EAGLE-3 replication','scale400':'Rust EAGLE-3 data and training scale','small400':'Rust EAGLE-3 data-size control at 400 updates','mixed400':'Rust EAGLE-3 mixed-data comparison'}
 for j,arm in enumerate(arms):
  selected={r['cohort']:r for r in rows if r['candidate']==arm}
  y=[i+(j-(len(arms)-1)/2)*.22 for i in range(6)]
  x=[100*selected[c]['median_paired_latency_reduction'] for c in cohorts]
  lo=[v-100*selected[c]['bootstrap_95pct_interval'][0] for c,v in zip(cohorts,x)]
  hi=[100*selected[c]['bootstrap_95pct_interval'][1]-v for c,v in zip(cohorts,x)]
  ax.errorbar(x,y,xerr=[lo,hi],fmt='o',capsize=3,label={'seed2':'Rust draft: second seed','rust-draft':'512 Rust / 100 updates','scale100':'2048 Rust / 100 updates','scale400':'2048 Rust / 400 updates','small400':'512 Rust / 400 updates','mixed400':'1536 Rust + 512 general / 400 updates'}.get(arm,arm),color=colors[j%3])
 ax.axvline(0,color='#777777',lw=1);ax.set_yticks(range(6),labels);ax.invert_yaxis();ax.grid(axis='x',alpha=.15)
 ax.set_xlabel('Median paired latency reduction vs original draft (%)\nPositive = faster; intervals bootstrap prompts, conditional on these runs')
 ax.set_title(titles[data['group']]+': counterbalanced sessions',loc='left',fontweight='bold',pad=20)
 ax.legend(loc='best');ax.spines[['top','right']].set_visible(False)
 fig.suptitle('Single-request serving · 1024-token cap · general diagnostics are small samples',fontsize=10,y=.94)
 a.output.parent.mkdir(parents=True,exist_ok=True)
 for ext in ['png','svg']:fig.savefig(a.output.with_suffix('.'+ext),dpi=170)
if __name__=='__main__':main()
