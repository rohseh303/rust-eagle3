"""Create exportable pilot figures from measured results only."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    report=json.loads(a.report.read_text())
    runs={r['path']:r for r in report['runs']}
    selected=[]
    for c in report['comparisons']:
        base,candidate=runs[c['baseline']],runs[c['candidate']]
        if base['arm']=='general-draft' and candidate['arm']=='rust-draft' and base['suite']!='validation':
            selected.append((base,c))
    selected.sort(key=lambda x:(x[0]['suite']!='heldout',x[0]['concurrency']))
    if len(selected)!=4:
        raise SystemExit('Expected four complete final comparisons')
    labels=[f"{'Rust' if r['suite']=='heldout' else 'Python'} · {r['concurrency']} concurrent"+
            (' *' if c['prompts_with_all_outputs_identical']<c['paired_prompts'] else '') for r,c in selected]
    centers=[100*c['median_paired_latency_reduction'] for r,c in selected]
    lower=[v-100*c['bootstrap_95pct_interval'][0] for v,(r,c) in zip(centers,selected)]
    upper=[100*c['bootstrap_95pct_interval'][1]-v for v,(r,c) in zip(centers,selected)]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
    fig,ax=plt.subplots(figsize=(9,4.5))
    ax.axvline(0,color='#7a8697',linewidth=1)
    ax.errorbar(centers,range(4),xerr=[lower,upper],fmt='o',color='#146b75',capsize=5,markersize=8,linewidth=2)
    ax.set_yticks(range(4),labels);ax.invert_yaxis()
    ax.set_xlabel('Median paired latency reduction vs original draft (%)')
    ax.set_title('Rust adaptation of an EAGLE-3 draft',loc='left',fontweight='bold',pad=18)
    ax.grid(axis='x',alpha=.2)
    for spine in ['top','right','left']:ax.spines[spine].set_visible(False)
    fig.text(.02,.04,'32 prompts per case · 3 repeats · H100 · exploratory 95% prompt-bootstrap intervals',fontsize=9,color='#526070')
    fig.text(.02,.012,'* Some outputs differ: timing differences are not a clean equivalent-output speedup.',fontsize=9,color='#526070')
    fig.tight_layout(rect=(0,.085,1,1))
    a.output.mkdir(parents=True,exist_ok=True)
    fig.savefig(a.output/'latency-reduction.png',dpi=180)
    fig.savefig(a.output/'latency-reduction.svg')
    plt.close(fig)


if __name__=='__main__':main()
