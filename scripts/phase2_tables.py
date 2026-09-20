"""Export measured comparisons and training exposure without inventing rows."""
import csv
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    comparisons=[]
    for group in ['replication','scale','control','mixed']:
        p=ROOT/f'phase2/{group}.json'
        if not p.exists():continue
        report=json.loads(p.read_text())
        for c in report['comparisons']:
            lo,hi=c['bootstrap_95pct_interval'];cl=c.get('crate_bootstrap_95pct_interval',[None,None])
            comparisons.append({'study':group,'baseline':c['baseline'],'candidate':c['candidate'],'cohort':c['cohort'],
                'prompts':c['paired_prompts'],'distinct_crates':c.get('distinct_crates'),
                'latency_reduction_pct':100*c['median_paired_latency_reduction'],
                'prompt_ci_low_pct':100*lo,'prompt_ci_high_pct':100*hi,
                'crate_ci_low_pct':100*cl[0] if cl[0] is not None else None,
                'crate_ci_high_pct':100*cl[1] if cl[1] is not None else None,
                'identical_output_prompts':c['prompts_with_all_outputs_identical'],
                'session0_reduction_pct':100*c['by_session'][0]['median_paired_latency_reduction'],
                'session1_reduction_pct':100*c['by_session'][1]['median_paired_latency_reduction']})
    out=ROOT/'phase2/comparisons.csv'
    with out.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(comparisons[0]));writer.writeheader();writer.writerows(comparisons)
    training=[]
    for variant in ['seed2','scale100','scale400','small400','mixed400']:
        p=ROOT/f'phase2/audits/{variant}.json'
        if not p.exists():continue
        audit=json.loads(p.read_text());best=audit['metadata'][f'{variant}.best_meta.json'];step=best['step']
        exposure=audit.get('exposure_by_optimizer_step',{}).get(str(step),{})
        if variant=='seed2':
            ref=ROOT/'phase2/audits/512-reference-exposures.json'
            if ref.exists():exposure=json.loads(ref.read_text())['20260919']
        training.append({'variant':variant,'selected_step':step,'validation_proxy':best['score'],
            'seed':audit['config']['training']['seed'],
            **{k:exposure.get(k) for k in ['example_exposures','unique_examples_seen','teacher_prompt_token_exposures','teacher_completion_token_exposures']}})
    with (ROOT/'phase2/training-exposure.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(training[0]));writer.writeheader();writer.writerows(training)
    print(json.dumps({'comparison_rows':len(comparisons),'training_rows':len(training)}))
if __name__=='__main__':main()
