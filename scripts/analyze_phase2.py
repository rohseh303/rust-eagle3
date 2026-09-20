"""Analyze predeclared cohorts without treating repeats as independent prompts."""
import argparse
import hashlib
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from analyze_results import compare
from benchmark import read_rows
ROOT=Path(__file__).resolve().parents[1]

def crate_sensitivity(a,b,crates):
    groups=[]
    for run in (a,b):
        grouped=defaultdict(list)
        for repeat in run['repeats']:
            for row in repeat:grouped[row['id']].append(row['latency_s'])
        groups.append(grouped)
    by_crate=defaultdict(list)
    for key in sorted(groups[0]):
        delta=1-statistics.median(groups[1][key])/statistics.median(groups[0][key])
        by_crate[crates[key]].append(delta)
    rng=random.Random(20260918);keys=sorted(by_crate)
    boot=sorted(statistics.median([v for k in rng.choices(keys,k=len(keys)) for v in by_crate[k]]) for _ in range(2000))
    return {'distinct_crates':len(keys),'crate_bootstrap_95pct_interval':[boot[49],boot[1949]],
            'crate_analysis_note':'Supplementary sensitivity analysis, added after replication; primary predeclared gate still uses prompt bootstrap.'}

def analyze(results,group):
    source=ROOT/'phase2/data/evaluation.jsonl'
    prompts=read_rows(source)
    cohorts={r['id']:r['task'] for r in prompts}
    hashes={r['id']:hashlib.sha256(r['prompt'].encode()).hexdigest() for r in prompts}
    crates={r['id']:r['crate'] for r in prompts if r['task'].startswith('rust_')}
    runs={};sessions=[]
    for block in (0,1):
        d=Path(results)/f'{group}-block{block}'
        session=json.loads((d/'session.json').read_text());sessions.append(session)
        for arm in session['arms']:
            config=json.loads((d/arm/'run.json').read_text())
            rows=read_rows(d/arm/'raw-0.jsonl')
            assert len(rows)==len(cohorts) and {r['id'] for r in rows}==set(cohorts)
            assert all(r['ok'] for r in rows), 'Failed requests invalidate this analysis'
            assert config['repeats']==1 and config['concurrency']==1
            assert config['input_sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
            assert all(r['prompt_sha256']==hashes[r['id']] for r in rows)
            runs[block,arm]=rows
    comparisons=[]
    arms=sessions[0]['arms']
    pairs=[('general-draft',a) for a in arms if a!='general-draft']
    if group=='seed2':pairs.append(('rust-draft','seed2'))
    elif group=='scale400':pairs.extend([('rust-draft','scale100'),('rust-draft','scale400'),('scale100','scale400')])
    elif group=='mixed400':pairs.append(('scale400','mixed400'))
    elif group=='small400':pairs.append(('small400','scale400'))
    for baseline,candidate in pairs:
        for cohort in sorted(set(cohorts.values())):
            selected=lambda b,a:[r for r in runs[b,a] if cohorts[r['id']]==cohort]
            aa={'repeats':[selected(b,baseline) for b in (0,1)]}
            bb={'repeats':[selected(b,candidate) for b in (0,1)]}
            per=[compare({'repeats':[aa['repeats'][b]]},{'repeats':[bb['repeats'][b]]}) for b in (0,1)]
            sensitivity=crate_sensitivity(aa,bb,crates) if cohort.startswith('rust_') else {}
            comparisons.append({'baseline':baseline,'candidate':candidate,'cohort':cohort,**compare(aa,bb),'by_session':per,**sensitivity})
    out={'group':group,'sessions':sessions,'requests':sum(map(len,runs.values())),'failures':0,
         'capped_requests':sum(r['finish_reason']=='length' for rows in runs.values() for r in rows),
         'comparisons':comparisons,
         'limitation':'Prompt-bootstrap intervals condition on two sessions and available seeds; not independent training-corpus replications. Small diagnostic cohorts do not prove absence of regression.'}
    if group=='seed2':
        primary=[c for c in comparisons if c['baseline']=='general-draft' and c['cohort']=='rust_fresh']
        out['scale_gate_passed']=len(primary)==2 and all(c['bootstrap_95pct_interval'][0]>0 and c['prompts_with_all_outputs_identical']==32 and all(s['median_paired_latency_reduction']>0 for s in c['by_session']) for c in primary)
    else:
        probes=read_rows(ROOT/'phase2/data/acceptance.jsonl')
        probe_ids={r['id'] for r in probes}
        native=defaultdict(list)
        texts=defaultdict(set)
        for block in (0,1):
            for arm in arms:
                rows=read_rows(Path(results)/f'{group}-block{block}'/arm/'acceptance.jsonl')
                assert len(rows)==24 and {r['id'] for r in rows}==probe_ids
                for row in rows:
                    assert row['prompt_sha256']==hashes[row['id']]
                    native[arm,cohorts[row['id']]].append(row['response']['meta_info'])
                    texts[row['id']].add(row['response']['text'])
        out['native_acceptance']=[]
        for (arm,cohort),rows in sorted(native.items()):
            accepted=sum(r['spec_num_correct_drafts'] for r in rows)
            proposed=sum(r['spec_num_proposed_drafts'] for r in rows)
            out['native_acceptance'].append({'arm':arm,'cohort':cohort,'requests':len(rows),
                'accepted':accepted,'proposed':proposed,'acceptance_rate':accepted/proposed,
                'verification_steps':sum(r['spec_verify_ct'] for r in rows),
                'completion_tokens':sum(r['completion_tokens'] for r in rows)})
        out['native_prompts_with_all_outputs_identical']=sum(len(v)==1 for v in texts.values())
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--group',choices=['seed2','scale400','mixed400','small400'],required=True)
    p.add_argument('--results',type=Path,default=ROOT/'phase2/results');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=analyze(a.results,a.group);a.output.write_text(json.dumps(out,indent=2))
    print(json.dumps({k:v for k,v in out.items() if k!='comparisons'},indent=2))
    for c in out['comparisons']:
        print(c['baseline'],c['candidate'],c['cohort'],round(100*c['median_paired_latency_reduction'],2),[round(100*x,2) for x in c['bootstrap_95pct_interval']],c['prompts_with_all_outputs_identical'])
if __name__=='__main__':main()
