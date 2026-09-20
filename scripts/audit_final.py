"""Check final-run completeness and record observed output disagreements."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from analyze_results import load_runs


def main():
    p=argparse.ArgumentParser()
    p.add_argument('results',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    runs=[r for r in load_runs(a.results) if Path(r['config']['input']).stem!='validation']
    expected={(arm,suite,c) for arm in ('target','general-draft','rust-draft')
              for suite,c in [('heldout',1),('heldout',4),('heldout',16),('humaneval-python',1)]}
    observed=set();errors=[];groups=defaultdict(lambda:defaultdict(list))
    for run in runs:
        cfg=run['config'];suite=Path(cfg['input']).stem
        key=(cfg['arm'],suite,cfg['concurrency'])
        if key in observed:errors.append(f'Duplicate case: {key}')
        observed.add(key)
        if len(run['repeats'])!=3:errors.append(f'Expected three repeats: {key}')
        for repeat in run['repeats']:
            if len(repeat)!=32 or len({r['id'] for r in repeat})!=32:
                errors.append(f'Expected 32 unique requests: {key}')
            for row in repeat:
                if not row['ok']:errors.append(f'Failed request: {key} {row["id"]}')
                groups[(suite,cfg['concurrency'])][row['id']].append({'arm':cfg['arm'],**row})
    if observed!=expected:errors.append(f'Missing/extra cases: {expected^observed}')
    disagreements=[];counts={}
    for case,prompts in groups.items():
        same=0
        repeatable={arm:0 for arm in ('target','general-draft','rust-draft')}
        for pid,rows in prompts.items():
            outputs={r['output'] for r in rows}
            if len(rows)!=9:errors.append(f'Expected nine outputs: {case} {pid}')
            if len({r['prompt_sha256'] for r in rows})!=1:errors.append(f'Prompt differs: {case} {pid}')
            same+=len(outputs)==1
            for arm in repeatable:
                arm_rows=[r for r in rows if r['arm']==arm]
                repeatable[arm]+=len(arm_rows)==3 and len({r['output'] for r in arm_rows})==1
            if len(outputs)>1:
                byarm={arm:[r for r in rows if r['arm']==arm] for arm in ('target','general-draft','rust-draft')}
                disagreements.append({'suite':case[0],'concurrency':case[1],'id':pid,
                    'distinct_outputs_by_arm':{arm:len({r['output'] for r in rr}) for arm,rr in byarm.items()},
                    'completion_tokens_by_arm':{arm:[r.get('completion_tokens') for r in rr] for arm,rr in byarm.items()}})
        counts[f'{case[0]}-c{case[1]}']={'prompts':len(prompts),'all_nine_outputs_identical':same,
                                       'repeatable_prompts_by_arm':repeatable}
    probes={}
    probe_rows={}
    for arm in ('general-draft','rust-draft'):
        path=a.results/f'{arm}-acceptance/requests.jsonl'
        if not path.exists():errors.append(f'Missing acceptance probe: {arm}');continue
        rows=[json.loads(x) for x in path.read_text().splitlines()]
        probe_rows[arm]={r['id']:r['response'] for r in rows}
        meta=[r['response']['meta_info'] for r in rows]
        fields=('spec_num_correct_drafts','spec_num_proposed_drafts','spec_verify_ct')
        if len(rows)!=16 or any(any(k not in m for k in fields) for m in meta):
            errors.append(f'Incomplete acceptance probe: {arm}');continue
        totals={k:sum(m[k] for m in meta) for k in fields}
        proposed=totals['spec_num_proposed_drafts'];accepted=totals['spec_num_correct_drafts']
        if not 0<=accepted<=proposed or not proposed:errors.append(f'Invalid acceptance counters: {arm}')
        probes[arm]={'requests':len(rows),**totals,'acceptance_rate':accepted/proposed if proposed else None,
                     'completion_tokens':sum(m['completion_tokens'] for m in meta)}
    probe_agreement={}
    if len(probe_rows)==2:
        left,right=probe_rows['general-draft'],probe_rows['rust-draft']
        if left.keys()!=right.keys():errors.append('Acceptance probes used different prompt IDs')
        common=left.keys() & right.keys()
        probe_agreement={'paired_prompts':len(common),
                         'identical_text':sum(left[k]['text']==right[k]['text'] for k in common)}
    report={'completeness_errors':errors,'case_output_agreement':counts,
            'disagreements':disagreements,'acceptance_probes':probes,
            'probe_output_agreement':probe_agreement}
    a.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    if errors:raise SystemExit('Final evaluation is incomplete; see audit')


if __name__=='__main__':main()
