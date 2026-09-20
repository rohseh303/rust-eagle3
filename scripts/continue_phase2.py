"""Sequential, budget-guarded continuation after capture; stops on any error.

Run with a Python containing the Modal SDK. No detached jobs or automatic retries.
"""
import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--group',choices=['scale','mixed','control'],required=True);a=p.parse_args()
    if a.group=='scale':
        stages=[('train100',['--stage','phase2-train','--variant','scale100']),
                ('train400-part0',['--stage','phase2-train','--variant','scale400','--block','0']),
                ('train400-part1',['--stage','phase2-train','--variant','scale400','--block','1']),
                ('evaluate-scale-block0',['--stage','phase2-evaluate','--variant','scale400','--block','0']),
                ('evaluate-scale-block1',['--stage','phase2-evaluate','--variant','scale400','--block','1'])]
        variant='scale400'
    elif a.group=='control':
        stages=[('train-small-part0',['--stage','phase2-train','--variant','small400','--block','0']),
                ('train-small-part1',['--stage','phase2-train','--variant','small400','--block','1']),
                ('evaluate-small-block0',['--stage','phase2-evaluate','--variant','small400','--block','0']),
                ('evaluate-small-block1',['--stage','phase2-evaluate','--variant','small400','--block','1'])]
        variant='small400'
    else:
        stages=[('generate-general',['--stage','phase2-generate','--variant','mixed400']),
                ('capture-general',['--stage','phase2-capture','--variant','mixed400']),
                ('train-mixed-part0',['--stage','phase2-train','--variant','mixed400','--block','0']),
                ('train-mixed-part1',['--stage','phase2-train','--variant','mixed400','--block','1']),
                ('evaluate-mixed-block0',['--stage','phase2-evaluate','--variant','mixed400','--block','0']),
                ('evaluate-mixed-block1',['--stage','phase2-evaluate','--variant','mixed400','--block','1'])]
        variant='mixed400'
    progress=ROOT/f'phase2/{a.group}-progress.json'
    if progress.exists():raise SystemExit('Progress already exists: review before any explicit recovery')
    state={'group':a.group,'steps':[]}
    for name,args in stages:
        entry={'name':name,'args':args,'state':'running','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
        state['steps'].append(entry);progress.write_text(json.dumps(state,indent=2));print('START',name,flush=True)
        log=ROOT/f'phase2/{name}-launch.log'
        with log.open('x') as f:
            code=subprocess.call([sys.executable,ROOT/'scripts/run_budgeted.py',*args],stdout=f,stderr=subprocess.STDOUT)
        entry.update(state='finished' if code==0 else 'failed',exit_code=code,ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        progress.write_text(json.dumps(state,indent=2));print('END',name,code,flush=True)
        if code:raise SystemExit(code)
    for remote,output in [('/phase2/results','results'),('/phase2/audits','audits'),('/phase2/regenerated','regenerated')]:
        subprocess.run([sys.executable,ROOT/'scripts/sync_results.py','--remote',remote,'--output',ROOT/'phase2'/output],check=True)
    subprocess.run([sys.executable,ROOT/'scripts/analyze_phase2.py','--group',variant,'--output',ROOT/f'phase2/{a.group}.json'],check=True)
    state['analysis_complete']=True;progress.write_text(json.dumps(state,indent=2))
if __name__=='__main__':main()
