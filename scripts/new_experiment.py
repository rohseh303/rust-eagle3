"""Copy code and frozen prompts to an empty destination, without previous results or budget."""
import argparse
import shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('destination',type=Path);a=p.parse_args()
    dest=a.destination.resolve()
    if dest==ROOT or ROOT in dest.parents:p.error('Choose a destination outside the evidence repository')
    dest.mkdir(parents=True,exist_ok=False)
    for name in ('scripts','tests','configs','patches','data','phase2/data'):
        shutil.copytree(ROOT/name,dest/name,ignore=shutil.ignore_patterns('__pycache__','*.log'))
    for name in ('modal_job.py','requirements-local.txt','requirements-report.txt','NOTICE.md','LICENSE','REPRODUCE.md'):
        shutil.copy2(ROOT/name,dest/name)
    print(f'Created {dest}. No prior gate, weights, results, ledger, or cloud work copied.')
if __name__=='__main__':main()
