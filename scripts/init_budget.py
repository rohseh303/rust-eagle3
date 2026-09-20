"""Initialize an explicit new budget; never overwrite a ledger or launch a job."""
import argparse
import datetime
import json
import math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--usd',type=float,required=True);a=p.parse_args()
    if not math.isfinite(a.usd) or not 0<a.usd<=60:p.error('Choose a finite positive ceiling no greater than $60; historical operating limit is $55')
    directory=ROOT/'run';directory.mkdir(exist_ok=True)
    with (directory/'budget.json').open('x') as f:
        json.dump({'authorized_usd':a.usd,'user_absolute_ceiling_usd':a.usd,
                   'authorization':'Explicit local initialization by the operator',
                   'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   'safety_reserve_usd':5,'runs':[]},f,indent=2)
    print('Budget initialized. No cloud work started. Existing ledgers are never reset.')
if __name__=='__main__':main()
