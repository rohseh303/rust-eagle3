"""Launch one bounded Modal stage, with a persistent conservative cost ledger.

This is a local launch guard, not a provider billing limit. Never run concurrent
launchers or bypass the user's recorded cumulative authorization.
"""
import datetime
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'run'
BUDGET = 55.0  # Historical operating limit, including reserve. New runs also need an explicit ledger.
RESERVE = 5.0  # Untouched allowance for storage/build/termination uncertainty.
STAGE_RESERVATION = 5.0  # 40-minute outer timeout, conservatively priced.
HOURLY_ESTIMATE = 6.0  # Historical experiment allowance; check current provider rates before a new run.


def main():
    args = sys.argv[1:]
    stage = args[args.index('--stage')+1] if '--stage' in args else ''
    variant = args[args.index('--variant')+1] if '--variant' in args else 'seed2'
    if stage.startswith('phase2-') and variant != 'seed2':
        gate = ROOT / 'phase2/replication.json'
        if not gate.exists() or json.loads(gate.read_text()).get('scale_gate_passed') is not True:
            raise SystemExit('Scaling guard: the predeclared replication gate has not passed.')
    RUN.mkdir(exist_ok=True)
    with (RUN / 'budget.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = RUN / 'budget.json'
        if not path.exists():
            raise SystemExit('Initialize an explicit budget first: python scripts/init_budget.py --usd AMOUNT')
        ledger = json.loads(path.read_text())
        authorized = float(ledger.get('user_absolute_ceiling_usd', ledger.get('authorized_usd', 50.0)))
        ceiling = min(BUDGET, authorized)
        committed = sum(r['charged_estimate_usd'] for r in ledger['runs'])
        if committed + STAGE_RESERVATION + RESERVE > ceiling:
            raise SystemExit(f'Budget guard: ${committed:.2f} committed; insufficient safe allowance for another stage.')
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        previous_ceiling = ledger.get('operating_ceiling_usd', 50.0)
        if previous_ceiling != ceiling:
            ledger.setdefault('operating_ceiling_changes', []).append({
                'recorded_utc': now, 'from_usd': previous_ceiling, 'to_usd': ceiling,
                'absolute_authorization_usd': authorized, 'reserve_usd': RESERVE,
                'reason': 'Apply the lower of configured operating limit and recorded budget, preserving all prior entries.'})
        ledger['operating_ceiling_usd'] = ceiling
        entry = {'started_utc': now, 'args': sys.argv[1:],
                 'charged_estimate_usd': STAGE_RESERVATION, 'state': 'reserved',
                 'operating_ceiling_usd': ceiling}
        ledger['runs'].append(entry)
        path.write_text(json.dumps(ledger, indent=2))
        started = time.monotonic()
        # No detach: losing the local launcher stops its Modal app.
        try:
            with subprocess.Popen(['modal', 'run', 'modal_job.py', *sys.argv[1:]], cwd=ROOT) as result:
                while result.poll() is None:
                    elapsed = time.monotonic() - started
                    if elapsed >= 2400:
                        result.kill()
                        result.wait()
                        raise subprocess.TimeoutExpired(result.args, 2400)
                    try:
                        result.wait(timeout=min(30, 2400-elapsed))
                    except subprocess.TimeoutExpired:
                        elapsed = time.monotonic() - started
                        entry['live_elapsed_seconds'] = elapsed
                        entry['live_estimate_usd'] = elapsed / 3600 * HOURLY_ESTIMATE + 0.5
                        path.write_text(json.dumps(ledger, indent=2))
                        print('COST_MONITOR', json.dumps({
                            'elapsed_seconds': round(elapsed),
                            'cumulative_estimate_usd': round(committed + entry['live_estimate_usd'], 2),
                            'cumulative_reserved_usd': round(committed + STAGE_RESERVATION, 2),
                            'operating_ceiling_usd': ceiling}), flush=True)
        except subprocess.TimeoutExpired:
            entry.update(state='outer_timeout', elapsed_seconds=time.monotonic()-started)
            path.write_text(json.dumps(ledger, indent=2))
            raise SystemExit('Outer timeout: reservation retained. Confirm this ephemeral Modal app is stopped before another launch.')
        elapsed = time.monotonic() - started
        entry.update(state='finished' if result.returncode == 0 else 'failed',
                     exit_code=result.returncode, elapsed_seconds=elapsed)
        # Keep full reservation after a failure: remote termination is uncertain.
        if result.returncode == 0:
            entry['charged_estimate_usd'] = max(0.5, elapsed / 3600 * HOURLY_ESTIMATE + 0.5)
        path.write_text(json.dumps(ledger, indent=2))
        print('BUDGET', json.dumps({'committed_estimate_usd': sum(r['charged_estimate_usd'] for r in ledger['runs']),
                                    'ceiling_usd': ceiling, 'reserve_usd': RESERVE}), flush=True)
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
