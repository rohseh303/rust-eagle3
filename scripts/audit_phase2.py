"""Audit complete follow-up studies directly from saved request records."""
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    source = ROOT / 'phase2/data/evaluation.jsonl'
    prompts = {r['id']: r for r in rows(source)}
    outputs, counts, token_counts = defaultdict(set), defaultdict(int), defaultdict(set)
    studies = []
    for name, variant in [('replication', 'seed2'), ('scale', 'scale400'),
                          ('control', 'small400'), ('mixed', 'mixed400')]:
        report_path = ROOT / f'phase2/{name}.json'
        if not report_path.exists():
            continue
        report = json.loads(report_path.read_text())
        failures, capped, timed, native, fresh_caps = [], [], 0, 0, []
        sessions = []
        for block in (0, 1):
            base = ROOT / f'phase2/results/{variant}-block{block}'
            session = json.loads((base / 'session.json').read_text())
            sessions.append(session)
            for arm in session['arms']:
                config = json.loads((base / arm / 'run.json').read_text())
                assert config['concurrency'] == config['repeats'] == 1
                assert config['max_tokens'] == 1024
                assert config['sampling'] == {'temperature': 0, 'enable_thinking': False}
                assert config['input_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
                requests = rows(base / arm / 'raw-0.jsonl')
                assert len(requests) == len(prompts)
                assert {r['id'] for r in requests} == set(prompts)
                for r in requests:
                    key = r['id']
                    assert r['prompt_sha256'] == hashlib.sha256(prompts[key]['prompt'].encode()).hexdigest()
                    timed += 1
                    assert math.isfinite(r['latency_s']) and r['latency_s'] > 0
                    if not r['ok']:
                        failures.append([block, arm, key])
                    if r['finish_reason'] == 'length':
                        capped.append([block, arm, key])
                        if prompts[key]['task'] == 'rust_fresh':
                            fresh_caps.append([block, arm, key])
                    outputs[key].add(r['output'])
                    token_counts[key].add(r['completion_tokens'])
                    counts[key] += 1
                probe = base / arm / 'acceptance.jsonl'
                if probe.exists():
                    native += len(rows(probe))
        assert timed == report['requests'] and len(capped) == report['capped_requests']
        assert not failures and not fresh_caps
        assert sessions[0]['arms'] == list(reversed(sessions[1]['arms']))
        studies.append({'study': name, 'timed_requests': timed, 'native_requests': native,
                        'failed_requests': failures, 'capped_requests': capped,
                        'fresh_rust_capped_requests': fresh_caps, 'sessions': sessions})
    out = {'evaluation_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
           'studies': studies, 'timed_requests': sum(x['timed_requests'] for x in studies),
           'native_requests': sum(x['native_requests'] for x in studies),
           'unique_prompts': len(prompts),
           'prompts_identical_across_all_saved_timed_outputs': sum(len(v) == 1 for v in outputs.values()),
           'differing_prompt_ids': [k for k, v in outputs.items() if len(v) != 1],
           'prompts_with_identical_completion_token_counts': sum(len(v) == 1 for v in token_counts.values()),
           'timed_observations_per_prompt': dict(counts),
           'note': 'Exact greedy text equality on these finite tests is not a proof of sampled-distribution equivalence.'}
    (ROOT / 'phase2/final-audit.json').write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k not in ('studies', 'timed_observations_per_prompt')}, indent=2))


if __name__ == '__main__':
    main()
