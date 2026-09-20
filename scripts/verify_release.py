"""Recompute the distributed result with no network, GPU, or third-party packages."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from analyze_phase2 import analyze
from analyze_results import compare, load_runs
from benchmark import read_rows

ROOT = Path(__file__).resolve().parents[1]

def verify():
    manifest = json.loads((ROOT/'ARTIFACT_SHA256.json').read_text())
    for name, digest in manifest.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, f'Artifact changed: {name}'
    outputs, tokens, observations = defaultdict(set), defaultdict(set), defaultdict(int)
    count = native = comparisons = 0
    for name, group in [('replication','seed2'),('scale','scale400'),('control','small400'),('mixed','mixed400')]:
        actual = analyze(ROOT/'phase2/results', group)
        saved = json.loads((ROOT/f'phase2/{name}.json').read_text())
        assert actual == saved, f'Reanalysis differs: {name}'
        comparisons += len(actual['comparisons'])
        count += actual['requests']
        for block in (0,1):
            d = ROOT/f'phase2/results/{group}-block{block}'
            session = json.loads((d/'session.json').read_text())
            assert session['arms'] == actual['sessions'][block]['arms']
            for arm in session['arms']:
                for row in read_rows(d/arm/'raw-0.jsonl'):
                    outputs[row['id']].add(row['output'])
                    tokens[row['id']].add(row['completion_tokens'])
                    observations[row['id']] += 1
                if (d/arm/'acceptance.jsonl').exists():native += len(read_rows(d/arm/'acceptance.jsonl'))
    assert count == 3120 and native == 480 and comparisons == 90
    assert len(outputs) == 120 and set(observations.values()) == {26}
    assert all(len(x) == 1 for x in outputs.values())
    assert all(len(x) == 1 for x in tokens.values())
    # Recompute every saved pilot pair, including concurrency cases with output drift.
    pilot = json.loads((ROOT/'run/final-comparison.json').read_text())
    runs = {Path(r['path']).name:r for r in load_runs(ROOT/'run/final-results')}
    for pair in pilot['comparisons']:
        a, b = (runs[Path(pair[k]).name] for k in ('baseline','candidate'))
        expected = {k:v for k,v in pair.items() if k not in {'baseline','candidate'}}
        assert compare(a,b) == expected, 'Pilot reanalysis differs'
    summary = {'verified_artifacts':len(manifest),'followup_comparisons_reproduced':comparisons,
               'pilot_comparisons_reproduced':len(pilot['comparisons']),
               'timed_followup_requests':count,'native_probe_requests':native,
               'unique_prompts_with_identical_outputs_and_token_counts':len(outputs),
               'network_requests':0,'gpu_jobs':0}
    print(json.dumps(summary, indent=2))
    return summary

if __name__ == '__main__':verify()
