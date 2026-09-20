"""Summarize measured runs, paired latency differences, and exact output parity."""
import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from benchmark import percentile


def load_runs(root):
    runs = []
    for path in sorted(Path(root).glob('*/run.json')):
        config = json.loads(path.read_text())
        repeats = []
        for raw in sorted(path.parent.glob('raw-*.jsonl')):
            repeats.append([json.loads(x) for x in raw.read_text().splitlines()])
        if repeats:
            runs.append({'path': str(path.parent), 'config': config, 'repeats': repeats})
    return runs


def compare(a, b):
    # Per-prompt median across repeats; bootstrap prompts, not correlated repeats.
    def group(run):
        out = defaultdict(list)
        for repeat in run['repeats']:
            for row in repeat:
                if row['ok']:
                    out[row['id']].append(row)
        return out
    aa, bb = group(a), group(b)
    common = sorted(aa.keys() & bb.keys())
    reductions = []
    identical_reductions = []
    output_matches = 0
    for key in common:
        base = statistics.median(r['latency_s'] for r in aa[key])
        candidate = statistics.median(r['latency_s'] for r in bb[key])
        reductions.append(1 - candidate / base)
        identical = len({r['output'] for r in aa[key] + bb[key]}) == 1
        output_matches += identical
        if identical:
            identical_reductions.append(reductions[-1])
    if not reductions:
        return {'paired_prompts': 0}
    rng = random.Random(20260918)
    boot = sorted(statistics.median(rng.choices(reductions, k=len(reductions)))
                  for _ in range(2000))
    return {'paired_prompts': len(common),
            'median_paired_latency_reduction': statistics.median(reductions),
            'bootstrap_95pct_interval': [boot[49], boot[1949]],
            'prompts_with_all_outputs_identical': output_matches,
            'median_reduction_on_identical_outputs': statistics.median(identical_reductions) if identical_reductions else None,
            'missing_from_baseline': sorted(bb.keys()-aa.keys()),
            'missing_from_candidate': sorted(aa.keys()-bb.keys()),
            'note': 'Positive reduction is faster. Exact equality across all repeats; no sampled-distribution claim.'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('results', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    runs = load_runs(args.results)
    report = {'runs': [], 'comparisons': []}
    for run in runs:
        rows = [r for repeat in run['repeats'] for r in repeat]
        good = [r for r in rows if r['ok']]
        summary_file = Path(run['path'])/'summary.json'
        summaries = json.loads(summary_file.read_text()) if summary_file.exists() else []
        wall = sum(s['wall_s'] for s in summaries)
        report['runs'].append({'path': run['path'], 'arm': run['config']['arm'],
            'suite': Path(run['config']['input']).stem,
            'concurrency': run['config']['concurrency'],
            'requests': len(rows), 'failures': len(rows)-len(good),
            'truncated': sum(r['finish_reason']=='length' for r in good),
            'median_latency_s': statistics.median(r['latency_s'] for r in good) if good else None,
            'p95_latency_s': percentile([r['latency_s'] for r in good], .95),
            'median_ttft_s': percentile([r['ttft_s'] for r in good], .5),
            'median_tpot_s': percentile([r['tpot_s'] for r in good if r['tpot_s'] is not None], .5),
            'aggregate_output_tokens_per_s': sum(r['completion_tokens'] for r in good)/wall if wall else None,
            'completed_repeats': len(run['repeats']),
            'expected_repeats': run['config']['repeats'],
            'median_completion_tokens': statistics.median(r['completion_tokens'] for r in good) if good else None})
    for i, a in enumerate(runs):
        for b in runs[i+1:]:
            if all(a['config'].get(k) == b['config'].get(k)
                   for k in ('input_sha256','concurrency','max_tokens','limit','sampling')):
                # Orient target -> general -> Rust irrespective of run directory ordering.
                order = {'target': 0, 'general-draft': 1, 'rust-draft': 2}
                first, second = sorted([a,b], key=lambda r: order[r['config']['arm']])
                report['comparisons'].append({'baseline': first['path'], 'candidate': second['path'],
                                               **compare(first, second)})
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
