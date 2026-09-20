"""Closed-loop streaming benchmark. Saves failures and raw outputs, never invented metrics."""
import argparse
import concurrent.futures
import hashlib
import json
import random
import statistics
import time
import urllib.request
from pathlib import Path


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def chat_token_ids(tokenizer, prompt):
    ids = tokenizer.apply_chat_template([{'role':'user','content':prompt}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
    if not isinstance(ids, list) or not all(isinstance(x, int) for x in ids):
        raise TypeError('Expected a flat token-ID list from the pinned tokenizer')
    return ids


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * q
    lo = int(position)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def request_one(url, model, row, max_tokens, temperature=0.0):
    payload = {
        "model": model, "messages": [{"role": "user", "content": row["prompt"]}],
        "temperature": temperature, "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": True, "stream_options": {"include_usage": True},
        "return_spec_tokens_details": True,
        "seed": int(hashlib.sha256(str(row['id']).encode()).hexdigest()[:8],16) % (2**31),
    }
    request = urllib.request.Request(url.rstrip('/') + '/v1/chat/completions',
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    result = {"id": row["id"], "task": row.get("task"), "prompt_sha256": hashlib.sha256(row['prompt'].encode()).hexdigest()}
    parts, first, usage, reason, done = [], None, None, None, False
    spec_details = None
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            for line in response:
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    done = True
                    break
                event = json.loads(data)
                if event.get("error"):
                    raise RuntimeError(str(event['error']))
                if event.get("usage"):
                    usage = event["usage"]
                if (event.get('sglext') or {}).get('spec_tokens_details'):
                    spec_details = event['sglext']['spec_tokens_details']
                for choice in event.get("choices", []):
                    text = choice.get("delta", {}).get("content") or ""
                    if text:
                        first = first if first is not None else time.perf_counter() - start
                        parts.append(text)
                    reason = choice.get("finish_reason") or reason
        if not done or not usage or not reason or first is None:
            raise RuntimeError("Incomplete stream, empty output, or missing token usage/finish reason")
        elapsed = time.perf_counter() - start
        count = usage['completion_tokens']
        result.update(ok=True, output=''.join(parts), latency_s=elapsed, ttft_s=first,
            completion_tokens=count, prompt_tokens=usage['prompt_tokens'], finish_reason=reason,
            spec_tokens_details=spec_details,
            tpot_s=(elapsed-first)/(count-1) if count > 1 else None)
    except Exception as exc:
        result.update(ok=False, error=f"{type(exc).__name__}: {exc}", output=''.join(parts), latency_s=time.perf_counter()-start)
    return result


def summarize(results, duration):
    good = [r for r in results if r['ok']]
    spec = [r.get('spec_tokens_details') for r in good]
    has_counts = bool(spec) and all(isinstance(s,dict) and
        'spec_num_correct_drafts' in s and 'spec_num_proposed_drafts' in s for s in spec)
    proposed = sum(s['spec_num_proposed_drafts'] for s in spec) if has_counts else 0
    accepted = sum(s['spec_num_correct_drafts'] for s in spec) if has_counts else 0
    return {
        "requests": len(results), "successes": len(good), "failures": len(results)-len(good),
        "truncated": sum(r['finish_reason']=='length' for r in good),
        "wall_s": duration,
        "output_tokens_per_s": sum(r['completion_tokens'] for r in good)/duration,
        "median_latency_s": percentile([r['latency_s'] for r in good], .5),
        "p95_latency_s": percentile([r['latency_s'] for r in good], .95),
        "median_ttft_s": percentile([r['ttft_s'] for r in good], .5),
        "p95_ttft_s": percentile([r['ttft_s'] for r in good], .95),
        "median_tpot_s": percentile([r['tpot_s'] for r in good if r['tpot_s'] is not None], .5),
        "acceptance_rate": accepted/proposed if proposed else None,
        "accepted_draft_tokens": accepted if has_counts else None,
        "proposed_draft_tokens": proposed if has_counts else None,
        "acceptance_note": "Ratio of summed per-request accepted/proposed draft counters when available; otherwise unavailable. Bonus tokens excluded.",
    }


def save_metrics(url, path):
    try:
        with urllib.request.urlopen(url.rstrip('/')+'/metrics', timeout=10) as response:
            path.write_bytes(response.read())
    except Exception as exc:
        path.write_text(f"UNAVAILABLE: {type(exc).__name__}: {exc}\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--url', default='http://127.0.0.1:30000')
    p.add_argument('--model', default='rust-target')
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--arm', required=True)
    p.add_argument('--concurrency', type=int, default=1)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--limit', type=int)
    p.add_argument('--max-tokens', type=int, default=1024)
    a=p.parse_args()
    if min(a.concurrency,a.repeats,a.max_tokens) < 1:
        p.error('Counts must be positive')
    a.output.mkdir(parents=True, exist_ok=False)
    rows=read_rows(a.input)
    random.Random(20260918).shuffle(rows)
    if a.limit:
        rows=rows[:a.limit]
    if not rows:
        p.error('No prompts')
    metadata={**vars(a), 'input_sha256':hashlib.sha256(a.input.read_bytes()).hexdigest(),
              'measurement':'closed-loop HTTP streaming; TTFT is first non-empty content event; natural EOS',
              'sampling':{'temperature':0,'enable_thinking':False},
              'required_server_settings':'prefix cache disabled; fixed target revision, precision, GPU, scheduler settings'}
    (a.output/'run.json').write_text(json.dumps(metadata,default=str,indent=2)+'\n')
    # Warm the actual request shape and concurrency, excluded from measurements.
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.concurrency) as pool:
        warm=list(pool.map(lambda r: request_one(a.url,a.model,r,a.max_tokens),rows[:max(a.concurrency,2)]))
    if not all(r['ok'] for r in warm):
        (a.output/'warmup-errors.json').write_text(json.dumps(warm,indent=2))
        raise SystemExit('Warmup failed; benchmark aborted')
    summaries=[]
    for repeat in range(a.repeats):
        order=list(rows)
        random.Random(20260918+repeat).shuffle(order)
        save_metrics(a.url,a.output/f'metrics-{repeat}-before.txt')
        start=time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.concurrency) as pool:
            results=list(pool.map(lambda r: request_one(a.url,a.model,r,a.max_tokens),order))
        elapsed=time.perf_counter()-start
        save_metrics(a.url,a.output/f'metrics-{repeat}-after.txt')
        with (a.output/f'raw-{repeat}.jsonl').open('w') as f:
            for result in results:
                f.write(json.dumps(result)+'\n')
        summaries.append(summarize(results,elapsed))
    (a.output/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    print(json.dumps(summaries,indent=2))
    if any(s['failures'] for s in summaries):
        raise SystemExit('Requests failed; results retained but this run is invalid')


if __name__=='__main__':
    main()
