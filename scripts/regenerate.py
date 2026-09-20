"""Collect frozen-target responses, preserving exact chat-template rendering."""
import argparse
import concurrent.futures
import json
from pathlib import Path
from benchmark import read_rows, request_one, chat_token_ids


def render_training_text(prefix, result, eos_token):
    if not result['ok'] or result['finish_reason'] not in {'stop','length'}:
        raise ValueError('Only successful target responses can enter training')
    # A capped answer is a valid target-generated prefix. Do not invent EOS at
    # its cap, or discard harder/longer examples based on the target response.
    suffix=eos_token+'\n' if result['finish_reason']=='stop' else ''
    return prefix+result['output']+suffix


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--target-path',required=True)
    p.add_argument('--url',default='http://127.0.0.1:30000')
    p.add_argument('--concurrency',type=int,default=4)
    p.add_argument('--max-tokens',type=int,default=1024)
    p.add_argument('--max-seq-len',type=int,default=3072)
    a=p.parse_args()
    if a.output.exists():
        raise SystemExit('Refusing to overwrite an existing generation corpus')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(a.target_path,trust_remote_code=False,local_files_only=True)
    rows=read_rows(a.input)
    if a.input.stem not in {'train','validation'}:
        raise SystemExit('Regeneration accepts train/validation files only; benchmark data is evaluation-only')
    excluded=[]
    eligible=[]
    for row in rows:
        prefix=chat_token_ids(tokenizer,row['prompt'])
        if len(prefix)+a.max_tokens+1 > a.max_seq_len:
            excluded.append({'id':row['id'],'prompt_tokens':len(prefix),'reason':'sequence_length_budget'})
        else:
            eligible.append(row)
    rows=eligible
    if not rows:
        raise SystemExit('No prompts fit the sequence length budget')
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.concurrency) as pool:
        results=list(pool.map(lambda r:request_one(a.url,'rust-target',r,a.max_tokens,0.8),rows))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    diagnostics=a.output.with_suffix('.requests.jsonl')
    diagnostics.write_text(''.join(json.dumps(r)+'\n' for r in results))
    a.output.with_suffix('.excluded.json').write_text(json.dumps(excluded,indent=2))
    failures=[r for r in results if not r['ok'] or r['finish_reason'] not in {'stop','length'}]
    if failures:
        raise SystemExit(f'{len(failures)} failed responses. Raw requests saved; no training corpus emitted.')
    temporary=a.output.with_suffix('.partial')
    with temporary.open('w') as f:
        for row,result in zip(rows,results):
            prefix=tokenizer.apply_chat_template([{'role':'user','content':row['prompt']}],
                tokenize=False,add_generation_prompt=True,enable_thinking=False)
            # Verify this token matches the pinned target, never assume another model's EOS.
            if tokenizer.eos_token != '<|im_end|>':
                raise RuntimeError('Unexpected Qwen EOS token')
            text=render_training_text(prefix,result,tokenizer.eos_token)
            if len(tokenizer.encode(text,add_special_tokens=False))>a.max_seq_len:
                raise RuntimeError('Rendered training sequence exceeds capture limit')
            f.write(json.dumps({'id':row['id'],'text':text,'finish_reason':result['finish_reason']})+'\n')
    temporary.replace(a.output)
    print(f'Wrote {len(rows)} target-generated conversations to {a.output}')
    print(f"Capped prefixes without synthetic EOS: {sum(r['finish_reason']=='length' for r in results)}")


if __name__=='__main__':
    main()
