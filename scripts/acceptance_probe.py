"""Read per-request speculative counters from SGLang's native endpoint.

SGLang 0.5.18 does not expose these fields in its OpenAI streaming response.
These are separate diagnostic requests, excluded from latency measurements.
"""
import argparse
import hashlib
import json
import random
import urllib.request
from pathlib import Path
from benchmark import chat_token_ids


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--target-path',required=True)
    p.add_argument('--limit',type=int,default=16)
    p.add_argument('--url',default='http://127.0.0.1:30000')
    a=p.parse_args()
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(a.target_path,local_files_only=True,trust_remote_code=False)
    rows=[json.loads(x) for x in a.input.read_text().splitlines()]
    random.Random(20260918).shuffle(rows)
    results=[]
    for row in rows[:a.limit]:
        ids=chat_token_ids(tokenizer,row['prompt'])
        payload={'input_ids':ids,'sampling_params':{'temperature':0,'max_new_tokens':1024},
                 'stream':False}
        req=urllib.request.Request(a.url.rstrip('/')+'/generate',data=json.dumps(payload).encode(),
            headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=300) as response:result=json.load(response)
        results.append({'id':row['id'],'prompt_sha256':hashlib.sha256(row['prompt'].encode()).hexdigest(),
                        'response':result})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(''.join(json.dumps(x)+'\n' for x in results))
    metadata=[x['response']['meta_info'] for x in results]
    proposed=sum(x.get('spec_num_proposed_drafts',0) for x in metadata)
    accepted=sum(x.get('spec_num_correct_drafts',0) for x in metadata)
    print(json.dumps({'requests':len(results),'accepted':accepted,'proposed':proposed,
                      'acceptance_rate':accepted/proposed if proposed else None}))


if __name__=='__main__':main()
