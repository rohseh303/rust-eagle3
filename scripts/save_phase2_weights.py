"""Save completed draft exports locally, with streaming checksums."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1]

def save(variant):
    out=ROOT/'weights'/variant
    if out.exists():raise RuntimeError(f'{out} exists; refusing an implicit overwrite')
    out.mkdir()
    volume=modal.Volume.from_name(os.environ['RUST_DRAFT_NAMESPACE']);manifest={}
    for name in ['config.json','model.safetensors']:
        p=out/name;tmp=p.with_suffix(p.suffix+'.partial')
        with tmp.open('wb') as f:volume.read_file_into_fileobj('/phase2/exports/'+variant+'/'+name,f)
        tmp.replace(p);h=hashlib.sha256()
        with p.open('rb') as f:
            while chunk:=f.read(8*1024*1024):h.update(chunk)
        manifest[name]={'bytes':p.stat().st_size,'sha256':h.hexdigest()}
    (out/'SHA256.json').write_text(json.dumps(manifest,indent=2))
    return {variant:manifest}

def main():
    p=argparse.ArgumentParser();p.add_argument('variants',nargs='+',choices=['scale100','scale400','small400','mixed400']);a=p.parse_args()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(save,a.variants):print(json.dumps(result),flush=True)
if __name__=='__main__':main()
