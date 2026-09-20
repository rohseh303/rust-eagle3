"""Retrieve pilot result/log files with the Modal SDK, without model weights.

Run with a Python environment containing modal==1.5.5. One recursive directory
listing avoids the per-file VolumeListFiles calls made by the CLI downloader.
"""
import argparse
import concurrent.futures
import json
import os
import time
from pathlib import Path
import modal
from modal.types import FileEntryType


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--remote',default='/results')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.remote not in {'/results','/final-results','/final-results-v2','/logs','/regenerated','/phase2/results','/phase2/audits','/phase2/regenerated'}:
        p.error('Only pilot results, logs, or regenerated text may be synced here')
    volume=modal.Volume.from_name(os.environ['RUST_DRAFT_NAMESPACE'])
    files=[entry.path for entry in volume.listdir(a.remote,recursive=True)
           if entry.type==FileEntryType.FILE]
    def get(remote):
        relative=Path(remote).relative_to(a.remote.lstrip('/'))
        dest=a.output/relative
        dest.parent.mkdir(parents=True,exist_ok=True)
        temporary=dest.with_name(dest.name+'.download')
        for attempt in range(5):
            try:
                with temporary.open('wb') as file:
                    volume.read_file_into_fileobj(remote,file)
                temporary.replace(dest)
                return str(dest)
            except Exception as exc:
                if 'rate limit' not in str(exc).lower() or attempt==4:raise
                time.sleep(min(10*2**attempt,40))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(get,files))
    print(json.dumps({'files':len(results),'output':str(a.output)}))


if __name__=='__main__':main()
