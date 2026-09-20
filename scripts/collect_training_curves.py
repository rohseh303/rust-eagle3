"""Retrieve phase-2 training logs and parse recorded validation checkpoints."""
import ast
import json
import os
import re
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1]

def main():
    v=modal.Volume.from_name(os.environ['RUST_DRAFT_NAMESPACE'])
    out=ROOT/'phase2/cloud-logs';out.mkdir(exist_ok=True)
    curves={}
    for entry in v.listdir('/logs'):
        if not entry.path.startswith('logs/phase2-train-') or not entry.path.endswith('.log'):continue
        target=out/Path(entry.path).name
        with target.open('wb') as f:v.read_file_into_fileobj(entry.path,f)
        text=target.read_text(errors='replace')
        match=re.search(r'/phase2/([^/]+)/train(?:-part\d)?\.json',text)
        if not match:continue
        name=match.group(1);curve=curves.setdefault(name,{})
        for line in text.splitlines():
            m=re.match(r'step (\d+): (\{.*\})',line)
            if not m:continue
            values=ast.literal_eval(m.group(2))
            if 'eval/simulated_acc_len' in values:
                curve[int(m.group(1))]={k:values[k] for k in ('eval/avg_loss','eval/avg_acc','eval/simulated_acc_len')}
    (ROOT/'phase2/training-curves.json').write_text(json.dumps(curves,indent=2))
    print(json.dumps({k:sorted(v) for k,v in curves.items()}))
if __name__=='__main__':main()
