"""Explicit, resumable GPU stages. No cloud launch is performed by this script."""
import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCES=json.loads((ROOT/'configs/sources.json').read_text())


def command(args,cwd=None):
    print('RUN:', ' '.join(map(str,args)),flush=True)
    subprocess.run(list(map(str,args)),cwd=cwd,check=True)


def training_config(run_root,steps):
    root=Path(run_root).resolve()
    return {
        'model': {'target_model_path':str(root/'models/target'),
            'draft_model_config':str(root/'models/general-draft/config.json'),
            'draft_checkpoint_path':str(root/'models/general-draft'),
            'vocab_mapping_path':str(root/'baseline-vocab.pt'),
            'torch_dtype':'bfloat16'},
        'data': {'hidden_states_path':str(root/'features/train'),
            'eval_hidden_states_path':str(root/'features/validation'),
            'max_length':3072,'chat_template':'qwen3-instruct','cache_dir':str(root/'cache')},
        'training': {'strategy':'eagle3','batch_size':1,'accumulation_steps':8,'learning_rate':1e-5,
            'num_epochs':3,'max_steps':steps,'total_steps':steps,
            'max_grad_norm':0.5,'ttt_length':7,'attention_backend':'sdpa',
            'save_interval':25,'eval_interval':25,'log_interval':5,'seed':20260918},
        'tracking':{'report_to':'none'},
        'run_id':'rust-eagle3-pilot','output_dir':str(root/'training'),
        'deployment':{'mode':'local_colocated','trainer':{'nnodes':1,'nproc_per_node':1}},
    }


@contextlib.contextmanager
def server(root,arm,port=30000,draft_path=None):
    import socket
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        try:
            probe.bind(('127.0.0.1',port))
        except OSError as exc:
            raise RuntimeError(f'Port {port} is already occupied; refusing to contact another server') from exc
    args=[sys.executable,'-m','sglang.launch_server','--model-path',root/'models/target',
          '--served-model-name','rust-target','--host','127.0.0.1','--port',str(port),
          '--dtype','bfloat16','--context-length','8192','--mem-fraction-static','0.7',
          '--random-seed','20260918',
          '--max-running-requests','16','--disable-radix-cache','--enable-metrics']
    if arm!='target':
        if draft_path is None and arm not in {'general-draft','rust-draft'}:
            raise ValueError('Unknown draft arm without an explicit checkpoint')
        draft=Path(draft_path) if draft_path else root/('models/general-draft' if arm=='general-draft' else 'export/rust-draft')
        args += ['--speculative-algorithm','EAGLE3','--speculative-draft-model-path',draft,
                 '--speculative-num-steps','3','--speculative-eagle-topk','1',
                 '--speculative-num-draft-tokens','4']
    logs=root/'logs';logs.mkdir(parents=True,exist_ok=True)
    logfile=logs/f'{arm}-{time.time_ns()}.log'
    with logfile.open('w') as log:
        proc=subprocess.Popen(list(map(str,args)),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            deadline=time.monotonic()+600
            while time.monotonic()<deadline:
                if proc.poll() is not None:
                    raise RuntimeError(f'Server exited: inspect {logfile}')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=2) as response:
                        if response.status==200:
                            break
                except Exception:
                    time.sleep(2)
            else:
                raise TimeoutError(f'Server startup timed out: {logfile}')
            yield
        finally:
            import signal
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            # The serving parent may exit before its worker processes. Reap
            # the entire process group created by this context before switching.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid,signal.SIGKILL)
            time.sleep(2)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['download','regenerate','capture','train','export','benchmark','evaluate'])
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--specforge',type=Path,required=True)
    p.add_argument('--steps',type=int,default=100)
    p.add_argument('--arm',choices=['target','general-draft','rust-draft'],default='target')
    p.add_argument('--suite',default='validation',choices=['validation','heldout','humaneval-rust','humaneval-python'])
    p.add_argument('--limit',type=int,default=32)
    p.add_argument('--concurrency',type=int,default=1)
    a=p.parse_args();root=a.run_root.resolve();root.mkdir(parents=True,exist_ok=True)
    if a.stage=='download':
        from huggingface_hub import snapshot_download
        from safetensors import safe_open
        import torch
        for key,directory in [('target','target'),('baseline_draft','general-draft')]:
            source=SOURCES[key]
            snapshot_download(source['repo'],revision=source['revision'],local_dir=root/'models'/directory,
                allow_patterns=['*.json','*.safetensors','pytorch_model*.bin','*.model','*.txt','*.jinja','LICENSE*','README.md'])
        # Preserve the warm-start draft's original vocabulary. Recomputing it
        # from Rust would silently change the meaning of pretrained head rows.
        mapping={}
        for file in (root/'models/general-draft').glob('*.safetensors'):
            with safe_open(file,framework='pt',device='cpu') as f:
                for key in ('t2d','d2t'):
                    if key in f.keys():mapping[key]=f.get_tensor(key)
        if not mapping:
            for file in (root/'models/general-draft').glob('pytorch_model*.bin'):
                state=torch.load(file,map_location='cpu',weights_only=True)
                for key in ('t2d','d2t'):
                    if key in state:mapping[key]=state[key]
                print('Draft checkpoint keys:', sorted(state),flush=True)
        if set(mapping)!={'t2d','d2t'}:
            raise RuntimeError('Baseline lacks required vocabulary mapping; do not start training')
        torch.save(mapping,root/'baseline-vocab.pt')
        (root/'sources.json').write_text(json.dumps(SOURCES,indent=2))
    elif a.stage=='regenerate':
        with server(root,'target'):
            for split in ('train','validation'):
                command([sys.executable,ROOT/'scripts/regenerate.py','--input',ROOT/f'data/{split}.jsonl',
                    '--output',root/f'regenerated/{split}.jsonl','--target-path',root/'models/target'])
    elif a.stage=='capture':
        command(['git','apply','--check',ROOT/'patches/specforge-preformatted-capture.patch'],cwd=a.specforge)
        command(['git','apply',ROOT/'patches/specforge-preformatted-capture.patch'],cwd=a.specforge)
        for split in ('train','validation'):
            command(['torchrun','--standalone','--nproc_per_node=1',a.specforge/'scripts/prepare_hidden_states.py',
                '--strategy','eagle3','--target-model-path',root/'models/target',
                '--draft-model-config',root/'models/general-draft/config.json',
                '--data-path',root/f'regenerated/{split}.jsonl','--output-path',root/f'features/{split}',
                '--is-preformatted','--chat-template','qwen3-instruct','--max-length','3072',
                '--tp-size','1','--batch-size','1','--sglang-mem-fraction-static','0.7',
                '--sglang-context-length','4096','--build-dataset-num-proc','2'],cwd=a.specforge)
    elif a.stage=='train':
        if a.steps<1:raise ValueError('steps must be positive')
        output=root/'training'
        if output.exists() and any(output.iterdir()):
            raise RuntimeError('Training output already exists. Use a new run root or an explicit verified resume config.')
        cfg=root/'train.json';cfg.write_text(json.dumps(training_config(root,a.steps),indent=2))
        command(['specforge','train','--config',cfg],cwd=a.specforge)
    elif a.stage=='export':
        checkpoint=root/'training/rust-eagle3-pilot-best'
        if not checkpoint.exists():
            raise RuntimeError('Validation-selected best checkpoint missing; inspect training before exporting')
        command(['specforge','export','--to','sglang','--checkpoint',checkpoint,
            '--draft-config',root/'models/general-draft/config.json','--output-dir',root/'export/rust-draft'],cwd=a.specforge)
        import torch
        from safetensors import safe_open
        mapping=torch.load(root/'baseline-vocab.pt',map_location='cpu',weights_only=True)
        original=torch.load(root/'models/general-draft/pytorch_model.bin',map_location='cpu',weights_only=True)
        checks={}
        weights=root/'export/rust-draft'
        tensor_count=0;parameter_count=0
        for file in weights.glob('*.safetensors'):
            with safe_open(file,framework='pt',device='cpu') as handle:
                for key in handle.keys():
                    tensor_count+=1
                    tensor=handle.get_tensor(key)
                    if key not in ('t2d','d2t'):parameter_count+=tensor.numel()
                    if key in mapping:
                        checks[key+'_unchanged']=torch.equal(tensor,mapping[key])
                    if key in ('fc.weight','lm_head.weight'):
                        checks[key+'_changed']=not torch.equal(tensor,original[key])
                        checks[key+'_relative_l2_change']=(
                            (tensor.float()-original[key].float()).norm()/original[key].float().norm()).item()
        if not all(checks.get(k) for k in ('t2d_unchanged','d2t_unchanged','fc.weight_changed','lm_head.weight_changed')):
            raise RuntimeError(f'Export audit failed: {checks}')
        audit={'selected_checkpoint':str(checkpoint.resolve()),'tensor_count':tensor_count,
               'stored_parameter_count_excluding_vocab_buffers':parameter_count,'checks':checks}
        (weights/'export-audit.json').write_text(json.dumps(audit,indent=2))
        print(json.dumps(audit),flush=True)
    elif a.stage=='benchmark':
        with server(root,a.arm):
            command([sys.executable,ROOT/'scripts/benchmark.py','--input',ROOT/f'data/{a.suite}.jsonl',
                '--output',root/f'results/{a.arm}-{a.suite}-c{a.concurrency}-{time.time_ns()}',
                '--arm',a.arm,'--concurrency',a.concurrency,'--limit',a.limit])
    elif a.stage=='evaluate':
        final_root=root/'final-results-v2'
        if final_root.exists() and any(final_root.iterdir()):
            raise RuntimeError('Final results already exist; preserve them and select a new run root explicitly')
        # All final arms share this physical GPU; held-out selection is fixed
        # before outputs are opened. No tuning uses these results.
        for index,arm in enumerate(('general-draft','target','rust-draft')):
            port=30000+index
            url=f'http://127.0.0.1:{port}'
            with server(root,arm,port=port):
                if arm=='general-draft':
                    command([sys.executable,ROOT/'scripts/acceptance_probe.py','--input',ROOT/'data/heldout.jsonl',
                        '--output',final_root/'preflight-acceptance/requests.jsonl',
                        '--target-path',root/'models/target','--limit','1','--url',url])
                for suite,concurrency in [('heldout',1),('humaneval-python',1),('heldout',4),('heldout',16)]:
                    command([sys.executable,ROOT/'scripts/benchmark.py','--input',ROOT/f'data/{suite}.jsonl',
                        '--output',final_root/f'{arm}-{suite}-c{concurrency}-{time.time_ns()}',
                        '--arm',arm,'--concurrency',concurrency,'--limit',a.limit,'--url',url])
                if arm!='target':
                    command([sys.executable,ROOT/'scripts/acceptance_probe.py','--input',ROOT/'data/heldout.jsonl',
                        '--output',final_root/f'{arm}-acceptance/requests.jsonl',
                        '--target-path',root/'models/target','--limit','16','--url',url])


if __name__=='__main__':
    main()
