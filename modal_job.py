"""One bounded GPU stage per invocation; all launches use the local budget guard."""
import json
import os
from pathlib import Path
import modal

ROOT=Path(__file__).resolve().parent if modal.is_local() else Path('/opt/rust-draft')
SOURCES=json.loads((ROOT/'configs/sources.json').read_text())
REV=SOURCES['specforge']['revision']
namespace=os.environ.get('RUST_DRAFT_NAMESPACE')
if not namespace:
    raise RuntimeError('Set RUST_DRAFT_NAMESPACE to a fresh experiment name before cloud use')
image=(modal.Image.from_registry('nvidia/cuda:12.9.1-devel-ubuntu22.04',add_python='3.11')
    .env({'RUST_DRAFT_NAMESPACE':namespace})
    .apt_install('git')
    .run_commands('git clone https://github.com/sgl-project/SpecForge.git /opt/SpecForge',
                  f'git -C /opt/SpecForge checkout {REV}')
    .pip_install('/opt/SpecForge'))
# Package immutable inputs and executable code, not reports written during runs.
for folder in ('configs','scripts','patches','data','phase2/data'):
    image=image.add_local_dir(ROOT/folder,remote_path=f'/opt/rust-draft/{folder}',
                              ignore=['__pycache__','.pytest_cache','*.log'])
app=modal.App(namespace)
volume=modal.Volume.from_name(namespace,create_if_missing=True)


@app.function(image=image,gpu='H100',cpu=4,memory=32768,timeout=1800,
              max_containers=1,retries=0,volumes={'/data':volume})
def run_stage(stage: str,arm: str='target',suite: str='validation',steps: int=100,
              concurrency: int=1,limit: int=32,variant: str='seed2',block: int=0):
    import subprocess
    import sys
    import time
    valid={'download','regenerate','capture','train','export','benchmark','evaluate',
           'phase2-train','phase2-evaluate','phase2-generate','phase2-capture'}
    if stage not in valid or not 1<=steps<=400 or not 1<=concurrency<=16 or not 1<=limit<=164:
        raise ValueError('Stage or pilot resource limits are invalid')
    started=time.monotonic()
    logdir=Path('/data/logs');logdir.mkdir(parents=True,exist_ok=True)
    logfile=logdir/f'{stage}-{time.time_ns()}.log'
    import importlib.metadata
    import hashlib
    environment={'stage':stage,'packages':{p:importlib.metadata.version(p)
        for p in ('torch','transformers','sglang','specforge')},
        'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total',
            '--format=csv,noheader'],text=True).strip(),
        'script_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in Path('/opt/rust-draft/scripts').glob('*.py')}}
    logfile.with_suffix('.environment.json').write_text(json.dumps(environment,indent=2))
    print(json.dumps(environment),flush=True)
    if not Path('/data/environment-pip-freeze.txt').exists():
        Path('/data/environment-pip-freeze.txt').write_text(
            subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    args=[sys.executable,'/opt/rust-draft/scripts/gpu_workflow.py',stage,
          '--run-root','/data','--specforge','/opt/SpecForge','--arm',arm,
          '--suite',suite,'--steps',str(steps),'--concurrency',str(concurrency),'--limit',str(limit)]
    if stage.startswith('phase2-'):
        args=[sys.executable,'/opt/rust-draft/scripts/phase2_workflow.py',stage,
              '--run-root','/data','--specforge','/opt/SpecForge',
              '--variant',variant,'--block',str(block)]
    import threading
    stop_commit=threading.Event()
    def publish_progress():
        while not stop_commit.wait(30):
            try:volume.commit()
            except Exception as exc:print(f'Progress log sync: {type(exc).__name__}',flush=True)
    progress_thread=threading.Thread(target=publish_progress,daemon=True)
    progress_thread.start()
    try:
        with logfile.open('w') as log:
            result=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,timeout=1700)
        if result.returncode:
            raise RuntimeError(logfile.read_text()[-8000:])
        return {'stage':stage,'elapsed_s':time.monotonic()-started,'log':str(logfile),
                'tail':logfile.read_text()[-2500:]}
    finally:
        stop_commit.set()
        progress_thread.join(timeout=60)
        volume.commit()


@app.local_entrypoint()
def main(stage: str,arm: str='target',suite: str='validation',steps: int=100,
         concurrency: int=1,limit: int=32,variant: str='seed2',block: int=0):
    print(run_stage.remote(stage,arm,suite,steps,concurrency,limit,variant,block))
