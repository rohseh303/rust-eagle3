"""Bounded replication/scaling stages; preserves every original pilot artifact."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from gpu_workflow import ROOT,command,server,training_config
from benchmark import read_rows

VARIANTS={'seed2':(100,20260919),'scale100':(100,20260918),'scale400':(400,20260918),'mixed400':(400,20260918),'small400':(400,20260918)}

def variant_config(root,variant):
    steps,seed=VARIANTS[variant]
    cfg=training_config(root,steps)
    cfg['training'].update(seed=seed,save_interval=25 if steps==100 else 100,eval_interval=25 if steps==100 else 100)
    cfg['output_dir']=str(root/'phase2'/variant/'training')
    cfg['run_id']=variant
    if variant=='small400':
        cfg['training']['num_epochs']=7
    elif variant!='seed2':
        cfg['data']['hidden_states_path']=str(root/'phase2/features'/('mixed2048' if variant=='mixed400' else 'rust2048'))
    return cfg

def train(root,variant,specforge,block=0):
    cfg=variant_config(root,variant)
    work=root/'phase2'/variant
    work.mkdir(parents=True,exist_ok=True)
    split_run=VARIANTS[variant][0]==400
    if block not in (0,1) or (block==1 and not split_run):raise ValueError('Invalid training segment')
    if block==0:
        if (work/'training').exists():raise RuntimeError('Training directory already exists; refuse overwrite')
        if split_run:cfg['training']['max_steps']=200
    else:
        resume=work/'training'/f'{variant}-step200'
        if not resume.exists() or (work/'train-part1.json').exists():
            raise RuntimeError('Missing step-200 checkpoint or segment already launched')
        cfg['model']['draft_checkpoint_path']=None
        cfg['training']['resume_from']=str(resume)
    path=work/(f'train-part{block}.json' if split_run else 'train.json')
    path.write_text(json.dumps(cfg,indent=2))
    command(['specforge','train','--config',path],cwd=specforge)
    if split_run and block==0:
        if not (work/'training'/f'{variant}-step200').exists():raise RuntimeError('Missing continuation checkpoint')
        print('Completed first 200 updates of a fixed 400-update schedule; explicit full-state resume required.',flush=True)
        return
    checkpoint=work/'training'/f'{variant}-best'
    if not checkpoint.exists():raise RuntimeError('Validation-selected checkpoint missing')
    export=root/'phase2/exports'/variant
    command(['specforge','export','--to','sglang','--checkpoint',checkpoint,
        '--draft-config',root/'models/general-draft/config.json','--output-dir',export],cwd=specforge)
    import torch
    from safetensors import safe_open
    mapping=torch.load(root/'baseline-vocab.pt',map_location='cpu',weights_only=True)
    checks={}
    for file in export.glob('*.safetensors'):
        with safe_open(file,framework='pt',device='cpu') as h:
            for key,value in mapping.items():
                if key in h.keys():checks[key]=torch.equal(h.get_tensor(key),value)
    if set(checks)!=set(mapping) or not all(checks.values()):raise RuntimeError(f'Vocabulary audit failed: {checks}')
    audit=root/'phase2/audits';audit.mkdir(exist_ok=True)
    metadata={p.name:json.loads(p.read_text()) for p in (work/'training').glob('*.json')}
    selected_step=metadata[f'{variant}.best_meta.json']['step']
    exposure={str(step):training_exposure(root,cfg,step) for step in {selected_step,VARIANTS[variant][0]}}
    (audit/f'{variant}.json').write_text(json.dumps({'config':cfg,'stage_configs':{p.name:json.loads(p.read_text()) for p in work.glob('train*.json')},'selected_checkpoint':str(checkpoint.resolve()),'vocab_unchanged':checks,'metadata':metadata,'exposure_by_optimizer_step':exposure},indent=2))
    if variant=='scale100':
        references={}
        for seed in (20260918,20260919):
            reference=training_config(root,100);reference['training']['seed']=seed
            references[str(seed)]=training_exposure(root,reference,100)
        (audit/'512-reference-exposures.json').write_text(json.dumps(references,indent=2))

def training_exposure(root,cfg,steps):
    """Reconstruct the pinned offline sampler; count teacher-token exposures."""
    from specforge.launch import _distributed_sampler_indices
    from specforge.runtime.data_plane.offline_reader import list_feature_files
    lookup={}
    corpora=[(root/'features/train',root/'regenerated/train.requests.jsonl'),
             (root/'phase2/features/extra-rust',root/'phase2/regenerated/extra-rust/train.requests.jsonl'),
             (root/'phase2/features/general',root/'phase2/regenerated/general/train.requests.jsonl')]
    for features,requests in corpora:
        if not features.exists() or not requests.exists():continue
        files=feature_files(features);rows=read_rows(requests)
        assert len(files)==len(rows)
        lookup.update({str(file.resolve()):row for file,row in zip(files,rows)})
    files=list_feature_files(cfg['data']['hidden_states_path'])
    count=steps*cfg['training']['batch_size']*cfg['training']['accumulation_steps']
    order=[];epoch=0
    while len(order)<count:
        order.extend(_distributed_sampler_indices(len(files),dp_rank=0,dp_size=1,seed=cfg['training']['seed'],epoch=epoch));epoch+=1
    rows=[lookup[str(Path(files[i]).resolve())] for i in order[:count]]
    return {'example_exposures':len(rows),'unique_examples_seen':len({r['id'] for r in rows}),
            'teacher_prompt_token_exposures':sum(r['prompt_tokens'] for r in rows),
            'teacher_completion_token_exposures':sum(r['completion_tokens'] for r in rows),
            'note':'Reconstructed from pinned SpecForge sampler, not an optimizer FLOP count; teacher usage tokens are not identical to the multi-step loss-mask count.'}

def evaluate(root,variant,block):
    if variant=='seed2':
        arms=['seed2','rust-draft','general-draft']
    elif variant=='scale400':
        arms=['scale400','scale100','rust-draft','general-draft']
    elif variant=='mixed400':
        arms=['mixed400','scale400','general-draft']
    elif variant=='small400':
        arms=['small400','scale400','general-draft']
    else:raise ValueError('Evaluation group not defined')
    if block==1:arms.reverse()
    elif block!=0:raise ValueError('Only two counterbalanced sessions permitted')
    base=root/'phase2/results'/f'{variant}-block{block}'
    if base.exists():raise RuntimeError('Refusing to overwrite evaluation session')
    base.mkdir(parents=True)
    (base/'session.json').write_text(json.dumps({'arms':arms,'block':block,'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name','--format=csv,noheader'],text=True)},indent=2))
    for i,arm in enumerate(arms):
        port=30100+i
        draft=root/'phase2/exports'/arm if arm in VARIANTS else None
        with server(root,arm,port=port,draft_path=draft):
            command([sys.executable,ROOT/'scripts/benchmark.py','--input',ROOT/'phase2/data/evaluation.jsonl',
                '--output',base/arm,'--arm',arm,'--concurrency','1','--repeats','1','--url',f'http://127.0.0.1:{port}'])
            if variant!='seed2':
                command([sys.executable,ROOT/'scripts/acceptance_probe.py',
                    '--input',ROOT/'phase2/data/acceptance.jsonl','--output',base/arm/'acceptance.jsonl',
                    '--target-path',root/'models/target','--limit','24','--url',f'http://127.0.0.1:{port}'])

def corpus_for(variant):
    if variant=='scale400':return 'extra-rust'
    if variant=='mixed400':return 'general'
    raise ValueError('Generation/capture only support extra Rust or general corpus')

def generate(root,variant):
    corpus=corpus_for(variant)
    with server(root,'target'):
        command([sys.executable,ROOT/'scripts/regenerate.py',
            '--input',ROOT/f'phase2/data/{corpus}/train.jsonl',
            '--output',root/f'phase2/regenerated/{corpus}/train.jsonl',
            '--target-path',root/'models/target'])

def feature_files(path):
    return sorted(path.rglob('data_*.ckpt'),key=lambda p:int(p.stem.split('_')[-1]))

def capture(root,variant,specforge):
    corpus=corpus_for(variant)
    text=root/f'phase2/regenerated/{corpus}/train.jsonl'
    out=root/f'phase2/features/{corpus}'
    if out.exists():raise RuntimeError('Capture output exists; refuse implicit resume')
    command(['git','apply','--check',ROOT/'patches/specforge-preformatted-capture.patch'],cwd=specforge)
    command(['git','apply',ROOT/'patches/specforge-preformatted-capture.patch'],cwd=specforge)
    command(['torchrun','--standalone','--nproc_per_node=1',specforge/'scripts/prepare_hidden_states.py',
        '--strategy','eagle3','--target-model-path',root/'models/target',
        '--draft-model-config',root/'models/general-draft/config.json',
        '--data-path',text,'--output-path',out,'--is-preformatted',
        '--chat-template','qwen3-instruct','--max-length','3072','--tp-size','1',
        '--batch-size','1','--sglang-mem-fraction-static','0.7',
        '--sglang-context-length','4096','--build-dataset-num-proc','2'],cwd=specforge)
    expected=1536 if corpus=='extra-rust' else 512
    captured=feature_files(out)
    raw=read_rows(text.with_suffix('.requests.jsonl'))
    excluded=json.loads(text.with_suffix('.excluded.json').read_text())
    if len(captured)!=expected or len(raw)!=expected or excluded:
        raise RuntimeError(f'Incomplete corpus: {len(captured)} features, {len(raw)} responses, {len(excluded)} exclusions')
    old=feature_files(root/'features/train')
    assert len(old)==512
    if corpus=='extra-rust':
        sources=old+captured;name='rust2048'
    else:
        extra=feature_files(root/'phase2/features/extra-rust');assert len(extra)==1536
        sources=old+extra[:1024]+captured;name='mixed2048'
    combined=root/'phase2/features'/name
    combined.mkdir(exist_ok=False)
    for i,file in enumerate(sources):
        (combined/f'data_{i:05d}.ckpt').symlink_to(file)
    assert len(feature_files(combined))==2048
    audit=root/'phase2/audits';audit.mkdir(exist_ok=True)
    (audit/f'capture-{corpus}.json').write_text(json.dumps({
        'corpus':corpus,'captured':len(captured),'combined_features':len(sources),
        'feature_sources':[str(s) for s in sources],
        'new_prompt_tokens':sum(r['prompt_tokens'] for r in raw),
        'new_answer_tokens':sum(r['completion_tokens'] for r in raw),
        'new_capped':sum(r['finish_reason']=='length' for r in raw),
        'excluded':excluded},indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('stage');p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--specforge',type=Path,required=True);p.add_argument('--variant',choices=list(VARIANTS),default='seed2');p.add_argument('--block',type=int,default=0)
    a=p.parse_args();root=a.run_root.resolve()
    if a.stage=='phase2-train':train(root,a.variant,a.specforge,a.block)
    elif a.stage=='phase2-evaluate':evaluate(root,a.variant,a.block)
    elif a.stage=='phase2-generate':generate(root,a.variant)
    elif a.stage=='phase2-capture':capture(root,a.variant,a.specforge)
    else:raise ValueError('Stage not yet implemented')
if __name__=='__main__':main()
