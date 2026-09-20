import json
import hashlib
from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from benchmark import read_rows
from phase2_workflow import variant_config
from analyze_phase2 import analyze

def test_nested_scaling_preserves_holdouts():
    a=read_rows(ROOT/'data/train.jsonl')
    b=read_rows(ROOT/'phase2/data/rust2048/train.jsonl')
    assert a==b[:512] and len(b)==2048
    for split in ['validation','heldout']:
        held=read_rows(ROOT/f'data/{split}.jsonl')
        assert held==read_rows(ROOT/f'phase2/data/rust2048/{split}.jsonl')
        assert not {r['crate'] for r in b}&{r['crate'] for r in held}
    evaluation=read_rows(ROOT/'phase2/data/evaluation.jsonl')
    assert not {r['id'] for r in b}&{r['id'] for r in evaluation}

def test_training_comparisons_are_explicit():
    cfg={v:variant_config(Path('/tmp/phase2-test'),v) for v in ['seed2','scale100','scale400','mixed400','small400']}
    assert cfg['seed2']['training']['seed']==20260919
    assert cfg['scale100']['training']['max_steps']==100
    assert cfg['scale400']['training']['max_steps']==400
    assert cfg['mixed400']['training']['max_steps']==400
    assert cfg['scale100']['data']['hidden_states_path']==cfg['scale400']['data']['hidden_states_path']
    assert cfg['small400']['training']['num_epochs']==7
    assert cfg['small400']['training']['max_steps']==400
    assert cfg['small400']['data']['hidden_states_path']==cfg['seed2']['data']['hidden_states_path']
    assert len({v['output_dir'] for v in cfg.values()})==5
    assert len({v['model']['draft_checkpoint_path'] for v in cfg.values()})==1

def test_replication_gate_rejects_fresh_output_drift(tmp_path):
    cohorts=read_rows(ROOT/'phase2/data/evaluation.jsonl')
    arms=['seed2','rust-draft','general-draft']
    for block in (0,1):
        d=tmp_path/f'seed2-block{block}';d.mkdir()
        (d/'session.json').write_text(json.dumps({'arms':arms,'block':block,'gpu':'fixture'}))
        for arm in arms:
            p=d/arm;p.mkdir()
            (p/'run.json').write_text(json.dumps({'repeats':1,'concurrency':1,'input_sha256':hashlib.sha256((ROOT/'phase2/data/evaluation.jsonl').read_bytes()).hexdigest()}))
            rows=[{'id':r['id'],'prompt_sha256':hashlib.sha256(r['prompt'].encode()).hexdigest(),'ok':True,'output':'fixed','latency_s':10. if arm=='general-draft' else 9.,'finish_reason':'stop'} for r in cohorts]
            (p/'raw-0.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    assert analyze(tmp_path,'seed2')['scale_gate_passed']
    path=tmp_path/'seed2-block1/seed2/raw-0.jsonl'
    rows=read_rows(path);key=next(r['id'] for r in cohorts if r['task']=='rust_fresh')
    next(r for r in rows if r['id']==key)['output']='different'
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    assert not analyze(tmp_path,'seed2')['scale_gate_passed']

def test_budget_guard_blocks_scaling_before_gate(monkeypatch,tmp_path):
    import pytest
    import run_budgeted
    monkeypatch.setattr(run_budgeted,'ROOT',tmp_path)
    monkeypatch.setattr(run_budgeted,'RUN',tmp_path/'run')
    monkeypatch.setattr(sys,'argv',['run_budgeted.py','--stage','phase2-train','--variant','scale100'])
    with pytest.raises(SystemExit,match='replication gate has not passed'):
        run_budgeted.main()
    assert not (tmp_path/'run').exists()

def test_budget_guard_keeps_cumulative_reserve(monkeypatch,tmp_path):
    import pytest
    import run_budgeted
    run=tmp_path/'run';run.mkdir()
    p=run/'budget.json';original=json.dumps({'runs':[{'charged_estimate_usd':42.0}]});p.write_text(original)
    phase=tmp_path/'phase2';phase.mkdir();(phase/'replication.json').write_text(json.dumps({'scale_gate_passed':True}))
    monkeypatch.setattr(run_budgeted,'ROOT',tmp_path)
    monkeypatch.setattr(run_budgeted,'RUN',run)
    monkeypatch.setattr(sys,'argv',['run_budgeted.py','--stage','phase2-train','--variant','scale100'])
    def forbidden(*args,**kwargs):raise AssertionError('No cloud launch is allowed')
    monkeypatch.setattr(run_budgeted.subprocess,'Popen',forbidden)
    with pytest.raises(SystemExit,match='insufficient safe allowance'):
        run_budgeted.main()
    assert p.read_text()==original

@pytest.mark.parametrize('authorization,committed',[(60.,47.),(50.,42.),(40.,31.)])
def test_budget_guard_respects_recorded_authorization(monkeypatch,tmp_path,authorization,committed):
    import run_budgeted
    run=tmp_path/'run';run.mkdir()
    p=run/'budget.json'
    original=json.dumps({'user_absolute_ceiling_usd':authorization,'runs':[{'charged_estimate_usd':committed}]})
    p.write_text(original)
    phase=tmp_path/'phase2';phase.mkdir();(phase/'replication.json').write_text(json.dumps({'scale_gate_passed':True}))
    monkeypatch.setattr(run_budgeted,'ROOT',tmp_path)
    monkeypatch.setattr(run_budgeted,'RUN',run)
    monkeypatch.setattr(sys,'argv',['run_budgeted.py','--stage','phase2-evaluate','--variant','mixed400'])
    def forbidden(*args,**kwargs):raise AssertionError('No cloud launch is allowed')
    monkeypatch.setattr(run_budgeted.subprocess,'Popen',forbidden)
    with pytest.raises(SystemExit,match='insufficient safe allowance'):
        run_budgeted.main()
    assert p.read_text()==original
