import importlib.util
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import benchmark
import prepare_data
import gpu_workflow
import analyze_results
import regenerate


def test_no_crate_or_prompt_leakage():
    sets={s:benchmark.read_rows(ROOT/f'data/{s}.jsonl') for s in ['train','validation','heldout']}
    for a,b in [('train','validation'),('train','heldout'),('validation','heldout')]:
        assert not ({r['crate'] for r in sets[a]} & {r['crate'] for r in sets[b]})
        assert not ({r['id'] for r in sets[a]} & {r['id'] for r in sets[b]})
    for rows in sets.values():
        assert all('output_data' not in r and 'canonical_solution' not in r for r in rows)


def test_literal_parser_does_not_execute_code():
    import pytest
    with pytest.raises((ValueError,SyntaxError)):
        prepare_data.parse_mapping("__import__('os').system('echo unsafe')")


def test_failures_and_truncation_remain_visible():
    rows=[{'ok':False}, {'ok':True,'finish_reason':'length','completion_tokens':10,
        'latency_s':2.,'ttft_s':.2,'tpot_s':.2}]
    summary=benchmark.summarize(rows,4)
    assert summary['failures']==1 and summary['truncated']==1
    assert summary['output_tokens_per_s']==2.5
    assert summary['acceptance_rate'] is None


def test_streaming_usage_counts_tokens_not_chunks(monkeypatch):
    class Response:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def __iter__(self):
            events=[{'choices':[{'delta':{'content':'ten tokens in one event'},'finish_reason':None}]},
                    {'choices':[{'delta':{},'finish_reason':'stop'}],
                     'usage':{'completion_tokens':10,'prompt_tokens':20}}]
            yield from [b'data: '+json.dumps(e).encode()+b'\n' for e in events]
            yield b'data: [DONE]\n'
    monkeypatch.setattr(benchmark.urllib.request,'urlopen',lambda *a,**k:Response())
    result=benchmark.request_one('http://localhost','test',{'id':'a','prompt':'test'},20)
    assert result['ok'] and result['completion_tokens']==10


def test_warm_start_does_not_recompute_vocabulary():
    config=gpu_workflow.training_config('/tmp/rust-draft-contract',100)
    assert config['model']['vocab_mapping_path'].endswith('baseline-vocab.pt')
    assert config['model']['draft_checkpoint_path'].endswith('models/general-draft')
    assert config['training']['max_steps']==100


def test_paired_analysis_detects_output_drift_and_latency_change():
    import pytest
    baseline={'repeats':[[{'id':'x','ok':True,'output':'same','latency_s':10}]]}
    candidate={'repeats':[[{'id':'x','ok':True,'output':'changed','latency_s':8}]]}
    result=analyze_results.compare(baseline,candidate)
    assert result['median_paired_latency_reduction']==pytest.approx(.2)
    assert result['paired_prompts']==1
    assert result['prompts_with_all_outputs_identical']==0


def test_capped_training_prefix_does_not_invent_eos():
    partial={'ok':True,'finish_reason':'length','output':'fn unfinished('}
    assert regenerate.render_training_text('prompt',partial,'<eos>')=='promptfn unfinished('
    complete={'ok':True,'finish_reason':'stop','output':'fn done() {}'}
    assert regenerate.render_training_text('prompt',complete,'<eos>').endswith('<eos>\n')


def test_pinned_tokenizer_returns_serializable_ids():
    import os
    import pytest
    model_path=os.environ.get('QWEN_TOKENIZER_PATH')
    if not model_path:
        pytest.skip('Set QWEN_TOKENIZER_PATH to the pinned local Qwen tokenizer for this integration check')
    transformers=pytest.importorskip('transformers')
    tokenizer=transformers.AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    ids=benchmark.chat_token_ids(tokenizer,'Write a Rust function that adds two integers.')
    assert len(ids)>10
    assert json.loads(json.dumps({'input_ids':ids}))['input_ids']==ids
    rendered=tokenizer.apply_chat_template([{'role':'user','content':'Write a Rust function that adds two integers.'}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    assert ids==tokenizer.encode(rendered,add_special_tokens=False)


def test_server_context_can_restart_same_port(tmp_path,monkeypatch):
    import socket
    import subprocess
    import sys
    original_popen=subprocess.Popen
    program='''import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
HTTPServer(("127.0.0.1",int(sys.argv[1])),Handler).serve_forever()
'''
    with socket.socket() as reserve:
        reserve.bind(('127.0.0.1',0));port=reserve.getsockname()[1]
    def standin(args,**kwargs):
        configured_port=args[args.index('--port')+1]
        return original_popen([sys.executable,'-c',program,configured_port],**kwargs)
    monkeypatch.setattr(gpu_workflow.subprocess,'Popen',standin)
    for arm in ('target','general-draft'):
        with gpu_workflow.server(tmp_path,arm,port=port):
            with socket.create_connection(('127.0.0.1',port),timeout=2):pass
        with socket.socket() as check:
            assert check.connect_ex(('127.0.0.1',port))!=0
