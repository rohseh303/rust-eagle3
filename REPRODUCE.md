# Reproducing the result

## 1. Reanalyze the recorded evidence (CPU only)

Use Python 3.11 or newer from the repository root:

```sh
python scripts/verify_release.py
python -m pip install pytest==8.4.2
python -m pytest -q
```

The verifier uses the standard library only and checks saved hashes, all 90 follow-up comparisons, all pilot paired comparisons, native acceptance totals, and follow-up output/token parity. It never downloads models or contacts a cloud provider. CI runs this path only. The optional tokenizer integration test is skipped unless `QWEN_TOKENIZER_PATH` points to the pinned Qwen3-8B tokenizer; install the local requirements to run it.

To recompute an individual follow-up into a new file:

```sh
python scripts/analyze_phase2.py --group small400 --output /tmp/rust-control.json
```

Other groups: `seed2` (replication), `scale400` (scaling), `mixed400` (mixture). `phase2/comparisons.csv` contains all cohorts and both session estimates. Positive reduction means faster. Prompt-bootstrap intervals are not uncertainty intervals over independently retrained models.

To regenerate plots, install `requirements-report.txt`, then run:

```sh
python scripts/plot_phase2.py phase2/control.json --output /tmp/rust-control
python scripts/plot_phase2.py phase2/mixed.json --output /tmp/rust-mixed
```

`plot_training_curves.py` writes back into `phase2/figures`; use a scratch copy to avoid altering the evidence manifest. The exact observed GPU package list is evidence, not a guaranteed portable lockfile.

## 2. Serve a released draft (requires a compatible GPU environment)

Use the pinned source/model revisions in `configs/sources.json` and observed SGLang 0.5.18 environment. Extract a model bundle into a local directory and verify `SHA256.json` before loading it. Download the target **at revision** `b968826d9c46dd6066d109eabc6255188de91218`; do not silently use latest. Set `QWEN_TARGET_PATH` and `RUST_DRAFT_PATH` to those local directories.

```sh
python -m sglang.launch_server \
  --model-path "$QWEN_TARGET_PATH" --served-model-name rust-target \
  --host 127.0.0.1 --port 30000 --dtype bfloat16 \
  --context-length 8192 --mem-fraction-static 0.7 --random-seed 20260918 \
  --max-running-requests 16 --disable-radix-cache --enable-metrics \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path "$RUST_DRAFT_PATH" \
  --speculative-num-steps 3 --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4
```

In another terminal, benchmark the frozen follow-up prompts:

```sh
python scripts/benchmark.py --input phase2/data/evaluation.jsonl \
  --output /tmp/rust-serving-check --arm scale400 \
  --concurrency 1 --repeats 1 --max-tokens 1024
```

The client fixes temperature zero and disables thinking. It reports output caps and failures. `scale400` above is a label, not an automatic checkpoint download. For valid paired timing, run the upstream draft on the same allocation and reverse order in another session, as in `phase2_workflow.py`. A single run of the command is a smoke check, not an independent reproduction of the confidence interval. Model load time is excluded from the reported request latency.

## 3. Repeat training as a new experiment (paid, operator-controlled)

No account access, cloud IDs, or authorization to spend accompanies this repository. Keep the evidence tree unchanged. First create an empty experiment workspace:

```sh
python scripts/new_experiment.py ../rust-eagle3-rerun
cd ../rust-eagle3-rerun
python -m pip install -r requirements-local.txt
```

Authenticate your own Modal account. Choose a fresh `RUST_DRAFT_NAMESPACE` for this run, and initialize an explicit budget:

```sh
export RUST_DRAFT_NAMESPACE=rust-eagle3-my-rerun
python scripts/init_budget.py --usd 50
```

The initializer refuses an existing ledger. The launcher preserves cumulative charges/reservations, locks out concurrent launchers, reserves failed attempts until inspected, and uses the lower of a historical $55 operating limit and the explicit ceiling. Its $6/hour estimate plus per-stage allowance is a **local guard, not a provider billing cap**. Verify current rates and provider-side limits before running it. Compute/image/storage charges and failures can differ from this historical experiment. See [Modal volumes](https://modal.com/docs/guide/volumes) for retained storage behavior.

The cloud recipe uses a single H100, one container at a time, no automatic retries, a 1,700-second subprocess limit and 1,800-second function limit. The local wrapper has a 2,400-second outer limit. Keep the launcher connected; do not bypass it with detached calls.

Run the pilot stages sequentially:

```sh
python scripts/run_budgeted.py --stage download
python scripts/run_budgeted.py --stage regenerate
python scripts/run_budgeted.py --stage capture
python scripts/run_budgeted.py --stage train --steps 100
python scripts/run_budgeted.py --stage export
python scripts/run_budgeted.py --stage evaluate
```

`download` fetches pinned target/draft weights and extracts the original vocabulary mapping. `regenerate` creates frozen-target training/validation conversations. `capture` applies the included SpecForge patch in that stage's container and records features. `train` warm-starts the draft; `export` selects the validation-best checkpoint and checks vocabulary preservation. `evaluate` includes the historical concurrency checks, whose output variation must not be hidden.

Then repeat the seed and two follow-up sessions, using the already frozen prompt splits:

```sh
python scripts/run_budgeted.py --stage phase2-train --variant seed2
python scripts/run_budgeted.py --stage phase2-evaluate --variant seed2 --block 0
python scripts/run_budgeted.py --stage phase2-evaluate --variant seed2 --block 1
python scripts/sync_results.py --remote /phase2/results --output phase2/results
python scripts/analyze_phase2.py --group seed2 --output phase2/replication.json
```

Only proceed if the **new run's** replication gate passes. The new-workspace helper does not copy the original successful gate. Generate the extra Rust responses/features and run the bounded continuation driver:

```sh
python scripts/run_budgeted.py --stage phase2-generate --variant scale400
python scripts/run_budgeted.py --stage phase2-capture --variant scale400
python scripts/continue_phase2.py --group scale
python scripts/continue_phase2.py --group control
python scripts/continue_phase2.py --group mixed
```

The last group is optional and subordinate to the declared budget. Drivers stop on errors and refuse existing progress files. Do not erase a ledger or progress file to force an unreviewed retry. The scale and control groups compare pool sizes at 100 and 400 updates; two 200-update stages resume a single 400-update schedule. The mixed group generates/captures general answers and trains on 1,536 Rust + 512 general prompts. No test timing selects checkpoints.

This recipe records the executed workflow, with publication-only namespace and budget-initialization safeguards. The original cached image resolved dependencies once; a fresh GPU image rebuild and a complete fresh training rerun were **not** validated during publication preparation. All original measurements remain tied to their recorded environment. Intermediate feature tensors and optimizer states are not release assets; reconstruct them for a new run. Saved teacher conversations are included to inspect the original training text and token accounting.
