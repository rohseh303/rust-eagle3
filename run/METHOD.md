# What this pilot establishes

The experiment tests whether a small Rust adaptation improves an existing
EAGLE-3 draft's serving speed with frozen Qwen3-8B. The contribution is the
adaptation experiment, measured serving comparison, reproducible artifacts, and
an upstream data-loader fix. EAGLE-3, SpecForge, SGLang, and the original draft
are upstream work, not new algorithms developed here.

## Training

512 prompts were chosen as a bounded feasibility pilot, not through a statistical
power calculation. They come from the pinned synthetic Strandset-Rust-v1 dataset.
The length-filtered set contains 511 function-generation examples and one code
completion example. It is not representative of arbitrary Rust programming.
Train, validation, and held-out sets use disjoint crate names; prompt deduplication
and benchmark-name/spec exclusions are exact, not semantic decontamination.

Qwen regenerated 108,939 answer tokens for the 512 training prompts at temperature
0.8, thinking disabled. All requests succeeded. Five answers reached the 1,024
token cap and were retained as prefixes without synthetic EOS. Validation used
64 separate prompts and 15,575 answer tokens, including two capped prefixes.
Source reference answers were used only to select longer tasks, not as training
targets. Training and validation capture retained every example.

Only draft weights were optimized: BF16, learning rate 1e-5, eight single-example
microbatches per update, 100 optimizer updates, seven unrolled draft positions,
and gradient clipping at 0.5. Target features were captured offline. The original
32,000-token draft vocabulary mapping and feature-layer choices were preserved.
The training objective uses target probabilities reconstructed from saved target
states and the frozen output head, rather than only hard next-token labels.

Checkpoint selection uses the highest validation simulated acceptance length,
evaluated every 25 updates. This is an offline proxy formed from per-position
accuracies, not measured serving acceptance. Step 100 won among the four tested
checkpoints. There was no baseline step-zero evaluation of this proxy, so its
trajectory alone cannot establish improvement over the original draft.

## Serving evaluation

The final comparison uses three arms on the same physical H100: Qwen alone,
Qwen with the original draft, and Qwen with the adapted draft. The execution order
is fixed original/target/adapted, which leaves time/order effects uncontrolled.
The final data and sample sizes were fixed before inspecting held-out outputs.

Each arm receives 32 crate-disjoint Rust prompts at concurrency 1, 4, and 16,
and 32 Python HumanEvalPack prompts at concurrency 1, with three repeats per case.
Each repeat shuffles the same prompt set with a recorded seed. Two or more warmup
requests are excluded. Prefix caching is disabled; temperature is zero; thinking
is disabled; all arms share a 1,024-token cap and stop naturally at EOS.

The primary statistic is the median per-prompt latency reduction: compute each
prompt's median latency across repeats in each arm, then take the median of
1 - adapted/original. A 2,000-resample prompt bootstrap gives an exploratory 95%
interval. Repeats are not treated as independent prompts. Raw outputs, failures,
truncations, client TTFT, token usage, and server metrics are retained. Concurrent
runs are closed-loop throughput tests, not production arrival-rate simulations.

Acceptance is measured separately on 16 held-out Rust prompts through SGLang's
native endpoint. The installed OpenAI-compatible streaming endpoint does not
expose the relevant counters. The denominator is proposed draft tokens; bonus
tokens are excluded. Those diagnostic requests are not part of latency timing.

Exact greedy-output comparisons check for observed behavioral differences. They
do not prove distribution preservation under stochastic sampling. Rust compiler
and unit-test evaluation, a stochastic distribution check, GPU peak-memory
profiling, and separate draft/verify kernel profiling are not performed here.
No claim that the target has become better at Rust is warranted: it is frozen.

## Environment and scope

Models, source code, and data revisions are in ../configs/sources.json. The GPU
image resolved torch 2.13.0, transformers 5.12.1, SGLang 0.5.18, and SpecForge
0.2.0. Full package versions and per-stage script hashes are preserved. Feature
capture used its default FlashInfer backend; serving used its default FA3 backend.
Training used SDPA. These backend differences can introduce numerical variation.

A preformatted-data bug in SpecForge's capture script was fixed: its loader
previously built a conversation column even when downstream code required text.
The exact patch is retained in ../patches/. This repair did not change the target,
draft architecture, optimizer, or evaluation prompts.

There is one training seed, one small synthetic corpus, one target, one draft,
one precision, and a small evaluation sample. Benchmark contamination by the
upstream target/draft cannot be ruled out. Positive results support this pilot
configuration; negative results do not establish that Rust adaptation never helps.
Quantization and the separate DSpark LoRA project are outside this run.

## Evaluation restart and tokenizer audit

The first final-run attempt completed the original-draft timing cases but stopped
in the separate acceptance probe: Transformers 5.12.1 returns BatchEncoding by
default, which was not JSON serializable as input_ids. Explicit return_dict=False
fixes this. The exact pinned tokenizer now passes a local integration test.
The restarted final run writes to final-results/; the aborted attempt's results
remain in results/ and are excluded from the primary comparison. No model or
serving hyperparameter was tuned from those partial results. An extra untimed
one-request acceptance preflight runs before the original arm's normal warmup.

The same default-return behavior made the initial generation prefilter count
fields rather than tokens. Every rendered training sequence was still checked
against the true token limit before capture. A retrospective exact-tokenizer audit
confirmed that all 512 training and 64 validation prompts also pass the intended
prompt-plus-generation-budget filter: maximum prompt lengths were 976 and 471,
and maximum rendered sequence lengths were 1,743 and 1,250. No corpus change or
retraining was needed. The generation prefilter is now corrected.

The second attempt completed the original arm, including acceptance, but failed
on a port-in-use check while switching to the target-only server. The harness
now permits safe socket reuse, explicitly terminates its own worker process group,
and assigns ports 30000/30001/30002 to the three arms. A local HTTP-server
integration test verifies repeated startup and teardown. These infrastructure
repairs do not alter model weights or sampling. The third attempt writes to
final-results-v2/; the earlier partial directories are retained and excluded from
the final comparison.
