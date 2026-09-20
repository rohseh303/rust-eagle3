# Authorized pilot execution record

User authorized running the Rust EAGLE-3 pilot on September 18, 2026, with a
maximum $50 budget and no broader data-scaling experiments. No public release
or DSpark GPU work is included in this run.

The initial cloud stage downloads the pinned public target and draft. Environment
compatibility and target/general-draft validation measurements precede training.
No new account or credential is needed: the existing Modal workspace was queried
successfully before launch.

## Frozen initial recipe

- Qwen3-8B BF16 target, unchanged; existing matching EAGLE-3 draft warm start.
- The prepared 512 Rust training prompts and 64 validation prompts, with an
  auditable sequence-length exclusion rule. Report actual retained examples and
  tokens after generation, not just these input counts.
- Generated answers hitting the token cap remain valid training prefixes. Retain
  them without an artificial EOS marker, report their count, and do not filter
  target answers based on compilation success. This policy was fixed before
  generating any target responses.
- Learning rate 1e-5, eight single-sequence microbatches per optimizer update,
  at most 100 updates, at most three epochs. Thus the update cap permits up to
  800 example exposures, rather than silently training on only 100 examples.
- Preserve pretrained draft vocabulary and target feature layers.
- Target and both drafts: natural EOS, maximum 1,024 generated tokens,
  temperature zero, thinking disabled, prefix caching disabled.
- Initial draft chain: three speculative steps, top-k one, four draft slots.
- First baseline comparison uses 16 validation prompts, three repeats. Final
  sample sizes were fixed after baseline timing, before opening held-out outputs:
  32 crate-disjoint Rust holdout prompts and 32 Python control prompts, each with
  three repeats. Rust concurrency 1/4/16; Python concurrency 1. All three final
  arms run sequentially in the same H100 container in general/target/Rust order.
- The first target-only run was interrupted during its third repeat. Its two
  complete repeats are preliminary evidence only; final evaluation has three.
- Per-request acceptance diagnostics use the native endpoint on 16 holdout
  prompts outside timed measurements because SGLang 0.5.18 does not expose those
  counters through its OpenAI streaming response. Bonus tokens are excluded.
- The adapted checkpoint is chosen by SpecForge's validation simulated acceptance
  length using its `best` pointer (the verified upstream default). Validation
  latency is measured after export, before held-out tests.
- Primary comparison: adapted versus unchanged draft latency. Target-only is a
  separate baseline. Ten-percent validation improvement is provisional success;
  no improvement remains a valid reported pilot result.
- Numerical output agreement, cross-language Python timing, and concurrent
  requests are required checks before broad performance claims. Unimplemented
  or unaffordable checks must be explicitly reported as limitations.

## Budget enforcement

All GPU invocations use the local budget launcher. One H100 container per stage,
zero retries, 30-minute remote timeout and 40-minute outer launch timeout. Reserve $5 before each stage and retain
$5 of the $50 ceiling for uncertainty. Successful stages are conservatively
accounted at $6 per hour of the entire local invocation plus $0.50; failed or
interrupted stages retain their full reservation pending investigation. These
are estimates and reservations, not claims about the provider's actual invoice.

The launcher refuses further stages when insufficient safe allowance remains.
It does not establish a provider-level billing cap. Stop compute before writing
the report, and distinguish observed runtime from estimated spend.

## Infrastructure-only restarts

Final evaluation was restarted after a tokenizer payload incompatibility, then
a server-port collision between arms. No model, sampling, prompt selection, or
training hyperparameter changed. Partial results remain in the original results/
and final-results/ volume directories; the final retry uses final-results-v2/.
See METHOD.md for the exact repairs and tests. The extra native preflight request
is excluded from timing. Server ports differ by arm after the transition repair.
