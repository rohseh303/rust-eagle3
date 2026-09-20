# Rust EAGLE-3: a controlled draft-adaptation experiment

**Hugging Face:** [Rust specialist (`scale400`)](https://huggingface.co/RSRS64/qwen3-8b-eagle3-rust) · [Rust/prose mixture (`mixed400`)](https://huggingface.co/RSRS64/qwen3-8b-eagle3-rust-mixed). Both require the pinned Qwen3-8B target; each page includes direct weight downloads and serving instructions. All six experimental checkpoints remain available in the GitHub release.

An existing EAGLE-3 draft adapted for **frozen Qwen3-8B** achieved **15.63% lower paired latency than the upstream draft** on 32 fresh Rust prompts in the matched-control study (exploratory 95% interval: 14.26–17.11%). H100 80GB, BF16, greedy generation, one request at a time.

**[Read the full write-up →](WRITEUP.md)** · [Reproduce the analysis](REPRODUCE.md) · [Browse artifacts](ARTIFACTS.md) · [Model cards](models/README.md) · [Download weights](https://github.com/rohseh303/rust-eagle3/releases/tag/v0.1.0)

![Matched training-budget comparison](phase2/figures/control.png)

## What the experiment establishes

- **Replication:** the initial 512-example / 100-update gain repeated at about 10.7% with two training seeds and new serving sessions.
- **Data versus training:** a 2,048-example pool added little at 100 updates. At 400 updates, it was 1.76% faster than the 512-example control (0.40–2.84%). Longer training produced the larger gain.
- **Trade-offs:** Rust-only adaptation slowed an eight-prompt JSON diagnostic. Mixed data improved prose relative to Rust-only training, while slowing Rust slightly; JSON recovery was inconclusive.
- **Output checks:** 3,120 follow-up requests, 120 unique prompts, identical text and completion-token counts across all tested drafts/sessions. This is finite greedy-output agreement, not proof of sampling equivalence or functional correctness.

This is draft adaptation and evaluation, not a new architecture, target fine-tuning, quantization, or a better Rust coding model. EAGLE-3, the pretrained draft, SGLang, and SpecForge are upstream work. Concurrency-dependent output variation in the original pilot remains unresolved.

## Verify without a GPU

Python 3.11 or later, standard library only:

```sh
python scripts/verify_release.py
```

This recomputes all 90 follow-up comparisons and native counters from the distributed per-request evidence, checks output parity and artifact hashes, and reproduces the pilot comparisons. It makes no network requests and starts no training.

Optional local checks: `python -m pip install pytest==8.4.2` then `python -m pytest -q`. One optional integration check requires the pinned Qwen tokenizer. Charts use the separately listed plotting dependency.

## Contents and scope

Source code, pinned prompts, generated conversations, raw measurements, training audits, plots, and machine-readable results are included. Weights belong in separate release assets, not Git history; see [model cards and hashes](models/README.md). The source snapshot is approximately tens of megabytes. Each draft is about 799 MB and needs the pinned target model.

The headline is a within-study paired comparison, not a subtraction of medians across different sessions. Confidence intervals are exploratory and conditional on the tested models/prompts. Full methods, the failed concurrency check, data provenance, and limitations are in the write-up.

Completed provider usage: $17.27 before credits; conservative internal estimate: $41.88; cap: $60. These are different accounting measures. No active GPU containers remained at closeout; retained storage is separate. Cloud reproduction needs a new explicit budget and namespace. A clean GPU image rebuild has not been independently validated.

[Licensing and attribution](NOTICE.md) · [Publication changes](PUBLICATION_CHANGES.md) · [Citation](CITATION.cff)
