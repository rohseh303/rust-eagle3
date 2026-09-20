# Artifact index

| Artifact | Where | What it supports |
|---|---|---|
| Full report | [WRITEUP.md](WRITEUP.md) | Methods, results, counterexamples, limits |
| Fixed protocol | [phase2/PROTOCOL.md](phase2/PROTOCOL.md) | Gate, endpoint, chronological addenda |
| Model and source pins | [configs/sources.json](configs/sources.json) | Exact upstream revisions |
| Observed GPU versions | [configs/gpu-constraints.txt](configs/gpu-constraints.txt) | Execution environment, not a tested fresh lockfile |
| Pilot input splits | [data/](data/) | Crate-disjoint train/validation/holdout and Python |
| Follow-up inputs | [phase2/data/](phase2/data/) | Nested Rust pool, mixed-source prompts, fixed evaluation |
| Teacher conversations | [run/regenerated/](run/regenerated/), [phase2/regenerated/](phase2/regenerated/) | Actual training text, capped prefixes, token usage |
| Training audits | [phase2/audits/](phase2/audits/) | Configurations, selection, resume and exposure records |
| Pilot requests | [run/final-results/](run/final-results/) | Three-arm comparison and concurrency caveat |
| Follow-up requests | [phase2/results/](phase2/results/) | Raw per-request latency, output, run settings, native counters |
| All 90 paired comparisons | [phase2/comparisons.csv](phase2/comparisons.csv) | Prompt and crate intervals, per-session estimates |
| Training exposure | [phase2/training-exposure.csv](phase2/training-exposure.csv) | Examples and teacher-token exposure |
| Figures | [phase2/figures/](phase2/figures/) | PNG and SVG, including saved validation curves |
| Output audit | [phase2/final-audit.json](phase2/final-audit.json) | 3,120 successful requests, 120 fixed prompts |
| Cost summary | [phase2/cost-summary.json](phase2/cost-summary.json) | Usage/estimate distinction and shutdown snapshot |
| Model cards and weight hashes | [models/](models/) | Six selected checkpoints; binaries are separate release assets |
| Artifact checksums | [ARTIFACT_SHA256.json](ARTIFACT_SHA256.json) | Distributed data/results/configuration integrity |
| Analysis and experiment code | [scripts/](scripts/) | Offline verification and explicit GPU workflows |
| Capture fix | [patches/specforge-preformatted-capture.patch](patches/specforge-preformatted-capture.patch) | Preformatted input handling |

Raw text is included for auditability. Native response records retain their original counters; cloud volume paths and hardware UUIDs are sanitized. H100 labels consistently distinguish the eight recorded devices. Account billing, cloud app IDs, credentials, local paths, intermediate feature tensors, optimizer states, and failed-job logs are not distributed. The original private record remains unchanged. Weight hashes allow verification of separately downloaded model binaries; they do not reconstruct omitted optimizer state.
