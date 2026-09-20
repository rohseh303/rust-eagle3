# Checkpoints

[Download all six checkpoint bundles from the v0.1.0 release](https://github.com/rohseh303/rust-eagle3/releases/tag/v0.1.0). Model binaries are distributed as release assets and excluded from Git history. Each tar contains the model, configuration, model card, checksums, and notices. Extract it into a new directory and use it with the pinned target; it is not a standalone model.

| Variant | Purpose |
|---|---|
| [scale400](scale400/README.md) | Strongest measured Rust candidate: 2,048 Rust, 400 updates |
| [mixed400](mixed400/README.md) | Rust/prose trade-off: 1,536 Rust + 512 general, 400 updates |
| [small400](small400/README.md) | Matched 512-Rust / 400-update control |
| [scale100](scale100/README.md) | Larger data pool, 100-update control |
| [seed2](seed2/README.md) | Second training seed for the initial adaptation |
| [pilot](pilot/README.md) | Original 512-Rust / 100-update adaptation |

[Release-asset manifest](release-assets.json) records file sizes and SHA-256 hashes. All six binaries match the original execution record's saved checksums. Each is approximately 799 MB; the full set is approximately 4.8 GB. Keep them as release downloads rather than Git blobs. [Reproduction instructions](../REPRODUCE.md) and [full results](../WRITEUP.md).
