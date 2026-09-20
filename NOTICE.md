# Attribution and third-party terms

The original experiment code and writing use the root MIT license. That grant does not replace upstream terms. This artifact builds on existing EAGLE-3, Qwen, Tengyunw, SGLang, and SpecForge work; it does not claim their architecture or pretrained acceleration as a new contribution.

| Source | Pinned revision | Declared terms / use |
|---|---|---|
| [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | `b968826d9c46dd6066d109eabc6255188de91218` | Apache-2.0; frozen target, downloaded separately |
| [Tengyunw Qwen3-8B EAGLE-3](https://huggingface.co/Tengyunw/qwen3_8b_eagle3) | `2a1059d51f622b8cad7d7d72840153ffea5488a0` | MIT declared in source card; warm-start draft |
| [Strandset-Rust-v1](https://huggingface.co/datasets/Fortytwo-Network/Strandset-Rust-v1) | `0a8d223302712a2b34a6ad4ce1fd679031894b3d` | Apache-2.0; selected Rust instructions/context |
| [HumanEvalPack](https://huggingface.co/datasets/bigcode/humanevalpack) | `9a41762f73a8cb23bb5811b73d5aab164efcf378` | MIT; selected evaluation prompts |
| [Databricks Dolly 15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a` | CC-BY-SA-3.0; general instructions/context, original answers not used |
| [SpecForge](https://github.com/sgl-project/SpecForge) | `ed64d275bac8a48e126adb2827368603566a3029` | Its included license; training and feature-capture framework |
| [SGLang](https://github.com/sgl-project/sglang) | `0.5.18` | Upstream serving framework, installed separately |

Upstream dataset/model cards are retained under [licenses/](licenses/). They document provenance and declarations; they are not a warranty about third-party content. The target license and SpecForge license are included there. The pinned draft repository supplies a MIT declaration in its model card without a separate LICENSE file; the exact card is retained in each model bundle. New model cards document the adaptation and all binary hashes.

## Data modifications and attribution

Rust records were selected by deterministic crate split and formatted into instructions. Their source reference solutions are excluded. HumanEvalPack records were reformatted into generation prompts. Teacher answers were generated with frozen Qwen; some capped prefixes were retained. The source identity, revision, prompt hashes, IDs, and selection manifests remain available. These transforms do not relicense source material under the root MIT grant.

Dolly data is copyright (2023) Databricks, Inc. Some source material is from Wikipedia editors and contributors; retain the full [Dolly source notice](licenses/dolly-source-card.md). The selected and reformatted Dolly instructions/context in `phase2/data/general/`, and their occurrences in the mixed generated-conversation files, are redistributed under [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/legalcode). Changes: balanced subset selection, chat-template formatting, and replacement of reference answers with Qwen outputs. The root MIT license does not supersede these terms. A copy of the CC license is included in `licenses/CC-BY-SA-3.0.txt`.

Model files are adaptations of the upstream MIT-declared draft; Qwen is required separately under its own terms. The mixed checkpoint's training provenance includes Dolly and is explicitly disclosed. No blanket claim is made that every source record or upstream component was authored by the experiment author.

The JSON/prose/reasoning diagnostic prompts were authored for this study and are not standardized benchmarks. No external source code repositories or private organizational material were used as training data in this experiment.
