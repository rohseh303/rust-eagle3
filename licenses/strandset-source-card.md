---
dataset_info:
  features:
  - name: crate_name
    dtype: string
  - name: input_data
    dtype: string
  - name: output_data
    dtype: string
  - name: task_category
    dtype: string
  - name: test
    dtype: string
  splits:
  - name: train
    num_bytes: 276805901
    num_examples: 191008
  - name: test
    num_bytes: 1069557
    num_examples: 225
  download_size: 109175212
  dataset_size: 277875458
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
license: apache-2.0
tags:
- code
size_categories:
- 100K<n<1M
---

# Strandset-Rust-v1

## Overview

**Strandset-Rust-v1** is a large, high-quality synthetic dataset built to advance code modeling for the Rust programming language.  
Generated and validated through **Fortytwo’s Swarm Inference**, it contains **191,008 verified examples** across **15 task categories**, spanning code generation, bug detection, refactoring, optimization, documentation, and testing.

Rust’s unique ownership and borrowing system makes it one of the most challenging languages for AI-assisted code generation. However, due to its relative modernity and rapid evolution, **there is still a lack of large, well-structured Rust datasets**.  
Strandset-Rust-v1 addresses this gap by combining multi-model generation, peer-review validation, and response aggregation-level filtering to deliver **the most comprehensive Rust-specific dataset to date**.

## Key Features

- **191,008 fully validated Rust task examples**
- **15 diverse categories** covering the full Rust development lifecycle  
- **94.3% compilation success rate** verified with `rustc`
- **Peer-reviewed via Fortytwo’s Swarm Inference** consensus network  
- **Structured JSON format** for easy fine-tuning and evaluation  
- **Compatible with Qwen, Llama, and other code LLMs**

---

## Data Generation

### Swarm Inference & Peer Review

The dataset was generated using **Fortytwo’s Swarm Inference network**, where multiple SLMs collaborate to generate, critique, and rank candidate examples.  
Each example passes through a **peer-review consensus** process ensuring correctness and idiomatic style before inclusion.

### Pipeline Summary

1. **Source Extraction:** Parsed over **2,300 popular crates** from [crates.io](https://crates.io) to collect real-world idioms.  
2. **Distributed Generation:** Swarm Inference network generated over 200K candidate examples.  
3. **Peer Validation:** Nodes evaluated examples for syntax, semantics, and idiomatic accuracy.  
4. **Consensus Filtering:** Retained only examples with ≥0.7 agreement score.  
5. **Compilation Testing:** Verified executable correctness with `rustc`.  

---

## Dataset Composition

| Task Category | Examples | Description |
|----------------|-----------|--------------|
| `code_generation` | 17,241 | Generate full Rust functions from text specs |
| `docstring_generation` | 16,889 | Produce Rust-style API documentation |
| `code_explanation` | 16,505 | Explain what given Rust code does |
| `comment_generation` | 16,143 | Add meaningful inline comments |
| `code_summarization` | 15,884 | Summarize function purpose concisely |
| `function_naming` | 15,776 | Suggest idiomatic Rust function names |
| `variable_naming` | 15,754 | Generate semantic variable names |
| `code_review` | 15,195 | Provide critique and improvements |
| `code_completion` | 14,527 | Fill in missing Rust code sections |
| `code_refactoring` | 14,324 | Improve readability while preserving logic |
| `bug_detection` | 12,765 | Identify and fix real-world bugs |
| `code_optimization` | 12,569 | Optimize algorithms or patterns |
| `code_search` | 3,766 | Return relevant code for a natural query |
| `test_generation` | 3,180 | Generate unit tests from specs |
| `api_usage_prediction` | 490 | Predict next API call or usage pattern |

**Total:** 191,008 validated examples  
**Compilation rate:** 94.3%  
**Consensus acceptance:** 73.2%

---

## Data Format

Each record is a JSON object with a unified schema:

```json
{
  "crate_name": "serde_json",
  "task_category": "code_generation",
  "input_data": {
    "title": "Serialize struct into JSON string",
    "description": "Given a Rust struct, generate code that serializes it into a JSON string.",
    "code_context": "use serde::{Serialize, Deserialize};"
  },
  "output_data": {
    "code": "let serialized = serde_json::to_string(&my_struct)?;"
  }
}
```

---

## Validation & Quality Control

Each example underwent a **multi-layered validation** process:

- **Syntax validation** (`rustc` compilation success)
- **Ownership and lifetime verification**
- **Trait-bound and type inference checks**
- **Peer-review scoring** by 3–5 independent SLM nodes
- **Cross-consensus filtering** for idiomatic correctness

Non-code tasks (e.g., docstrings or naming) were validated through **LLM-based semantic scoring** using `Claude Sonnet 4` and `GPT-4o` as reference evaluators.

---

## Statistics

| Metric | Value | Description |
|---------|--------|-------------|
| Total examples | 191,008 | Final curated set |
| Initial generated samples | 200,000+ | Before filtering |
| Average example length | 127 tokens | Compact, diverse inputs |
| Compilation success | 94.3% | Rust `rustc` verified |
| Consensus acceptance | 73.2% | Peer agreement threshold |
| Feature coverage | 89% | Of Rust language constructs |
| Diversity index | 0.82 | Token-level uniqueness measure |

---

## Example Use

### Load with Hugging Face Datasets

```python
from datasets import load_dataset

dataset = load_dataset("Fortytwo-Network/Strandset-Rust-v1")
print(dataset["train"][0])
```

---

## Applications

- Fine-tuning language models for Rust programming  
- Training specialized code copilots or agents  
- Evaluation of Rust reasoning and syntax understanding  
- Data augmentation for compiler-based AI systems  

---

## License

This dataset is released under the **Apache 2.0 License**, allowing unrestricted research and commercial use with attribution.

---

## Citation

```bibtex
@misc{Strandset-Rust-v1,
  title={Strand-Rust-Coder-v1: Rust Coding Model Fine-Tuned on Peer-Ranked Synthetic Data},
  author={Ivashov, Aleksei and Larin, Vladyslav and Tripathi, Vishesh and Nikitin, Ivan}, 
  year={2025}, 
  publisher={Hugging Face},
  url={https://huggingface.co/datasets/Fortytwo-Network/Strandset-Rust-v1} 
}
```

---

## 🌐 Related Resources

- [Strand-Rust-Coder-v1: Technical Report](https://huggingface.co/blog/Fortytwo-Network/strand-rust-coder-tech-report)
- [Fortytwo-Network/Strand-Rust-Coder-14B-v1](https://huggingface.co/Fortytwo-Network/Strand-Rust-Coder-14B-v1)
- [Fortytwo: Swarm Inference with Peer-Ranked Consensus (arXiv)](https://arxiv.org/abs/2510.24801)
- [Self-Supervised Inference of Agents in Trustless Environments (arXiv)](https://arxiv.org/abs/2409.08386)  
- [fortytwo.network](https://fortytwo.network)