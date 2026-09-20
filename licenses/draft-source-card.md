
腾云智算（Tenyunw）专注于大模型推理优化和GPU云基础设施。

我们在做什么：
- 推理加速：实现了 Eagle3 for training on Qwen3-8B（20,000+ downloads），显著提升LLM推理速度
- GPU云平台：为AIGC应用提供性价比最优的推理部署方案
- 量化优化：基于QAT+ DPO的训练和推理框架，帮助客户在Blackwell架构硬件上翻倍并发能力

核心团队来自腾讯云、华为云、面壁智能等，在AI基础设施领域有15年以上实战经验。

🎁 限时福利：
如果你在做大模型应用部署，我们提供免费的推理优化咨询（30分钟技术诊断），帮你分析：
- 当前部署方案的性能瓶颈
- 成本优化的具体路径
- 适合你场景的推理加速方案

适合谁：
- 月推理费用 > $5K的团队
- 需要降低推理成本30%+
- 考虑自建推理集群

联系方式：

- 官网：https://www.tenyunw.com/


We help AI builders deploy faster and cheaper. Let's talk.

---
license: mit
base_model:
- Qwen/Qwen3-8B
---

## Introduce
We adapted the official speculative sampling training method, Eagle3, for training on Qwen3-8B. 

After implementing Eagle3, the inference performance of Qwen3-8B using the SGLang framework on a single H200 GPU improved from 187 tokens/s to 365 tokens/s.

The TPS (tokens per second) improvement reached nearly 100%.

Amazingly, on a single RTX 5090, the TPS (transactions per second) of Qwen3-8B-Eagle3 increased from 90 to 220.

The TPS (tokens per second) improvement reached nearly 140%.

| model | gpu | tps | 
|---------|---------|---------|
| qwen3-8b   | 5090   | 90   | 
| qwen3-8b-eagle3   | 5090   | 220   |
| qwen3-8b   | h200   | 187   |
| qwen3-8b-eagle3  | h200  | 365  |

Join our AI computing power cloud platform now and enjoy the best AI cloud service experience. The link is as follows: https://tenyunn.com/

## How to use

To use Eagle3 with SGLang, first replace the qwen3.py file in SGLang’s directory (sglang/python/sglang/srt/models/) with the qwen3.py file from this project.


The launch command for using Eagle3 with SGLang is:

```python
python3 -m sglang.launch_server --model Qwen/Qwen3-8B --speculative-algorithm EAGLE3  --speculative-draft-model-path  Tengyunw/qwen3_8b_eagle3 --speculative-num-steps 6        --speculative-eagle-topk 10 --speculative-num-draft-tokens 32 --mem-fraction 0.9         --cuda-graph-max-bs 2 --dtype bfloat16

```

## How to train

Training Dataset:
ultrachat_200k.
Only the prompts from these datasets were utilized for data synthesis. This synthesized data is used to train the Eagle modules. 

dataset nums: 600K samples,1B tokens

Evaluation Dataset:
ShareGPT,GSM8K,HUAMEVAL,MT-BENCH,APLCA

Our Sharegpt test data is located in the eagle_data.jsonl file under this directory.