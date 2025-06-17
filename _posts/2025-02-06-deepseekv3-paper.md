---
layout: post
title: DeepSeekV3 简单概述
date: 2025-02-06 22:00:00
summary: DeepSeekv3 技术报告的简单概述和总结。
categories: Transformer
---

## 1. 介绍

DeepSeek-V3 是一款强大的 Mixture-of-Experts (`MoE`) 语言模型，**总参数量为 `671B`，每个 token 激活的参数为 37B**。为了实现高效推理和成本效益的训练，DeepSeek-V3 采用了在 DeepSeek-V2 中经过充分验证的 Multi-head Latent Attention (MLA) 和 DeepSeekMoE 架构。此外，**DeepSeek-V3 首创了无辅助损失的负载平衡策略，并引入了 `Multi-Token Prediction` 技术来实现多 token 预测的训练目标，用以提高性能**。我们在 14.8 万亿多样且高质量的 token 上进行了 DeepSeek-V3 的预训练，随后通过监督微调和强化学习阶段，充分发挥其能力。全面评估结果表明，DeepSeek-V3 在性能上超越了其他开源模型，且与领先的闭源模型的性能相当。尽管表现出色，DeepSeek-V3 仅需 2.788M H800 GPU 小时进行全程训练。此外，其训练过程非常稳定。在整个训练过程中，我们没有遇到任何不可恢复的损失峰值，也没有进行任何回滚。

## 2. 模型总结

**架构：创新的负载平衡策略和训练目标**

在 DeepSeek-V2 高效架构的基础上，作者**首创了一种无辅助损失的负载平衡策略**，最大限度减少了因鼓励负载平衡而带来的性能下降。另外还研究了一种**多 token 预测（MTP）目标**，并证明它对模型性能有益。它还可以用于推测性解码，从而加速推理过程。

**预训练：迈向极致的训练效率**

**设计了一个 FP8 混合精度训练框架**，并首次验证了在超大规模模型上进行 FP8 训练的可行性和有效性。通过**算法、框架和硬件的联合设计**，克服了跨节点 MoE 训练中的通信瓶颈，**几乎实现了计算与通信的完全重叠**。这显著提高了 DeepSeek-V3 训练效率并降低了训练成本，使得能够在没有额外开销的情况下进一步扩大模型规模。以仅 2.664M H800 GPU 小时的经济成本，完成了 DeepSeek-V3 在 14.8T tokens 上的预训练，产生了目前最强的开源基础模型。预训练后的后续训练阶段仅需要 0.1M GPU 小时。

**后训练：来自 DeepSeek-R1 的知识蒸馏**

通过引入一种创新方法，旨在将推理能力从长链式思考（CoT）模型（特别是 DeepSeek R1 系列模型之一）提炼到标准大型语言模型（LLMs）中（尤其是 DeepSeek-V3）。

## 参考资料

- [github-DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3)