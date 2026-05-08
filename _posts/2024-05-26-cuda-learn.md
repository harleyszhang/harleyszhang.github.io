---
layout: post
title: cuda 教程推荐
date: 2024-09-26 11:00:00
summary: 推荐一些不错的 cuda 编程教程。
categories: Hpc
---

### 深度学习基础课程

1，推荐几个比较好的深度学习模型压缩与加速的仓库和课程资料：

1. [神经网络基本原理教程](https://github.com/microsoft/ai-edu/blob/master/%E5%9F%BA%E7%A1%80%E6%95%99%E7%A8%8B/A2-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86/%E7%AC%AC8%E6%AD%A5%20-%20%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/17.1-%E5%8D%B7%E7%A7%AF%E7%9A%84%E5%89%8D%E5%90%91%E8%AE%A1%E7%AE%97%E5%8E%9F%E7%90%86.md)
2. [AI-System](https://microsoft.github.io/AI-System/): 深度学习系统，主要从底层方向讲解深度学习系统等原理、加速方法、矩阵成乘加计算等。
3. [pytorch-deep-learning](https://github.com/mrdbourke/pytorch-deep-learning)：很好的 pytorch 深度学习教程。

2，一些笔记好的博客链接：

- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/): 国内比较好的博客大都参考这篇文章。
- [C++ 并发编程（从C++11到C++17）](https://paul.pub/cpp-concurrency/): 不错的 C++ 并发编程教程。
- [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- [annotated_deep_learning_paper_implementations](https://github.com/labmlai/annotated_deep_learning_paper_implementations)

### 高性能编程学习资料推荐

英伟达 gpu cuda 编程语法和特性学习资料推荐：

- [GPU Architecture and Programming](https://homepages.laas.fr/adoncesc/FILS/GPU.pdf): 了解 GPU 架构和 cuda 编程的入门文档资料，学完可以理解 gpu 架构的基本原理和理解 cuda 编程模型（cuda 并行计算的基本流程）。建议当作学习 cuda 高性能计算编程的第一篇文档（文章）。
- [CUDA Tutorial](https://cuda-tutorial.github.io/): CUDA 教程，分成四部分：CUDA 基础、GPU 硬件细节、最近的特性和趋势和基于任务的编程实例，提供了完整清晰的 PDF 文档和 cuda 代码实例。**建议当作系统性学习 cuda 编程的教程**。
- [learn-cuda](https://github.com/rshipley160/learn-cuda?tab=readme-ov-file): 完整的 cuda 学习教程，包含高级异步方法内容，特点是有性能实验的代码实例。建议当作学习 cuda 高级特性的教程。
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf)：内容很全，直接上手学习比较难，建议当作查缺补漏和验证细节的 cuda 百科全书，目前版本是 12.6。
- 《CUDA C编程权威指南》：翻译的国外资料，说实话很多内容翻译的非常不行，我最开始跟着这个学习的，学了一周，只是了解了线程、内存概念和编程模型的概述，但是细节和系统性思维没学到，而且翻译的不行，内容也比较过时，完全不推荐，我已经替大家踩过坑了。
- 《CUDA 编程：基础与实践_樊哲勇》：国内自己写的教材，我查资料时候挑着看了一点，基本逻辑是通的，虽然很多原理、概念都讲的特别啰嗦，但实践需要的关键知识点都有讲到，想学中文教程的，可以当作当作了解一个方向的快速阅读资料。
- [CUDA-Kernels-Learn-Notes](https://github.com/DefTruth/CUDA-Learn-Notes/tree/main)： CUDA 内核编程笔记及实战代码，有很强的实践性，后期可以重点学习，我也准备认真看下代码和文档。

kernel 编写笔记资料：

- 最基本的通用矩阵乘法（gemm）：https://zhuanlan.zhihu.com/p/657632577

## 参考资料

- [CUDA-Kernels-Learn-Notes](https://github.com/DefTruth/CUDA-Learn-Notes/tree/main)
- [CUDA and Applications to Task-based Programming](https://cuda-tutorial.github.io/)
- [transformer inference arithmetic](https://kipp.ly/transformer-inference-arithmetic/)
- [LLM Inference Unveiled: Survey and Roofline Model Insights](https://arxiv.org/pdf/2402.16363)
- [CUDATutorial](https://github.com/RussWong/CUDATutorial/tree/main)
- [NVIDIA CUDA Knowledge Base](https://github.com/rshipley160/learn-cuda/wiki)
- [cuda_programming](https://github.com/CoffeeBeforeArch/cuda_programming/tree/master)
- [GitHub Repo for CUDA Course on FreeCodeCamp](https://github.com/Infatoshi/cuda-course/tree/master)