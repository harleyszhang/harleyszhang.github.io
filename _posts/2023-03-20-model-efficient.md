---
layout: post
title: 模型压缩-轻量化网络总结
date: 2023-03-20 19:00:00
summary: 轻量级网络的核心是在尽量保持精度的前提下，从模型体积和速度两方面对网络进行轻量化改造。
categories: Model_Compression
---

- [前言](#前言)
- [一些关键字理解](#一些关键字理解)
  - [计算量 FLOPs](#计算量-flops)
  - [内存访问代价 MAC](#内存访问代价-mac)
  - [GPU 内存带宽](#gpu-内存带宽)
  - [Latency and Throughput](#latency-and-throughput)
  - [Volatile GPU Util](#volatile-gpu-util)
  - [英伟达 GPU 架构](#英伟达-gpu-架构)
- [CNN 架构的理解](#cnn-架构的理解)
- [手动设计高效 CNN 架构建议](#手动设计高效-cnn-架构建议)
  - [一些结论](#一些结论)
  - [一些建议](#一些建议)
- [轻量级网络模型部署总结](#轻量级网络模型部署总结)
- [轻量级网络论文解析文章汇总](#轻量级网络论文解析文章汇总)
- [参考资料](#参考资料)

> 文章同步发于 [github 仓库](https://github.com/HarleysZhang/cv_note/blob/master/7-model_compression/%E8%BD%BB%E9%87%8F%E7%BA%A7%E7%BD%91%E7%BB%9C%E8%AE%BA%E6%96%87%E8%A7%A3%E6%9E%90/%E8%BD%BB%E9%87%8F%E7%BA%A7%E6%A8%A1%E5%9E%8B%E8%AE%BE%E8%AE%A1%E4%B8%8E%E9%83%A8%E7%BD%B2%E6%80%BB%E7%BB%93.md)，最新版以 `github` 为主。本人水平有限，文章如有问题，欢迎及时指出。如果看完文章有所收获，一定要先点赞后收藏。毕竟，**赠人玫瑰，手有余香**。

## 前言

**轻量级网络的核心是在尽量保持精度的前提下，从模型体积和速度两方面对网络进行轻量化改造**。关于如何手动设计轻量级网络的研究，目前还没有广泛通用的准则，只有一些指导思想，和针对不同芯片平台（不同芯片架构）的一些设计总结，建议大家从经典论文中吸取指导思想和建议，然后自己实际做各个硬件平台的部署和模型性能测试。

对于卷积神经网络，典型的工作有 `Mobilenet` 系列网络、`ShuffleNet` 系列网络、`RepVGG`、`CSPNet`、`VoVNet` 等论文。轻量级网络论文解析可阅读 [github 专栏-轻量级网络](./3-classic_backbone/efficient_cnn)。

## 一些关键字理解
### 计算量 FLOPs

- `FLOPs`：`floating point operations` 指的是浮点运算次数，理解为**计算量**，可以用来衡量**算法/模型时间的复杂度**。
- `FLOPS`：（全部大写），`Floating-point Operations Per Second`，每秒所执行的浮点运算次数，理解为计算速度, 是一个衡量硬件性能/模型速度的指标，即一个芯片的算力。
- `MACCs`：`multiply-accumulate operations`，乘-加操作次数，`MACCs` 大约是 `FLOPs` 的一半。将 w[0]∗x[0]+... 视为一个乘法累加或 `1` 个 `MACC`。

### 内存访问代价 MAC

`MAC`: `Memory Access Cost` 内存访问代价。指的是输入单个样本（一张图像），模型/卷积层完成**一次前向传播所发生的内存交换总量**，即模型的空间复杂度，单位是 `Byte`。

> `FLOPs` 和 `MAC` 的计算方式，请参考我之前写的文章 [神经网络模型复杂度分析](https://github.com/HarleysZhang/cv_note/blob/79740428b6162630eb80ed3d39052cac52f60c32/9-model_deploy/B-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E6%A8%A1%E5%9E%8B%E5%A4%8D%E6%9D%82%E5%BA%A6%E5%88%86%E6%9E%90.md)。

### GPU 内存带宽

- `GPU` 的内存带宽决定了它将数据从内存 (`vRAM`) 移动到计算核心的速度，是比 `GPU` 内存速度更具代表性的指标。
- `GPU` 的内存带宽的值取决于**内存和计算核心之间的数据传输速度，以及这两个部分之间总线中单独并行链路的数量**。

`NVIDIA RTX A4000` 建立在 `NVIDIA`  `Ampere` 架构之上，其芯片规格如下所示:

<div align="center">
<img src="../images/efficient_model/A4000_specifications.png" width="60%" alt="">
</div>

`A4000` 芯片配备 `16 GB` 的 `GDDR6` 显存、`256` 位显存接口（`GPU` 和 `VRAM` 之间总线上的独立链路数量），因为这些与显存相关的特性，所以 `A4000` 内存带宽可以达到 `448 GB/s`。


### Latency and Throughput
> 参考英伟达-Ashu RegeDirector of Developer Technology 的 `ppt` 文档 [An Introduction to Modern GPU Architecture](https://download.nvidia.com/developer/cuda/seminar/TDCI_Arch.pdf)。

深度学习领域**延迟** `Latency` 和**吞吐量** `Throughput`的一般解释：

+ 延迟 (`Latency`): 人和机器做决策或采取行动时都需要反应时间。延迟是指**提出请求与收到反应之间经过的时间**。大部分人性化软件系统（不只是 AI 系统），延迟都是以**毫秒**来计量的。
+ 吞吐量 (`Throughput`): 在给定创建或部署的深度学习网络规模的情况下，可以传递多少推断结果。简单理解就是在**一个时间单元（如：一秒）内网络能处理的最大输入样例数**。

**`CPU` 是低延迟低吞吐量处理器；`GPU` 是高延迟高吞吐量处理器**。

### Volatile GPU Util

一般，很多人通过 `nvidia-smi` 命令查看 `Volatile GPU Util` 数据来得出 `GPU` 利用率，但是！关于这个利用率(`GPU Util`)，容易产生**两个误区**：

- 误区一: `GPU` 的利用率 = `GPU` 内计算单元干活的比例。利用率越高，算力就必然发挥得越充分。
- 误区二: 同条件下，利用率越高，耗时一定越短。

但实际上，`GPU Util` 的本质只是反应了，在采样时间段内，一个或多个内核（`kernel`）在 `GPU` 上执行的时间百分比，采样时间段取值 `1/6s~1s`。
> 原文为 Percent of time over the past sample period during which one or more kernels was executing on the GPU. The sample period may be between 1 second and 1/6 second depending on the product. 来源文档 [nvidia-smi.txt](https://developer.download.nvidia.com/compute/DCGM/docs/nvidia-smi-367.38.pdf)

通俗来讲，就是，在一段时间范围内， `GPU` 内核运行的时间占总时间的比例。比如 `GPU Util` 是 `69%`，时间段是 `1s`，那么在过去的 `1s` 中内，`GPU` 内核运行的时间是 `0.69s`。如果 `GPU Util` 是 `0%`，则说明 `GPU` 没有被使用，处于空闲中。

也就是说**它并没有告诉我们使用了多少个 `SM` 做计算，或者程序有多“忙”，或者内存使用方式是什么样的，简而言之即不能体现出算力的发挥情况**。
> `GPU Util` 的本质参考知乎文章-[教你如何继续压榨GPU的算力](https://zhuanlan.zhihu.com/p/346389176) 和 [stackoverflow 问答](https://stackoverflow.com/questions/40937894/nvidia-smi-volatile-gpu-utilization-explanation)。

### 英伟达 GPU 架构

`GPU` 设计了更多的晶体管（`transistors`）用于数据处理（`data process`）而不是数据缓冲（`data caching`）和流控（`flow control`），因此 `GPU` 很适合做**高度并行计算**（`highly parallel computations`）。同时，`GPU` 提供比 `CPU` 更高的**指令吞吐量**和**内存带宽**（`instruction throughput and memory bandwidth`）。

`CPU` 和 `GPU` 的直观对比图如下所示

<div align="center">
<img src="../images/efficient_model/gpu-devotes-more-transistors-to-data-processing.png" width="60%" alt="distribution of chip resources for a CPU versus a GPU">
</div>
> 图片来源 [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)

最后简单总结下英伟达 `GPU` 架构的一些特点:
- `SIMT` (`Single Instruction Multiple Threads`) 模式，即多个 `Core` 同一时刻只能执行同样的指令。虽然看起来与现代 `CPU` 的 `SIMD`（单指令多数据）有些相似，但实际上有着根本差别。
- 更适合**计算密集**与**数据并行**的程序，原因是缺少 `Cache` 和 `Control`。

`2008-2020` 英伟达 `GPU` 架构进化史如下图所示:

<div align="center">
<img src="../images/efficient_model/gpu_architecture.png" width="60%" alt="`2008-2020` 英伟达 `GPU` 架构进化史">
</div>

另外，英伟达 `GPU` 架构从 `2010` 年开始到 `2020` 年这十年间的架构演进历史概述，可以参考知乎的文章-[英伟达GPU架构演进近十年，从费米到安培](https://zhuanlan.zhihu.com/p/413145211)。

`GPU` 架构的深入理解可以参考博客园的文章-[深入GPU硬件架构及运行机制](https://www.cnblogs.com/timlly/p/11471507.html#41-gpu%E6%B8%B2%E6%9F%93%E6%80%BB%E8%A7%88)。

## CNN 架构的理解

**网络宽度**，即通道(`channel`)的数量，也是滤波器（3 维）的数量；**网络深度**，即 `layer` 的层数，如 `resnet18` 有 `18` 层网络。注意我们这里说的网络宽度和宽度学习一类的模型没有关系，而是特指深度卷积神经网络的（通道）宽度。

- **网络深度的意义**：`CNN` 的网络层能够对输入图像数据进行逐层抽象，比如第一层学习到了图像边缘特征，第二层学习到了简单形状特征，第三层学习到了目标形状的特征。即**深度决定了网络的表达（抽象）能力，网络越深学习能力越强**。
- **网络宽度的意义**：网络越宽，对目标特征的提取能力越强，即这一层网络能学习到更加丰富的特征，比如不同方向、不同频率的纹理特征等。即**宽度决定了网络在某一层学到的信息量**。

总而言之，在一定的程度上，网络越深越宽，性能越好，但关键在于如何取 `trade-off`。
## 手动设计高效 CNN 架构建议

### 一些结论

1. **分析模型的推理性能得结合具体的推理平台**（常见如：英伟达 `GPU`、移动端 `ARM` `CPU`、端侧 `NPU` 芯片等）；目前已知影响 `CNN` 模型推理性能的因素包括: 算子计算量 `FLOPs`（参数量 `Params`）、卷积 `block` 的内存访问代价（访存带宽）、网络并行度等。但相同硬件平台、相同网络架构条件下， `FLOPs` 加速比与推理时间加速比成正比。
2. 建议对于轻量级网络设计应该考虑直接 `metric`（例如速度 `speed`），而不是间接 `metric`（例如 `FLOPs`）。
3. **`FLOPs` 低不等于 `latency` 低，尤其是在有加速功能的硬体 (`GPU`、`DSP` 与 `TPU`)上不成立，得结合具硬件架构具体分析**。
4. 不同网络架构的 `CNN` 模型，即使是 `FLOPs` 相同，但其 `MAC` 也可能差异巨大。
5. **`Depthwise` 卷积操作对于流水线型 `CPU`、`ARM` 等移动设备更友好，对于并行计算能力强的 `GPU` 和具有加速功能的硬件（专用硬件设计-NPU 芯片）上比较没有效率**。`Depthwise` 卷积算子实际上是使用了大量的低 `FLOPs`、高数据读写量的操作。因为这些具有高数据读写量的操作，再加上**多数时候  `GPU` 芯片算力的瓶颈在于访存带宽**，使得模型把大量的时间浪费在了从显存中读写数据上，从而导致 `GPU` 的算力没有得到“充分利用”。结论来源知乎文章-[FLOPs与模型推理速度](https://zhuanlan.zhihu.com/p/122943688)和论文 [G-GhostNet](https://arxiv.org/pdf/2201.03297.pdf)。

### 一些建议

1. 在大多数的硬件上，`channel` 数为 `16` 的倍数比较有利高效计算。如海思 `351x` 系列芯片，当输入通道为 `4` 倍数和输出通道数为 `16` 倍数时，时间加速比会近似等于 `FLOPs` 加速比，有利于提供 `NNIE` 硬件计算利用率。(来源海思 `351X` 芯片文档和 `MobileDets` 论文)
2. 低 `channel` 数的情况下 (如网路的前几层)，在有加速功能的硬件使用普通 `convolution` 通常会比 `separable convolution` 有效率。（来源 [MobileDets 论文](https://medium.com/ai-blog-tw/mobiledets-flops%E4%B8%8D%E7%AD%89%E6%96%BClatency-%E8%80%83%E9%87%8F%E4%B8%8D%E5%90%8C%E7%A1%AC%E9%AB%94%E7%9A%84%E9%AB%98%E6%95%88%E6%9E%B6%E6%A7%8B-5bfc27d4c2c8)）
3. [shufflenetv2 论文](https://arxiv.org/pdf/1807.11164.pdf) 提出的**四个高效网络设计的实用指导思想**: G1同样大小的通道数可以最小化 `MAC`、G2-分组数太多的卷积会增加 `MAC`、G3-网络碎片化会降低并行度、G4-逐元素的操作不可忽视。
4. `GPU` 芯片上 $3\times 3$ 卷积非常快，其计算密度（理论运算量除以所用时间）可达 $1\times 1$ 和 $5\times 5$ 卷积的四倍。（来源 [RepVGG 论文](https://zhuanlan.zhihu.com/p/344324470)）
5. **从解决梯度信息冗余问题入手**，提高模型推理效率。比如 [CSPNet](https://arxiv.org/pdf/1911.11929.pdf) 网络。
6. 从解决 `DenseNet` 的密集连接带来的高内存访问成本和能耗问题入手，如 [VoVNet](https://arxiv.org/pdf/1904.09730.pdf) 网络，其由 `OSA`（`One-Shot Aggregation`，一次聚合）模块组成。

## 轻量级网络模型部署总结

在阅读和理解经典的轻量级网络 `mobilenet` 系列、`MobileDets`、`shufflenet` 系列、`cspnet`、`vovnet`、`repvgg` 等论文的基础上，做了以下总结：

1. 低算力设备-手机移动端 `cpu` 硬件，考虑 `mobilenetv1`(深度可分离卷机架构-低 `FLOPs`)、低 `FLOPs` 和 低`MAC`的`shuffletnetv2`（`channel_shuffle` 算子在推理框架上可能不支持）
2. 专用 `asic` 硬件设备-`npu` 芯片（地平线 `x3/x4` 等、海思 `3519`、安霸`cv22` 等），分类、目标检测问题考虑 `cspnet` 网络(减少重复梯度信息)、`repvgg2`（即 `RepOptimizer`: `vgg` 型直连架构、部署简单）
3. 英伟达 `gpu` 硬件-`t4` 芯片，考虑 `repvgg` 网络（类 `vgg` 卷积架构-高并行度有利于发挥 `gpu` 算力、单路架构省显存/内存，问题: `INT8 PTQ` 掉点严重）

`MobileNet block` (深度可分离卷积 `block`, `depthwise separable convolution block`)在有加速功能的硬件（专用硬件设计-`NPU` 芯片）上比较没有效率。
> 这个结论在 [CSPNet](https://arxiv.org/pdf/1911.11929.pdf) 和 [MobileDets](https://arxiv.org/pdf/2004.14525.pdf) 论文中都有提到。

除非芯片厂商做了定制优化来提高深度可分离卷积 `block` 的计算效率，比如地平线机器人 `x3` 芯片对深度可分离卷积 `block` 做了定制优化。

下表是 `MobileNetv2` 和 `ResNet50` 在一些常见 `NPU` 芯片平台上做的性能测试结果。

<div align="center">
<img src="../images/efficient_model/model_perf.png" width="60%" alt="深度可分离卷积和常规卷积模型在不同NPU芯片平台上的性能测试结果">
</div>


以上，均是看了轻量级网络论文总结出来的一些**不同硬件平台部署轻量级模型的经验**，实际结果还需要自己手动运行测试。

## 轻量级网络论文解析文章汇总

- [MobileNetv1论文详解](https://github.com/HarleysZhang/cv_note/blob/master/7-model_compression/%E8%BD%BB%E9%87%8F%E7%BA%A7%E7%BD%91%E7%BB%9C%E8%AE%BA%E6%96%87%E8%A7%A3%E6%9E%90/MobileNetv1%E8%AE%BA%E6%96%87%E8%AF%A6%E8%A7%A3.md)
- [ShuffleNetv2论文详解](https://github.com/HarleysZhang/cv_note/blob/master/7-model_compression/%E8%BD%BB%E9%87%8F%E7%BA%A7%E7%BD%91%E7%BB%9C%E8%AE%BA%E6%96%87%E8%A7%A3%E6%9E%90/ShuffleNetv2%E8%AE%BA%E6%96%87%E8%AF%A6%E8%A7%A3.md)
- [RepVGG论文详解](https://github.com/HarleysZhang/cv_note/blob/master/7-model_compression/%E8%BD%BB%E9%87%8F%E7%BA%A7%E7%BD%91%E7%BB%9C%E8%AE%BA%E6%96%87%E8%A7%A3%E6%9E%90/RepVGG%E8%AE%BA%E6%96%87%E8%AF%A6%E8%A7%A3.md)
- [CSPNet论文详解](https://github.com/HarleysZhang/cv_note/blob/master/7-model_compression/%E8%BD%BB%E9%87%8F%E7%BA%A7%E7%BD%91%E7%BB%9C%E8%AE%BA%E6%96%87%E8%A7%A3%E6%9E%90/CSPNet%E8%AE%BA%E6%96%87%E8%AF%A6%E8%A7%A3.md)
- [VoVNet论文解读](https://github.com/HarleysZhang/cv_note/blob/master/7-model_compression/%E8%BD%BB%E9%87%8F%E7%BA%A7%E7%BD%91%E7%BB%9C%E8%AE%BA%E6%96%87%E8%A7%A3%E6%9E%90/VoVNet%E8%AE%BA%E6%96%87%E8%A7%A3%E8%AF%BB.md)

## 参考资料

- [An Introduction to Modern GPU Architecture](https://download.nvidia.com/developer/cuda/seminar/TDCI_Arch.pdf)
- [轻量级网络论文解析合集](https://github.com/HarleysZhang/cv_note/tree/79740428b6162630eb80ed3d39052cac52f60c32/7-model_compression/%E8%BD%BB%E9%87%8F%E7%BA%A7%E7%BD%91%E7%BB%9C%E8%AE%BA%E6%96%87%E8%A7%A3%E6%9E%90)
