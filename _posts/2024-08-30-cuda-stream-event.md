---
layout: post
title: CUDA 流介绍
date: 2024-10-10 15:00:00
summary: CUDA Stream 用来组织异步操作，并在满足条件时实现计算与数据传输、多个任务之间的重叠执行。
categories: Hpc
---

## CUDA Stream 原理

CUDA 程序的并行，不只发生在 kernel 内部。一方面，kernel 内部会通过 thread、warp、block 在 GPU 上并行执行；另一方面，**kernel 之外的 CUDA 操作**，例如内存拷贝、kernel 启动、内存清零，也可以通过 **CUDA Stream** 进行组织，并在硬件允许、依赖关系满足时实现重叠执行。

初学者可能会把 stream 简单理解成“让 kernel 外部并行起来的工具”，这个说法不算错，但不够准确。更准确的说法是：**CUDA Stream 是一条操作队列。主机线程把 CUDA 操作按顺序提交到某个 stream 中；同一条 stream 内保持顺序，不同 stream 之间没有默认顺序约束，因此有机会并发执行。**

这里的重点不是“天然并行”，而是**异步提交 + 有条件重叠**。

### 可以重叠什么

从程序行为上看，stream 常见的作用主要有几类：

1. **kernel 计算与数据传输重叠**
2. **主机计算与设备端工作重叠**
3. **不同 stream 中的 kernel 交错或并发执行**
4. **不同 stream 中的数据传输与计算交错执行**

但要注意，是否真的并发，不只取决于你用了几个 stream，还取决于：

- GPU 是否支持相应并发能力
- 操作之间是否存在依赖
- 是否使用了异步 API
- 主机内存是否为 pinned memory
- kernel 是否已经吃满了 GPU 资源

所以，**stream 提供的是调度上的独立性，不等于一定带来并行收益**。

### 为什么很多时候 stream 看起来“不重要”

如果程序设计得足够“GPU 化”，通常会尽量做到两点：

- 尽量减少主机和设备之间的数据传输
- 尽量把计算留在设备端完成，避免主机频繁介入

在这种情况下：

- 主机计算与设备计算的重叠，价值会下降
- 主机与设备频繁 copy 的优化空间，也会变小

另外，**如果单个 kernel 已经足够大，能够很好地占满 SM、寄存器、shared memory 和带宽**，那么同时再跑多个 kernel，收益也未必明显，甚至可能相互争抢资源。

所以在很多高吞吐场景里，优化重点往往不是“开更多 stream”，而是：

- 提高单个 kernel 的效率
- 减少不必要的数据搬运
- 提升访存质量
- 做 kernel fusion
- 减少同步点

不过，在下面这些场景中，stream 依然非常重要：

- 推理服务中的请求级并发
- **pipeline 式的数据预取与计算重叠**
- **多 batch 分块执行**（Deepseekv3 的推理优化方案，实现计算和通信的重叠）
- H2D / D2H copy 与 kernel overlap
- 多阶段任务链路的异步调度

### CUDA Stream 的基本语义

一个 CUDA Stream，可以理解为**设备上的一条命令队列**。主机线程把 CUDA 操作提交到某个 stream 后：

- **同一 stream 内的操作按提交顺序执行**
- **不同 stream 之间默认没有先后依赖**。因此，不同 stream 中的操作可能：并发执行、交错执行、也可能仍然串行，取决于硬件资源和运行时条件

Stream 的一些关键理解：
1. 同一 stream 内是有序的：比如先发起 memcpy，再启动 kernel，那么这个 kernel 会等前面的 memcpy 在同一 stream 中完成后再执行。
2. **不同 stream 之间默认无序**：如果两个操作在两个不同 stream 中，CUDA 运行时不会自动为它们建立顺序关系。它们可能重叠，也可能因为资源不足而排队。
3. stream 解决的是“调度关系”，不是“自动提速”：**stream 本身不创造算力**。它只是给运行时一个机会，**让原本被顺序提交的操作，变成可以异步推进、可能重叠**的工作流。

## CUDA Stream 实践

**任何 CUDA 操作都属于某个 stream**：

- 要么属于**默认流**（default stream）
- 要么属于显式创建的**非默认流**

如果不手动指定 stream，大多数 CUDA Runtime API 会把操作提交到默认流。

> 早期资料里经常把默认流叫做 **null stream** 或 **空流**。这个名字历史上常见，但在写文章时，直接称为“默认流”更清晰。

### 创建 CUDA Stream

CUDA Runtime API 中，stream 由 `cudaStream_t` 表示，可以这样创建：

```cpp
__host__ cudaError_t cudaStreamCreate(cudaStream_t* pStream);
```

更常见的写法是：

```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);
```

使用完成后销毁：

```cpp
cudaStreamDestroy(stream);
```

一个最小例子

下面的代码创建了一条非默认流，并把 kernel 提交到这条 stream 中：

```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);

kernel<<<grid, block, 0, stream>>>(...);

cudaStreamDestroy(stream);
```

这里第四个执行配置参数就是 stream。  
如果不写这个参数，kernel 默认提交到默认流。

如果基于 stream 做 CUDA 性能优化，关于 stream，优先关注这几件事：

1. **先确认是否真的存在可重叠的工作**
2. **优先减少不必要的数据传输**
3. **需要重叠 copy 与 compute 时，使用 `cudaMemcpyAsync` + pinned memory**
4. **显式使用非默认 stream，避免默认流语义干扰**
5. **用 profiler 看时间线，而不是凭感觉判断是否并发**

一个简单判断标准是：

> 如果时间线中 copy、kernel、主机侧准备工作可以形成流水线，stream 往往值得做；  
> 如果 kernel 本身已经很重，先优化 kernel，通常比增加 stream 更有效。

小结：CUDA Stream 本质上不是“并行开关”，而是**组织异步操作的机制**。

- 同一 stream 内，操作按顺序执行
- 不同 stream 之间，没有默认顺序约束
- 是否真正重叠，要看 API、依赖关系、内存类型和硬件资源

## CUDA Stream 与 Event

如果说 stream 用来组织工作，那么 event 更像是用来描述“工作做到哪里了”。在 CUDA 中，`event` 常用于两件事：

1. **做同步**
2. **做计时**

单靠 stream，只能表达“同一条队列里的先后顺序”。如果要在**不同 stream 之间建立依赖关系**，通常就要配合 event。

一个典型用法是：

- 在 stream A 中记录一个 event
- 让 stream B 等待这个 event
- 这样就建立了 “A 的某一步完成之后，B 再继续” 的关系

这比直接调用全局同步更细，也更适合做流水线。

例如：

```cpp
cudaStream_t streamA, streamB;
cudaEvent_t event;

cudaStreamCreate(&streamA);
cudaStreamCreate(&streamB);
cudaEventCreate(&event);

kernelA<<<gridA, blockA, 0, streamA>>>(...);
cudaEventRecord(event, streamA);

cudaStreamWaitEvent(streamB, event, 0);
kernelB<<<gridB, blockB, 0, streamB>>>(...);
```

上述代码表达的是：kernelA 在 streamA 中运行，当 kernelA 前面的工作执行到 event 时，记录这个事件，streamB 等待该事件完成后，再启动 kernelB。这样做的好处是，依赖关系只约束必要的部分，不会把整个设备都停下来。

Event 和同步的关系
- cudaDeviceSynchronize()：等待整个设备上的工作完成
- cudaStreamSynchronize(stream)：等待某个 stream 完成
- cudaEventSynchronize(event)：等待某个 event 完成
- cudaStreamWaitEvent(stream, event)：让一个 stream 等另一个 stream 的事件

