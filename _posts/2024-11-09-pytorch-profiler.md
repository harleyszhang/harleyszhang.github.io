---
layout: post
title: Pytorch 性能分析器使用探究
date: 2024-11-09 20:50:00
summary: Pytorch 性能分析器使用探究。
categories: DeepLearning
---

## 用法

Pytorch 提供了 profiler 分析器模块来分析模型性能。

1, 先导入相应模块和包 

```python
import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
```

2, 准备一个用于性能分析测试的模型

```python
model = models.resnet18()
inputs = torch.randn(5, 3, 224, 224)
```

3, 使用 `profiler` 分析执行时间, 它的使用代码示例如下所示：

```python
with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("model_inference"):
        model(inputs)
```

`profile` 函数有很多参数，其中最有用的一些是

- `activities` - 要分析的活动列表
    - ProfilerActivity.CPU - PyTorch 运算符、TorchScript 函数和用户定义的代码标签（参见下面的 record_function）；
    - ProfilerActivity.CUDA - 设备上的 CUDA 内核；

- `record_shapes` - 是否记录运算符输入的形状；
- `profile_memory` - 是否报告模型张量消耗的内存量；

> 注意：在使用 CUDA 时，profiler 还会显示主机上发生的运行时 CUDA 事件。