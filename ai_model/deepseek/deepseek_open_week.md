## DeepSeek Open Week Review

ref: https://github.com/deepseek-ai/open-infra-index.git

### 1. FlashMLA


### DeepEP


### DeepGEMM


### Optimized Parallelism Strategies


### 3FS, Thruster for All DeepSeek Data Access


### DeepSeek-V3/R1 Inference System Overview
> https://zhuanlan.zhihu.com/p/27181462601

优化思路梳理：
- 目标：高吞吐、低延迟
- 优化点：
    - 大规模跨节点专家并行（Expert Parallelism / EP）
        - WHY：
            - EP 使得 batch size 大大增加，提高矩阵乘的效率；
            - EP 将专家分不到不同 GPU 上，每个 GPU 只需要计算很少的专家，降低访存需求，降低延迟
        - 复杂性：
            - EP 引入跨节点通信，需要设计使得通信和计算重叠
            - EP 涉及多个节点，需要 DP 和负载均衡


#### 关于成本核算

> 在最近的 24 小时里（北京时间 2025/02/27 12:00 至 2025/02/28 12:00），DeepSeek V3 和 R1 推理服务占用节点总和，峰值占用为 278 个节点，平均占用 226.75 个节点（每个节点为 8 个 H800 GPU）。假定 GPU 租赁成本为 2 美金/小时，总成本为 $87,072/天。

87072 = 226.75 * 8 * 2 * 24

这里对应的成本换算一下：
- 2 美金/小时/H800卡
- 16 美金/小时/节点
- 384 美金/天/节点
- 11520 美金/月/节点 （30天）
- 86400 人民币/月/节点 （30天，汇率 7.5）


> - 输入 token 总数为 608B，其中 342B tokens（56.3%）命中 KVCache 硬盘缓存。
> - 输出 token 总数为 168B。平均输出速率为 20~22 tps，平均每输出一个 token 的 KVCache 长度是 4989。
> - 平均每台 H800 的吞吐量为：对于 prefill 任务，输入吞吐约 73.7k tokens/s（含缓存命中）；对于 decode 任务，输出吞吐约 14.8k tokens/s。
> 如果所有 tokens 全部按照 DeepSeek R1 的定价[1]计算，理论上一天的总收入为 $562,027，成本利润率 545%。
> DeepSeek R1 的定价：$0.14 / 百万输入 tokens (缓存命中)，$0.55 / 百万输入 tokens (缓存未命中)，$2.19 / 百万输出 tokens。

562100 = 342e9 * 0.14 / 1000000 + (608e9 - 342e9) * 0.55 / 1000000 + 168e9 * 2.19 / 1000000

成本与理论收入图中，峰值在 30k $ 左右，对应 278 个节点。
按照比例等比拆分一下，对应：
562100/30000 = 18.74
input tokens = 608e9 / 18.74 = 32.4e9
cached input tokens = 342e9 / 18.74 = 18.25e9
output tokens = 168e9 / 18.74 = 9e9

avg input tps = 32.4e9 / 3600 = 9e6
avg output tps = 9e9 / 3600 = 2.5e6


278-18*13-4*11

> 平均每台 H800 的吞吐量为：对于 prefill 任务，输入吞吐约 73.7k tokens/s（含缓存命中）；对于 decode 任务，输出吞吐约 14.8k tokens/s。

prefill: 608e9/(73700*3600*24) = 95.5 台
decode: 168e9/(14800*3600*24) = 131.4 台

131.4 + 95.5 = 226.9 基本和全天均值 226.75 个节点一致



考虑到 deepseek 采用了 prefill 和 decode 分离部署的逻辑，每个 prefill 需要 4 node，每个 decode 需要 18 node。
平均需要部署 23.9 套 prefill 和 7.3 套 decode。
按这个比例，大概 3:1 的 prefill 和 decode 的部署比例，对应到节点数量，就是 12 node prefill + 18 node decode。

> 我们采用多机多卡间的专家并行策略来达到以下目的：
> Prefill：路由专家 EP32、MLA 和共享专家 DP32，一个部署单元是 4 节点，32 个冗余路由专家，每张卡 9 个路由专家和 1 个共享专家
> Decode：路由专家 EP144、MLA 和共享专家 DP144，一个部署单元是 18 节点，32 个冗余路由专家，每张卡 2 个路由专家和 1 个共享专家

猜测：
对于峰值 278 台，116 台用于 prefill, 162 台用于 decode。
或者 278 台，108 台用于 prefill，162 台用于 decode，空闲 8 台用于机动。

实际场景估计会在这两种配置之间做动态调整。










