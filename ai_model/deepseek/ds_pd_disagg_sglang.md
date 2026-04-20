Deploying DeepSeek with PD Disaggregation and Large-scale Expert Parallelism on 96 H100 GPUs
> https://lmsys.org/blog/2025-05-05-large-scale-ep/


DeepSeek 主要结构特征：MLA + MoE，对推理系统的高效运行带来挑战。

SGLang 达到了与 DS 报告的自有推理系统性能水平。
- 52.3k input token/s/node
- 22.3k output token/s/node
(2k input seq)
0.2$/1M output token

推理系统：
12 台 H100 PD 分离：
- 3 Prefill worker
- 9 Decode worker

合计：
- 3*52.3k = 156.9k，平均每台机器 13.075k input token tps
- 9*22.3k = 200.7k，平均每台机器 16.725k output token tps


Highlight:
- PD disagg
- large-scale EP: DeepEP, DeepGEMM, EPLB

Limitations and Future Work:
- Latency Optimization: current TTFT 2-5s, ITL 100ms
- Seq length constraints: 更多 gpu 可以扩展
- MTP integration: sglang 还没支持全面的 MTP with DP attention
- EPLB Distribution:
- [TODO] Flexible TP Sizes: 当前 SGLang 只支持纯 TP 或 DP，显存使用效率不够高。
- [TODO] Blackwell support

## Parallelism Design


- Attention Layers
    - WHY: DS 使用 MLA 建模 input seq 内部的复杂依赖关系
    - HOW: DP Attention, 消除不同设备间的 KV Cache 的重复(sglang v0.4) 。扩展到 hybrid data and tensor parallelism, 为小 batch 提供灵活性。
- Dense FFNs
    - WHY: 虽然只有 3 FFN Layer，但大幅提升了峰值显存消耗
    - HOW: adopt DP over TP
        - Enhanced Scalability:
            - intermediate dim 18432, 在 TP=32 下切分为 576 的碎片，不能被 128 整除，无法在 GPU 中对齐。
            - 使用 DP 可避免碎片，同时设备间负载均衡
        - Optimized Memory Efficiency
            - 一般而言，随着 worker 增加，TP 可减少显存用量，但 DP attention 下，并不是这样。
            - pure TP 下，单层 Transformer 的显存需求：
              $$ Memory=\frac{N_{param}}{TP}+(1+k)N_{hidden\_state}\cdot {DP} $$
              其中：
              $$ N_{hidden\_state}=n_{token}\cdot n_{hidden\_size} $$
              $$ N_{param}=n_{intermediate\_size}\cdot n_{hidden\_size} $$
              在 DP=TP 下，最优解为：
              $$ TP = \sqrt{\frac{N_{param}}{(1+k)N_{hidden\_state}}}=\sqrt{\frac{n_{intermediate\_size}}{(1+k)n_{token}}} $$
            - DS-V3 的 intermediate size 18432
            - prefill: 一般关掉 cuda graph (k=0), token/dev 经常超过 2k, 对应 TP <=3
            - decode: 可能 128 token/dev, k=3, 对应 TP=6
            - 因此，相比纯 TP，DP 可提供一个更具显存效率的方案。
        - Minimized Communication Overhead
            - pure TP, 每个 FFN 需要两个 all-reduce。 使用 DP，可优化为一个 reduce-scatter（在前一个 attn layer 后），加一个 all-gather （在下一个 attn-layer 之前）。减少 50% 通信。
            - 进一步 attn 也是 pure DP 时，完全消除了跨设备通信。 
        - 开启方式：--moe-dense-tp-size=1 
- Sparse FFNs
    - large scale EP
- LM Head
    - 使用 DP（镜像 dense FFN 的策略），减少 memory overhead, 简化通信


## Prefill and Decode Disagg.

传统 PD 集成方案的问题：
- prefill interrruption
- DP Attention Imbalance
    - 两个 DP worker 可能在分别处理 prefill 和 decode，导致 decode 延迟增加
- Incompatible with DeepEP
    - DeepEP 对 P/D 执行不同的分发模式，不兼容

实现方案：
> design doc (https://docs.google.com/document/d/1rQXJwKd5b9b1aOzLh98mnyMhBMhlxXA5ATZTHoQrwvc/edit?tab=t.0)
- Non blocking transfer
- RDMA based transfer
- Flexible API Integration：集成现有通信方案：Mooncake/NIXL

没有太多新意

## Large-scale EP

- EP with DeepEP
    - DeepEP 提供两种分法方式：
        - Normal: 处理长序列，适合 prefill，与 cuda graph 不兼容。不适合 decode 这种以 kernel launch 为 bound 的负载。
        - Low-latency: 适合 decode，支持 cuda graph, 需预分配一个固定的显存。超出会报错。
    - SGLang 默认为 auto mode，动态选择两种模式
    - PD 不分离时，无法在一个通信组同时支持两种模式，PD 分离就可以分别设置了。

- DeepGEMM Integration
    - 提供两个专用函数来处理 MoE 相关的矩阵乘（Grouped GEMMs），适配不同的推理过程
    - Grouped GEMMs(contiguous layout):
        - 面向动态的 input shape, 适合 inference 的 prefill 阶段
    - Grouped GEMMs(masked layout):
        - 固定的 input shape，使用一个 mask tensor 来计算有效的部分
        - 兼容 cuda graph，适合 decode 阶段 kernel launch bound 的情况
    - DeepGEMM 与 DeepEP 的分发模式可以平滑集成：
        - contiguous layout kernel: normal dispatch
            - 需要一个额外的步骤：对 output 排序
                - 参考 LightLLM，实现了一个 triton kernel 做排序
                - 将 normal dispatch 输出的 symbolic shape 排序，来适配 contiguous format
        - masked layout kernel: low-latency dispatch
    - SGLang 集成：也基于 tensor parallel 为 MoE 计算做了集成
    - general GEMM kernel 集成: 加速了 non-MoE 的计算，通过 `SGL_ENABLE_JIT_DEEPGEMM=1` 使用 
    
- Two-batch overlap
- EPLB

## Evaluation

## Toolkits


