
> personal note to
>  
> ref: 
> 
> https://zhuanlan.zhihu.com/p/27292649125
> https://zhuanlan.zhihu.com/p/29841050824
> https://zhuanlan.zhihu.com/p/29540042383


## DS V3 网络结构参数
- MLA:
    - $n_h = 128, d_h=128$
    - KV 压缩维度 $d_c=512$
    - Query压缩维度 $d_c^{'}s=1536$
    - $d_h^R=64$
- Expert: 3 个 GEMM, $d_h=7168, d_{expert}=2048$
    - up GEMM
    - down GEMM
    - gate GEMM

共 671B 参数，每个 token 激活 37B。

共 61 层 transformer layer:
- 1-3 layer: 非 MoE
- 4-61 layer: FFN 拓展为 MoE
    - 每个 MoE 层：1 共享专家 + 256 路由专家
    - h=2048
    - 每个 token 激活 8 个路由专家
    - 训练时：每个 token 最多发送 4 个 node 以减少节点间通信？？

参数量近似估计：
- 每个专家权重：44M=42MB
    - $7168*2048*3=44M$
- dense 部分权重：14B=13GB
- expert 权重总量：657B=612GB



## 推理系统

平均 sequence 长度：
- DS V3: 1k + 1k
- DS R1: 1k + 5k

目标：
- 基本部署模式：H20x8 / H800x16
- 问题：kv cache 非常小，导致 MoE 和 attention QKV gemm 的 batchsize 都很低，MoE 的计算完全是 memory bound / latency bound. 
- 优化方向：提高 batchsize，尽量达到 compute bound 的最佳设备数量
- 优化方法：scale out 到多机，减少每卡显存需要容纳的参数量，提升单卡用于 kv cache 的显存

原作者观点：
- TP 对 MLA 部分的 KVCache 不友好：
    - 问题：MLA 将多个 head 压缩到了同一个 hidden vector，无法在 TP 节点内做 head 切分，导致 KVCache 各卡的存储是冗余的。
    - 方案：为了做 scale out，基本假设是 Attention DP + MoE 部分 TP/EP，考虑跨卡通信效率，**Attention DP + MoE EP** 比较可行。


**问题：为了让 MoE layer 达到计算 bound，需要多少张 H800？**

简化条件：无冗余 expert，仅考虑 routed expert，不考虑 1-3 dense layers

### 显存约束分析

