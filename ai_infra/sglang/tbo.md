

## two_batch_overlap.py

TBO：将一个大的 batch 分成两个子 batch 并行处理，提高 GPU 利用率。




### class MaybeTboDeepEPDispatcher

通过封装多个 DeepEPDispatcher 实例，提供一个统一的接口来支持 TBO 功能。

既兼容 TBO 未开启的场景，又提供了 TBO 开启时的并行处理能力。

使用场景：
```python
# 第一个子 batch
dispatcher.dispatch_a(..., tbo_subbatch_index=0)
result_a = dispatcher.dispatch_b(tbo_subbatch_index=0)

# 第二个子 batch  
dispatcher.dispatch_a(..., tbo_subbatch_index=1)
result_b = dispatcher.dispatch_b(tbo_subbatch_index=1)
```

#### 设计思路

- 初始化阶段，根据 arg `enable_two_batch_overlap` 判断是将 `self._inners` 初始化为一个还是两个 dispatcher。
- 调用接口时，内部通过 `_execute` 统一两种模式下的调用方式。通过 `tbo_subbatch_index` 选择 dispatcher，通过 `getattr(dispatcher, name)` 获取 attr。
- 实际调用 `DeepEPDispatcher` 的 `dispatch_a/b` `combine_a/b` 等方法。
- 向下再次穿透，调用 `_DeepEPDispatcherImplLowLatency` 和 `_DeepEPDispatcherImplNormal` 的对应方法。

#### _DeepEPDispatcherImplNormal._dispatch_core

`_DeepEPDispatcherImplNormal`：
- `dispatch_a`: 开启 JIT DEEPGEMM 时，进行 group quant 计算
- `dispatch_b`: 根据路由把 token 分发给不同专家。核心部分通过 `_dispatch_core` 方法完成。

```python
# def _dispatch_core(...):
# 0. 获取全局 DeepEPBuffer，管理跨节点通信的缓冲区
buffer = self._get_buffer()
# 1. 布局计算：根据每个 token 选择的 top-k 专家索引，分析数据分布，计算通信量，内存预分配
... = buffer.get_dispatch_layout(...)
# 2. 实际分发
... = buffer.dispatch(...)
# 3. 监控/调试的数据记录
get_global_expert_distribution_recorder().on_deepep_dispatch_normal(...)
```

