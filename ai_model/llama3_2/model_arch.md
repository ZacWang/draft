
> ref: https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD_VISION.md

## 基本信息

- 提供 11B 和 90B, Released at 2024.9.25
- 输入：文本+图像，输出：文本
- 结构：基于 llama3.1 仅文本的模型
- 场景：视觉识别、图像推理、图像问答等
- context len: 128k
- GQA: YES
- Training on: 6B (image, text) pairs, knowledge cutoff 2023.12

## 模型结构的变化（与 llama3.1 对比）
相比于 llama3.1 的结构：
- 增加了一个基于 cross-attention 的 vision adapter
- core LLM 增加了 25% 的层数
- 输入的处理：（图像，文本）-> 图像部分经过 vision adapter 后，与文本部分拼接，一起输入 core LLM 模型。

具体的：
- vision 11B 模型：
    - core LLM 从原始的 32 层增加到 40 层
- vision 90B 模型：
    - core LLM 从原始的 80 层增加到 100 层




## Hardware

- Training Time (H100 Hours)
    - Pretraining: 11B: 147K, 90B: 885K
    - Annealing  : 11B:  98K, 90B: 885K
    - SFT        : 11B: 896 , 90B: 3072
    - RLHF       : 11B: 224 , 90B: 2048


## 模型架构

为了支持图像识别：
- 增加了一个独立训练的 vision adapter：
    - 由一组 cross-attention layers 组成
    - 将 image encoder repr 喂给 core LLM


### 模型基础结构

外层大结构：

CrossAttentionTransformer:
- vision_model: CrossAttentionTransformerVision
    - vision_encoder: VisionEncoder
        - conv1: ColumnParallelConv2dPatch
        - ln_post: LayerNorm
        - ln_pre: LayerNorm
        - transformer: ImageTransformer
            - resblocks: ModuleList(32 x ImageTransformerBlock)
        - global_transformer: ImageTransformer
            - resblocks: ModuleList(8 x ImageTransformerBlock)
        - pre_tile_pos_embed: TilePositionEmbedding
        - post_tile_pos_embed: TilePositionEmbedding
    - vision_projection: ColumnParallelLinear
- text_model: CrossAttentionTransformerText
    - tok_embeddings: VocabParallelEmbedding
    - norm: RMSNorm
    - output: ColumnParallelLinear
    - learnable_embedding: VocabParallelEmbedding
    - layers: ModuleList(32 x TransformerBlock)
    - cross_attention_layers: ModuleList(8 x CrossAttentionTransformerBlock)

基础组件：
- Huge Blocks:
    - ImageTransformerBlock
    - TransformerBlock
    - CrossAttentionTransformerBlock
- Basic Blocks:
    - LayerNorm
    - RMSNorm
    - ColumnParallelLinear
    - ColumnParallelConv2dPatch
- Embedding Blocks:
    - VocabParallelEmbedding
    - TilePositionEmbedding

```
CrossAttentionTransformer(
  (vision_model): CrossAttentionTransformerVision(
    (vision_encoder): VisionEncoder(
      (conv1): ColumnParallelConv2dPatch(
        (_unfold): Unfold(kernel_size=(14, 14), dilation=1, padding=0, stride=14)
        (_linear): ColumnParallelLinear()
      )
      (ln_post): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
      (ln_pre): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
      (transformer): ImageTransformer(
        (resblocks): ModuleList(
          (0-31): 32 x ImageTransformerBlock(
            (attn): ImageAttention(
              (wq): ColumnParallelLinear()
              (wk): ColumnParallelLinear()
              (wv): ColumnParallelLinear()
              (wo): RowParallelLinear()
            )
            (ln_1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
            (mlp): ImageFeedForward(
              (c_fc): ColumnParallelLinear()
              (c_proj): RowParallelLinear()
              (non_linearity): GELU(approximate='none')
            )
            (ln_2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
          )
        )
      )
      (global_transformer): ImageTransformer(
        (resblocks): ModuleList(
          (0-7): 8 x ImageTransformerBlock(
            (attn): ImageAttention(
              (wq): ColumnParallelLinear()
              (wk): ColumnParallelLinear()
              (wv): ColumnParallelLinear()
              (wo): RowParallelLinear()
            )
            (ln_1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
            (mlp): ImageFeedForward(
              (c_fc): ColumnParallelLinear()
              (c_proj): RowParallelLinear()
              (non_linearity): GELU(approximate='none')
            )
            (ln_2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
          )
        )
      )
      (pre_tile_pos_embed): TilePositionEmbedding()
      (post_tile_pos_embed): TilePositionEmbedding()
    )
    (vision_projection): ColumnParallelLinear()
  )
  (text_model): CrossAttentionTransformerText(
    (tok_embeddings): VocabParallelEmbedding()
    (norm): RMSNorm()
    (output): ColumnParallelLinear()
    (learnable_embedding): VocabParallelEmbedding()
    (layers): ModuleList(
      (0-31): 32 x TransformerBlock(
        (attention): Attention(
          (wq): ColumnParallelLinear()
          (wk): ColumnParallelLinear()
          (wv): ColumnParallelLinear()
          (wo): RowParallelLinear()
        )
        (feed_forward): FeedForward(
          (w1): ColumnParallelLinear()
          (w2): RowParallelLinear()
          (w3): ColumnParallelLinear()
        )
        (attention_norm): RMSNorm()
        (ffn_norm): RMSNorm()
      )
    )
    (cross_attention_layers): ModuleList(
      (0-7): 8 x CrossAttentionTransformerBlock(
        (attention): CrossAttention(
          (wq): ColumnParallelLinear()
          (wk): ColumnParallelLinear()
          (wv): ColumnParallelLinear()
          (wo): RowParallelLinear()
          (q_norm): RMSNorm()
          (k_norm): RMSNorm()
        )
        (attention_norm): RMSNorm()
        (feed_forward): FeedForward(
          (w1): ColumnParallelLinear()
          (w2): RowParallelLinear()
          (w3): ColumnParallelLinear()
        )
        (ffn_norm): RMSNorm()
      )
    )
  )
)
```


#### TransformerBlock vs ImageTransformerBlock
```
TransformerBlock(
    (attention): Attention()
    (attention_norm): LayerNorm()
    (feed_forward): FeedForward()
    (ffn_norm): LayerNorm()
)

ImageTransformerBlock(
    (attn): ImageAttention()
    (ln_1): LayerNorm()
    (mlp): ImageFeedForward()
    (ln_2): LayerNorm()
)
```
整体结构很像，区别主要在 Attention 和 FeedForward 的实现上。

attention 的两种实现结构类似，实现也类似。主要就差一个 kv cache.
```
Attention(
    (wq): ColumnParallelLinear()
    (wk): ColumnParallelLinear()
    (wv): ColumnParallelLinear()
    (wo): RowParallelLinear()
)
ImageAttention(
    (wq): ColumnParallelLinear()
    (wk): ColumnParallelLinear()
    (wv): ColumnParallelLinear()
    (wo): RowParallelLinear()
)
```

FeedForward 的两种实现区别比较大：
- FeedForward: w1 ColumnParallel, w2 RowParallel, w3 ColumnParallel
    - 使用 SwiGLU 架构（三个线性层），使用 SiLU 激活，不使用偏置项，无 dropout
- ImageFeedForward: c_fc ColumnParallel, c_proj RowParallel
    - 使用传统的两层架构，默认使用 GELU 激活，使用偏置项，有 dropout

> 拓展一下，SwiGLU 和传统两层架构的区别：
> - 传统两层：$$ FFN(x) = W_2(GELU(W_1x+b_1))+b_2 $$
> - SwiGLU：$$ FFN(x) = W_2(SiLU(W_1x) \odot W_3x) $$
> SwiGLU 使用额外的投影层 $W_3$ 来生成门控信号，参数量会多 50%，通常有更好的性能。
> 门控机制允许网络动态调整每个特征的重要性。


FeedForward 的实现差别不大。

```
FeedForward(
    (c_fc): ColumnParallelLinear()
    (c_proj): RowParallelLinear()
    (non_linearity): GELU(approximate='none')
)
ImageFeedForward(
    (c_fc): ColumnParallelLinear()
    (c_proj): RowParallelLinear()
    (non_linearity): GELU(approximate='none')
)
```

#### VocabParallelEmbedding vs TilePositionEmbedding

这两个差别还挺大的，用途不太一样。暂时先不展开了。


### 模块之间的连接


#### highlevel

最外层结构如下：
- CrossAttentionTransformer:
    - vision_model: CrossAttentionTransformerVision
    - text_model: CrossAttentionTransformerText

查看 Llama 类的 generate 方法，可以看到具体的调用方式：
- 调用 compute_vision_tokens_masks([images], [mask], total_len)，生成xattn_caches, xattn_masks, full_text_row_masked_out_mask
- 调用 forward(position_ids, tokens, xattn_masks, full_text_row_masked_out_mask, xattn_caches)

即，forward 其实是分成了两半，vision 部分需要独立调用，text 部分还是原来的方式调用 forward 时触发计算。

考虑 llama_models 仓库中调用流程，如下：
(vision, text) input pair:
    -> vision preprocess -|                     # vision input 被编码并缓存为 kv cache，目前独立于 model.forward 存在
    -> text embedding    -|                     # text input 被编码为 token ids
                          |-> text_model.foward

即，模型推理服务封装时，generate 方法需要自行调用 vision prerpocess 对应的方法。（compute_vision_tokens_masks）


#### lowlevel

to be added.

