
> ref: https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD_VISION.md

## 基本信息

- 提供 11B 和 90B, Released at 2024.9.25
- 输入：文本+图像，输出：文本
- 结构：基于 llama3.1 仅文本的模型
- 场景：视觉识别、图像推理、图像问答等
- context len: 128k
- GQA: YES
- Training on: 6B (image, text) pairs, knowledge cutoff 2023.12


## 模型架构

为了支持图像识别：
- 增加了一个独立训练的 vision adapter：
    - 由一组 cross-attention layers 组成
    - 将 image encoder repr 喂给 core LLM



## Hardware

Training Time
- Pretraining : 11B: 147K, 90B: 

