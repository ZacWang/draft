
## DeepSeek Model Series

- reasoning models:
    - DeepSeek-R1-Zero: 671B total params, 37B activated params.
        - 基于 DeepSeek-V3-Base，使用大规模强化学习，没有前置 SFT 步骤。
    - DeepSeek-R1: 671B total params, 37B activated params.
        - 针对 DeepSeek-R1-Zero 模型的无限重复、可读性差、语言混合等问题，通过在 RL 前合并冷启动数据，达到 openai-o1 的效果（数学、代码、推理等领域）。
    - DeepSeek-R1-Distill-xxx: 使用不同架构对 DeepSeek-R1 蒸馏得到的小模型，其中 DeepSeek-R1-Distill-Qwen-32B 在多领域超过 openai-o1-mini 的效果。

- base models:
    - DeepSeek-V3: 671B total params, 37B activated params, MoE.
        - Multihead Latent Attention + MoE
        - auxiliary-loss-free strategy for load balancing
        - multi-token prediction training objective for stronger performance
        - pretrain data: 15.8 trillion tokens
        - followed with sft + rl step
        - 2.788M H800 GPU hours for training
        - total params 685B = 671B main model + 14B multi-token prediction module



model tree:

- DeepSeek-V3-Base: 671B MoE, 37B activated params.
    - + RL => DeepSeek-R1-Zero
    - + cold start data, then RL => DeepSeek-R1
      - + Distill on QWen/Llama => DeepSeek-R1-Distill-xxx
      - + Distill on DeepSeek-V3-Base => DeepSeek-V3



## models


### DeepSeek-VL2

params & active params:
- DeepSeek-VL2-Tiny   :  3B, active: 0.6B LLM + 0.4B VE = 1B
- DeepSeek-VL2-Small  : 16B, active: 2.4B LLM + 0.4B VE = 2.8B
- DeepSeek-VL2        : 27B, active: 4.1B LLM + 0.4B VE = 4.5B

training time:
- DeepSeek-VL2-Tiny   : 21k  A800 GPU hours =  7 days on 16 A800 node
- DeepSeek-VL2-Small  : 63k  A800 GPU hours = 10 days on 33 A800 node
- DeepSeek-VL2        : 113k A800 GPU hours = 14 days on 42 A800 node

deploy:
- DeepSeek-VL2-Tiny   : 10GB GPU memory
- DeepSeek-VL2-Small  : 40GB GPU memory
- DeepSeek-VL2        : 80GB GPU memory

training data:
- vision-language alignment data: ShareGPT4V, 1.2M caption & conversation samples
- vision-language pretraining data: 
    - 70% VL data
        - interleaved image-text data: WIT + WikiHow + 30% random sample from OBELICS + chinese content from Wanjuan + inhouse collection of general real-world knowledge
        - image captioning data: opensource with recaption & caption quality filtering
        - optical character recognition data: LaTex OCR + 12M RenderedText + in-house OCR dataset (English & Chinese character recognition)
        - Visual QA data:
            - General VQA: inherit from DS-VL data
            - Table, chart & document: PubTabNet, FinTabNet, Docmatrix
            - Web-to-code and plot-to-Python generation: Websight + generated data by DS-V2.5
            - QA with visual prompt
        - Visual grounding data: arXiv 2306.14824(Z. Peng) + Objects365 (S. Shao)
        - Grounded conversation data : arXiv 2306.14824(Z. Peng)
    - 30% text-only data
        - from DeepSeek's base LLM pretraining corpus
- SFT data: 
    - General visual QA
    - OCR and document understanding
    - Table and chart understanding
    - Reasoning, logic and meth
    - Textbook and academic questions
    - Web-to-code and plot-to-Python generation
    - Visual grounding
    - Grounded conversation
    - Text-Only datasets



### DeepSeek-V3

- training cluster: 2048 H800 GPUs
- training framework: HAI-LLM framework, 16 way PP + 64 way EP spanning 8 nodes + ZeRO-1 DP
    - DualPipe algorithm: fewer PP bubbles, overlaps computation & communication


- pre-train data: 14.8T tokens



#### model architecture

- 671B total params, 37B activated params.
    - activated params 36.7B = 0.9B for embedding + 0.9B for output Head + 34.9B for transformer layers

- layers:
    - input embedding layer
    - 61 Transformer layers
        - 0-2 layers: ffn
        - 3-61 layers: moe
    - output embedding layer


on TP8 ckpt:
- input embedding(embed.weight): [16160, 7168] (16160*7168*8=0.9B)
- output embedding(head.weight): [16160, 7168] (16160*7168*8=0.9B)
- transformer layers:
    - 0-2 layers: 86.1e6 params/layer
        - attn: 36.6e6 params
            - wq: 15.7e6
                - a: scale [12,56] weight [1536,7168]
                - b: scale [24,12] weight [3072,1536]
            - wkv: 6.2e6
                - a: scale [5,56] weight [576,7168]
                - b: scale [32,4] weight [4096,512]
            - wo: scale [56,16] weight [7168,2048]
            - kv_norm: [512]
            - q_norm: [1536]
        - attn_norm: [7168]
        - ffn: 49.5e6 params
            - w1: scale [18,56] weight [2304,7168]
            - w2: scale [56,18] weight [7168,2304]
            - w3: scale [18,56] weight [2304,7168]
        - ffn_norm: [7168]
    - 3-60 layers: 1453.3e6 params/layer
        - attn:36.6e6 params (same as 0-2 layers)
            - wq: 15.7e6
                - a: scale [12,56] weight [1536,7168]
                - b: scale [24,12] weight [3072,1536]
            - wkv: 6.2e6
                - a: scale [5,56] weight [576,7168]
                - b: scale [32,4] weight [4096,512]
            - wo: scale [56,16] weight [7168,2048]
            - kv_norm: [512]
            - q_norm: [1536]
        - ffn: 1416.7e6 params
            - experts[i:i+32]: 1409.37e6 params
                - each [expert i]: 44e6 params
                    - w1: scale [16,56] weight [2048,7168]
                    - w2: scale [56,16] weight [7168,2048]
                    - w3: scale [16,56] weight [2048,7168]
            - shared_experts: 5.5e6 params
                - w1: scale [2,56] weight [256,7168]
                - w2: scale [56,2] weight [7168,256]
                - w3: scale [2,56] weight [256,7168]
            - gate: bias [256] weight [256,7168]
        - ffn_norm: [7168]


for each token, activate:
- input embedding: 0.9B
- output embedding: 0.9B
- transformer layers:
    - 0-2 layers: 86.1e6 params/layer * 8TP
    - 3-60 layers: 87.9e6 params/layer
        - attn:36.6e6 params (same as 0-2 layers) * 8TP
        - ffn: 51.3e6 params
            - experts[i:i+32]: 44e6 params(*)
                - each [expert i]: 44e6 params
                    - w1: scale [16,56] weight [2048,7168]
                    - w2: scale [56,16] weight [7168,2048]
                    - w3: scale [16,56] weight [2048,7168]
            - shared_experts: 5.5e6 params
                - w1: scale [2,56] weight [256,7168]
                - w2: scale [56,2] weight [7168,256]
                - w3: scale [2,56] weight [256,7168]
            - gate: bias [256] weight [256,7168]
        - ffn_norm: [7168]

* total activate 8 experts, so for TP=8 case, each TP candidate average activate 1 expert.


for each expert: 
- w1+w2+w3=44e6 params
- 58 moe layers -> total 44e6*58=2.552B params
shared export: 2.552B params

common params:
- 0-2 layers: 86.1e6 params/layer * 8TP
- 3-60 layers.attn: 36.6e6 params * 8TP
- total: (86.1e6*3+36.6e6*58)*8=19B

for each TP candidate:
- activate 1 expert -> 44e6 params
- activate 8 experts -> 44e6*8=352e6 params
