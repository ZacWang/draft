
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