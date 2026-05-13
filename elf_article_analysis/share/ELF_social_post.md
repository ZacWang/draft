# 社交软件分发版：ELF 是什么，为什么重要

这篇 MIT/何恺明团队的新工作 **ELF: Embedded Language Flows**，核心不是“第一次做扩散语言模型”，而是重新回答了一个关键问题：

> 语言最终必须输出离散 token，但生成过程一定要一直待在 token 空间里吗？

ELF 的答案是：不一定。

传统 GPT 类模型是从左到右逐 token 生成。很多扩散语言模型则在 token、MASK 或词表分布上反复填空、修正。ELF 走的是另一条路：先把文本表示成连续的 contextual embedding，然后在连续 embedding 空间里用 Flow Matching 从噪声逐步去噪，直到最后一步才把向量翻译回 token。

可以把它理解成：

```text
GPT：从左到右逐字打字
离散 DLM：在空格里反复填词、擦掉、再填
ELF：先形成连续的语义草稿，不断修清楚，最后落成文字
```

这件事重要在于：扩散/Flow 模型本来就擅长连续空间。ELF 尽量不让中间轨迹反复被词表约束，因此可以更自然地使用图像扩散里成熟的技术，比如 Flow Matching、self-conditioning 和 CFG。

论文结果显示，在中等规模扩散语言模型对比中，ELF 用更少训练 token、更少采样步数，超过了多种离散和连续 DLM baseline。它还不能说明 ELF 已经能替代 GPT/Claude/Gemini 这类大规模自回归模型，但它说明：**连续扩散语言模型这条路线并不是死路，而且可能还有很大空间。**

配图建议：

1. 先发 `assets/elf_teaser.gif`：说明“连续 embedding 空间里从噪声流向文本表示”。
2. 再发 `assets/elf_generation.gif`：说明“生成过程逐步变清晰”。
3. 需要完整阅读时，再发 `ELF_report_single.html`，这是单文件 HTML，GIF 已内嵌，浏览器打开即可看到动态效果。
