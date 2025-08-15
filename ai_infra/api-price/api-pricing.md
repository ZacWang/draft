
## 基于大模型 API 的服务如何定价


### 按 token 用量

目前看到的，主要还是基于 token 用量的方案。分别记录输入、输出 token 的量，进行计费。
细致一些的方案，会做如下设定：
- 峰谷定价：低谷期的单价便宜一些；
- cache 计费：写入 cache 的 token 进行计费，特别的，会设置不同 cache 保留时间，分别计费；
- 被 cache 住的 token，单独定价，更便宜一些。

目前大型 API 服务提供商都是按照这个思路来。

### 单纯包月

这个是相对常见的方案。在聊天形式、coding 形式的产品中，都出现过类似形态。
“订阅”形式，更容易被消费者接受/理解，每月固定消费金额，不会产生超出预期的消费。

### 包月+超额限制频率

这个是 chatgpt 曾经使用过的方案，也是常见的对高频使用进行限制的方案。

限制频率主要包括：
- 每天/每周固定的配额，超出配额将无法使用：例如每天可以向 gpt-5 发 20 个请求，超出的，将无法响应，需等待第二天刷新配额
- 滑动 24h/7d 做配额统计：是上边方案的平滑版本，但是理解成本可能反而会高一些？
- token 配额：在固定时间窗口内，最多消耗这些 token
- 金额配额：在固定时间窗口内，最多消耗这些金额
- 限制不同“套餐包”下的并发请求总数

最近几个月，cursor 在做一些复杂度比较高的方案探索，想来整体的成本压力比较大。
看目前的发展趋势，估计会按照固定金额包月的方案，通过金额配额来做：
- 每个月给 $20 的额度；
- 每次请求根据实际使用的模型、token 按定价计费，在额度内进行扣除；
- 用满 $20 后强制降级（策略在调整）；


### 背景信息梳理

- 不同版本的模型，价格差异巨大：by 25.8，GPT-5 = 5 * GPT-5-mini = 5 * 5 * GPT-5-nano
    - GPT-5: input $1.25/M token, output: $10/M token, cached input 0.125/1M token
    - GPT-5 mini: input $0.25/M token, output: $2/M token, cached input 0.025/1M token
    - GPT-5 nano: input $0.05/M token, output: $4/M token, cached input 0.005/1M token

所以 cursor 可以通过 auto 模式做降级，在指定的配额后，通过auto 模式，大幅降低“无上限方案”的成本。


### 如果我一定要提供“无上限”的方案
一些简单的计算。

如果限制单请求，一个订阅账户，最多能处理多少 token？

一些粗糙的估计：
根据我在 cursor 上一天的使用，平均每条请求：
- claude :
    - input (w/ cache write): 47k
    - input (w/o cache write): 0.5k
    - cache read: 33k
    - output: 2k
- gpt-5:
    - input (w/o cache write): 28k
    - cache read: 30k
    - output: 4k

claude pricing:
- sonnet-4 (prompt <=200K tokens):
    - input (w/ cache write): $3.75/M
    - input (w/o cache write): $3/M
    - cache read: $0.3/M
    - output: $15/M
- sonnet-4 (prompt >200K tokens):
    - input (w/ cache write): $7.5/M
    - input (w/o cache write): $6/M
    - cache read: $0.6/M
    - output: $22.5/M

我的请求基本都在 200k 以内，claude 单请求的价格平均在：
$$ (47*3.75+0.5*3+33*0.3+2*15)/1000 = 0.21765/req $$


gpt-5 pricing:
- GPT-5: input $1.25/M token, output: $10/M token, cached input 0.125/1M token

$$ (28*1.25+30*0.125+4*10)/1000 = 0.07875/req $$

汇总一下：
claude-sonnet-4: $0.21765/req, $20 额度可以用 90 个这种请求
gpt-5: $0.07875/req, $20 额度可以用 250 个这种请求

粗糙估计一下，gpt-5 可以在 22 个工作日，平均每天用 11 个请求。其实还是挺少的。


### 一些极限估计

30天24小时不间断，假定一直在 output：
- 10 tps: 10*3600*24*30 = 25.92 M tokens, = $259 (gpt-5) or $388 (claude-sonnet-5)
- 20 tps: 20*3600*24*30 = 51.84 M tokens, = $518 (gpt-5) or $776 (claude-sonnet-5)
- 30 tps: 30*3600*24*30 = 77.76 M tokens, = $777 (gpt-5) or $1164 (claude-sonnet-5)

> 一般来说，这个定价是按照峰值压力做的评估。在全天中，会有至少一般时间是相对低谷，定价一般按照半价甚至更低
> 按峰谷定价评估，价格会在上述价格的 3/4。
> - 10 tps: $194 (gpt-5) or $291 (claude-sonnet-5)
> - 20 tps: $388 (gpt-5) or $582 (claude-sonnet-5)
> - 30 tps: $582 (gpt-5) or $873 (claude-sonnet-5)

按 8 小时连续使用的频率：
- 10 tps: $86 (gpt-5) or $130 (claude-sonnet-5)
- 20 tps: $172 (gpt-5) or $260 (claude-sonnet-5)
- 30 tps: $258 (gpt-5) or $390 (claude-sonnet-5)

按 1:3 的使用、阅读理解分析时间：
- 10 tps: $22 (gpt-5) or $32 (claude-sonnet-5)
- 20 tps: $44 (gpt-5) or $64 (claude-sonnet-5)
- 30 tps: $66 (gpt-5) or $96 (claude-sonnet-5)

按 1:6 的使用、阅读理解分析时间：
- 10 tps: $12 (gpt-5) or $18 (claude-sonnet-5)
- 20 tps: $24 (gpt-5) or $36 (claude-sonnet-5)
- 30 tps: $36 (gpt-5) or $54 (claude-sonnet-5)


### 另外一种估计

按 30k input + 30k cache input + 4k output：
- input tps: 3k
- output tps: 20
一个请求大概需要 30k/3k + 4k/20= 210s

一个小时可以处理 17 req
按 1:3 的使用时间，4 req/h
按 1:6 的使用时间，2.5 req/h

