

## 要点整理

目标：基于用户反馈，改善框架

关键经验：
- 开发者需要可复用、可定制的 agent
- 需要灵活的协作模式，支持 chat 和非 chat 的协作
    - 例如，构建可预测的、有序的 workflow，通过非 chat 接口交互
- 需要更好的可调试性和可扩展性
- 代码质量提升

重新设计 AutoGen 框架：
- 新版本采用了演员模型，以支持分布式、高度可扩展的事件驱动代理系统。
    - 优势：
        - 可组合性：允许开发者使用不同的框架或编程语言构建 agent，以构建更强大的系统
        - 灵活性：支持构建确定性、有序的 workflow，或事件驱动的、去中心化的 workflow
        - 可调试性和可观测性：事件驱动的通信将消息传递从代理转移到集中组件，使得观察和调试代理活动变得更加容易
        - 可扩展性：事件驱动的架构支持分布式和云部署的代理
- 新版本将分为三个库：
    - Core：构建事件驱动代理系统的基本模块
    - AgentChat：接近原 0.2 版本的 API，基于 core 的高级 API，支持 group chat、代码执行、预置 agent 等
    - Extensions：core 接口的实现和第三方集成（例如，Azure 代码执行器和 OpenAI 模型客户端）


## 原文
> https://microsoft.github.io/autogen/0.2/blog/

> New AutoGen Architecture Preview
> 
> One year ago, we launched AutoGen, a programming framework designed to build agentic AI systems. The release of AutoGen sparked massive interest within the developer community. As an early release, it provided us with a unique opportunity to engage deeply with users, gather invaluable feedback, and learn from a diverse range of use cases and contributions. By listening and engaging with the community, we gained insights into what people were building or attempting to build, how they were approaching the creation of agentic systems, and where they were struggling. This experience was both humbling and enlightening, revealing significant opportunities for improvement in our initial design, especially for power users developing production-level applications with AutoGen.
>
> Through engagements with the community, we learned many lessons:
>
> - Developers value modular and reusable agents. For example, our built-in agents that could be directly plugged in or easily customized for specific use cases were particularly popular. At the same time, there was a desire for more customizability, such as integrating custom agents built using other programming languages or frameworks.
> - Chat-based agent-to-agent communication was an intuitive collaboration pattern, making it easy for developers to get started and involve humans in the loop. As developers began to employ agents in a wider range of scenarios, they sought more flexibility in collaboration patterns. For instance, developers wanted to build predictable, ordered workflows with agents, and to integrate them with new user interfaces that are not chat-based.
> - Although it was easy for developers to get started with AutoGen, debugging and scaling agent teams applications proved more challenging.
> - There were many opportunities for improving code quality.
>
> These learnings, along with many others from other agentic efforts across Microsoft, prompted us to take a step back and lay the groundwork for a new direction. A few months ago, we started dedicating time to distilling these learnings into a roadmap for the future of AutoGen. This led to the development of AutoGen 0.4, a complete redesign of the framework from the foundation up. AutoGen 0.4 embraces the actor model of computing to support distributed, highly scalable, event-driven agentic systems. This approach offers many advantages, such as:
>
> - Composability. Systems designed in this way are more composable, allowing developers to bring their own agents implemented in different frameworks or programming languages and to build more powerful systems using complex agentic patterns.
> - Flexibility. It allows for the creation of both deterministic, ordered workflows and event-driven or decentralized workflows, enabling customers to bring their own orchestration or integrate with other systems more easily. It also opens more opportunities for human-in-the-loop scenarios, both active and reactive.
> - Debugging and Observability. Event-driven communication moves message delivery away from agents to a centralized component, making it easier to observe and debug their activities regardless of agent implementation.
> - Scalability. An event-based architecture enables distributed and cloud-deployed agents, which is essential for building scalable AI services and applications.
>
> Today, we are delighted to share our progress and invite everyone to collaborate with us and provide feedback to evolve AutoGen and help shape the future of multi-agent systems.
>
> As the first step, we are opening a pull request into the main branch with the current state of development of 0.4. After approximately a week, we plan to merge this into main and continue development. There's still a lot left to do before 0.4 is ready for release though, so keep in mind this is a work in progress.
>
> Starting in AutoGen 0.4, the project will have three main libraries:
>
> - Core - the building blocks for an event-driven agentic system.
> - AgentChat - a task-driven, high-level API built with core, including group chat, code execution, pre-built agents, and more. This is the most similar API to AutoGen 0.2 and will be the easiest API to migrate to.
> - Extensions - implementations of core interfaces and third-party integrations (e.g., Azure code executor and OpenAI model client).
>
> AutoGen 0.2 is still available, developed and maintained out of the 0.2 branch. For everyone looking for a stable version, we recommend continuing to use 0.2 for the time being. It can be installed using:
>
> ```bash
> pip install autogen-agentchat~=0.2
> ```
>
> This new package name was used to align with the new packages that will come with 0.4: autogen-core, autogen-agentchat, and autogen-ext.
> 
> Lastly, we will be using GitHub Discussion as the official community forum for the new version and, going forward, all discussions related to the AutoGen project. We look forward to meeting you there.

