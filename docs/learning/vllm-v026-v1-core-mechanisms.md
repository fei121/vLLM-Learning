# vLLM v0.26 V1 核心机制：从 HTTP 请求到流式输出

> 本文是一份与源码绑定的学习笔记。它解释当前工作区的 vLLM V1 请求主线，而不是抽象的“典型 vLLM 架构”。

## 阅读范围与事实边界

本文基于当前检出快照：

- commit：96add737ff022430b17034ce50dc55504c04094f
- git describe：v0.26.1rc0-247-g96add737f
- 主题范围：v0.26 系列开发快照中的 V1 engine。
- 叙事基线：OpenAI-compatible Chat Completions、流式文本生成、decoder-only 文本模型、CUDA、默认不展开 speculative decoding、LoRA、多模态和 KV connector。
- 进程基线：API server 创建 AsyncLLM；AsyncLLM 通过 AsyncMPClient 与后台 EngineCore 通信；data parallel 分支只有在对应配置下才出现。

本文只使用三类表述：

- **源码事实**：当前 commit 的符号、分支、状态更新或测试可以直接确认。
- **设计推论**：由源码事实推出的性能或设计解释，不冒充代码保证。
- **配置相关**：只有特定模型、平台或配置才进入的分支。

文中的 V2 model runner 不是另一套 Engine，也不是所有模型的无条件默认值；选择逻辑以 `VllmConfig.use_v2_model_runner` 为准。

## 0. 如何使用这份学习文档

本文沿用学习路线中的阶段顺序，但将每个阶段展开为可从源码验证的机制说明。完成学习后，应能够：

- 说明请求提交路径、模型执行路径和输出回传路径之间的进程边界；
- 根据 `RequestStatus` 与 Token 计数推导请求下一步是否可调度以及可调度多少 Token；
- 解释 Scheduler 如何在 Token 预算、KV 容量和请求优先级之间做出选择；
- 从请求 Token 位置追踪到 KV block、block table、slot mapping 和 GPU KV tensor；
- 从 `SchedulerOutput` 追踪到前向计算、采样、`ModelRunnerOutput` 与流式文本；
- 分别解释正常完成、stop string、客户端取消、抢占和 engine error 的收尾路径。

不要按文件数量衡量进度，而应围绕“问题 → 状态 → 调度结果 → 状态变化”学习。每读一节，先尝试回答图上方的问题，再沿该节列出的源码符号核对结论，最后记录模块执行前后的状态变化。完成主路径后，再分别验证取消、抢占、0-token step、异步 in-flight 和错误处理路径。

~~~mermaid
---
config:
  theme: base
  look: classic
  themeVariables:
    background: "#FFFFFF"
    mainBkg: "#FFFFFF"
    primaryColor: "#EAF2FF"
    primaryBorderColor: "#2563EB"
    primaryTextColor: "#172033"
    secondaryColor: "#ECFDF5"
    tertiaryColor: "#FFF7ED"
    lineColor: "#475569"
    textColor: "#172033"
    clusterBkg: "#F8FAFC"
    clusterBorder: "#94A3B8"
---
flowchart LR
    q([从一个问题开始]) --> entry[入口与 IPC<br/>API → AsyncLLM → EngineCore]
    entry --> request[请求状态与 Token 进度<br/>Request / RequestStatus]
    request --> sched[本轮调度结果<br/>SchedulerOutput]
    sched --> kv[KV 所有权与地址<br/>KVCacheManager / BlockPool / BlockTables]
    kv --> run[GPU 执行<br/>runner → attention → sampler]
    run --> out[外部语义<br/>OutputProcessor → SSE]
    out --> edge{终止与异常路径是否已覆盖?}
    edge -->|否：回到对应状态| request
    edge -->|是| prove[用测试和状态变化验证结论]
    prove --> q

    classDef question fill:#FFF7ED,stroke:#C2410C,color:#172033
    classDef source fill:#EAF2FF,stroke:#2563EB,color:#172033
    classDef verify fill:#ECFDF5,stroke:#059669,color:#172033
    class q,edge question
    class entry,request,sched,kv,run,out source
    class prove verify
~~~

推荐按五个阶段阅读：

1. **系统边界**：阅读第 2、3、5、10 节，明确 API、EngineCore 与 worker 的进程和职责边界。
2. **请求状态与计算进度**：阅读第 4、6 节，手工推演 `status`、`num_computed_tokens` 和 `num_in_flight_tokens`。
3. **KV 地址映射**：阅读第 7、8 节，区分 token position、`KVCacheBlock.block_id`、block table row 和 slot。
4. **模型执行路径**：阅读第 9 节，从 `SchedulerOutput` 追踪到 `ModelRunnerOutput`。
5. **终止与异常处理**：重读第 10 节，分别画出 stop string、客户端取消、engine error 与正常完成路径。

### 源码证据索引

下表是全文结论的最小证据集。图和正文是在解释这些源码，不是替代它们；“性能收益”类语句均是基于这些事实的设计推论。

| 结论 | 当前源码证据 |
| --- | --- |
| HTTP 路由把 stream 结果包装成 `StreamingResponse` | [`api_router.py::create_chat_completion`](../../vllm/entrypoints/openai/chat_completion/api_router.py) |
| serving 先渲染输入和采样参数，再构造/消费 `AsyncLLM.generate` | [`serving.py::OpenAIServingChat._create_chat_completion`](../../vllm/entrypoints/openai/chat_completion/serving.py) |
| 前端先登记 `RequestState`，再跨进程提交请求 | [`async_llm.py::AsyncLLM._add_request`](../../vllm/v1/engine/async_llm.py) |
| ADD/ABORT 与输出使用共享 ZeroMQ 通道，不是每请求 RPC 返回值 | [`core_client.py::AsyncMPClient`](../../vllm/v1/engine/core_client.py) |
| 状态、token 计数与 in-flight 计数的定义 | [`request.py::Request / RequestStatus`](../../vllm/v1/request.py) |
| Scheduler 不划分固定 prefill/decode phase，并先处理 running | [`scheduler.py::Scheduler.schedule`](../../vllm/v1/core/sched/scheduler.py) |
| 调度后预先推进 computed/in-flight，输出返回时再结算和修正 | [`scheduler.py::_update_after_schedule / update_from_output`](../../vllm/v1/core/sched/scheduler.py) |
| 抢占释放请求 blocks、清零 computed，并以 `PREEMPTED` 重排队 | [`scheduler.py::_preempt_request`](../../vllm/v1/core/sched/scheduler.py) |
| engine step 的 execute、grammar、sample、abort、update 顺序 | [`core.py::EngineCore.step`](../../vllm/v1/engine/core.py) |
| KV 分配以 `None` 表示容量不足，并考虑 watermark/reserved blocks | [`kv_cache_manager.py::allocate_slots`](../../vllm/v1/core/kv_cache_manager.py) |
| prefix hit 将最大长度限制为 `num_tokens - 1`；fine-grained manager 还支持 partial-tail hit 与 CoW | [`kv_cache_manager.py::get_computed_blocks`](../../vllm/v1/core/kv_cache_manager.py)、[`single_type_kv_cache_manager.py`](../../vllm/v1/core/single_type_kv_cache_manager.py) |
| cached block 的引用、free queue 和实际淘汰是不同动作 | [`block_pool.py::touch / free_blocks / get_new_blocks`](../../vllm/v1/core/block_pool.py) |
| V2 runner 用 `BlockTables` 构造 batch tables/slots；V1 runner 有独立输入准备路径 | [`gpu/block_table.py::BlockTables`](../../vllm/v1/worker/gpu/block_table.py)、[`gpu_model_runner.py::_get_slot_mappings`](../../vllm/v1/worker/gpu_model_runner.py) |
| V1/V2 runner 的选择与两段式 execute/sample | [`vllm.py::use_v2_model_runner`](../../vllm/config/vllm.py)、[`gpu_model_runner.py`](../../vllm/v1/worker/gpu_model_runner.py)、[`gpu/model_runner.py`](../../vllm/v1/worker/gpu/model_runner.py) |
| stop string 在前端 detokenize 后可能触发反向 abort | [`output_processor.py::process_outputs`](../../vllm/v1/engine/output_processor.py)、[`detokenizer.py::check_stop_strings`](../../vllm/v1/engine/detokenizer.py) |

每个机制节都按同一顺序组织：**问题 → 机制图 → 源码事实 → 状态不变量 → 边界路径 → 自测问题**。如果某句话无法落回上表或该节追加的源码入口，就不应把它当作当前版本的结论。

## 1. 核心问题与状态模型

LLM 服务的难点不只是矩阵乘法。不同请求的 prompt 长度、生成长度和到达时间都不同。如果把一批请求固定成静态 batch，短请求结束后留下的空位会浪费计算；如果为每条请求预留连续的 KV Cache，又会带来显存碎片和过度预留。

vLLM 的两个核心机制分别回答两个问题：

1. continuous batching：每个 engine step 重新决定哪些请求、哪些 token 进入下一轮计算。
2. PagedAttention 相关的 KV block 管理：把请求的逻辑历史映射到物理 GPU KV blocks，而不是为每条请求预留一段连续显存。

前者主要改变调度和 batch 组成，后者主要改变 KV Cache 的分配、复用和 attention 读取方式。两者在 SchedulerOutput、worker block table 和 attention metadata 处汇合，但不是同一个机制。

### 贯穿全文的三类核心状态

阅读 V1 源码时，最容易混淆的是“请求是否还活着”“哪些 token 已算完”和“哪些显存仍被占用”。它们由不同对象维护，不能互相替代：

| 要回答的问题 | 主要事实来源 | 核心量或状态 | 为什么要分开 |
| --- | --- | --- | --- |
| 请求此刻能否被调度？ | `Request.status`、Scheduler 的 `waiting`、`skipped_waiting`、`running` | `WAITING`、`RUNNING`、`PREEMPTED`、`FINISHED_*` | 同一请求可能因 grammar、远端 KV 或流式输入而暂时不可运行。 |
| 该请求还有多少 Token 待计算？ | `Request` | `num_computed_tokens`、`num_tokens_with_spec`、`num_in_flight_tokens` | V1 通过目标 Token 位置与已调度计算位置之间的差值，统一描述 chunked prefill、decode 和 speculative token。 |
| 这些历史 token 占了哪些 KV 空间？ | `KVCacheManager`、`BlockPool`、`BlockTables` | 请求 block 映射、物理 block、GPU slot mapping | 请求结束不必然意味着缓存立即消失；反过来，逻辑历史也不是 GPU tensor 本身。 |

后续流程都围绕这三类状态展开：**Scheduler 根据请求状态和计算进度生成本轮调度结果，并分配所需 KV blocks；worker 根据 KV 地址信息读写缓存并执行模型；模型输出再更新请求状态与计算进度。**

## 2. 全景：一次请求经过哪些边界

这张图要回答的问题：一个 Chat Completions 请求从 HTTP 入口到流式响应，跨过哪些模块、任务和进程？

~~~mermaid
---
config:
  theme: base
  look: classic
  themeVariables:
    background: "#FFFFFF"
    mainBkg: "#FFFFFF"
    primaryColor: "#EAF2FF"
    primaryBorderColor: "#2563EB"
    primaryTextColor: "#172033"
    secondaryColor: "#ECFDF5"
    tertiaryColor: "#FFF7ED"
    lineColor: "#475569"
    textColor: "#172033"
    clusterBkg: "#F8FAFC"
    clusterBorder: "#94A3B8"
---
flowchart LR
    client([OpenAI 客户端])

    subgraph api[API 服务进程]
        router[API 路由<br/>create_chat_completion]
        serving[OpenAIServingChat]
        llm[AsyncLLM]
        input[InputProcessor]
        mp[AsyncMPClient]
        handler[后台 output_handler]
        output[OutputProcessor]
        collector[(RequestOutputCollector)]
    end

    subgraph coreproc[EngineCore 进程]
        proc[EngineCoreProc<br/>继承 EngineCore<br/>queues + step / post_step]
        sched[Scheduler]
        executor[Executor]
    end

    subgraph execution[执行层；进程边界由 Executor 后端决定]
        worker[GPU Worker]
        runner[选中的 GPUModelRunner<br/>V1 或 V2]
        attn[Attention backend]
        kv[(GPU KV cache tensors)]
    end

    client -->|HTTP| router --> serving
    serving -->|消费异步生成器| llm
    llm --> input
    llm -->|创建| collector
    llm -->|先登记前端 RequestState| output
    output -.->|持有 collector 并按 request_id put| collector
    llm -->|再提交 EngineCoreRequest| mp
    mp -.->|ZeroMQ ADD / ABORT IPC| proc
    proc --> sched
    proc --> executor --> worker --> runner --> attn
    attn <--> kv
    runner --> worker --> executor --> proc
    proc -.->|ZeroMQ EngineCoreOutputs IPC| mp
    mp -->|共享输出队列| handler --> output
    collector -->|RequestOutput| serving -->|SSE| client

    classDef control fill:#EAF2FF,stroke:#2563EB,color:#172033
    classDef data fill:#ECFDF5,stroke:#059669,color:#172033
    classDef gpu fill:#FFF7ED,stroke:#C2410C,color:#172033
    class router,serving,llm,input,mp,handler,output,proc,sched,executor control
    class collector data
    class worker,runner,attn,kv gpu
~~~

读图要点：

- `AsyncMPClient`、`InputProcessor`、`OutputProcessor` 和 collector 都位于 API 进程；`CoreClient` 的命名不代表该对象位于 EngineCore 进程。
- `EngineCoreProc` 继承 `EngineCore`，在同一个对象上增加输入/输出队列与 busy loop；worker 是否另起进程取决于具体 Executor 后端。
- 两条虚线是共享 ZeroMQ 输入/输出通道。ADD 请求本身不返回一个 per-request stream；前端按 `request_id` 把共享输出分流到 collector。
- GPU KV tensor 与 scheduler 维护的 `KVCacheBlock` 所有权映射是不同层次的数据。

### 进程边界与调用关系

整条链路跨越多个进程与异步任务，不能按单一同步调用栈理解。`AsyncLLM` 是 API 侧协调点：它通过 `AsyncMPClient` 的共享输入 socket 提交 `EngineCoreRequest`，并从本地 collector 读取按请求分流的结果。后台 `EngineCoreProc` 继承 `EngineCore`，在同一实例中处理输入、调度与执行；worker 负责准备 GPU 输入、读写 KV、前向计算和采样。

因此有两条方向相反的路径：

- **提交路径**：HTTP 请求经过渲染和输入处理后，作为请求消息送进 engine。
- **回传路径**：engine 产生 `EngineCoreOutputs`，API 侧后台任务把它加工成 `RequestOutput`，流式生成器才把它发给客户端。

这一区分解释了一个常见现象：HTTP handler 已经拿到 generator 并建立 `StreamingResponse`，不表示第一个 token 已经算出。`AsyncLLM.generate` 是异步生成器；开始迭代它时才执行 `add_request` 的正文。

源码入口：vllm/entrypoints/openai/api_server.py、vllm/entrypoints/openai/chat_completion/serving.py、vllm/v1/engine/async_llm.py、vllm/v1/engine/core_client.py、vllm/v1/engine/core.py。

## 3. API 请求如何变成 EngineCoreRequest

这张图要回答的问题：HTTP 请求什么时候完成校验、渲染和输入处理，什么时候才真正进入 V1 engine？

~~~mermaid
---
config:
  theme: base
  look: classic
  themeVariables:
    background: "#FFFFFF"
    mainBkg: "#FFFFFF"
    primaryColor: "#EAF2FF"
    primaryBorderColor: "#2563EB"
    primaryTextColor: "#172033"
    secondaryColor: "#ECFDF5"
    tertiaryColor: "#FFF7ED"
    lineColor: "#475569"
    textColor: "#172033"
---
sequenceDiagram
    autonumber
    actor User as 客户端
    participant Router as API route
    participant Serving as OpenAIServingChat
    participant LLM as AsyncLLM
    participant Input as InputProcessor
    participant Output as OutputProcessor
    participant MP as AsyncMPClient
    participant Proc as EngineCoreProc

    User->>Router: POST /v1/chat/completions
    Router->>Serving: create_chat_completion(request)
    Serving->>Serving: render_chat_request + SamplingParams
    alt 校验或渲染失败
        Serving-->>Router: ErrorResponse
        Router-->>User: JSON error
    else stream=true
        Serving->>LLM: 构造 generate(...) 异步生成器
        Serving-->>Router: chat_completion_stream_generator
        Router-->>User: 建立 StreamingResponse
        Note over Serving,LLM: 开始迭代 result_generator 后才进入 generate 正文
        Serving->>LLM: 请求下一个 RequestOutput
        LLM->>Input: process_inputs 或 process_inputs_async
        Input-->>LLM: EngineCoreRequest
        LLM->>LLM: assign_request_id + 创建 collector
        LLM->>Output: add_request，先登记 RequestState
        LLM->>MP: add_request_async
        MP-)Proc: ZeroMQ ADD 消息
        Note over Output,Proc: ADD 不返回“每请求通道”；输出按共享 IPC 回传并由 request_id 分流
    end
~~~

`OpenAIServingChat._create_chat_completion` 先调用 `render_chat_request`，再构造采样参数和 `AsyncLLM.generate(...)` 异步生成器。streaming 分支把它包装进 `chat_completion_stream_generator`；真正开始迭代 `result_generator` 时，`AsyncLLM.generate` 才调用 `add_request`、建立 collector，并提交请求。模型不在 serving 层运行。

当前 API server 的 build_async_engine_client_from_engine_args 会调用 AsyncLLM.from_vllm_config。AsyncLLM 初始化时创建 InputProcessor、OutputProcessor，并调用 EngineCoreClient.make_async_mp_client。data_parallel_size 为 1 时，该工厂返回 AsyncMPClient；数据并行配置会进入 DPAsyncMPClient 或 DPLBAsyncMPClient 分支。

因此，API handler 拿到的是 `AsyncGenerator[RequestOutput]`，而不是一次性模型结果。请求提交和请求输出消费是两个相互配合的异步方向；共享 IPC 输出队列与每请求 collector 也必须分开理解。

### 输入规范化与前端状态登记

`AsyncLLM.add_request` 接受的并不只是一种 prompt。已经渲染好的 `EngineInput` 走同步 `InputProcessor.process_inputs`；原始 prompt 则走 `process_inputs_async`，源码明确把 tokenization 和可能的多模态处理放到不会阻塞事件循环的路径。处理完成后，`assign_request_id` 赋予内部请求 ID，`OutputProcessor.add_request` 先在 API 侧登记输出状态，然后才调用 `engine_core.add_request_async` 跨进程提交。

这个“先登记、后提交”的顺序很关键：即使 engine 很快返回第一批输出，前端已经有对应的 collector 可以接收它。`SamplingParams.n > 1` 时，`add_request` 还会创建 `ParentRequest` 并 fan-out 为多个内部 child request；外部看到的仍是一个逻辑请求的多候选输出，而调度器实际处理的是多个可独立运行的子请求。

`EngineCoreRequest` 是进入 engine 前完成规范化的内部请求对象。它包含 Token 化后的 prompt 或 embedding、采样或池化参数、优先级，以及 LoRA、多模态和追踪等请求元数据。OpenAI HTTP 协议的解析已在前端完成，Scheduler 面向统一的内部数据结构和当前资源状态进行调度。

## 4. 请求生命周期与连续批处理

这张图要回答的问题：V1 Request 的状态如何从等待进入执行，又如何处理抢占、完成和错误？

~~~mermaid
---
config:
  theme: base
  look: classic
  themeVariables:
    background: "#FFFFFF"
    mainBkg: "#FFFFFF"
    primaryColor: "#EAF2FF"
    primaryBorderColor: "#2563EB"
    primaryTextColor: "#172033"
    secondaryColor: "#ECFDF5"
    tertiaryColor: "#FFF7ED"
    lineColor: "#475569"
    textColor: "#172033"
    clusterBkg: "#F8FAFC"
    clusterBorder: "#94A3B8"
---
flowchart LR
    start([创建 Request]) --> structured{包含 structured output?}
    structured -->|否| W[WAITING]
    structured -->|是| WG[WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR]
    WG -->|grammar ready| W
    WG -->|grammar exception| FE[FINISHED_ERROR]

    W -->|普通 admission| R[RUNNING]
    W -->|异步 remote KV load| WK[WAITING_FOR_REMOTE_KVS]
    WK --> ready{load 完成时曾被抢占?}
    ready -->|否| W
    ready -->|是| P[PREEMPTED]
    R -->|KV 不足，释放 blocks 并清零 computed| P
    P -->|重新 admission| R

    R -->|resumable 停止且暂无新输入| WS[WAITING_FOR_STREAMING_REQ]
    WS -->|新 streaming update| W

    R --> finish{check_stop / 请求级错误}
    finish -->|EOS 或 stop token| FS[FINISHED_STOPPED]
    finish -->|max tokens / model len| FL[FINISHED_LENGTH_CAPPED]
    finish -->|repetition detection| FR[FINISHED_REPETITION]
    finish -->|grammar / KV load / request error| FE

    active[任一未完成状态] -.->|finish_requests 外部中止| FA[FINISHED_ABORTED]

    classDef decision fill:#FFF7ED,stroke:#C2410C,color:#172033
    classDef waiting fill:#EAF2FF,stroke:#2563EB,color:#172033
    classDef running fill:#ECFDF5,stroke:#059669,color:#172033
    classDef finished fill:#F8FAFC,stroke:#64748B,color:#172033
    class structured,ready,finish decision
    class W,WG,WK,WS,P waiting
    class R running
    class FS,FL,FR,FA,FE finished
~~~

`RequestStatus` 的状态名直接来自 `vllm/v1/request.py`。图中只包含当前主要生成路径能够从源码确认的迁移。枚举还声明了 `FINISHED_IGNORED`，但当前 `vllm/v1` Python 源码没有将请求设置为该状态，因此图中不添加对应迁移。

`PREEMPTED` 不等于完成。`_preempt_request` 释放请求 blocks、把 `num_computed_tokens` 清零、设置 `PREEMPTED`，并将对象放回 waiting queue；重新 admission 时它直接进入 `RUNNING`。任何大于 `PREEMPTED` 的枚举值才满足 `RequestStatus.is_finished`。

### Prefill 与 decode 共享统一的进度模型

这是理解 V1 的关键。`Scheduler.schedule` 的源码注释明确说明：调度器不存在独立的“prefill phase”或“decoding phase”。对每个请求，它比较：

- `num_computed_tokens`：Scheduler 已推进到的计算位置。该值在调度后立即增加，因此在异步或 pipeline parallel 路径中可能包含尚未返回结果的 Token；
- `num_tokens_with_spec`：当前计算目标位置，等于 prompt、已生成 output 和可能存在的 speculative token 的总长度。

本轮候选计算量以两者之差为基础，再受 `token_budget`、`max_model_len`、长 prefill 阈值、encoder 预算和 KV 容量约束。长 prompt 因此可以拆分到多个 step；decode 请求通常只新增少量待计算 Token。两类请求可以同时出现在一个 `SchedulerOutput` 中。**Continuous batching** 指的正是 Scheduler 在每个 step 根据各请求的当前进度重新组成 batch，而不是维持固定的 prefill batch 和 decode batch。

这里还要区分两个容易混淆的计数：`_update_after_schedule` 同时增加 `num_computed_tokens` 与 `num_in_flight_tokens`；`update_from_output` 在结果返回后减少 in-flight，并在 speculative rejection 等情况下修正 `num_computed_tokens`。因此，前者表示 Scheduler 已推进的计算位置，后者表示其中尚未完成结果处理的 Token 数。

连续 batching 的关键不是“一个永远不变的 batch”，而是 engine step 之间的状态转换：

- 既有 running 请求优先消耗本轮预算。
- 只有本步没有发生抢占且 Scheduler 未暂停时，才进入普通 waiting admission。
- 某些请求可能因 structured output grammar、remote KV 或 streaming input 暂时阻塞。
- 完成请求从运行集合移出，相关请求态释放；缓存 block 是否保留由 KV cache 逻辑单独决定。

~~~mermaid
---
config:
  theme: base
  look: classic
  themeVariables:
    background: "#FFFFFF"
    mainBkg: "#FFFFFF"
    primaryColor: "#EAF2FF"
    primaryBorderColor: "#2563EB"
    primaryTextColor: "#172033"
    secondaryColor: "#ECFDF5"
    tertiaryColor: "#FFF7ED"
    lineColor: "#475569"
    textColor: "#172033"
---
flowchart TD
    start([新的 engine step]) --> delta[计算每个请求的待处理 Token 数<br/>num_tokens_with_spec + placeholders<br/>- num_computed_tokens]
    delta --> limits[按 token budget、model len、encoder budget<br/>和长 prefill 阈值裁剪]
    limits --> slots{allocate_slots 成功?}
    slots -->|否| preempt[抢占并重置 computed/KV<br/>或停止本轮选择]
    preempt --> result
    slots -->|是| result[构造 SchedulerOutput<br/>记录本 step 的调度结果]
    result --> advance[_update_after_schedule<br/>推进 computed 与 in_flight]
    advance --> execute[runner 执行 forward / sampling]
    execute --> settle[update_from_output<br/>in_flight 结算、拒绝回滚、追加 token]
    settle --> status{请求结束?}
    status -->|否| start
    status -->|是| free[释放请求引用；缓存块可继续留存]
    free --> start

    classDef decision fill:#FFF7ED,stroke:#C2410C,color:#172033
    classDef state fill:#ECFDF5,stroke:#059669,color:#172033
    class slots,status decision
    class result,advance,settle state
~~~

读图要点：

- Scheduler.schedule 先处理 running requests，再处理 waiting queue。
- 新请求可以在旧请求仍然 decode 时进入同一轮或后续轮次。
- 每一轮的 `SchedulerOutput` 都记录本 step 的调度决策，不是跨 step 持久存在的 batch。
- `_update_after_schedule` 在调度后推进计算位置与 in-flight 计数；`update_from_output` 在执行结果返回后结算 in-flight、修正被拒绝的 draft token，并决定请求是否完成以及何时释放资源。
- “连续”来自跨 step 的动态重组，而不是把所有请求一次性拼成固定 batch。

### Request 状态迁移及其触发条件

1. 请求创建时默认处于 `WAITING`；带 structured output 的请求会先处于 `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR`。
2. `schedule` 只有在预算、并发请求上限和 KV slot 都允许时，才把可运行请求放入 `RUNNING` 并记录到 `SchedulerOutput`。
3. `RUNNING` 不保证请求在每个 step 都产生进展：异步依赖、pipeline cadence、encoder 预算或 KV 容量不足，都可能使其本轮没有新增 Token。
4. KV 空间不足时，已运行请求可能被 `_preempt_request`，状态回到可重新排队的路径；它不是“丢弃用户请求”。
5. 只有停止条件、长度上限、中止或请求级错误等使状态进入 `FINISHED_*`，该请求才进入最终输出和释放收尾。

该状态机明确区分“暂时不可调度”“因资源压力被抢占”和“请求已完成”。这种区分是正确处理取消、抢占和远端 KV 等高并发边界条件的基础。

## 5. EngineCore 的主循环

这张图要回答的问题：EngineCore 如何把输入消息、调度、执行和输出串成不断重复的 step？

~~~mermaid
---
config:
  theme: base
  look: classic
  themeVariables:
    background: "#FFFFFF"
    mainBkg: "#FFFFFF"
    primaryColor: "#EAF2FF"
    primaryBorderColor: "#2563EB"
    primaryTextColor: "#172033"
    secondaryColor: "#ECFDF5"
    tertiaryColor: "#FFF7ED"
    lineColor: "#475569"
    textColor: "#172033"
---
flowchart TD
    poll[EngineCoreProc._process_input_queue<br/>处理 ADD / ABORT / UTILITY] --> work{has_work?}
    work -->|否| poll
    work -->|是| stepfn{是否启用 batch queue?}
    stepfn -->|否| step[EngineCore.step]
    stepfn -->|是| pp[step_with_batch_queue<br/>调度新 batch 或回收最旧 future]
    pp --> reclaimed{本次回收了完成 batch?}
    reclaimed -->|否：只把新 batch 入队| enqueue
    reclaimed -->|是| abortq
    step --> has{scheduler.has_requests?}
    has -->|否| empty[返回空输出，model_executed=false]
    has -->|是| schedule[Scheduler.schedule]
    schedule --> exec[execute_model non_block=True]
    exec --> grammar[取得 grammar bitmask]
    grammar --> future[等待 future.result]
    future --> sample{model_output is None?}
    sample -->|是| samplecall[Executor.sample_tokens]
    sample -->|否| abortq
    samplecall --> abortq[处理执行期间到达的 abort queue]
    abortq --> update[Scheduler.update_from_output]
    update --> enqueue[EngineCoreOutputs 入输出队列]
    empty --> enqueue
    enqueue --> post[post_step<br/>可选 draft token bookkeeping]
    post --> waitkv{0-token 且仍有 scheduler work?}
    waitkv -->|是| yield[短暂让出 GIL，等待后台 KV transfer]
    waitkv -->|否| poll
    yield --> poll

    classDef decision fill:#FFF7ED,stroke:#C2410C,color:#172033
    classDef async fill:#ECFDF5,stroke:#059669,color:#172033
    class work,stepfn,reclaimed,has,sample,waitkv decision
    class exec,future,enqueue,yield async
~~~

`EngineCoreProc.run_busy_loop` 每轮先排空输入，再调用 `self.step_fn`。`EngineCore.__init__` 根据 `max_concurrent_batches` 把 `step_fn` 设为普通 `step` 或 `step_with_batch_queue`；后者是 pipeline parallel / async scheduling 能让多个 batch 同时 in-flight 的关键边界。

普通 `step` 进入 `Scheduler.schedule`，产生 `SchedulerOutput` 并交给 executor；结果结算后调用 `Scheduler.update_from_output`。`_process_engine_step` 先把 `EngineCoreOutputs` 入队，再调用 `post_step`。若 Scheduler 仍持有请求但本轮没有模型工作（例如等待 remote KV 或 delayed free），源码会短暂 `sleep(0.001)` 让后台 transfer 线程前进；0-token step 不是“没有任何状态工作”。

这里需要留意一个边界：EngineCore client 的 get_output_async 与 EngineCoreProc 的 output socket/channel 是消息传递关系；EngineCore.step 内部的 Scheduler、Executor 和 model runner 则是 engine 执行链中的普通调用关系。

### `EngineCore.step` 的执行顺序

按源码顺序展开 `step`，可以避免两个常见误解：它并非只包含一次模型调用，采样也并非总在 worker 外部完成。当前实现的主要路径如下：

1. `scheduler.has_requests()` 为假时直接返回空输出和 `False`，不会虚构 GPU 工作。
2. 调用 `scheduler.schedule(...)` 得到 `SchedulerOutput`，其中包含每请求 Token 数、new/cached request data、block ID 变化、encoder/spec 信息和完成/抢占集合；其 `total_num_scheduled_tokens` 也可能为 0。
3. 调用 `model_executor.execute_model(scheduler_output, non_block=True)`，取得 future；随后从 Scheduler 取得 grammar bitmask。
4. 等待 future 的结果。若 executor 返回 `None`，才调用 `model_executor.sample_tokens(grammar_output)`；这说明采样的归属会随执行路径变化，不能把某个 runner 的实现当作唯一形态。
5. 在处理模型输出前，先处理模型执行期间进入 abort queue 的请求，避免已中止请求继续以正常结果回写状态。
6. 调用 `scheduler.update_from_output(scheduler_output, model_output)`，把模型结果变成 `EngineCoreOutputs`，并更新请求进度、完成状态和资源释放。

`EngineCoreProc._process_engine_step` 随后把每个输出放入输出队列，并调用 `post_step(model_executed)`。后者不是另一个调度阶段：只有 `check_for_draft_tokens`、非 async scheduling 且本轮实际执行过模型三项同时成立时，它才从 executor 取回 draft token IDs 并更新 Scheduler。

### Future 与 abort queue 的作用

`execute_model(..., non_block=True)` 允许 executor 采用适合其后端的异步或跨 worker 执行方式，同时 engine 仍按 `SchedulerOutput` 的调度顺序处理对应的 `model_output`。用户取消可能在模型执行期间到达；源码在 `update_from_output` 前显式处理 abort queue，避免将已取消请求按正常模型结果继续推进。这是模型执行与请求生命周期并发时必须处理的竞态边界。

## 6. Scheduler 如何在吞吐与资源之间做决定

这张图要回答的问题：一次 schedule 调用如何在 running、waiting、token budget 和 KV capacity 之间做选择？

~~~mermaid
---
config:
  theme: base
  look: classic
  themeVariables:
    background: "#FFFFFF"
    mainBkg: "#FFFFFF"
    primaryColor: "#EAF2FF"
    primaryBorderColor: "#2563EB"
    primaryTextColor: "#172033"
    secondaryColor: "#ECFDF5"
    tertiaryColor: "#FFF7ED"
    lineColor: "#475569"
    textColor: "#172033"
---
flowchart TD
    start([Scheduler.schedule]) --> budget[初始化 token / encoder budget<br/>kv_cache_manager.new_step_starts]
    budget --> running[遍历 running]
    running --> eligible{本请求本步可推进?}
    eligible -->|否：cadence / throttle / 0 token| nextrun[跳过它，继续较后请求]
    nextrun --> running
    eligible -->|是| calc[计算并裁剪 num_new_tokens]
    calc --> alloc{allocate_slots 成功?}
    alloc -->|是| record[记录 scheduled_running<br/>扣减 budgets]
    record --> running
    alloc -->|否| victim[选择最低优先级/队尾 victim<br/>_preempt_request]
    victim --> self{victim 是当前请求?}
    self -->|否| alloc
    self -->|是| nowait[结束 running loop]
    running -->|遍历完成或预算耗尽| preempted{本步发生过抢占?}
    nowait --> preempted
    preempted -->|是| build[跳过 waiting admission]
    preempted -->|否| waiting[从 waiting / skipped_waiting 选队头]
    waiting --> promote{阻塞状态已可提升?}
    promote -->|否| skip[放入本步 skipped，继续考察]
    skip --> waiting
    promote -->|是或普通 WAITING| guards{stale output / LoRA / 并发限制通过?}
    guards -->|否，可跳过| skip
    guards -->|是| prefix[若 computed == 0<br/>查本地/远端 prefix]
    prefix --> fit{预算、encoder 与 KV 容量可满足?}
    fit -->|否| build
    fit -->|远端异步 load| remotewait[WAITING_FOR_REMOTE_KVS<br/>可分配但不 forward]
    remotewait --> waiting
    fit -->|是| admit[移入 RUNNING，记录 new/resumed]
    admit --> waiting
    waiting -->|无候选或预算耗尽| build[构造 SchedulerOutput]
    build --> advance[_update_after_schedule<br/>推进 computed / in_flight]
    advance --> done([返回 SchedulerOutput])

    classDef decision fill:#FFF7ED,stroke:#C2410C,color:#172033
    classDef state fill:#ECFDF5,stroke:#059669,color:#172033
    class eligible,alloc,self,preempted,promote,guards,fit decision
    class skip,remotewait,admit,advance state
~~~

Scheduler 在构造时保存 max_num_running_reqs、max_num_scheduled_tokens、max_model_len 和 cache 配置，并维护 requests、waiting、skipped_waiting、running 等状态。它还持有 KVCacheManager，必要时创建 KV connector 和 encoder cache manager。

当前源码中，调度大致呈现如下因果链：

1. 先尝试推进已有 running requests。
2. 根据本轮剩余 token budget 和 KV blocks 决定是否能继续。
3. `allocate_slots` 返回 `None` 时，从 running 中选 victim 抢占并重试；若 victim 就是当前请求，结束 running loop。
4. 只有本步没有抢占且未暂停时，才从 `waiting` / `skipped_waiting` 继续 admission。
5. 对 grammar、remote KV、streaming 等阻塞状态，保留在 skipped waiting 或对应状态。
6. 被接纳的请求变成 RUNNING，并写入 SchedulerOutput。

这解释了为什么 token budget 不是“每请求最大生成长度”：它是当前 engine step 能交给执行层的工作量预算。

### Running 请求的优先处理顺序

源码先遍历 `running`，只有在**本轮没有发生抢占**且 engine 未暂停时，才继续接纳 `waiting`。“优先推进已运行请求可以减少其等待、但可能推迟新请求 admission”是由遍历顺序得到的设计推论；具体尾延迟效果仍需压测验证。KV 不足时的抢占则是明确的资源恢复路径。

对每个 running 请求，Scheduler 先计算待处理 Token 数，再按当前 `token_budget` 和 `max_model_len` 裁剪，最后调用 `kv_cache_manager.allocate_slots`。返回 `None` 不表示模型执行失败，而是表示当前资源不足以为本轮新增 Token 分配所需 slots。此时源码会从 `running` 中选择一个请求调用 `_preempt_request`：优先级策略按 `(priority, arrival_time)` 选择最低优先级候选，其他策略选择运行集合末尾的候选。若当前目标请求自身成为 victim，说明继续抢占也无法满足本轮分配条件，该请求不会在本轮执行。

### Waiting 请求的接纳条件

进入 waiting 阶段前，Scheduler 还会检查 `max_num_running_reqs`，并在 `waiting` 与 `skipped_waiting` 之间选择待考察队列。以下情况会导致请求被跳过而非直接完成：

- structured output grammar、远端 KV、流式输入等状态还不能提升为可运行；
- 上一次抢占留下的可交付 stale output 尚在飞行，立即恢复可能重采样同一个位置；
- 引入新的 LoRA 会超过 `max_loras`；
- 对 `token_budget`、encoder budget、KV slots 不满足的情况，源码通常 `break` 并停止本轮 admission，而不是一律放入 `skipped_waiting`。

首次接纳且 `num_computed_tokens == 0` 的请求会尝试 prefix-cache lookup；命中的历史 Token 会减少本轮待计算量和需要新增的 block 数。对于 waiting/preempted admission，只有本 step 已存在 scheduled request 时，`allocate_slots` 才应用 watermark；异步 remote-KV load 还通过 `reserved_blocks` 为其他 in-flight prefill 保留空间。因此，`SchedulerOutput` 不只是 request ID 列表，而是执行层所需的完整本轮调度结果。

### 吞吐与延迟的交换在哪里发生

- 增大 `max_num_scheduled_tokens` 会提高单轮允许的工作量；是否提高吞吐、是否损害新请求延迟属于需基准验证的设计推论。
- 更积极地接纳 waiting 请求能提高并发，却会更快消耗 KV 空间，增加抢占概率。
- chunked prefill 把长 prompt 拆开，避免一个长请求独占 token budget；代价是它要跨更多 step 维护进度和 block。
- priority 策略改变的是资源压力下“谁先让出位置”，不是绕过 KV 容量限制。

这些都是基于当前 Scheduler 代码路径的推论；具体阈值、是否启用 prefill throttle、KV connector 与并行配置会改变实际行为。

## 7. KV Cache：逻辑映射、物理 block 和 GPU tensor

这张图要回答的问题：一个请求的 token 历史如何从 scheduler 里的逻辑对象走到 GPU 上实际读写的 KV tensor？

~~~mermaid
---
config:
  theme: base
  look: classic
  themeVariables:
    background: "#FFFFFF"
    mainBkg: "#FFFFFF"
    primaryColor: "#EAF2FF"
    primaryBorderColor: "#2563EB"
    primaryTextColor: "#172033"
    secondaryColor: "#ECFDF5"
    tertiaryColor: "#FFF7ED"
    lineColor: "#475569"
    textColor: "#172033"
    clusterBkg: "#F8FAFC"
    clusterBorder: "#94A3B8"
---
flowchart LR
    subgraph logical[请求的逻辑视图]
        positions[token positions<br/>0..num_tokens-1]
        hashes[完整前缀的 block_hashes]
    end

    subgraph scheduler[Scheduler / KV manager 的所有权视图]
        coord[KVCacheCoordinator]
        mapping[每个 KV group 的 req_to_blocks<br/>list of KVCacheBlock]
        pool[BlockPool<br/>统一拥有 KVCacheBlock 对象]
        block[KVCacheBlock.block_id<br/>物理池索引 + ref_cnt + hash]
    end

    subgraph worker[Worker 的执行视图]
        schedout[SchedulerOutput<br/>new/cached request block IDs]
        persistent[持久 BlockTables<br/>按请求索引保存 block_id]
        gathered[本 batch 的 input block tables]
        slots[slot mapping<br/>kernel block_id × kernel_block_size + offset]
        tensor[(GPU KV cache tensors)]
    end

    positions --> coord
    hashes -->|prefix lookup| coord
    coord --> mapping
    mapping --> block
    pool --- block
    block -->|get_block_ids| schedout
    schedout -->|append / staged writes| persistent
    persistent -->|gather_block_tables| gathered
    persistent -->|positions + query_start_loc| slots
    gathered -->|读历史 K/V| tensor
    slots -->|写本轮 K/V| tensor

    classDef logicalStyle fill:#EAF2FF,stroke:#2563EB,color:#172033
    classDef ownerStyle fill:#FFF7ED,stroke:#C2410C,color:#172033
    classDef gpuStyle fill:#ECFDF5,stroke:#059669,color:#172033
    class positions,hashes logicalStyle
    class coord,mapping,pool,block ownerStyle
    class schedout,persistent,gathered,slots,tensor gpuStyle
~~~

当前 V1 的 KV 管理至少有三层：

- `KVCacheManager` 通过 coordinator 协调一个或多个 KV cache group；每组 manager 的 `req_to_blocks` 保存 `list[KVCacheBlock]`。
- `BlockPool` 统一拥有这些 `KVCacheBlock` 对象；`block_id` 是物理池索引，对象还带 `ref_cnt` 与 cache hash 元数据。
- worker 的 `BlockTables` 接收 block IDs，维护持久表，再为当前 batch gather rows 并计算 slot mappings。

因此不存在图中常见但源码没有的独立“逻辑 block ID → 物理 block ID”对象转换。逻辑层是 token position 及它在请求 block 列表中的序号；`req_to_blocks` 中的对象已经指向物理 pool block；GPU KV tensor 才是 attention 实际读写的存储。

### 分页的本质：把“连续序列”拆成两次间接寻址

对请求来说，历史 token 是按位置递增的序列；manager 按所属 KV group 的 block size 把位置映射到 `req_to_blocks` 中的 `KVCacheBlock`。进入 worker 后，`append_block_ids` / `apply_staged_writes` 更新持久 block table，`gather_block_tables` 抽取本 batch 的行，`compute_slot_mappings` 再结合 positions、block table 与 kernel block size 计算写入 slots。

所以 attention 不要求每个请求独占一段按最大长度预留的连续 KV 区域：它按 block table 读取历史 K/V，并按 slot mapping 写本轮 K/V。源码事实是统一 block pool 与间接寻址；“减少碎片和过度预留”是由该布局得到的设计收益，实际幅度取决于请求长度分布和 block size。

### `allocate_slots` 的容量检查与分配阶段

`KVCacheManager.allocate_slots` 的源码把一次分配拆成明确阶段：

1. 根据已计算 token、刚命中的本地 prefix、可能由 connector 提供的外部 KV、本轮新 token 和 speculative lookahead，计算需要覆盖的范围。
2. 先按已处理 token 边界移除 attention 不再需要的 blocks；计算边界会扣除 `num_in_flight_tokens`，避免异步 step 仍在读取时过早释放。
3. 计算需要新增的 block 数。watermark 只用于 waiting/preempted 且本步已有 scheduled request 的 admission；`reserved_blocks` 由调用者为异步 remote-KV load 保留。
4. 空间不足返回 `None`；空间足够时，先把可复用的 computed blocks 接入请求映射，再为需要计算的 token 分配新 block。
5. 启用缓存时，只把已经最终确定的 token 对应 blocks 提交到缓存；源码会以 `request.num_tokens` 封顶，避免尚可能被拒绝的 speculative token 被错误地当成可复用前缀。

这个顺序解释了两件事：KV 分配失败是 Scheduler 的正常资源信号，不是异常；而“已被模型看到过”也不自动等于“已经可以安全成为 prefix cache”。

~~~mermaid
---
config:
  theme: base
  look: classic
  themeVariables:
    background: "#FFFFFF"
    mainBkg: "#FFFFFF"
    primaryColor: "#EAF2FF"
    primaryBorderColor: "#2563EB"
    primaryTextColor: "#172033"
    secondaryColor: "#ECFDF5"
    tertiaryColor: "#FFF7ED"
    lineColor: "#475569"
    textColor: "#172033"
---
flowchart LR
    scope[Model Runner V2 的具体 block-table 路径]
    sched[SchedulerOutput<br/>block IDs + num_scheduled_tokens]
    state[runner 更新持久请求状态]
    append[BlockTables.append_block_ids<br/>apply_staged_writes]
    gather[gather_block_tables<br/>抽取当前 batch 行并清零 padding]
    slots[compute_slot_mappings<br/>positions → 物理 slots]
    meta[attention metadata<br/>长度、block tables、slot mappings]
    backend[Attention backend forward]
    kv[(GPU KV cache tensors)]
    hidden[hidden states]

    scope --> sched --> state --> append
    append --> gather --> meta
    append --> slots --> meta
    meta --> backend
    backend -->|写新 K/V 到 slots| kv
    kv -->|按 block tables 读历史 K/V| backend
    backend --> hidden

    classDef control fill:#EAF2FF,stroke:#2563EB,color:#172033
    classDef address fill:#FFF7ED,stroke:#C2410C,color:#172033
    classDef gpu fill:#ECFDF5,stroke:#059669,color:#172033
    class scope,sched,state control
    class append,gather,slots,meta address
    class backend,kv,hidden gpu
~~~

图中展示 **Model Runner V2** 的具体 block-table 路径。Model Runner V1 在 `gpu_model_runner.py` / `gpu_input_batch.py` 中组织同类信息，但使用不同的对象和输入准备函数，因此 `gpu/block_table.py::BlockTables` 只用于解释 V2 路径。V2 路径分为：

- SchedulerOutput 提供本轮请求相关的 block IDs。
- BlockTables.gather_block_tables 组织 GPU 侧 block table。
- BlockTables.compute_slot_mappings 计算新 token 对应的物理 slot。
- attention metadata 将这些信息交给具体 backend。
- backend 负责把新 K/V 写入 slots，并依据 block table 读取历史 K/V。

在当前代码中，“PagedAttention”表示分页式 KV 管理与 attention 访问协议的整体机制，不对应某个唯一的 `PagedAttention` Python 类。

### 写入和读取在同一轮如何配合

`SchedulerOutput` 把调度阶段确认过的 block IDs 带到 worker。V2 runner 更新持久 `BlockTables`，再 gather 当前 batch rows、计算 slot mappings 和 attention metadata；新增 token 的 K/V 写入对应 slots，attention 又通过 block tables 访问历史 K/V。V1 runner 的类名和准备路径不同，但也必须向 attention backend 提供等价的长度与地址语义。

不同 attention backend、V1/V2 runner、CUDA graph 和 pipeline parallel 配置会改变具体 tensor 布局与输入准备路径。因此，本文只总结这些实现共同遵循的职责链：Scheduler → block IDs → BlockTables/metadata → KV tensor，不将某个 backend 的 kernel 细节推广为所有模型的固定实现。

## 8. Prefix caching 为什么能复用历史

这张图要回答的问题：已有 prompt 前缀何时被命中、保留、复用或淘汰？

~~~mermaid
---
config:
  theme: base
  look: classic
  themeVariables:
    background: "#FFFFFF"
    mainBkg: "#FFFFFF"
    primaryColor: "#EAF2FF"
    primaryBorderColor: "#2563EB"
    primaryTextColor: "#172033"
    secondaryColor: "#ECFDF5"
    tertiaryColor: "#FFF7ED"
    lineColor: "#475569"
    textColor: "#172033"
---
flowchart TD
    lookup[首次 admission<br/>get_computed_blocks] --> enabled{prefix cache read 启用?}
    enabled -->|否| miss[命中 0 token]
    enabled -->|是| cap[最大查找长度 = request.num_tokens - 1]
    cap --> longest[find_longest_cache_hit<br/>最长兼容 token 前缀]
    longest --> hit{存在命中?}
    hit -->|否| miss
    hit -->|是| touch[BlockPool.touch<br/>移出 free queue，ref_cnt++]
    touch --> partial{命中落在 manager block 内部?}
    partial -->|否| allocate
    partial -->|是，fine-grained manager| cow[安排 partial-tail copy-on-write<br/>后续写入私有 block]
    cow --> allocate[allocate_slots<br/>接入命中 blocks 并补足 slots]
    miss --> allocate
    allocate --> compute[runner 计算缺失 token]
    compute --> commit[cache_blocks / cache_full_blocks<br/>提交 finalized 的完整 token blocks]
    commit --> release[请求结束或抢占<br/>free_blocks 递减 ref_cnt]
    release --> refs{ref_cnt == 0?}
    refs -->|否| shared([仍由其他请求引用])
    refs -->|是且无 hash| uncached([free queue 前部<br/>优先分配])
    refs -->|是且有 hash| cached[保留 hash，进入 free queue 尾部<br/>可命中，也是淘汰候选]
    cached --> fate{先发生什么?}
    fate -->|新请求 hash 命中| retouch([touch：重新成为 in-use])
    fate -->|get_new_blocks 取走| evict([删除旧 hash<br/>物理 block 重新分配])

    classDef decision fill:#FFF7ED,stroke:#C2410C,color:#172033
    classDef cache fill:#ECFDF5,stroke:#059669,color:#172033
    class enabled,hit,partial,refs,fate decision
    class touch,commit,cached,retouch,shared cache
~~~

本地 prefix cache 的主要流程如下：

- `Request.update_block_hashes` 为新达到的 hash 边界补充 `block_hashes`，admission 时按这些 hashes 查最长兼容前缀。
- 普通 dense cache 通过 `cache_full_blocks` 提交完整 manager blocks；支持 fine-grained lookup 的 manager 还能报告落在 block 内部的精确 token hit。
- partial-tail hit 不能让新请求直接覆盖共享尾块，`SingleTypeKVCacheManager` 会预留私有 block 并安排 copy-on-write。
- 未命中部分获得新 slots 并执行；`allocate_slots` 只把不超过 `request.num_tokens` 的 finalized token 提交为 cache。
- 请求释放引用时只减少 `ref_cnt`；hash 索引与物理块是否仍可复用由 `BlockPool` 继续管理。

因此，“请求完成”与“KV 立即清空”不是同义词。cached block 的 `ref_cnt` 降到 0 后会进入 free queue，但仍保留 hash：此时它既能再次被 `touch` 命中，也是分配新 block 时的淘汰候选。`get_new_blocks` 真正取走它时，`_maybe_evict_cached_block` 才删除 cache hash。

partial block、copy-on-write、KV offload 和 remote KV transfer 都有额外条件。本文主线只解释本地 prefix cache；其他能力在最后的能力地图中定位。

### Prefix cache 的命中上限与尾部重算

`KVCacheManager.get_computed_blocks` 把最大查找长度限制到 `request.num_tokens - 1`：即使整段输入已有缓存，也必须留下最后一个 token 重新取得下一 token 的 logits。对普通 block-aligned lookup，这可能造成整块重算；对当前支持 fine-grained hash lookup 的 manager，最长命中可以落在 manager block 内部，但继续写共享尾块时必须走 CoW。两条路径不能压成“永远只命中完整 block”一句话。

这也说明 prefix cache 的收益与边界：共享系统提示词、相同文档前缀会减少真正需要前向计算的 token；但不同的最后 token、未满 block 的尾部、被显存压力淘汰的 block，都会让命中止步于某个边界。它不是任意字符串子串缓存，也不是把一个请求的输出文本永久保存。

### 跳过 prefix-cache lookup 的条件

`prefix_cache_lookup_enabled` 会在缓存未启用，或请求明确跳过读取时返回假。源码特别提到，某些需要 prompt logprobs 的请求以及特定 pooling 情形会跳过 prefix-cache read；这是为了保持输出语义正确，而不是“缓存越多越好”。因此评估 prefix caching 时，应同时观察命中率、实际少算 token 数和功能约束，不能只看 BlockPool 里是否还有缓存。

### 引用释放与缓存淘汰

请求结束时，`free_blocks` 递减引用。`ref_cnt == 0` 的 hashed block 会进入 free queue 尾部，同时仍留在 cache 索引；未带 hash 的 free block 被放到更早淘汰的位置。后续 hash hit 会通过 `touch` 将其重新标记为 in-use；物理块重新分配时，旧 hash 才会被删除。因此，引用计数归零表示请求不再占有该块，并不等同于缓存条目已经淘汰。

## 9. 从 SchedulerOutput 到模型前向和采样

这张图要回答的问题：调度结果如何进入输入准备、模型 forward、logits 和采样？

~~~mermaid
---
config:
  theme: base
  look: classic
  themeVariables:
    background: "#FFFFFF"
    mainBkg: "#FFFFFF"
    primaryColor: "#EAF2FF"
    primaryBorderColor: "#2563EB"
    primaryTextColor: "#172033"
    secondaryColor: "#ECFDF5"
    tertiaryColor: "#FFF7ED"
    lineColor: "#475569"
    textColor: "#172033"
---
flowchart TD
    config{VllmConfig.use_v2_model_runner}
    v1[Model Runner V1<br/>vllm/v1/worker/gpu_model_runner.py]
    v2[Model Runner V2<br/>vllm/v1/worker/gpu/model_runner.py]
    schedout[SchedulerOutput]
    state[finish / free / add / update<br/>持久请求与 block-table 状态]
    tokens{total_num_scheduled_tokens > 0?}
    noforward[EMPTY / kv_connector no_forward]
    prepare[准备 input IDs、positions、block tables<br/>slot mappings、attention metadata]
    mode{运行模式}
    eager[eager model forward]
    piece[piecewise compiled graph]
    full[full CUDA graph replay]
    pp{最后一个 PP rank?}
    intermediate[返回 IntermediateTensors]
    logits[hidden states → compute_logits]
    phase[execute_model 通常返回 None<br/>EngineCore 再调用 sample_tokens]
    grammar[应用 grammar bitmask]
    sampler[Sampler.forward<br/>logits processors / penalties / temperature<br/>top-k / top-p / random or greedy]
    result[ModelRunnerOutput<br/>sampled_token_ids + logprobs]

    config -->|false| v1
    config -->|true| v2
    schedout --> v1
    schedout --> v2
    v1 --> state
    v2 --> state
    state --> tokens
    tokens -->|否| noforward
    tokens -->|是| prepare --> mode
    mode -->|NONE| eager
    mode -->|PIECEWISE| piece
    mode -->|FULL| full
    eager --> pp
    piece --> pp
    full --> pp
    pp -->|否| intermediate
    pp -->|是| logits --> phase --> grammar --> sampler --> result

    classDef decision fill:#FFF7ED,stroke:#C2410C,color:#172033
    classDef resultStyle fill:#ECFDF5,stroke:#059669,color:#172033
    class config,tokens,mode,pp decision
    class schedout,phase,result resultStyle
~~~

GPU runner 的选择不是固定字符串。VllmConfig.use_v2_model_runner 会综合环境变量、模型架构、Triton 是否可用、并行配置、speculative 配置和 unsupported features 判断结果。

因此，当前文章把两套同名 `GPUModelRunner` 实现明确画成配置分支：

- Model Runner V1：`vllm/v1/worker/gpu_model_runner.py`。
- Model Runner V2：`vllm/v1/worker/gpu/model_runner.py`。

对选中的 runner，主流程仍然是：

1. 读取 `SchedulerOutput`，先 finish/free/add/update 持久请求状态。
2. `total_num_scheduled_tokens == 0` 时返回 empty/no-forward 结果；非空调度结果进入 inputs、block tables、slots 与 attention metadata 的准备阶段。
3. 根据 dispatch 进入 eager、piecewise graph 或 full CUDA graph，最后一个 PP rank 计算 logits；其他 PP rank 返回中间张量。
4. 常见生成路径的 `execute_model` 返回 `None`，EngineCore 随后调用 executor 的 `sample_tokens` 阶段。
5. `Sampler.forward` 应用 logits processors，再在 `sample` 中执行 penalties、temperature、top-k/top-p 与 random/greedy 选择。
6. sampled token IDs、logprobs 与 connector 等结果组成 `ModelRunnerOutput`。

两套 runner 都有 connector、并行、多模态、speculative 与 graph 等配置分支，但状态组织和阶段拆分不同；这些不应被误写成每个最小文本模型都会执行。

### `SchedulerOutput` 向 runner 提供哪些信息

`SchedulerOutput` 至少向 runner 提供：本轮参与的 request、每个请求调度的 Token 数、对应的 block IDs，以及可能存在的 encoder 输入和 speculative 信息。GPU runner 先更新持久 batch state，再据此准备 input IDs、positions、每请求长度、block tables、slot mappings 与 attention metadata。模型前向因此同时获得本轮计算范围、历史 K/V 地址和新增 K/V 写入位置。

`gpu_model_runner.py::execute_model` 还包含若干配置相关边界：本轮没有 scheduled tokens 时通常返回空 `ModelRunnerOutput`，KV connector 场景可能进入专门的 no-forward 路径；启用 CUDA graph 或 microbatch 时，runner 会为图捕获和 tensor shape 添加 padding；多模态、encoder-decoder、pipeline parallel 与 speculative decoding 会在输入准备或前向计算前后插入额外分支。图中只保留这些实现共有的输入准备、前向计算、logits 与采样阶段。

### 从 logits 到 sampled token

前向 logits 只是候选分布。`Sampler.forward` 先将 logits 转为 float32，应用 processors，再执行 penalties、temperature、top-k/top-p 与 random/greedy 选择；输出 token tensor 最终转为 `int32` 并扩成每请求一行。随后 token IDs 与可选 logprob 信息随 `ModelRunnerOutput` 回到 Scheduler 结算。

源码事实是多个请求共享一次 runner batch，同时保留每请求位置、KV 映射和采样 metadata；“由此提高 GPU 利用率”是设计推论，幅度需要按模型、batch 形状和 graph mode 测量。

## 10. 从 token 到流式文本，以及取消和错误

这张图要回答的问题：模型输出如何变成用户看到的增量文本，取消和错误如何收尾？

~~~mermaid
---
config:
  theme: base
  look: classic
  themeVariables:
    background: "#FFFFFF"
    mainBkg: "#FFFFFF"
    primaryColor: "#EAF2FF"
    primaryBorderColor: "#2563EB"
    primaryTextColor: "#172033"
    secondaryColor: "#ECFDF5"
    tertiaryColor: "#FFF7ED"
    lineColor: "#475569"
    textColor: "#172033"
---
sequenceDiagram
    autonumber
    participant Core as EngineCoreProc
    participant MP as AsyncMPClient
    participant Handler as AsyncLLM output_handler
    participant Processor as OutputProcessor
    participant Detok as IncrementalDetokenizer
    participant Q as RequestOutputCollector
    participant Gen as AsyncLLM.generate
    participant SSE as chat stream generator

    Core--)MP: 共享 IPC 返回 EngineCoreOutputs
    MP->>Handler: get_output_async
    loop 按 VLLM_V1_OUTPUT_PROC_CHUNK_SIZE 分块
        Handler->>Processor: process_outputs(slice)
        Processor->>Detok: update(new_token_ids, stop_terminated)
        Detok-->>Processor: 文本增量 / matched stop string
        Processor->>Q: put(RequestOutput)
    end
    Q-->>Gen: RequestOutput
    Gen-->>SSE: yield RequestOutput
    SSE-->>SSE: 编码为 SSE chunk

    alt EngineCore 已给 finish_reason
        Processor->>Processor: _finish_request 移除前端 RequestState
    else detokenizer 首先命中 stop string
        Processor->>Processor: 先生成 finish_reason=STOP 的最终输出并移除状态
        Processor-->>Handler: reqs_to_abort
        Handler-)MP: abort_requests_async
        MP-)Core: ABORT IPC，终止仍在 engine 的请求
    else 客户端断开 / generator 取消
        Gen->>Processor: abort_requests，移除状态并产生 ABORT 输出
        Gen-)MP: abort_requests_async
        MP-)Core: ABORT IPC
        Gen->>Q: finally close collector
    else output handler / engine 错误
        Handler->>Processor: propagate_error(exception)
        Processor->>Q: 为所有未完成请求放入异常
        Q-->>Gen: raise error
    end
~~~

OutputProcessor 位于 AsyncLLM 前端侧。后台 output handler 从 EngineCore client 获取 EngineCoreOutputs，调用 OutputProcessor.process_outputs，再通过每请求的 RequestOutputCollector 交给 API streaming generator。

RequestState 负责保存请求级输出状态。IncrementalDetokenizer 根据 token IDs 产生增量文本；check_stop_strings 处理依赖文本的停止条件。停止 token、最大长度、重复等完成原因由 RequestStatus 和输出处理逻辑共同表达，不能把所有停止条件都归于 detokenizer。

取消路径与正常结束不同：`AsyncLLM.generate` 捕获 `CancelledError` / `GeneratorExit` 后调用 `abort(..., internal=True)`；`AsyncLLM.abort` 先让 `OutputProcessor.abort_requests` 移除前端状态并构造 ABORT final output，再向 EngineCore 发送内部 IDs；`generate` 的 `finally` 最后关闭 collector。由于原 generator 正在取消，这个 ABORT output 不保证仍被客户端消费。

Engine 级异常则通过 error output 或 dead/error 状态传播，不能伪装成普通的 FINISHED_ABORTED。

### API 侧输出处理的职责

EngineCore 的职责是高频地调度和执行；它返回的是适合内部状态更新的 `EngineCoreOutputs`。`AsyncLLM._run_output_handler` 在 API 侧后台持续 `get_output_async()`，并按 `VLLM_V1_OUTPUT_PROC_CHUNK_SIZE` 切块调用 `OutputProcessor.process_outputs`，必要时在 chunk 之间让出事件循环。这样一次包含很多请求输出的 engine 回传不会长期阻塞其他协程。

`OutputProcessor` 再把 engine 级输出投影到每请求的 `RequestState`：追加 token、维护 logprobs/统计、调用 `IncrementalDetokenizer` 得到可安全发出的文本增量，并构造 `RequestOutput` 放入对应 `RequestOutputCollector`。collector 既是 `AsyncLLM.generate` 等待的对象，也是把多候选子请求重新聚合为外部输出的位置。

### 流式输出有两个独立的节流点

不要把“每轮 engine step”理解为“必然向客户端发送一个 token”。`RequestState.make_request_output` 会结合 `output_kind` 与 `stream_interval` 决定是否发出：

- `FINAL_ONLY` 在请求未完成时不产生用户可见输出；
- `stream_interval > 1` 时，首个 token、累计达到间隔、或请求完成才会产生输出；
- delta 模式只从上次发送偏移量之后截取 token，非 delta 模式则可携带全部已生成内容。

因此 GPU 的调度粒度、内部 `EngineCoreOutputs` 粒度和 HTTP/SSE 的可见粒度是三层不同的节奏。

### Stop string 的前端终止语义与 EngineCore 清理

EOS、最大长度等停止条件可以在 engine 侧确定；stop string 则依赖增量文本。当前 `process_outputs` 一旦命中，先把前端 `finish_reason` 改为 `STOP`、构造最终 `RequestOutput` 并移除 `RequestState`；若 `engine_core_output.finished` 仍为假，再把内部 ID 放进 `reqs_to_abort`，由后台 handler 通知 EngineCore。于是外部语义是 STOP，反向 ABORT 负责停止仍在 engine/in-flight 的工作和释放资源，两者不能合并成一个状态。

客户端取消也需要双侧处理，但顺序不同于 stop string：`OutputProcessor.abort_requests` 先生成 ABORT final output 并移除前端状态，EngineCore 再收到 abort，collector 最终在 `generate` 的 `finally` 关闭。只取消 HTTP generator 而不通知 engine，会让后台继续为无人消费的请求工作。

## 11. 高级能力在主流程中的位置

这张图要回答的问题：speculative decoding、structured output、LoRA、KV offload 和分布式执行分别插入主路径哪里？

~~~mermaid
---
config:
  theme: base
  look: classic
  themeVariables:
    background: "#FFFFFF"
    mainBkg: "#FFFFFF"
    primaryColor: "#EAF2FF"
    primaryBorderColor: "#2563EB"
    primaryTextColor: "#172033"
    secondaryColor: "#ECFDF5"
    tertiaryColor: "#FFF7ED"
    lineColor: "#475569"
    textColor: "#172033"
---
flowchart LR
    input[输入规范化]
    client[EngineCore client / IPC routing]
    request[Request 与状态]
    sched[SchedulerOutput]
    kv[KV / encoder cache]
    runner[GPU runner]
    sample[sampling]
    output[OutputProcessor]
    input --> client --> request --> sched --> kv --> runner --> sample --> output

    mm[多模态] -.-> input
    mm -.-> kv
    structured[structured output] -.-> request
    structured -.-> sample
    lora[LoRA] -.-> sched
    lora -.-> runner
    spec[speculative decoding] -.-> sched
    spec -.-> runner
    spec -.-> sample
    transfer[KV transfer / offload] -.-> sched
    transfer -.-> kv
    tp[Tensor Parallel] -.-> runner
    pp[Pipeline Parallel / batch queue] -.-> sched
    pp -.-> runner
    dp[Data Parallel / routing] -.-> client
    dp -.-> sched

    classDef main fill:#EAF2FF,stroke:#2563EB,color:#172033
    classDef capability fill:#ECFDF5,stroke:#059669,color:#172033
    class input,client,request,sched,kv,runner,sample,output main
    class mm,structured,lora,spec,transfer,tp,pp,dp capability
~~~

读图要点：

- speculative decoding 主要改变 proposal、验证和 sampled token 的生成路径。
- structured output 会影响 grammar、logits processor 和等待状态。
- LoRA 进入请求和 worker/model runner 的权重适配路径。
- multimodal 需要 encoder cache、输入处理和 GPU runner 的额外分支。
- KV transfer/offload 改变 KV block 的来源、释放时机或远端依赖。
- distributed execution 改变 executor、通信和 data/pipeline/tensor parallel 边界。
- V2 model runner 是执行层的实现变体，不是独立于 V1 engine 的另一套请求生命周期。

## 12. 可验证的整体机制模型

当你再看任意一段 V1 推理代码时，可以连续问五个问题：

1. **它在改变请求资格吗？** 看 `RequestStatus`、waiting/running/skipped 队列和 abort。
2. **它在改变请求的计算进度吗？** 看 `num_computed_tokens`、`num_tokens_with_spec` 和本轮 `num_scheduled_tokens`。
3. **它在改变 KV 所有权或地址吗？** 看 `req_to_blocks`、`BlockPool`、`allocate_slots`、block table 与 slot mapping。
4. **它在真正运行 GPU 工作吗？** 看 `SchedulerOutput` 如何进入 executor/runner、输入准备、attention metadata、forward 与 sampling。
5. **它在把内部结果变成外部语义吗？** 看 `update_from_output`、`OutputProcessor`、detokenizer、collector 与 abort。

从用户提交 prompt 开始，`InputProcessor` 生成规范化的 `EngineCoreRequest`；Scheduler 根据请求状态、计算进度和 KV 容量生成本轮 `SchedulerOutput`；runner 据此执行前向计算、写入新增 KV 并完成采样；Scheduler 再根据 `ModelRunnerOutput` 更新请求状态和进度；最后，`OutputProcessor` 将内部 Token 转换为客户端可见的增量文本。下一个 step 不继承固定 batch，而是根据更新后的三类核心状态重新调度。这构成了 vLLM V1 请求处理的核心循环。

### 学习结果自检

- [ ] 能解释为什么 `AsyncMPClient` 在 API 进程，以及 ADD 为什么不返回 per-request stream。
- [ ] 能根据 counters 计算一条请求本轮的候选 Token 数，并指出 in-flight 部分在哪里结算。
- [ ] 能解释一次抢占会释放什么、重置什么、保留什么，以及 stale output 为什么存在。
- [ ] 能逐分支说明 running、waiting、skipped waiting、0-token remote load 如何进入 `SchedulerOutput`。
- [ ] 能说明 `req_to_blocks` 为什么不是“逻辑 ID 到物理 ID”的第二张映射表。
- [ ] 能解释 prefix partial-tail hit 为什么需要 CoW，以及 hashed free block 为什么同时是 cache 与淘汰候选。
- [ ] 能分别在 V1/V2 runner 中找到 inputs、attention metadata、forward、logits、sample 的边界。
- [ ] 能区分 engine STOP、前端 stop string、client cancel 与 engine error 的外部语义和内部清理路径。

## 13. 分阶段源码学习路线

按下表逐站推进；“产出”不是写摘要，而是证明你能从输入状态推出输出状态。

| 站点 | 按顺序读的符号 | 必须产出 | 最便宜的验证 |
| --- | --- | --- | --- |
| 0. 进程边界 | route `create_chat_completion` → serving `_create_chat_completion` → `AsyncLLM` → `AsyncMPClient` | 一张提交/回传反向图，标出 API、Core、worker 边界 | 对照 `core_client.py` 的 input/output socket，而不是启动模型 |
| 1. 请求规范化 | `AsyncLLM.add_request` → `_add_request` → `Request.from_engine_core_request` | 写出 HTTP → `EngineInput` → `EngineCoreRequest` → `Request` 的字段变化 | `tests/v1/engine/test_async_llm.py` |
| 2. 核心状态 | `Request` properties → `Scheduler._update_after_schedule` → `update_from_output` | 手算短 prompt、chunked prefill、preempt 三类请求连续三个 step 的 counters | `tests/v1/core/test_async_scheduler.py` 与 scheduler tests |
| 3. EngineCore 循环 | `EngineCore.step` → `step_with_batch_queue` → `_process_engine_step` → `post_step` | 精确复述 execute、grammar、sample、abort、update、enqueue 的顺序 | 用源码分支解释 0-token step 与 batch queue，不需 GPU |
| 4. Scheduler | `schedule` 的 running loop → waiting loop → `_preempt_request` → `_update_after_schedule` | 每次 schedule 前后列出 queues、budget、scheduled tokens、victim | `tests/v1/core/test_scheduler.py`、`test_priority_scheduler_random.py` |
| 5. KV 与 prefix | `get_computed_blocks` → `allocate_slots` → per-group manager → `BlockPool` → V2 `BlockTables` / V1 input batch | 画两个请求共享前缀时的对象引用、ref count、free queue 与 worker slots | `test_prefix_caching.py`、`prefix_cache/test_partial_prefix_cache_hits.py`、`test_gpu_block_table.py` |
| 6. runner | `use_v2_model_runner` → 选中的 `GPUModelRunner.execute_model` / `sample_tokens` → `Sampler.forward` | 从一个 `SchedulerOutput` 字段追到对应 tensor/state，再到 sampled ID | 先做静态追踪；只有环境完备时再跑 worker tests |
| 7. 输出与收尾 | `_run_output_handler` → `process_outputs` → detokenizer → `generate` / `abort` | 分开画 normal、stop string、client cancel、engine error 四条路径 | `tests/v1/engine/test_output_processor.py`、`test_async_llm.py` |
| 8. 高级能力 | 从第 11 节能力图任选一条虚线反向追踪 | 说明它影响哪类核心状态、插入主流程的哪个阶段 | 只读该能力对应 tests；不要同时展开所有分支 |

每个阶段结束时回答四个问题：模块的输入和输出是什么？它读取哪些状态？它直接修改哪些状态？发生失败、取消、异步执行或部分成功时，哪些不变量仍然成立？如果无法回答，应回到对应符号和测试继续核对，再进入更底层的 kernel 实现。

## 14. 版本限制与可维护性

本文只对上述 commit 的源码负责。以下内容可能因配置或后端改变：

- AsyncMPClient、DP client 与其他 EngineCore client 的选择。
- V1/V2 GPU model runner。
- attention backend 和 Triton/CUDA kernel。
- prefix cache、KV connector、KV offload。
- data parallel、pipeline parallel、tensor parallel。
- speculative decoding、structured output、多模态和 LoRA。
- 不同模型架构的 KV cache group 和 attention layout。

更新本文时，若 `.codegraph/` 存在，先重新执行 CodeGraph 符号与调用链核对；Serena 可用于补充符号总览和引用查询。每张图的 `.mmd` 文件是唯一源文件，Markdown 中的代码块必须与它逐字同步。

## 验证记录

- CodeGraph：已用当前 `.codegraph/` 追踪 API、AsyncLLM、EngineCore、Scheduler、KV manager、runner 和输出处理的跨文件关系。
- Serena：已确认仓库存在 `.serena/project.yml` 与本地 CLI；本次结论不依赖其缓存，事实源仍是当前源码与 CodeGraph 返回的逐行源码。
- Mermaid CLI：`/Users/seer/.local/bin/mmdc`，版本 `11.16.0`；13 张 `.mmd` 均成功渲染。
- 同步检查：Markdown 内 13 个 Mermaid block 与对应 `.mmd` 逐字一致；20 个本地源码链接均存在。
- 视觉检查：已逐张检查 SVG 的白色画布、文字、节点、箭头、分区和裁切；状态图与 prefix-cache 图在首轮检查后重排并再次验证。仓库只保留 SVG 预览。
