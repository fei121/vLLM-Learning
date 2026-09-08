# vLLM v0.26 V1 源码学习路线

> 目标：用最短路径掌握 vLLM 推理服务的核心机制，并为后续重写系统化学习博客提供源码大纲、验证问题和实践任务。

## 1. 学习范围

本文基于当前工作区源码：

- commit：`96add737ff022430b17034ce50dc55504c04094f`
- git describe：`v0.26.1rc0-247-g96add737f`
- 主线：V1 Engine、OpenAI-compatible 流式文本生成、decoder-only 模型、CUDA
- 执行层：同时认识传统 GPU runner 与 Model Runner V2，但不把 V2 runner 误认为独立的 V2 Engine

本文不是源码文件清单。每一阶段都围绕一个机制问题展开：先建立心智模型，再阅读主路径，最后用测试或小实验验证。

## 2. 学习完成后应具备的能力

完成这条路线后，应能不依赖架构图回答以下问题：

1. 一条 HTTP 请求如何变成 `EngineCoreRequest`，又如何以 SSE 增量返回？
2. 为什么 V1 Scheduler 没有固定的 prefill phase 和 decode phase？
3. `num_computed_tokens`、`num_tokens_with_spec` 和 `num_in_flight_tokens` 分别表示什么？
4. Scheduler 为什么选择这些请求和 token，而不是其他请求？
5. KV 空间不足为什么表现为 `allocate_slots()` 返回 `None`，随后可能触发抢占？
6. 请求的逻辑 token 历史如何映射到物理 block、GPU block table 和 slot mapping？
7. prefix cache 命中后为什么仍可能重算最后一个 token？
8. `SchedulerOutput` 如何进入 GPU runner，经过前向与采样变成 `ModelRunnerOutput`？
9. stop token、stop string、长度上限、客户端取消和 engine error 如何分别收尾？
10. Tensor Parallel、Pipeline Parallel、Data Parallel、speculative decoding 等能力插入主线的什么位置？

## 3. 贯穿全部源码的三个账本

学习时始终区分下面三类状态。很多看似复杂的分支，本质上是在维护其中一个账本，或让三个账本保持一致。

| 账本 | 回答的问题 | 主要源码对象 |
| --- | --- | --- |
| 请求资格 | 请求现在能否运行、被阻塞、被抢占还是已结束？ | `Request.status`、`waiting`、`skipped_waiting`、`running` |
| token 进度 | 模型已经计算到哪里，本轮还欠多少计算？ | `num_computed_tokens`、`num_tokens_with_spec`、`num_in_flight_tokens` |
| KV 所有权与地址 | 请求引用哪些 blocks，物理显存在哪里，本轮写入哪些 slots？ | `KVCacheManager`、`BlockPool`、`BlockTables`、slot mapping |

V1 的主循环可以压缩为：Scheduler 读取三个账本，生成本轮执行合同；GPU runner 执行合同；Scheduler 和 OutputProcessor 再把结果写回账本。

## 4. 推荐学习顺序

### 阶段 0：先建立仓库边界

#### 学习目标

知道 V1 Engine、Model Runner V1/V2、attention backend 和 kernel 分别处在哪一层，避免一开始进入 CUDA 细节。

#### 源码入口

- `vllm/entrypoints/openai/api_server.py`
- `vllm/v1/engine/async_llm.py`
- `vllm/v1/engine/core.py`
- `vllm/v1/core/sched/scheduler.py`
- `vllm/v1/worker/gpu_model_runner.py`
- `vllm/v1/worker/gpu/model_runner.py`
- `vllm/config/vllm.py`：`VllmConfig.use_v2_model_runner`

#### 必须理解

```text
V1 Engine
  ├── Scheduler、KV 管理、请求生命周期
  └── GPU 执行层
       ├── 传统 GPUModelRunner
       └── Model Runner V2
```

V2 runner 是 V1 Engine 内部的执行层实现，不是另一套独立请求生命周期。

#### 完成标准

能画出 API、EngineCore、Scheduler、Executor、Worker、OutputProcessor 的边界，并标出哪些连接是进程间消息，哪些是进程内调用。

---

### 阶段 1：追踪一次请求的端到端路径

#### 核心问题

用户调用 Chat Completions 后，第一个 token 如何返回？

#### 阅读顺序

1. `vllm/entrypoints/openai/chat_completion/api_router.py`
   - `create_chat_completion`
2. `vllm/entrypoints/openai/chat_completion/serving.py`
   - `OpenAIServingChat._create_chat_completion`
   - `chat_completion_stream_generator`
3. `vllm/v1/engine/async_llm.py`
   - `AsyncLLM.add_request`
   - `AsyncLLM._add_request`
   - `AsyncLLM.generate`
   - `AsyncLLM._run_output_handler`
4. `vllm/v1/engine/core_client.py`
   - `EngineCoreClient.make_async_mp_client`
   - `AsyncMPClient`
5. `vllm/v1/engine/core.py`
   - `EngineCoreProc.run_busy_loop`

#### 阅读重点

- 已渲染 `EngineInput` 与原始 prompt 为什么分别走同步和异步输入处理？
- 为什么 `OutputProcessor.add_request` 在 API 进程登记状态后，才向 EngineCore 提交请求？
- `SamplingParams.n > 1` 为什么会展开为多个内部 child requests？
- `AsyncLLM.generate` 为什么返回异步生成器，而不是一次性结果？
- output handler 为什么独立于提交请求的协程？

#### 实践任务

为一条请求记录下面的对象转换：

```text
HTTP request
→ EngineInput / PromptType
→ EngineCoreRequest
→ Request
→ SchedulerOutput
→ ModelRunnerOutput
→ EngineCoreOutputs
→ RequestOutput
→ SSE chunk
```

#### 完成标准

能够解释请求提交路径和输出回传路径为什么方向相反，以及客户端取消如何沿反方向通知 EngineCore。

---

### 阶段 2：掌握 Request 状态与 token 进度

#### 核心问题

Scheduler 如何知道一条请求还欠多少工作？

#### 源码入口

- `vllm/v1/request.py`
  - `Request.__init__`
  - `RequestStatus`
  - `num_tokens`
  - `num_tokens_with_spec`
  - `append_output_token_ids`
- `vllm/v1/core/sched/scheduler.py`
  - `Scheduler.add_request`
  - `_preempt_request`
  - `_free_request`

#### 必须掌握的量

- `num_prompt_tokens`：prompt 或 prompt embeddings 的逻辑长度。
- `num_tokens`：prompt 加已经确认的 output token。
- `num_tokens_with_spec`：再加 speculative tokens 后的目标长度。
- `num_computed_tokens`：调度器认为已经计算到的位置。
- `num_in_flight_tokens`：已调度但结果尚未完全结算的 token。
- `is_prefill_chunk`：当前是否作为未完成的 prefill chunk 调度。

#### 关键结论

当前 `Scheduler.schedule` 明确没有固定的 prefill/decode 两阶段。每一轮都让 `num_computed_tokens` 追赶 `num_tokens_with_spec`，再由 token budget、模型长度、encoder 预算和 KV 容量裁剪。

#### 实践任务

手工推演三类请求在连续三个 step 中的计数变化：

1. 短 prompt 正常 decode；
2. 长 prompt 被 chunked prefill 拆分；
3. decode 请求在 KV 压力下被抢占后重新进入 waiting。

#### 完成标准

看到一个 Request 时，不依赖“prefill/decode”标签，也能根据计数判断下一轮需要计算多少 token。

---

### 阶段 3：理解 EngineCore 的执行闭环

#### 核心问题

一次 engine step 精确做了什么？

#### 源码入口

- `vllm/v1/engine/core.py`
  - `EngineCore.step`
  - `EngineCore.post_step`
  - `EngineCoreProc._process_input_queue`
  - `EngineCoreProc._process_engine_step`
  - `EngineCoreProc.run_busy_loop`

#### 必须掌握的顺序

```text
Scheduler.schedule
→ Executor.execute_model(non_block=True)
→ 等待 future
→ 必要时 sample_tokens
→ 处理执行期间到达的 abort queue
→ Scheduler.update_from_output
→ EngineCoreOutputs 入队
→ post_step 的可选 draft-token 更新
```

#### 阅读重点

- 没有 scheduler work 时为什么返回空输出和 `False`？
- grammar bitmask 在执行和采样之间处于什么位置？
- 为什么必须在 `update_from_output` 前处理模型执行期间到达的 abort？
- `post_step` 为什么不负责 `update_from_output`？
- pipeline parallel 或 async scheduling 如何让多个 batch 同时处于 in-flight 状态？

#### 完成标准

能够按源码顺序解释 `EngineCore.step`，并区分调度、GPU 执行、采样、状态回写和 post-step bookkeeping。

---

### 阶段 4：深入 Scheduler

#### 核心问题

本轮为什么选择这些请求和 token？

#### 源码入口

- `vllm/v1/core/sched/scheduler.py`
  - `Scheduler.__init__`
  - `Scheduler.schedule`
  - `Scheduler.update_from_output`
  - `Scheduler._preempt_request`
- `vllm/v1/core/sched/output.py`
  - `SchedulerOutput`
- `tests/v1/core/test_scheduler.py`
- `tests/v1/core/test_priority_scheduler_random.py`

#### 阅读顺序

1. 先看 Scheduler 维护的 `requests`、`running`、`waiting`、`skipped_waiting`。
2. 跟踪 `token_budget` 如何初始化和消耗。
3. 阅读 running requests 的调度循环。
4. 跟踪 `allocate_slots()` 返回 `None` 后的抢占路径。
5. 阅读 waiting requests 的接纳条件和阻塞状态提升。
6. 最后看 `SchedulerOutput` 的构造和 `update_from_output` 的状态回写。

#### 必须掌握

- running 请求优先于 waiting 请求推进。
- 本轮发生抢占后，不会继续走普通 waiting admission。
- token budget 是本 step 的工作预算，不是单请求最大生成长度。
- KV 容量、max running requests、LoRA 上限、grammar、remote KV、streaming input 都可能影响接纳。
- `SchedulerOutput` 是执行合同，不只是 request ID 列表。

#### 实践任务

选择 `tests/v1/core/test_scheduler.py` 中一个测试，写出每次 `schedule()` 前后的：

- running/waiting 请求集合；
- token budget 消耗；
- 每请求 `num_scheduled_tokens`；
- 新分配 blocks；
- 抢占或完成请求。

#### 完成标准

能够解释吞吐、首 token 延迟、长请求尾延迟和抢占之间的取舍发生在源码的哪些分支。

---

### 阶段 5：掌握 KV Cache 与 PagedAttention

#### 核心问题

连续的 token 序列如何映射到不连续的 GPU KV blocks？

#### 源码入口

- `vllm/v1/core/kv_cache_manager.py`
  - `get_computed_blocks`
  - `allocate_slots`
  - `free`
- `vllm/v1/core/single_type_kv_cache_manager.py`
  - `SingleTypeKVCacheManager`
  - `req_to_blocks`
- `vllm/v1/core/block_pool.py`
  - `BlockPool`
  - `get_new_blocks`
  - `cache_full_blocks`
  - `touch`
  - `free_blocks`
- `vllm/v1/worker/gpu/block_table.py`
  - `BlockTables.gather_block_tables`
  - `BlockTables.compute_slot_mappings`
- `tests/v1/core/test_prefix_caching.py`
- `tests/v1/core/prefix_cache/test_partial_prefix_cache_hits.py`

#### 必须区分的四层对象

```text
请求 token 位置
→ 请求到逻辑 blocks 的映射
→ BlockPool 中的物理 block ID
→ Worker block table / slot mapping
→ GPU KV tensor
```

#### `allocate_slots` 阅读框架

1. 合并已计算、本地 cache hit、外部 KV、本轮新 token 和 lookahead。
2. 先释放 sliding window 等已不再需要的 blocks。
3. 计算需要分配的 block 数。
4. 加上 reserved blocks 和 admission watermark 检查容量。
5. 容量不足返回 `None`，让 Scheduler 决定是否抢占。
6. 接入 cache-hit blocks，再分配本轮新 blocks。
7. 只把已经最终确认的 token 提交为可复用缓存，避免 speculative token 污染 prefix cache。

#### Prefix caching 必须理解

- 只复用完整、已计算并可按 hash 匹配的 blocks。
- 全部 prompt 命中时仍限制到 `request.num_tokens - 1`，最后 token 需要重新计算 logits。
- 请求释放引用和缓存 block 被淘汰不是同一件事。
- prompt logprobs 或特定 pooling 语义可能主动跳过 prefix-cache read。

#### 实践任务

画出两个请求共享前三个 blocks、尾部各自不同的映射，并分别推演：

- 第二个请求首次 admission；
- 第一个请求完成；
- 显存压力触发缓存淘汰；
- 第二个请求继续 decode。

#### 完成标准

能够清楚解释“PagedAttention 降低碎片”背后的实际数据结构，而不是只复述宣传语。

---

### 阶段 6：理解 GPU Runner、前向与采样

#### 核心问题

`SchedulerOutput` 如何变成真正的 GPU 工作？

#### 源码入口

- 传统 runner：`vllm/v1/worker/gpu_model_runner.py`
  - `execute_model`
  - `_prepare_inputs`
  - `_get_slot_mappings`
  - `_build_attention_metadata`
- Model Runner V2：`vllm/v1/worker/gpu/model_runner.py`
  - `execute_model`
  - `prepare_inputs`
  - `prepare_attn`
  - `sample_tokens`
- `vllm/v1/sample/sampler.py`
  - `Sampler.forward`
- `vllm/config/vllm.py`
  - `VllmConfig.use_v2_model_runner`

#### 共同执行主线

```text
更新持久请求/batch 状态
→ 准备 input IDs、positions 和长度
→ 准备 block table、slot mapping、attention metadata
→ 选择 eager / piecewise graph / full CUDA graph
→ model forward
→ logits
→ sampling
→ ModelRunnerOutput
```

#### 对比 runner 时关注什么

- 不比较文件大小，比较状态所有权和阶段边界。
- 传统 runner 的输入、attention 和 preprocess 逻辑更集中。
- V2 runner 显式拆出 `RequestState`、`InputBuffers`、`ModelState` 和 `BlockTables`，并分阶段 finish/free/add/update requests。
- 两者都属于 V1 Engine，共享 Scheduler 和 KV 语义。
- V2 runner 是否启用由配置、模型架构、Triton 和功能支持情况共同决定，不能假定始终使用 V2。

#### Sampling 必须掌握

- logits 还不是 token。
- logits processors、temperature、penalty、top-k/top-p 或 greedy sampling 在采样阶段生效。
- 每请求可以共享同一次 GPU 前向，但仍保留独立采样参数和状态。

#### 实践任务

从 `SchedulerOutput.num_scheduled_tokens` 开始，分别在两个 runner 中追踪到 `ModelRunnerOutput.sampled_token_ids`，记录每一步新建或更新的 tensor/状态对象。

#### 完成标准

能够解释为什么 runner 不是简单的 `model(input_ids)`，以及 block table、attention metadata、CUDA Graph 和 sampling 分别解决什么问题。

---

### 阶段 7：理解输出、停止、取消和错误

#### 核心问题

内部 token 如何变成用户可见文本，并保证各种结束路径正确？

#### 源码入口

- `vllm/v1/engine/output_processor.py`
  - `OutputProcessor.process_outputs`
  - `RequestState.make_request_output`
  - `_finish_request`
  - `RequestOutputCollector`
- `vllm/v1/engine/detokenizer.py`
  - `IncrementalDetokenizer`
  - `check_stop_strings`
- `vllm/v1/engine/async_llm.py`
  - `_run_output_handler`
  - `generate`
  - `abort`
- `tests/v1/engine/test_output_processor.py`
- `tests/v1/engine/test_async_llm.py`

#### 必须区分

- stop token：可在 token/engine 状态层判断。
- stop string：依赖 detokenized text，可能由 API 侧发现并反向 abort engine。
- max tokens：进入长度上限完成状态。
- client cancellation：先关闭前端 collector，再通知 EngineCore 中止内部请求。
- engine error：错误传播，不能伪装成普通 abort。

#### 阅读重点

- output handler 为什么按 chunk 处理 `EngineCoreOutputs` 并让出事件循环？
- `FINAL_ONLY`、delta 输出和 `stream_interval` 如何改变用户可见频率？
- 为什么一次 engine step 不等于一次 SSE 输出？
- `SamplingParams.n > 1` 的内部 child outputs 如何重新聚合？

#### 完成标准

能够画出正常完成、stop string、客户端取消和 engine error 四条不同收尾路径。

---

### 阶段 8：最后再进入并行和高级能力

先掌握主线，再按需求选择分支：

| 能力 | 插入位置 | 首要问题 |
| --- | --- | --- |
| Tensor Parallel | 模型层和 collective | 单层权重与计算如何切分？ |
| Pipeline Parallel | EngineCore batch queue、runner 中间张量 | 多个 batch 如何填充 pipeline 并保持顺序？ |
| Data Parallel | EngineCore client、DP coordinator、负载均衡 | 请求如何路由到不同 DP rank？ |
| Speculative decoding | Scheduler lookahead、runner proposal/verification、sampling | draft token 如何验证，拒绝后状态如何回滚？ |
| Structured output | Request 等待状态、grammar bitmask、logits processor | grammar 何时就绪，如何限制 token？ |
| LoRA | 请求元数据、Scheduler 限额、runner adapter 激活 | 同一 batch 如何选择不同 adapter？ |
| Multimodal | InputProcessor、encoder cache、runner embeddings | encoder 输出如何进入 decoder attention？ |
| KV transfer/offload | Scheduler KV connector、worker connector、释放时机 | 本地与远端 KV 的所有权如何协调？ |

这一阶段不要试图一次读完。每次选择一项能力，先标出它改变三个账本中的哪一个，再追踪它插入主线的边界。

## 5. 不推荐的学习方式

- 一开始进入 attention CUDA kernel：会看见地址和张量，但不知道它们为何这样组织。
- 只读 README 或旧架构图：V1 调度器、runner 选择和 KV 分支变化较快，必须回到当前源码。
- 把 prefill 和 decode 视为两个永久固定的 Scheduler 阶段。
- 把 Request 的逻辑 blocks、BlockPool 物理 blocks 和 GPU KV tensor 当成同一个对象。
- 把 Model Runner V2 当作 V2 Engine。
- 只看正常生成路径，不看抢占、取消、stop string、错误和异步 in-flight 状态。

## 6. 建议的源码验证方法

每学习一个阶段，完成以下四步：

1. 用 CodeGraph 追踪入口符号、关键调用链和动态 dispatch 边界。
2. 阅读对应测试，用断言验证自己对状态变化的理解。
3. 手工记录一次请求在该模块进入前和离开后的状态差异。
4. 用一句因果链总结：为什么需要该模块、它改变了什么、收益和代价是什么。

推荐优先阅读的测试：

- `tests/v1/core/test_scheduler.py`
- `tests/v1/core/test_prefix_caching.py`
- `tests/v1/core/prefix_cache/test_partial_prefix_cache_hits.py`
- `tests/v1/engine/test_output_processor.py`
- `tests/v1/engine/test_async_llm.py`
- `tests/v1/core/test_async_scheduler.py`

## 7. 用这份路线重写博客的建议结构

后续博客可以直接沿用下面的大纲：

1. vLLM 要解决的资源利用问题。
2. 三个账本：请求资格、token 进度、KV 所有权。
3. 一次请求的端到端旅程与进程边界。
4. V1 如何统一描述 prefill、decode 和 chunked prefill。
5. Scheduler 如何用 token budget、KV 容量和状态作决定。
6. KV Cache、BlockPool、BlockTables 与 PagedAttention。
7. Prefix caching 的命中、对齐、留存与淘汰。
8. SchedulerOutput 如何进入 GPU runner、前向与采样。
9. 输出、detokenization、停止、取消和错误。
10. Model Runner V1/V2 的执行层差异。
11. 并行、speculative decoding、LoRA、多模态和 KV transfer 的能力地图。
12. 从请求到 token 的最终心智模型与源码阅读路线。

每章保持同一种节奏：

```text
具体问题
→ 机制图
→ 源码事实
→ 关键状态或数据结构
→ 性能收益从哪里产生
→ 代价和配置边界
→ 建议继续阅读的符号与测试
```

## 8. 最终掌握检查表

- [ ] 能解释 API 进程和 EngineCore 进程之间的请求/输出双向通道。
- [ ] 能根据 Request 的 token 计数推导本轮欠账，而不是只说 prefill/decode。
- [ ] 能逐分支解释 `Scheduler.schedule` 的 running、waiting、抢占和阻塞处理。
- [ ] 能区分逻辑 block、物理 block、block table、slot mapping 和 KV tensor。
- [ ] 能解释 `allocate_slots()` 返回 `None` 的含义及后续抢占路径。
- [ ] 能解释 prefix cache 为什么只命中完整 blocks，并重算最后 token。
- [ ] 能从 `SchedulerOutput` 追踪到模型前向、sampling 和 `ModelRunnerOutput`。
- [ ] 能说明传统 runner 与 Model Runner V2 的边界和选择逻辑。
- [ ] 能区分 stop token、stop string、长度上限、取消和 engine error。
- [ ] 能把任一高级能力放回三个账本和端到端主线中定位。

如果这些问题都能脱离文档回答，就已经掌握了 vLLM V1 最核心的机制逻辑，可以开始按主题深入 kernel、并行后端或特定模型实现。
