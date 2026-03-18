# 当前状态

1. 仓库已经有完整的 speculative decoding 主干：
   - 算法枚举在 `python/sglang/srt/speculative/spec_info.py`，当前包含 `EAGLE`、`EAGLE3`、`STANDALONE`、`NGRAM`，以及被映射为 `EAGLE` 的 `NEXTN`。
   - 主 worker 路径已经拆分为 target verify、draft、draft-extend、CUDA graph、KV cache 分配与回收、tree verify 等模块。
   - `EAGLEWorkerV2` 是当前更重要的实现路径；`SpecV2`/overlap scheduler 已经限定 `topk=1`，这与 `PEAGLE` 的线性并行 draft 形态高度匹配。
2. 仓库已有两类与 `PEAGLE` 最接近的基础设施：
   - `python/sglang/srt/speculative/eagle_worker_v2.py`：现有 `EAGLE/EAGLE3` 单 draft-runner 路径，当前仍是串行多步 draft。
   - `python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py`：已有“一次调度中驱动多个 draft runner”的骨架，证明仓库已经支持多步 speculative 的并行化调度思路，但它服务的是 MTP/多 runner 形态，不是 `PEAGLE` 的单模型并行 draft。
3. 模型侧现状：
   - `python/sglang/srt/models/llama_eagle3.py` 是现有 EAGLE3 draft model，明确限制 `num_hidden_layers == 1`，说明当前仓库还没有 `PEAGLE` 这种多层 parallel-drafting draft model。
   - `python/sglang/srt/configs/model_config.py` 已经支持“当作为 draft model 加载时，把目标模型架构改写成 speculative 专用实现”，适合新增 `PEAGLE` draft model 路由。

# 问题背景

1. P-EAGLE 论文（arXiv:2602.01469，提交日期为 2026-02-01）说明：它把 EAGLE 从 autoregressive drafting 改成 parallel drafting，通过一次 forward 直接生成多个 draft token。
2. Hugging Face 上的公开模型卡进一步给出两个对实现很关键的信息：
   - `P-EAGLE` 生成 `K` 个 draft token 只需要一次 forward。
   - 它沿用了 EAGLE3 的隐藏状态输入方式；例如公开 GPT-OSS 模型卡明确写到“follows the vanilla EAGLE3 using three layers of hidden states from the target model”。
3. vLLM 文档暴露出的运行时要点也和论文一致：
   - parallel drafting 需要额外的 padded draft slots；
   - draft model config 里需要 `ptd_token_id` 或 `pard_token` 之类的占位 token；
   - 对 EAGLE 类模型，还需要一个 learnable masked hidden state，用来填充并行 draft 的隐藏状态槽位。

# 需求拆解

1. 不能把 `PEAGLE` 做成 `EAGLE3` 的简单别名，因为那样不会减少 draft 侧 forward 次数，也无法兑现性能提升。
2. 不能直接复用 `multi_layer_eagle_worker_v2.py` 作为最终实现，因为它依赖多个 draft runner 或逐步传播隐藏状态，仍不是“单次 forward 产生 K 个 token”的 `PEAGLE`。
3. 真正需要补齐的是三部分能力：
   - 新的 speculative 算法入口与约束；
   - 新的 parallel-drafting draft model；
   - 新的 draft 输入构造与采样路径，使一次 forward 的多个输出位置能直接变成 verify 输入。

# 推荐实现方案

1. 第一优先级：新增 `PEAGLE` 算法，作为 `EAGLE3` 家族下的并行 draft 变体接入。
   - 运行时复用现有 `EAGLE` verify / accept / KV cache / sampling / scheduler 主链路。
   - draft 阶段改为“padded linear draft”，一次 forward 产生长度为 `K` 的候选链。
   - 初版显式限制 `topk=1`，优先贴合论文与公开模型，也能直接适配当前 `SpecV2` 的限制。
2. 第二优先级：新增 `PEAGLE` draft model 实现，而不是扩展现有 `llama_eagle3.py` 的单层假设。
   - 推荐先以 LLaMA 风格 draft model 为起点实现 `LlamaForCausalLMPeagle`。
   - 设计上沿用 EAGLE3 的 hidden-state 输入方式，但支持多层、masked hidden state 和并行 draft slot。
   - 后续如需扩展到更多 draft 架构，再在 `model_config.py` 中按架构追加映射。
3. 第三优先级：在 worker 中新增 parallel draft 输入展开逻辑，而不是改写 verify 逻辑。
   - verify 阶段继续走现有 `EagleVerifyInput` / `EagleDraftInput` 和 `topk=1` 的线性链路。
   - draft 阶段新增“把每个请求展开为真实 token + 并行占位 token”的输入准备逻辑，并从多个输出位置收集 K 个 token。

# 备选方案

1. 方案 B：把 `PEAGLE` 挂到 `multi_layer_eagle_worker_v2.py`，通过多个 runner 并行执行。
   - 优点：复用更多现有代码。
   - 缺点：语义上更接近 MTP，不是单模型 parallel drafting；性能收益也不具备 `PEAGLE` 的关键特征。
   - 结论：不推荐作为主方案。
2. 方案 C：仅新增参数或别名，让现有 `EAGLE3` 在配置层暴露 `PEAGLE` 名称。
   - 优点：改动最小。
   - 缺点：没有真实新能力，无法满足“实现 `PEAGLE` 并达到性能提升”。
   - 结论：不可接受。

# 测试策略

1. 第一优先级：纯逻辑/单元测试。
   - 验证 `SpeculativeAlgorithm`、`server_args`、`model_config` 的新分支与约束。
   - 验证 parallel draft 输入展开函数的 token/position/slot 布局。
   - 验证 `PEAGLE` 强制 `topk=1` 等边界条件。
2. 第二优先级：运行时集成测试。
   - 参考现有 `test/registered/spec/eagle/` 结构，为 `PEAGLE` 增加至少一个基础集成测试。
   - 如果当前环境拿不到真实 `PEAGLE` checkpoint，则先通过 mock / 轻量 fake model 验证“一次 draft forward 产出 K token”的运行链路。
3. 第三优先级：性能验证。
   - 最理想：对真实 `PEAGLE` checkpoint，复用现有 benchmark 或 server_info 指标比较 `avg_spec_accept_length`、tokens/s、decode latency。
   - 若当前环境不适合下载和跑大模型，至少补一个可复现 benchmark 入口，并用 draft forward 次数减少作为最小性能代理指标。
