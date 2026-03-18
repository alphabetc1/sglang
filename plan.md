# 问题背景

当前 `sglang` 已经具备完整的 speculative decoding 主链路，但还没有 `PEAGLE`。现有 `EAGLE3` 在 runtime 中仍然需要多次串行 draft forward，而 `P-EAGLE` 的核心收益来自把这部分改成单次 parallel draft forward。用户还要求不能照抄 `vLLM` 的实现，因此方案必须沿用 `sglang` 已有的 worker、verify、KV cache、scheduler 和 model loader 机制，做增量式扩展。

# 当前状态

1. `python/sglang/srt/speculative/spec_info.py` 负责算法枚举和 worker 路由，当前没有 `PEAGLE`。
2. `python/sglang/srt/server_args.py` 负责 speculative 参数约束、`NEXTN -> EAGLE` 映射、是否启用 overlap scheduler 等逻辑。
3. `python/sglang/srt/speculative/eagle_worker_v2.py` 已经实现：
   - draft -> build verify input -> target verify -> draft extend 的完整闭环；
   - `topk=1` 场景下的线性 verify；
   - draft extend 一次 forward 处理 `speculative_num_steps + 1` 个 token 的能力。
4. `python/sglang/srt/models/llama_eagle3.py` 仅支持单层 EAGLE3 draft model，不足以承载 `PEAGLE` 的多层 parallel-drafting drafter。
5. `python/sglang/srt/configs/model_config.py` 已经有“draft model 架构重写”入口，适合新增 `PEAGLE` 专用 model class 映射。

# 选定方案

选择“新增 `PEAGLE` 算法 + 复用 `EAGLE` verify 主链 + 新增 LLaMA 风格 `PEAGLE` draft model + 在 `EAGLEWorkerV2` 内新增 parallel draft 分支”。

选择原因：

1. 这条路径能最大化复用 `sglang` 已有 speculative 主干，避免复制或平行实现 verify / accept / KV cache 逻辑。
2. `PEAGLE` 的关键变化只发生在 draft model 与 draft 输入构造，不需要重做 target verify。
3. 当前 `SpecV2` 已经天然适配 `topk=1` 的线性链路，适合作为 `PEAGLE` 的首个支持路径。
4. 先支持 LLaMA 风格 `PEAGLE` checkpoint，能把实现范围控制在可落地的最小闭环；后续扩展到更多模型家族时，可以复用同一套 runtime 接口。

# 实现范围

初版 `PEAGLE` 的明确范围如下：

1. 新增 `--speculative-algorithm PEAGLE`。
2. 初版仅支持 `topk=1`。
3. 初版强制走 overlap worker，也就是 `EAGLEWorkerV2` 路径，不支持 legacy `EAGLEWorker`。
4. 初版新增 LLaMA 风格 `PEAGLE` draft model：
   - 通过 draft model config 中的 `ptd_token_id` 标记 parallel draft 占位 token；
   - 支持 learnable `mask_hidden`；
   - 支持多层 drafter。
5. verify / accept / output processing / scheduler 复用现有 `EAGLE` 线性链路。

# 计划改动

## 1. 算法枚举与参数约束

修改文件：

1. `python/sglang/srt/speculative/spec_info.py`
2. `python/sglang/srt/server_args.py`
3. `python/sglang/srt/managers/tokenizer_manager.py`
4. `python/sglang/srt/model_executor/model_runner.py`
5. 视需要补充少量 `is_eagle()` 相关分支文件

具体改动：

1. 在 `SpeculativeAlgorithm` 中新增 `PEAGLE`。
2. 让 `PEAGLE` 复用大多数 `is_eagle()` 路径，但保留 `is_peagle()` 单独分支，用于：
   - 参数校验；
   - worker 内切换 parallel draft；
   - target hidden-state capture 按 `EAGLE3` 规则处理。
3. 在 `server_args.py` 中增加 CLI choice 和约束：
   - `PEAGLE` 必须有 draft model；
   - `PEAGLE` 强制 `speculative_eagle_topk == 1`；
   - `speculative_num_draft_tokens` 自动调整为 `speculative_num_steps + 1`；
   - 自动启用 overlap scheduler，并给出日志说明。
4. 让 tokenizer reserved token 计算把 `PEAGLE` 视作 `EAGLE` 类算法。
5. 让 target model 在 `PEAGLE` 下也按 `EAGLE3` 方式读取 draft config，决定是否捕获 aux hidden states。

## 2. Draft model 接入

修改文件：

1. 新增 `python/sglang/srt/models/llama_peagle.py`
2. `python/sglang/srt/configs/model_config.py`

具体改动：

1. 新增 `LlamaForCausalLMPeagle`。
2. 模型设计采用“输入 token embedding + target hidden states 融合后进入多层 draft decoder”的结构，保留 `EAGLE3` 所需的 hidden-state 输入模式，但允许多层。
3. 模型中新增：
   - `mask_hidden`：用于并行占位 slot 的 learnable hidden state；
   - `ptd_token_id`：从 config 读取，用于 parallel draft placeholder token；
   - 可选的 `draft_vocab_size` / `target_hidden_size` 兼容逻辑。
4. 在 `ModelConfig._config_draft_model()` 中加入：
   - 当 `is_draft_model` 且 `speculative_algorithm == "PEAGLE"`，如果 draft checkpoint 的 architecture 是 `LlamaForCausalLM`，将其改写为 `LlamaForCausalLMPeagle`。

## 3. Parallel draft 运行时

修改文件：

1. `python/sglang/srt/speculative/eagle_worker_v2.py`
2. 视复杂度决定是否新增 `python/sglang/srt/speculative/peagle_utils.py`

具体改动：

1. 在 `EagleDraftWorker` 中增加 `PEAGLE` 分支，复用同一个 worker 类而不是新建平行 worker。
2. 新增并行 draft 输入准备逻辑：
   - 对每个请求，把已有 token 序列展开为“真实 token + 额外并行 slot”；
   - 额外 slot 的 `input_ids` 使用 `ptd_token_id`；
   - 额外 slot 的 `hidden_states` 使用 draft model 的 `mask_hidden`；
   - 位置编码按当前 seq_len 之后的连续位置展开。
3. 使用单次 draft forward 得到多个位置的 logits，并从这些位置直接采样出长度为 `K = speculative_num_steps + 1` 的线性 draft token 链。
4. 将这组线性 draft token 重组为现有 `EagleVerifyInput` 期望的格式：
   - 因为 `topk=1`，tree 退化为线性链；
   - 仍复用已有 verify、accept 和 KV cache 更新路径。
5. 对 decode 后的 next draft input，复用现有 `next_draft_input` 结构，但将其改为保存 `PEAGLE` 需要的单次 parallel draft 结果，而不是串行下一步种子。

## 4. 文档与测试

修改文件：

1. `docs/advanced_features/speculative_decoding.md`
2. 新增/修改 `test/registered/spec/eagle/` 下的 `PEAGLE` 测试
3. 如有必要，增加一个纯单元测试文件验证 parallel draft 输入展开逻辑

具体改动：

1. 文档中补充 `PEAGLE` 的启用方式、限制条件和模型要求。
2. 增加参数约束测试：
   - `PEAGLE + topk > 1` 必须报错；
   - 未提供 `ptd_token_id` 的 draft model 应报出清晰错误。
3. 增加运行逻辑测试：
   - parallel draft 输入展开后的 token / position / hidden-state 布局正确；
   - `PEAGLE` 只需一次 draft forward 即可得到 `K` 个候选 token。
4. 如环境允许，补一个基础集成测试；如真实 checkpoint 不可用，则用 mock/fake drafter 验证 runtime 主链。

# 边界条件与风险

1. `PEAGLE` 的公开收益依赖专门训练过的 draft checkpoint；如果用普通 `EAGLE3` checkpoint 强行跑 parallel draft，接受率很可能失真。因此代码里需要明确依赖 `ptd_token_id` 和 `mask_hidden`。
2. 初版只支持 `topk=1`，不做 tree branching 版本，避免把实现扩展成新的树搜索系统。
3. 当前网络和硬件环境不一定适合下载和运行真实的大型 `PEAGLE` checkpoint，因此性能验证可能只能做到：
   - 代码路径上减少 draft forward 次数；
   - 保留真实 benchmark 入口和运行方式；
   - 在可行测试中比较关键指标。

# 验证计划

1. 运行最相关的单元测试或新增测试，确保：
   - 算法枚举和 server args 正常；
   - draft model 路由正确；
   - parallel draft 输入构造正确。
2. 对受影响的 Python 文件至少做一次静态检查式验证，例如导入/编译级测试。
3. 如果能构造 lightweight fake drafter，则验证：
   - 单次 forward 产出 `speculative_num_steps + 1` 个 draft token；
   - verify 路径仍然接受并返回正确的 `next_draft_input`。
4. 若真实模型 benchmark 无法运行，在 review 中明确说明限制，并把“减少 draft forward 次数”作为最小性能证据保留下来。
