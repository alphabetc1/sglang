# 任务目标

在当前 `sglang` 仓库中实现 `PEAGLE` 推测解码能力，并在现有 speculative decoding 框架内获得可验证的性能提升。实现需要基于 `sglang` 当前架构、调度方式、worker 生命周期、图捕获和 speculative 相关代码完成，不能照抄 `vLLM` 中的实现细节；可以参考其公开设计思想与算法目标，但最终代码结构、接口设计和执行路径必须符合 `sglang` 的已有实现风格。

# 背景

当前仓库已经具备较完整的 speculative decoding 基础设施，包括 `EAGLE`、`EAGLE3`、`NGRAM`、`NEXTN/MTP` 等相关实现、模型适配、kernel 与测试。用户需求是在这个基础上增加 `PEAGLE` 支持，而不是引入一套平行且割裂的新框架。

`PEAGLE` 的目标是改进草稿生成/验证流程，提高接受长度、降低 decode latency 或提升吞吐，从而在实际服务场景中优于现有非 `PEAGLE` 路径。由于用户明确要求“不要照抄 vllm”，实现时需要优先复用 `sglang` 已有抽象，尽量以增量方式扩展已有 speculative 代码路径。

# 明确要求

1. 在 `sglang` 当前代码库中实现 `PEAGLE` 所需的核心运行时逻辑。
2. 设计必须贴合 `sglang` 当前 speculative 架构，优先复用已有的：
   - speculative algorithm 枚举和路由
   - worker / model runner / draft runner / graph runner 抽象
   - batch metadata、tree 构建、accept/reject、KV cache、采样等流程
   - 现有 `EAGLE` / `EAGLE3` 测试与 benchmark 组织方式
3. 不直接复制 `vllm` 的实现代码；如果参考其思路，需要转化为 `sglang` 风格的实现。
4. 补齐必要的配置、文档、测试和可执行验证路径。
5. 给出性能验证方法，并尽可能在当前环境中运行最相关的验证，证明该实现相对基线存在性能收益；若受限于环境无法完整复现，至少要提供可复现脚本、指标和结果说明。

# 约束

1. 代码注释、标识符、接口命名遵循仓库现有风格，代码本身使用英文；阶段文档使用中文。
2. 不破坏现有 speculative decoding 算法，尤其不能引入对 `EAGLE` / `EAGLE3` / `NGRAM` / `NEXTN` 的行为回归。
3. 需要考虑 CUDA Graph、draft extend、verify、调度元数据、TP/DP 以及现有模型适配层可能受到的影响。
4. 若方案依赖特定模型权重或导出产物，需要明确最小接入方式，并避免把需求建立在当前仓库不存在的专用资产之上。
5. 如果实现过程中发现原始需求与仓库现状存在结构性冲突，需要在后续计划阶段明确指出，而不是隐式偏离目标。

# 预期交付

1. `PEAGLE` 运行时实现代码。
2. 必要的接口/配置接入，使其可以通过现有服务启动参数或内部配置启用。
3. 覆盖关键路径的测试，至少包括：
   - 算法路由/配置接入
   - 关键运行逻辑正确性
   - 与现有 speculative 基础设施的集成检查
4. 更新后的文档或说明，解释如何启用和验证 `PEAGLE`。
5. 性能验证结果或可复现的性能验证方案。

# 验证期望

1. 运行与改动最相关的单元测试、集成测试或已有 speculative 测试。
2. 若仓库已有可复用 benchmark 脚本，优先复用并补充 `PEAGLE` 配置。
3. 对性能结论给出明确对比对象，例如：
   - 同模型下的普通 decode
   - 同模型下的现有 `EAGLE` / `EAGLE3` 路径
4. 至少说明使用的指标，例如 tokens/s、accept length、decode latency、端到端 latency。

# 重要假设

1. 当前仓库中尚未有 `PEAGLE` 实现，需要从现有 speculative 框架扩展。
2. 当前环境可能无法直接获取完整的线上模型与大规模 benchmark 资源，因此实现应优先保证架构正确、测试可运行、性能验证路径清晰。
3. 最优实现方向大概率是以当前 `EAGLE`/`EAGLE3` 代码路径为基础演进，而不是新造独立 speculative 子系统；是否最终选择该方向，需要在 research 阶段结合仓库现状确认。
