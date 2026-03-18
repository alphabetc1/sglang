# Review 范围

本轮 review 针对 `stage4` 引入的 `PEAGLE` 支持做了端到端复查，重点检查以下链路：

1. `PEAGLE` 的算法枚举、参数约束和 overlap worker 路由是否完整。
2. `LlamaForCausalLMPeagle` 的 draft model 装载入口是否能被现有 model loader 正确识别。
3. `EagleDraftWorker` 的 parallel draft 路径是否能复用现有 verify/accept 主链。
4. `PEAGLE` 在 scheduler 的 batch filter / merge 路径上是否与现有 `EagleDraftInput` 数据结构兼容。

# 检查与验证

已完成的检查：

1. 复查 `PEAGLE` 相关实现文件，确认 draft 侧已经从多次串行 forward 改为单次 parallel forward 生成线性 draft token 链。
2. 复查 `ScheduleBatch.filter_batch()` / `merge_batch()` 对 `spec_info` 的调用点，确认 `PEAGLE` 会经过这条路径。
3. 复查 `EagleDraftInput` 的批处理逻辑，确认当前实现原先默认 `topk_p/topk_index` 一定存在。

已运行验证：

1. `PYTHONPATH=python python -m pytest -q test/registered/unit/spec/test_eagle_draft_input.py`
2. `PYTHONPATH=python python -m pytest -q test/registered/unit/server_args/test_server_args.py -k PEagle`
3. `PYTHONPATH=python python -m pytest -q test/registered/spec/utils/test_build_peagle_tree.py`
4. `PYTHONPATH=python python -m compileall python/sglang/srt/speculative/eagle_info.py test/registered/unit/spec/test_eagle_draft_input.py`

# Blocking issues

本轮发现并修复了 1 个 blocking issue：

1. `python/sglang/srt/speculative/eagle_info.py` 中 `EagleDraftInput.filter_batch()` 和 `merge_batch()` 原先无条件访问/拼接 `topk_p` 与 `topk_index`。
2. `PEAGLE` 的设计是复用 `EAGLE` verify 链路，但 draft 输入只依赖 `hidden_states` 和 `verified_id`，不会生成 `topk_p/topk_index`。
3. 这会导致 `PEAGLE` 请求在 overlap scheduler 的 batch filter 或 batch merge 阶段触发 `NoneType` 相关异常，属于运行时阻断问题。
4. 已修复为：
   - `filter_batch()` 对 `topk_p/topk_index` 做可选张量过滤；
   - `merge_batch()` 允许 `PEAGLE` 的 `topk_*` 为空，并在不一致场景下显式报错；
   - 新增 `test/registered/unit/spec/test_eagle_draft_input.py` 覆盖“无 topk 的 filter/merge”路径。

# 非阻断关注点

1. 当前环境没有现成的真实 `PEAGLE` checkpoint 和端到端 benchmark 输入，因此没有完成真实吞吐/时延对比测试。
2. 现阶段能确认的最小性能证据是：`PEAGLE` runtime 已将 draft 侧从多次串行 forward 收敛为单次 parallel forward，这满足本次代码实现的性能改进方向，但不等同于特定模型上的最终 benchmark 数字。

# 最终结论

经过本轮修复和复测，当前实现不存在已知 blocking issues。`PEAGLE` 的参数约束、tree 构造以及无 `topk_*` 的调度兼容路径已经通过验证，可以作为当前仓库中的首版 `PEAGLE` 支持提交。
