# Prefill 执行侧优化机会（codex-exec）

分析基线：`18107e38d266a98f59a8e3c766f8ddaf9b723ded`。
仅使用本地代码检索和阅读；未运行 GPU、安装依赖、启动服务或执行性能测试。下述优先级是静态分析判断，不是实测收益排序。代码行号均对应基线。

## 执行路径与边界

- `python/sglang/srt/model_executor/forward_batch_info.py:739`，`ForwardBatch.init_new`：主机 extend 长度转设备张量，构造 positions / extend_start_loc。
- `python/sglang/srt/model_executor/model_runner.py:1654`，`ModelRunner._forward_raw`：准备 batch 后，选择 split prefill、prefill graph 或 eager。当前版本没有独立的 `ModelRunner.forward_extend`。
- `python/sglang/srt/model_executor/runner/eager_runner.py:251`，`EagerRunner._execute_extend`：准备模型相关元数据，调用 attention backend 的 `init_forward_metadata`，再执行模型。
- `python/sglang/srt/layers/attention/{flashinfer_backend,flashattention_backend,triton_backend}.py`：每批次构造元数据；各层调用 `forward_extend` 消费元数据。本文没有把批次初始化误算成逐层初始化。
- `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py:964,1081,1625`，`_prepare_forward_metadata_for_replay` / `can_run_graph` / `_execute_body_capture`：BCG/Full 支持多请求 body capture，LM-head/logits 尾部仍 eager；元数据是否捕获取决于 backend，不能断言所有 graph 都把元数据留在图外。TC piecewise 也不能与 BCG/Full 混为一谈。

## 1. 优先：FlashInfer 滑窗元数据中的主机同步

证据：`python/sglang/srt/layers/attention/flashinfer_backend.py:1850-1988`，`FlashInferIndicesUpdaterPrefill.update_sliding_window`、`_build_swa_prefix_custom_mask`。

```python
# update_sliding_window，ragged 分支 :1894，paged 分支 :1906
paged_kernel_lens_sum = paged_kernel_lens.sum().item()
# _build_swa_prefix_custom_mask :1958-1960
prefix_lens_cpu = prefix_lens.detach().cpu().tolist()
extend_lens_cpu = (seq_lens - prefix_lens).detach().cpu().tolist()
kv_start_cpu = kv_start_idx.detach().cpu().tolist()
# 请求循环内 :1976-1978
block = (k_abs >= (q_abs - window)).to(torch.uint8)
if not bool(block.all()):
    need_mask = True
```

**触发条件：** FlashInfer 的滑窗 wrapper 分派；mask 构造进一步限定为 ragged prefill。`FlashInferAttnBackend.init_forward_metadata`（:947-1001）在 deterministic、TC piecewise、use_paged、multimodal/MIS 等条件下会禁用 ragged，因此不能把 mask 开销推广到所有请求。

**可优化原因：** CUDA 张量读回 CPU 和 GPU 布尔转 Python 会产生等待；循环还逐请求创建 arange、比较 mask，最后 cat。mask 大小为各请求 `extend_len * min(prefix_len, window)` 的和；这只是张量元素数量推导，未测量耗时或实际分配峰值。

**建议：** 普通 extend 已将 `seq_lens_cpu`、`extend_prefix_lens_cpu` 传入 updater（:982-998）。让滑窗分支也使用镜像计算长度和 `kv_start=max(prefix-window,0)`，向 mask helper 传主机长度；可从整数区间预判是否需要 mask，消除逐请求 `block.all()` 判断，再考虑单次批量 mask 构造。对于非空 query/key 区间，“所有 key 对所有 query 均有效”等价于 `kv_start >= prefix_len + extend_len - 1 - window`。空前缀/空 extend 必须单独处理。

**预期影响：** 缩短滑窗 prefill 元数据关键路径，尤其多请求、小 chunk；不声称 attention 主 kernel 变快。缺少主机镜像的 speculative 路径需保留独立处理；不得混用 verify tree mask 与 ragged prefix mask。验证需覆盖 window 边界、prefix=0、长 prefix、混合 full/SWA 层和 allocator 翻译语义。

**现有优化不应重做：** 同文件 `update_single_wrapper`（:1818-1826）已经在有 CPU prefix 镜像时用 `sum(extend_prefix_lens_cpu)` 避免 D2H。

## 2. 共享 extend 前缀和，减少位置计算重复工作

证据链：

- `python/sglang/srt/model_executor/forward_batch_info.py:909-933`，`ForwardBatch.init_new`：两个 `torch.tensor(...).to(device, non_blocking=True)` 后调用 `compute_position`；:1772-1788 的 `compute_position` 在 `support_triton` 成立时调 Triton 实现。
- `python/sglang/kernels/ops/attention/position.py:32-61`，`compute_position_kernel`：

```python
pid = tl.program_id(0).to(tl.int64)
cumsum_start = tl.cast(0, tl.int64)
for i in range(pid):
    cumsum_start += tl.load(extend_seq_lens + i)
```

对于 B 个请求，源码表达的循环加法次数合计 `B(B-1)/2`；这是算法复杂度，不是实测 GPU 指令或 DRAM 读取数量，缓存/编译优化会影响实际执行。

- `python/sglang/srt/layers/attention/flashattention_backend.py:994-1014`，`FlashAttentionBackend.init_forward_metadata`：有 prefix 的普通 extend 再 `pad(cumsum(extend_seq_lens))` 得到 `cu_seqlens_q`；无 prefix 已复用 `cu_seqlens_k`。
- `python/sglang/srt/layers/attention/triton_backend.py:942-944`，`TritonAttnBackend.init_forward_metadata`：再次 cumsum 填 `qo_indptr`。
- `python/sglang/srt/layers/logits_processor.py:460-465`，`LogitsProcessor._get_pruned_states`：无 input logprobs 时再次 `cumsum(extend_seq_lens)-1` 获取末 token。

**建议：** 单次 scan 生成长度为 B+1 的 extend offsets，positions、attention 的 query indptr 和末 token 索引共享；大 batch 的 position kernel 直接读取 offset，避免每请求串行扫描前面所有长度。主机长度已知时也可比较批量传输 offsets 的方案。两个独立小 H2D 可作为附带研究项，不能仅凭 `non_blocking=True` 推断传输已完全异步或已与计算重叠。

**预期影响：** 大 batch、每请求短 extend 更可能受益；小 batch 增加 scan launch 可能抵消节省，不能无条件替换。注意 DP/CP padding、graph 静态地址、draft override positions 与整数 dtype；`compute_position` 即使已有 positions 仍需提供 `extend_start_loc`，不能直接整体跳过。

## 3. input-logprob 尾部：主机逐 token 索引与每 chunk 全量筛选

证据：`python/sglang/srt/layers/logits_processor.py:514-582`，`LogitsProcessor._get_pruned_states`：

```python
token_to_seq_idx.extend([idx] * (extend_len - start_len))
input_logprob_indices.extend([
    input_logprob_indices_pt + i
    for i in range(extend_len - extend_logprob_start_len)
])
pruned_states = torch.cat(pruned_states_list)
```

随后已有 pinned host memory + non-blocking H2D（:571-582），不是待补的空白优化。裁剪按请求生成 view，再 cat；before-norm / auxiliary hidden states 开启时还有相应复制。

`python/sglang/srt/layers/logprob_processor.py:527-573`，`InputLogprobProcessor._forward_by_chunk`：

```python
for i in range(num_chunks):
    chunk_mask = (input_logprob_indices >= start_idx) & (
        input_logprob_indices < end_idx
    )
    global_indices = input_logprob_indices[chunk_mask]
    mask_indices = torch.nonzero(chunk_mask, as_tuple=True)[0]
    # ... LM head ...
    if chunk_sample_mask.any():
        ...
```

**可优化原因：** CPU 索引列表随裁剪 token 数增长；每 chunk 又扫描整个 input-logprob 索引列表。若索引数为 L、chunk 数 C，mask 比较工作为 O(C·L)；GPU nonzero 的动态输出和 `.any()` 的 Python 分支还引入同步机会。未测量这些成本在 LM-head 时间中的占比。

**建议：** 索引天然按序且由主机长度决定，可按 request offsets 预计算每 chunk 的索引区间与采样请求范围，替代全量 mask/nonzero 和 device-dependent Python 分支；用请求区间边界表示 `token_to_seq_idx`，在 chunk 两端查找所属请求，避免逐 token Python 列表。全保留且布局兼容时可直接复用 hidden_states，否则保留正确 gather。保持 chunking 控制 logits 峰值内存，不能为了少循环改成完整大 logits。

**影响范围与边界：** 仅 input logprobs/scoring 请求；普通生成已只挑最后 token 做 LM head。BCG/Full 在 `PrefillCudaGraphRunner._execute_body_capture`（`python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py:1625-1635`）明确让 tail eager，graph 命中也不自动消除这类成本。现有 fast logsumexp/top-k 和 chunking（`InputLogprobProcessor.__init__` / `forward`，:436-480）应保留；验证 chunk 跨请求、零 logprob 行但仍需采样、top-k/token-id logprob、LoRA、DP collective 顺序及 deterministic 模式。

## 后续验证建议（本轮未执行）

先分别验证以上三项的输出等价，再测各自触发的工作负载：滑窗 prefix/chunk 组合；固定总 token 改变请求数的短 extend；开启 input logprobs 并改变 chunk 数。对 graph/eager 单独记录，不能把 graph 开启当作全部执行路径已捕获。

所有后续 SGLang 性能数字必须经用户指定的 `benchmark/engine_steps/bench_engine_steps.py`，记录脚本 revision、完整命令、模型/backend、请求/长度/prefix/logprob/graph 配置及验证结果；本轮未确认该脚本在此 checkout 中的可用性。固定批次 engine-step 吞吐不能作为应用 E2E/TTFT 的实测替代。
