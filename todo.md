# SGLang Python frontend overhead 与 Rust frontend 可行性调研

日期：2026-06-01
工作目录：`/root/code/origin_sglang/front-rust`
本次用于测量的代码：必须显式使用 `PYTHONPATH=/root/code/origin_sglang/front-rust/python`，否则会 import 到另一份 checkout。

## 结论

在本机 `/models/Qwen/Qwen3-0.6B`、单 GPU H20、radix cache 关闭、`max_running_requests=64` 的条件下，SGLang Python frontend 不是主要瓶颈。

硬指标来自 `/generate` 返回的内部时间戳。`api_server_dispatch_finish_ts - request_received_ts` 包含 Python 侧请求规范化、校验、tokenize（如果传 text）、构造 tokenized object、ZMQ `send_pyobj` 到 scheduler 的时间：

- `input_ids` 路径：约 `0.07-0.11 ms/request`。
- `text` 路径：128 token prompt 约 `0.33-0.39 ms/request`，512 token prompt 约 `0.99 ms/request`。
- 并发 32、128 in / 32 out 时，`text` 路径 frontend dispatch 约 `0.33 ms`，服务端 E2E 约 `117 ms`，占比约 `0.28%`。
- 最短、最有利于 frontend 显著化的场景是并发 1、512 in / 1 out，`text` dispatch 约 `0.99 ms`，服务端 E2E 约 `13.59 ms`，占比约 `7.3%`。这已经是小模型、短输出、无 cache 的偏前端敏感场景；真实较大模型或较长 decode 下占比应继续下降。

第一轮 OpenAI chat 只能用 client latency 做上界。低并发下 chat 相对 native `input_ids` 多约 `0.4-1.3 ms`，并发 32 / 128 in / 1 out 下上界变成约 `12 ms` mean、`14 ms` median；这混入了更多 prompt token（Qwen chat template 本次从 128 变 136 tokens）、HTTP/Pydantic/response serialization、客户端并发调度和 scheduler batching 影响，不能直接等同于 “Python frontend 可以被 Rust 消掉的收益”。后续已补 opt-in OpenAI frontend timing，把 handler 内 validation / convert / template-tokenize / response build 拆开。

因此，按现有数据，直接把整个 SGLang Python frontend 搬到 Rust，预计不会带来类似 “端到端大幅提速” 的确定收益。它可能在这些 workload 才有意义：极短输出、极高 QPS、小模型、CPU/frontend 饱和、复杂 tool schema/chat template、慢 tokenizer、多租户 HTTP gateway 压力很大。

二次验证已按这个策略补了可开关 OpenAI chat 分段 timing。结果显示：OpenAI chat 的模板/tokenize/convert 中位数约 `0.35-0.45 ms/request`，response build 约 `0.02-0.03 ms/request`；相比并发短输出下 `40-80 ms` 级别的 tokenizer_manager/model 等待，E2E latency 收益很小。高并发 c64 tiny-output 压力下，API 主进程会接近单核 `~96%`，说明 Rust/更轻网关可能提升极短请求的最大 QPS；但这更像 throughput/capacity 优化，不是常规单请求 latency 大幅优化。

把 server 继续绑到 `4c` 后，普通 OpenAI chat 仍没有退化：同样 c64、12000 requests、128 in / 1 out 下，throughput 约 `635 req/s`，`openai_convert` p50/p95 为 `0.392/0.463 ms`，`process_messages` p50/p95 为 `0.353/0.419 ms`。这不支持“192c 掩盖了普通 frontend 大开销”。

但 CPU-heavy OpenAI payload 会显著改变结论。用 `oai_chat_tools` 增加 8 个 tool、每个 8 个 schema fields 后，4c 下同一 chat convert path 的 API pre-dispatch 从普通 chat 的 `~0.4 ms` p50 抬到 `~21 ms` p50，其中 request validation 约 `16 ms`，chat template/tokenize 约 `4.6 ms`。所以担心是成立的，但成立范围更准确地说是：tools/schema/function calling/复杂 template/spec 这类 CPU-heavy feature，可能把 frontend 变成值得优化的 CPU hot path；普通 chat 不是。

## vLLM Rust frontend 对比

参考 PR：https://github.com/vllm-project/vllm/pull/43283
vLLM 环境变量参考：https://docs.vllm.ai/en/latest/configuration/env_vars/

vLLM 的 Rust frontend 不是简单把所有 Python 推理栈改写成 Rust。它新增/集成 `vllm-frontend-rs`，通过 `VLLM_USE_RUST_FRONTEND` 启用，由 Python 侧 frontend process manager 启动 Rust frontend，再通过继承 socket fd 和 ZMQ 地址与 engine 通信。可迁移给 SGLang 的启发是边界设计：把 HTTP/OpenAI request parsing、chat template/tokenize、engine IPC 边界做成独立 frontend process，而不是一次性重写 scheduler/model executor。

但 SGLang 当前路径已经把 GPU scheduler/executor 放在子进程，Python API server 的可见成本在本次测量里偏小。是否值得做 Rust frontend，要看它能替代的具体阶段是否真的占总延迟或 CPU。

## SGLang 当前 frontend 路径

OpenAI chat 入口：

- `python/sglang/srt/entrypoints/openai/serving_base.py:79` 在 validate/convert 前记录 `received_time`。
- `python/sglang/srt/entrypoints/openai/serving_chat.py:455` 的 `_convert_to_internal_request()` 调 `_process_messages()`。
- `python/sglang/srt/entrypoints/openai/serving_chat.py:599` 走 `_apply_jinja_template()`。
- `python/sglang/srt/entrypoints/openai/serving_chat.py:770` 先 `apply_chat_template(tokenize=False)`，再 `tokenizer.encode()`。
- 非 multimodal 时，`python/sglang/srt/entrypoints/openai/serving_chat.py:490-493` 通常把 `prompt_ids` 作为 `input_ids` 传给 `GenerateReqInput`，因此 `TokenizerManager` 不会再重复 tokenize。
- `python/sglang/srt/entrypoints/openai/serving_chat.py:1154` 调 `tokenizer_manager.generate_request()`，`python/sglang/srt/entrypoints/openai/serving_chat.py:1277` 构建 OpenAI chat response。

Native `/generate` 路径：

- `python/sglang/srt/managers/tokenizer_manager.py:543` 进入 `generate_request()`。
- `python/sglang/srt/managers/tokenizer_manager.py:582` `_tokenize_one_request()`；如果传 `input_ids`，主要是 validate 和 object construction；如果传 `text`，会走 tokenizer。
- `python/sglang/srt/managers/tokenizer_manager.py:1002` `_create_tokenized_object()` 构造 sampling params 和 tokenized request。
- `python/sglang/srt/managers/tokenizer_manager.py:1219` `_send_one_request()` 在 `send_pyobj` 前后记录 API server dispatch 时间。
- `python/sglang/srt/managers/tokenizer_manager.py:1321` `_wait_one_response()` 等 scheduler/detokenizer 返回。

时间戳边界：

- `python/sglang/srt/observability/req_time_stats.py:371` 有 `tokenize_finish_time`，但当前 `convert_to_output_meta_info()` 没有导出它。
- `python/sglang/srt/observability/req_time_stats.py:424-439` 导出 `request_received_ts`、`api_server_dispatch_finish_ts`、`request_finished_ts`、`response_sent_to_client_ts`。
- 所以本次可直接测的硬指标是 `api_server_dispatch_finish_ts - request_received_ts`，不能再拆成 validation / template / tokenizer / ZMQ dispatch。

## 实验设置

服务启动命令：

```bash
PYTHONPATH=/root/code/origin_sglang/front-rust/python \
CUDA_VISIBLE_DEVICES=7 \
SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 \
python3 -m sglang.launch_server \
  --model-path /models/Qwen/Qwen3-0.6B \
  --host 127.0.0.1 --port 31041 \
  --trust-remote-code --dtype bfloat16 \
  --mem-fraction-static 0.45 \
  --max-total-tokens 8192 --max-running-requests 64 \
  --chunked-prefill-size -1 \
  --disable-radix-cache --enable-metrics \
  --skip-server-warmup \
  --log-level warning --log-level-http warning
```

Probe 脚本：`benchmark/frontend_overhead_probe.py`

对比模式：

- `native_input_ids`：`/generate` + `input_ids`，绕过 server tokenizer。
- `native_text`：`/generate` + `text`，由 server tokenizer 处理。
- `oai_completion`：`/v1/completions`。
- `oai_chat`：`/v1/chat/completions`，包含 OpenAI chat template/response 格式。

所有 `clean_*.json` 是干净结果；非 `clean_` 结果来自早期并发误跑，已丢弃，不用于结论。

## Benchmark 结果

单位：毫秒。第一轮 benchmark 中，`frontend dispatch` 只在 native `/generate` 可从 `meta_info` 直接计算；OpenAI public response 默认不带这些内部时间戳。后面的 timed benchmark 使用本地 opt-in header 返回 OpenAI frontend timing。

| artifact | mode | client mean/p50/p99 | server mean/p50 | frontend dispatch mean/p50/p99 | prompt/out tokens |
|---|---:|---:|---:|---:|---:|
| `clean_qwen06b_c1_in32_out1.json` | native_input_ids | 13.11 / 13.08 / 13.88 | 12.48 / 12.46 | 0.09 / 0.09 / 0.14 | 32 / 1 |
| `clean_qwen06b_c1_in32_out1.json` | native_text | 13.86 / 13.93 / 16.74 | 13.21 / 13.27 | 0.24 / 0.24 / 0.34 | 32 / 1 |
| `clean_qwen06b_c1_in32_out1.json` | oai_completion | 14.04 / 13.99 / 14.61 | - | - | 32 / 1 |
| `clean_qwen06b_c1_in32_out1.json` | oai_chat | 14.39 / 14.37 / 15.07 | - | - | 40 / 1 |
| `clean_qwen06b_c1_in128_out1.json` | native_input_ids | 14.10 / 14.12 / 14.66 | 13.48 / 13.50 | 0.09 / 0.09 / 0.11 | 128 / 1 |
| `clean_qwen06b_c1_in128_out1.json` | native_text | 14.25 / 14.22 / 14.92 | 13.61 / 13.59 | 0.39 / 0.39 / 0.41 | 128 / 1 |
| `clean_qwen06b_c1_in128_out1.json` | oai_completion | 14.50 / 14.34 / 15.79 | - | - | 128 / 1 |
| `clean_qwen06b_c1_in128_out1.json` | oai_chat | 14.54 / 14.54 / 15.19 | - | - | 136 / 1 |
| `clean_qwen06b_c1_in512_out1.json` | native_input_ids | 14.97 / 13.89 / 53.03 | 14.10 / 13.17 | 0.11 / 0.10 / 0.17 | 512 / 1 |
| `clean_qwen06b_c1_in512_out1.json` | native_text | 14.35 / 14.34 / 15.23 | 13.59 / 13.55 | 0.99 / 0.93 / 1.20 | 512 / 1 |
| `clean_qwen06b_c1_in512_out1.json` | oai_completion | 14.31 / 14.30 / 14.73 | - | - | 512 / 1 |
| `clean_qwen06b_c1_in512_out1.json` | oai_chat | 14.39 / 14.36 / 14.98 | - | - | 520 / 1 |
| `clean_qwen06b_c32_in128_out1.json` | native_input_ids | 40.40 / 38.17 / 72.05 | 35.36 / 33.94 | 0.07 / 0.07 / 0.12 | 128 / 1 |
| `clean_qwen06b_c32_in128_out1.json` | native_text | 48.81 / 46.90 / 78.54 | 41.75 / 40.38 | 0.34 / 0.33 / 0.43 | 128 / 1 |
| `clean_qwen06b_c32_in128_out1.json` | oai_completion | 47.15 / 46.98 / 63.62 | - | - | 128 / 1 |
| `clean_qwen06b_c32_in128_out1.json` | oai_chat | 52.45 / 52.40 / 77.01 | - | - | 136 / 1 |
| `clean_qwen06b_c32_in128_out32.json` | native_input_ids | 126.11 / 126.06 / 127.37 | 117.60 / 117.27 | 0.07 / 0.07 / 0.12 | 128 / 32 |
| `clean_qwen06b_c32_in128_out32.json` | native_text | 146.69 / 127.55 / 456.67 | 117.45 / 114.69 | 0.33 / 0.32 / 0.43 | 128 / 32 |
| `clean_qwen06b_c32_in128_out32.json` | oai_completion | 130.57 / 130.23 / 138.76 | - | - | 128 / 32 |
| `clean_qwen06b_c32_in128_out32.json` | oai_chat | 144.18 / 144.31 / 145.81 | - | - | 136 / 32 |

关键派生值：

- 并发 1、32 in / 1 out：`native_text` dispatch / server E2E = `1.84%`；chat client 相对 native `input_ids` 多 `1.28 ms` mean。
- 并发 1、128 in / 1 out：`native_text` dispatch / server E2E = `2.89%`；chat client 相对 native `input_ids` 多 `0.44 ms` mean。
- 并发 1、512 in / 1 out：`native_text` dispatch / server E2E = `7.26%`；tokenizer 成本随 prompt length 增长，但绝对值仍约 `1 ms`。
- 并发 32、128 in / 1 out：`native_text` dispatch / server E2E = `0.81%`；chat client 相对 native `text` 多 `3.64 ms` mean / `5.51 ms` median。
- 并发 32、128 in / 32 out：`native_text` dispatch / server E2E = `0.28%`；decode 变长后 frontend dispatch 被明显稀释。

`clean_qwen06b_c32_in128_out32.json` 的 `native_text` client mean/p99 有明显 client-side outlier，但 server E2E 和 frontend dispatch 没有同步放大；这个点不用于推导 frontend dispatch 占比，只保留为 artifact 事实。

## Profile 结果

Profile 命令：

```bash
py-spy record \
  --pid 2457173 \
  --output artifacts/frontend_overhead_probe/pyspy_frontend_oai_chat.raw \
  --format raw \
  --duration 25 \
  --rate 99 \
  --nonblocking \
  --threads \
  --full-filenames
```

压力流量：

```bash
PYTHONPATH=/root/code/origin_sglang/front-rust/python \
python3 benchmark/frontend_overhead_probe.py \
  --base-url http://127.0.0.1:31041 \
  --model /models/Qwen/Qwen3-0.6B \
  --tokenizer /models/Qwen/Qwen3-0.6B \
  --requests 12000 \
  --warmup 0 \
  --concurrency 64 \
  --input-len 128 \
  --output-len 1 \
  --modes oai_chat \
  --output artifacts/frontend_overhead_probe/profile_traffic_oai_chat_c64_in128_out1.json
```

Profile traffic 结果：12000/12000 OK，client latency mean `99.30 ms`，median `81.86 ms`，p99 `418.09 ms`，prompt/completion tokens `136/1`。

`py-spy` active Python samples：343。按一条 stack 归入一个类别：

- OpenAI chat conversion / Jinja template / tokenizer：168 samples，`49.0%`。
- HTTP framework parse / validation / serialization：155 samples，`45.2%`。
- TokenizerManager response metrics：11 samples，`3.2%`。
- TokenizerManager generate / dispatch / wait：4 samples，`1.2%`。
- event loop / ZMQ：3 samples，`0.9%`。
- other：2 samples，`0.6%`。

Top leaf：

- `asyncio/locks.py:13 __aenter__`：61 samples，`17.8%`。
- `transformers/tokenization_utils_tokenizers.py:959 _encode_plus`：59 samples，`17.2%`。
- `asyncio/runners.py:118 run`：29 samples，`8.5%`。
- `starlette/_exception_handler.py:23 wrap_app_handling_exceptions`：13 samples，`3.8%`。
- `pydantic/type_adapter.py:441 validate_python`：11 samples，`3.2%`。
- `zmq/sugar/socket.py:958 send_pyobj`：7 samples，`2.0%`。

Interpretation：

- CPU 活跃时，OpenAI chat 前处理和 HTTP/Pydantic 框架成本各占约一半；这说明如果只优化 `send_pyobj` 或 `TokenizerManager` dispatch，收益会很有限。
- 真正跟模型侧等待相关的 `generate_request()` / `_wait_one_response()` stack 很多不是 CPU hot path，而是 async 等待。
- 另一次带 `--idle` 的 profile 显示 API 主线程大多数采样停在 event loop 顶层，旁路线程停在 watchdog/cpu monitor wait；这与 benchmark 里 frontend dispatch 绝对值很小相符。

## 二次验证：OpenAI chat 分段 timing

按“先补观测再判断”的策略，本地新增了一个可开关的 OpenAI frontend timing：

- 请求头：`x-sglang-frontend-timing: 1`
- 响应位置：`metadata.sglang_frontend_timing`
- 默认行为：不带请求头时不返回 timing，仍只保留原有 `metadata.weight_version`
- 覆盖测试：`test/registered/unit/entrypoints/openai/test_serving_chat.py`

新增字段分两类：

- `segments_s`：handler 内部阶段，包括 semantic validation、OpenAI convert、`process_messages`（chat template + tokenizer encode）、sampling params、`GenerateReqInput` 构造、`TokenizerManager.generate_request()` 等待、response build。
- `engine_s`：复用已有 `meta_info` 推导的 API dispatch、post-dispatch-to-finish、response send lag、queue。

注意：FastAPI JSON parse/Pydantic request model 构造发生在 `handle_request()` 之前，不在 `segments_s` 里；这部分只能通过 CPU/profile 间接看。

### Timed benchmark

服务同样使用 `/models/Qwen/Qwen3-0.6B`，端口 `31042`，radix cache 关闭。artifact：

- `artifacts/frontend_overhead_probe/timed_qwen06b_c1_in128_out1.json`
- `artifacts/frontend_overhead_probe/timed_qwen06b_c32_in128_out1.json`
- `artifacts/frontend_overhead_probe/timed_qwen06b_c32_in128_out32.json`
- `artifacts/frontend_overhead_probe/timed_qwen06b_c64_in128_out1_cpu_probe2.json`

关键数值，单位毫秒：

| case | mode | client mean / p50 | API dispatch mean / p50 / p95 | OpenAI convert mean / p50 / p95 | process_messages mean / p50 / p95 | tokenizer_manager mean / p50 / p95 |
|---|---:|---:|---:|---:|---:|---:|
| c1, 128 in / 1 out | native_text | 14.08 / 13.99 | 0.404 / 0.394 / 0.480 | - | - | - |
| c1, 128 in / 1 out | oai_chat | 14.39 / 14.25 | 0.601 / 0.594 / 0.622 | 0.456 / 0.451 / 0.470 | 0.413 / 0.408 / 0.427 | 13.07 / 12.94 / 13.65 |
| c32, 128 in / 1 out | native_text | 48.69 / 46.63 | 0.339 / 0.325 / 0.416 | - | - | - |
| c32, 128 in / 1 out | oai_chat | 52.64 / 53.52 | 0.538 / 0.518 / 0.624 | 0.406 / 0.396 / 0.459 | 0.367 / 0.358 / 0.417 | 43.62 / 42.01 / 61.69 |
| c32, 128 in / 32 out | native_text | 126.48 / 126.47 | 0.325 / 0.317 / 0.391 | - | - | - |
| c32, 128 in / 32 out | oai_chat | 157.93 / 143.96 | 1.142 / 0.507 / 0.614 | 1.010 / 0.388 / 0.459 | 0.972 / 0.351 / 0.417 | 131.18 / 125.12 / 135.07 |
| c64, 128 in / 1 out | oai_chat | 103.79 / 84.44 | 0.678 / 0.514 / 0.617 | 0.435 / 0.392 / 0.463 | 0.397 / 0.355 / 0.420 | 82.20 / 66.23 / 135.13 |

Interpretation：

- OpenAI chat `process_messages` 基本就是本 workload 下最主要的 Python chat-template/tokenizer 可优化项，中位数稳定在 `0.35-0.41 ms`。
- `response_build` 中位数只有 `0.017-0.033 ms`，不是优先目标。
- `openai_convert` 的 p95 基本 `<0.5 ms`；个别 run 有 `~300 ms` outlier，表现为单请求 pause/调度异常，不应按 mean 估算收益。
- 并发 32、1-token 输出时，chat convert 中位数 / tokenizer_manager 中位数约 `0.396 / 42.0 = 0.94%`。
- 并发 32、32-token 输出时，chat convert 中位数 / tokenizer_manager 中位数约 `0.388 / 125.1 = 0.31%`。
- 并发 64、1-token 输出时，chat convert 中位数 / tokenizer_manager 中位数约 `0.392 / 66.2 = 0.59%`。

### CPU saturation check

在 c64、12000 个 OpenAI chat、128 in / 1 out 压力下：

- 吞吐约 `612 req/s`（12000 requests / `19.61s`）。
- API 主进程 `/proc/<pid>/stat` 瞬时 CPU 采样在流量真正开始后约 `91-98%` 单核。
- 这说明极短请求高 QPS 场景下，Python API process 可以成为单核 capacity 瓶颈。
- 但 handler 内可见的 OpenAI convert/template/tokenize 仍只有 `~0.4 ms` 中位数；剩余 CPU 多在 FastAPI/Pydantic/Starlette/HTTP/serialization/event-loop glue，以及 response lifecycle。Rust gateway 若要有收益，必须覆盖这些边界，而不是只改 `TokenizerManager` dispatch。

### CPU affinity: 32c / 8c 验证

为了验证“本机 192c 太强，掩盖了 frontend CPU 开销”这个假设，又把同一个 SGLang server 绑到较小 CPU set 跑了同款 c64、12000 requests、128 in / 1 out OpenAI chat workload。

设置：

- `server32c`：server 主进程、scheduler、detokenizer 全部 `taskset 0-31`。
- `server8c`：同一组进程和线程全部 `taskset 0-7`。
- `server4c`：同一组进程和线程全部 `taskset 0-3`。
- 压测 client 不限 CPU，避免 client 端 JSON/HTTP 生成成为干扰。

结果：

| CPU affinity | elapsed | throughput | client mean / p50 / p95 / p99 | openai_convert p50 / p95 | process_messages p50 / p95 |
|---|---:|---:|---:|---:|---:|
| unrestricted | `19.61s` | `612 req/s` | `103.8 / 84.4 / 375.2 / 430.4 ms` | `0.392 / 0.463 ms` | `0.355 / 0.420 ms` |
| server32c | `19.02s` | `631 req/s` | `100.6 / 82.9 / 383.7 / 430.4 ms` | `0.393 / 0.466 ms` | `0.356 / 0.423 ms` |
| server8c | `19.23s` | `624 req/s` | `101.8 / 83.8 / 378.5 / 407.9 ms` | `0.394 / 0.463 ms` | `0.356 / 0.420 ms` |
| server4c | `18.90s` | `635 req/s` | `100.0 / 85.1 / 369.6 / 419.6 ms` | `0.392 / 0.463 ms` | `0.353 / 0.419 ms` |

CPU sampling：

- `server32c` active window：API `~92%` mean active / scheduler `~100%` / detokenizer `~2%`。
- `server8c` active window：API `~88%` mean active / scheduler `~100%` / detokenizer `~2%`。
- `server4c` active window：API `~87%` mean active / scheduler `~100%` / detokenizer `~2%`。

Interpretation：

- 32c 并没有让 latency 或 throughput 变差；8c 和 4c 也基本没有变差。
- 因此当前数据不支持“192c 掩盖了 frontend 大开销”。这个 workload 的 CPU 形态更像是两个热单核：API event loop 一个核，scheduler 一个核；给 8c/32c 都足够。
- 这反而强化了 Rust frontend 的定位：如果要做，目标应该是降低 API 单核消耗、提高极短请求最大 QPS，而不是指望在常规请求上因为 CPU 核数减少产生大幅 E2E latency 收益。
- 真要观察 CPU 总核数不足导致的退化，需要继续压到 `<=2c`、限制 client CPU、或增加更重的 OpenAI payload（tools/schema/logprobs/streaming/large messages）。但 `4c` 对普通 chat 已经不是瓶颈。

### CPU-heavy OpenAI payload: tools/schema on 4c

为了覆盖“当前没有开 spec 等 CPU overhead feature，导致开销被掩盖”的担心，又加了 `oai_chat_tools` mode。payload 是 8 个 function tools、每个 tool 8 个 JSON-schema fields；再大一版 16x16 会把 prompt 撑到 `11088` tokens，超过本次 `8192` context，因此最终用 8x8。8x8 tools 已经让 Qwen chat prompt 从普通 `72` tokens 变成 `3202` tokens。

对照 workload：server 全进程 `taskset 0-3`，c16、512 requests、64 in / 1 out。

| mode | prompt tokens | client mean / p50 / p95 / p99 | API dispatch p50 / p95 | validation p50 / p95 | process_messages p50 / p95 | tokenizer_manager p50 / p95 |
|---|---:|---:|---:|---:|---:|---:|
| `oai_chat` | `72` | `35.5 / 34.3 / 48.3 / 55.7 ms` | `0.421 / 0.509 ms` | `0.005 / 0.006 ms` | `0.268 / 0.322 ms` | `30.0 / 40.0 ms` |
| `oai_chat_tools` | `3202` | `546.2 / 550.5 / 578.7 / 590.6 ms` | `20.970 / 21.921 ms` | `16.105 / 16.608 ms` | `4.594 / 4.698 ms` | `515.1 / 535.9 ms` |

tools-only CPU sampling：

- API active window：`~63%` mean active，p95 `~65%`，max `~68%`。
- Scheduler active window：`~100%`，说明大 prompt prefill 也把 scheduler/model side 拉满。
- Detokenizer 基本空闲。

Interpretation：

- 这个 case 证明 CPU-heavy OpenAI feature 可以让 frontend 成本从 `sub-ms` 级变成 `~20 ms/request` 级。
- 但 tools schema 同时把 prompt 撑到 `3202` tokens，模型侧 prefill/tokenizer_manager 等待也变成 `~515 ms` p50。因此即使 Rust gateway 完全消掉 `~21 ms` API pre-dispatch，单请求 E2E latency 的中位数收益也大约是 `3-4%`，不是数量级变化。
- Rust frontend 的更实际价值会是 CPU capacity/QPS：如果线上大量 tools/schema/function calling 请求，Python request validation/chat template/tokenize 会吃掉可观 CPU；这时值得做 prototype。prototype 必须覆盖 HTTP/Pydantic/schema validation/template/tokenizer/serialization 这些边界，只优化 `TokenizerManager` 或 ZMQ dispatch 不够。

### 其他可代码优化的瓶颈

按现有 profile 和分段 timing，除“完整 Rust frontend”外，比较值得看的代码优化点是：

1. Tool schema validation cache
   - 证据：4c tools case 里 `_validate_request()` 的 validation p50/p95 是 `16.105 / 16.608 ms`，主要来自每个 request 对每个 tool 执行 `normalize_json_schema_types()` 和 `Draft202012Validator.check_schema()`。
   - 代码位置：`python/sglang/srt/entrypoints/openai/serving_chat.py:420-430`。
   - 优化方向：按 tool schema 的稳定 hash 缓存“已 normalize + 已通过 check_schema”的结果；相同 tools payload 在连续请求里不重复做 JSON Schema 校验。需要注意当前 `normalize_json_schema_types()` 会原地改 schema，cache 设计要避免跨请求共享可变 dict 引发副作用。
   - 预期收益：对普通 chat 没影响；对 tools/schema 请求，理论上能从 API pre-dispatch 里削掉最多 `~16 ms/request`。这是本次最明确的 Python 局部优化点。

2. FastAPI / Pydantic / response serialization bypass
   - 证据：普通 c64 chat 的 API process py-spy active samples 中，HTTP/FastAPI/Starlette/Pydantic/serialization 占 `~60%`；top leaves 包括 `validate_python`、`jsonable_encoder`、FastAPI routing/dependency。handler 内 `response_build` p50 只有 `~0.02 ms`，但 FastAPI 后续把 Pydantic response 编码出去的成本不在 handler timing 里。
   - 代码位置：`python/sglang/srt/entrypoints/http_server.py:1506-1512` 返回 `ChatCompletionResponse`，由 FastAPI 继续序列化；streaming path 已经大量用 `model_dump_json()` 直接拼 SSE。
   - 优化方向：non-streaming OpenAI chat/completions 可以评估直接返回 `Response(content=response.model_dump_json(...), media_type="application/json")` 或等价的预序列化 fast path，绕过 FastAPI `jsonable_encoder` 二次处理。需要验证错误响应、metadata、exclude_none、OpenAI 兼容字段都完全一致。
   - 预期收益：普通 chat 的单请求 latency 可能仍只有亚毫秒到几毫秒，但高 QPS tiny-output 时能降低 API 单核占用；这是 capacity 优化。

3. Chat template / tokenizer encode path
   - 证据：普通 chat `process_messages` p50 约 `0.35 ms`，tools case 约 `4.6 ms`；py-spy 里 tokenizer `_encode_plus` 是 top leaf，占 active samples `~17%`。
   - 代码位置：`python/sglang/srt/entrypoints/openai/serving_chat.py:718-789` 每次构造 `openai_compatible_messages`，再 `apply_chat_template(tokenize=False)`，再 `tokenizer.encode()`。
   - 优化方向：对重复 tools/system prefix 做 rendered prefix 或 token prefix cache；减少重复 `model_dump()`；在不会 double-BOS 的 tokenizer 上评估 `apply_chat_template(tokenize=True)` fast path。当前 render+encode split 是为避免 special token 语义问题，不能无条件改回。
   - 预期收益：普通 chat 不大；tools/schema、长 system prompt、复杂 template 下有意义，且能配合 radix cache 进一步减少模型侧 prefill。

4. Scheduler hot core
   - 证据：32c/8c/4c affinity 下 scheduler process active window 都接近 `~100%`，普通 chat latency 未退化，但形态上 scheduler 是另一个热单核。
   - 当前不足：本次没有 scheduler py-spy/nsys profile，不能把它归因到某段 Python scheduling loop、CUDA stream wait、IPC 或 batch output handling。
   - 优化方向：下一步应单独 profile scheduler process，而不是直接猜代码改动。只有看到 scheduler CPU 真在 Python 逻辑而不是 CUDA/event wait，才适合做代码优化。

不太像优先瓶颈的点：

- `send_pyobj` / ZMQ dispatch：py-spy leaf 约 `2%`，native `/generate` dispatch `0.07-0.4 ms`，不是第一目标。
- TokenizerManager response metrics：API profile 约 `1-2%`，除非线上打开大量 metrics/custom labels/logprobs/spec stats，否则优先级低。
- OpenAI non-streaming response build：handler 内 p50 `~0.02 ms`，真正要看的是 FastAPI 出口序列化，不是 `_build_chat_response()` 本身。

### 已实现优化与 E2E 验证

本轮已实现两个局部优化：

1. Tool schema validation cache
   - 代码：`python/sglang/srt/entrypoints/openai/serving_chat.py`
   - 行为：对完全相同的 tool `parameters` JSON payload 做 LRU cache，cache miss 时 normalize + `Draft202012Validator.check_schema()`，cache hit 时复用 normalized schema。
   - 安全点：cache key 保持原始 JSON key order，不用 sorted dump；这样不会改变 chat template 渲染后的 prompt tokenization。单测覆盖了 repeated validation 只调用一次 `check_schema`，且 cache hit 后仍保留 normalized schema 和原始 key order。

2. OpenAI non-streaming response fast path
   - 代码：`python/sglang/srt/entrypoints/openai/serving_base.py`
   - 行为：non-streaming handler 如果返回 Pydantic model，直接 `model_dump_json()` 后返回 `fastapi.Response(media_type="application/json")`，绕过 FastAPI `jsonable_encoder` / Pydantic 二次序列化。
   - 覆盖范围：OpenAI chat 和 completions 的成功 non-streaming 响应；streaming 和 error response 保持原路径。

本地验证：

- Unit：`test/registered/unit/entrypoints/openai/test_serving_chat.py` + `test/registered/unit/entrypoints/openai/test_serving_completions.py` 全量通过：`79 passed, 9 subtests passed`。
- Compile：`python3 -m py_compile benchmark/frontend_overhead_probe.py python/sglang/srt/entrypoints/openai/serving_base.py python/sglang/srt/entrypoints/openai/serving_chat.py python/sglang/srt/entrypoints/openai/serving_completions.py` 通过。
- E2E server：Qwen3-0.6B，server/API/scheduler/detokenizer 全部 `taskset 0-3`，port `31045`，radix cache 关闭。

Tool schema cache before/after，c16、512 requests、64 in / 1 out、8x8 tools：

| mode | artifact | prompt tokens | client p50 | API dispatch p50 | validation p50 | process_messages p50 | tokenizer_manager p50 |
|---|---|---:|---:|---:|---:|---:|---:|
| before `oai_chat_tools` | `timed_qwen06b_c16_in64_out1_chat_vs_tools_server4c.json` | `3202` | `550.5 ms` | `20.970 ms` | `16.105 ms` | `4.594 ms` | `515.1 ms` |
| after `oai_chat_tools` | `optimized_qwen06b_c16_in64_out1_chat_vs_tools_server4c_order_preserved.json` | `3202` | `549.5 ms` | `5.038 ms` | `0.055 ms` | `4.712 ms` | `539.3 ms` |

Interpretation：

- schema validation cache 把 tools request 的 validation p50 从 `16.105 ms` 降到 `0.055 ms`，API pre-dispatch p50 从 `20.970 ms` 降到 `5.038 ms`。
- prompt tokens 保持 `3202`，说明修正后的 cache 没有改变 tool schema key order / prompt tokenization。
- E2E latency p50 基本不变，因为 `~3.2K` prompt 的模型侧 prefill 仍是 `~0.5s` 级；收益主要是 API CPU capacity 和 frontend pre-dispatch。

Response fast path before/after，普通 `oai_chat`，c64、12000 requests、128 in / 1 out：

| artifact | throughput | client p50 / p95 / p99 | API dispatch p50 / p95 | response_send_lag p50 / p95 | API active CPU |
|---|---:|---:|---:|---:|---:|
| before `timed_qwen06b_c64_in128_out1_server4c.json` | `635 req/s` | `85.1 / 369.6 / 419.6 ms` | `0.512 / 0.613 ms` | `2.099 / 6.826 ms` | `86.9%` mean active, p95 `98.0%` |
| after `optimized_qwen06b_c64_in128_out1_server4c.json` | `631 req/s` | `81.6 / 396.8 / 436.4 ms` | `0.514 / 0.618 ms` | `1.389 / 2.380 ms` | `84.5%` mean active, p95 `94.0%` |

Interpretation：

- response fast path 明显降低 response_send_lag：p50 `2.10 -> 1.39 ms`，p95 `6.83 -> 2.38 ms`。
- API active CPU 小幅下降：mean active `86.9% -> 84.5%`，p95 `98.0% -> 94.0%`。
- 端到端 latency/QPS 没有稳定提升，原因是该 workload 下 scheduler 仍接近 `100%` 单核，模型侧等待主导。
- Completion endpoint E2E sanity：`optimized_qwen06b_c16_in64_out1_completion_server4c.json`，512/512 OK，p50 `34.95 ms`。

### 二次验证后的收益估算

- 常规 latency：Rust frontend 能直接削掉的 handler 内 OpenAI chat 成本大约 `0.4-0.7 ms/request`，在并发短输出下通常 `<1-3% E2E`，长输出下 `<1%`。
- CPU-heavy tools/schema：本次 4c tools case 的 API pre-dispatch p50 约 `21 ms/request`，其中 validation 约 `16 ms`、template/tokenize 约 `4.6 ms`。这类 workload 上 Rust gateway 有明确优化空间，但按本次 `3202` prompt tokens 的 E2E p50 `~550 ms` 算，latency 收益约 `3-4%`；更可能的收益是降低 CPU/core 和提升稳定 QPS。
- 极短请求最大 QPS：API 单进程在约 `600 req/s` 会接近单核满载。Rust/更轻 HTTP gateway 有可能提升 capacity，但需要 prototype 证明 scheduler/detokenizer/GPU 不会成为新瓶颈。
- 完整 Python frontend -> Rust：不建议现在做。推荐只做可插拔 Rust gateway prototype，成功标准设为同等 workload 下最大稳定 QPS、CPU/core、P99 均明显改善。

## 是否应该做 Rust frontend

当前不建议把 “移植整个 Python frontend 到 Rust” 作为第一优先级，理由：

1. 可直接测的 SGLang frontend dispatch 绝对值太小。`input_ids` 约 `0.1 ms`，128-token text 约 `0.3-0.4 ms`，512-token text 约 `1 ms`。
2. 并发和输出 token 增加后，GPU scheduler/prefill/decode 迅速主导总时延；128 in / 32 out 时 frontend dispatch 占比约 `0.28%`。
3. OpenAI chat 额外 client latency 存在，但它混合了模板后 token 数增加、HTTP 框架、Pydantic 校验、响应构建、客户端并发调度和 batching。Rust 可以减少其中一部分，但不会消掉模型侧和调度侧等待。
4. vLLM Rust frontend 的收益前提可能来自 vLLM 自身 frontend/engine 边界和 workload；SGLang 不能直接套结论，需要用 SGLang 自身分段指标证明 CPU frontend 饱和。

可以考虑 Rust 或更底层实现的条件：

- OpenAI-compatible serving 的请求非常短，例如 `<=128 input / <=1-4 output`，QPS 很高，GPU 侧不是瓶颈。
- 大量 tool/function schema、JSON schema guided decoding、复杂 chat template、reasoning/tool parsing 让 Python/Jinja/Pydantic 占用明显升高。
- 多租户网关场景里 HTTP parse/validate/serialize 成本压满 CPU core。
- 已经通过分段指标看到 OpenAI conversion/template/tokenize 占端到端 latency 的 `>10-20%`，或 API process CPU 接近饱和并拖慢 scheduler feeding。

## 建议下一步

1. 优化 Python 局部热区：
   - cache 或预编译 chat template 相关路径，确认 `apply_chat_template(tokenize=False) + encode` 是否有重复工作。
   - 针对 OpenAI request model validation/serialization 做压测；必要时引入更轻的 request parser 或 `orjson` response path。
   - 对高并发 text path 评估 tokenizer batch encode 是否能覆盖 OpenAI chat 生成的 `input_ids` 路径，避免重复/低效 encode。

2. 扩大 benchmark 矩阵：
   - 等 GPU 空闲后跑 `/models/meta-llama/Llama-3.1-8B-Instruct` 和更大模型，验证 frontend 占比是否继续下降。
   - 增加 tool calls / JSON schema / streaming / logprobs / `n>1` 场景，因为这些更可能放大 OpenAI frontend。
   - 固定 request-rate 做 CPU saturation 曲线，而不是只跑最大并发。

3. 如果仍要做 Rust prototype，建议缩小边界：
   - 先做 OpenAI HTTP + validation + JSON serialization gateway，复用现有 tokenizer/scheduler IPC，测 CPU/core 和 P99。
   - 再评估 Rust tokenizer/chat-template path；不要一开始改 scheduler/model executor。
   - 成功标准应是：同等输出 token 下，OpenAI chat E2E mean/P99 或最大稳定 QPS 明显改善，且 profile 证明减少的是 Python frontend CPU，而不是 benchmark 噪声。

## 本次产物

- Probe 脚本：`benchmark/frontend_overhead_probe.py`
- OpenAI chat timing instrumentation：
  - `python/sglang/srt/entrypoints/openai/serving_base.py`
  - `python/sglang/srt/entrypoints/openai/serving_chat.py`
  - `test/registered/unit/entrypoints/openai/test_serving_chat.py`
- 干净 benchmark：
  - `artifacts/frontend_overhead_probe/clean_qwen06b_c1_in32_out1.json`
  - `artifacts/frontend_overhead_probe/clean_qwen06b_c1_in128_out1.json`
  - `artifacts/frontend_overhead_probe/clean_qwen06b_c1_in512_out1.json`
  - `artifacts/frontend_overhead_probe/clean_qwen06b_c32_in128_out1.json`
  - `artifacts/frontend_overhead_probe/clean_qwen06b_c32_in128_out32.json`
- Profile：
  - `artifacts/frontend_overhead_probe/pyspy_frontend_oai_chat.raw`
  - `artifacts/frontend_overhead_probe/pyspy_frontend_oai_chat_idle.raw`
  - `artifacts/frontend_overhead_probe/profile_traffic_oai_chat_c64_in128_out1.json`
  - `artifacts/frontend_overhead_probe/profile_traffic_oai_chat_idle_c64_in128_out1.json`
- 二次 timed benchmark：
  - `artifacts/frontend_overhead_probe/timed_qwen06b_c1_in128_out1.json`
  - `artifacts/frontend_overhead_probe/timed_qwen06b_c32_in128_out1.json`
  - `artifacts/frontend_overhead_probe/timed_qwen06b_c32_in128_out32.json`
  - `artifacts/frontend_overhead_probe/timed_qwen06b_c64_in128_out1_cpu_probe2.json`
  - `artifacts/frontend_overhead_probe/api_cpu_instant_c64_oai_chat.txt`
- CPU affinity benchmark：
  - `artifacts/frontend_overhead_probe/timed_qwen06b_c64_in128_out1_server32c.json`
  - `artifacts/frontend_overhead_probe/timed_qwen06b_c64_in128_out1_server8c.json`
  - `artifacts/frontend_overhead_probe/timed_qwen06b_c64_in128_out1_server4c.json`
  - `artifacts/frontend_overhead_probe/cpu_instant_c64_oai_chat_server32c.txt`
  - `artifacts/frontend_overhead_probe/cpu_instant_c64_oai_chat_server8c.txt`
  - `artifacts/frontend_overhead_probe/cpu_instant_c64_oai_chat_server4c.txt`
- CPU-heavy tools benchmark：
  - `artifacts/frontend_overhead_probe/sanity_qwen06b_oai_chat_tools_4c.json`
  - `artifacts/frontend_overhead_probe/timed_qwen06b_c16_in64_out1_chat_vs_tools_server4c.json`
  - `artifacts/frontend_overhead_probe/timed_qwen06b_c16_in64_out1_tools_server4c_cpu_probe.json`
  - `artifacts/frontend_overhead_probe/cpu_instant_c16_oai_chat_tools_server4c.txt`
- Optimized benchmark：
  - `artifacts/frontend_overhead_probe/optimized_qwen06b_c16_in64_out1_chat_vs_tools_server4c_order_preserved.json`
  - `artifacts/frontend_overhead_probe/optimized_qwen06b_c64_in128_out1_server4c.json`
  - `artifacts/frontend_overhead_probe/cpu_instant_c64_oai_chat_server4c_optimized.txt`
  - `artifacts/frontend_overhead_probe/optimized_qwen06b_c16_in64_out1_completion_server4c.json`

## Caveats

- 只在 Qwen3-0.6B 单模型上完成干净测量；更大模型应使 frontend 占比下降，但仍需要实测。
- OpenAI endpoint 默认不返回内部 timing；本地新增的 `x-sglang-frontend-timing: 1` 只用于 benchmark，默认不改变 public response。
- spec / speculative decoding 还没有跑。同模型 Qwen3-0.6B 本地没有匹配 draft model；现有 EAGLE3 draft 是 Llama 8B 体系，直接跑会同时改变模型、prompt、GPU prefill/decode 成本，不能和本组 Qwen0.6B 直接横比。
- 本次关闭 radix cache，避免 cache 命中掩盖 prefill；线上 cache 命中很高时，frontend 相对占比可能上升。
- 早期非 `clean_` artifact 来自服务/benchmark 并行干扰，已排除。
