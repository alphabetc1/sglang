# GLM-4.7 NextN Draft Quantization Regression

## Problem

On `GLM-4.7-FP8` with EAGLE enabled, the speculative draft model drops the checkpoint's auto-detected FP8 quantization and falls back to an unquantized draft path. Draft and target then diverge almost immediately, and `Accept length` collapses to about `1.0`.

## Root Cause

`python/sglang/srt/models/glm4_moe_nextn.py` decides whether to keep the draft `quant_config` by reading `server_args.speculative_draft_model_quantization`.

That is the wrong source of truth for this model:

- when the user does not explicitly pass `--speculative-draft-model-quantization`
- `server_args.speculative_draft_model_quantization` stays `None`
- but the loader has already auto-detected `compressed-tensors` from the checkpoint and passed a valid `quant_config`

The old logic treated `None` as "do not quantize draft" and discarded the valid loader-provided config.

## Reproduce

1. Check out `main` or commit `454228e07`.
2. Launch the server:

```bash
python3 -m sglang.launch_server \
  --model /models/ZhipuAI/GLM-4.7-FP8/ \
  --host 0.0.0.0 \
  --mem-fraction-static 0.55 --page-size 64 \
  --port 7000 \
  --reasoning-parser glm45 \
  --served-model-name GLM-4.7 \
  --speculative-algorithm EAGLE \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 8 \
  --speculative-num-steps 7 \
  --tool-call-parser glm47 \
  --tp 8 \
  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 64}' \
  --trust-remote-code
```

3. Run the benchmark:

```bash
python3 -m sglang.bench_serving \
  --backend sglang-oai \
  --dataset-name random \
  --num-prompts 1 \
  --model /models/ZhipuAI/GLM-4.7-FP8/ \
  --dataset-path /root/code/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
  --random-input-len 150000 \
  --random-output-len 1024 \
  --random-range-ratio 0.3 \
  --max-concurrency 1 \
  --warmup-requests 0 \
  --seed 77 \
  --host 127.0.0.1 --port 7000
```

4. Observe `Accept length: 1.00`.

Validation on this host additionally used:

```bash
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0
```

Those env vars only bypass local environment gates and are not part of the code bug.

## Fix

- Preserve the loader-provided `quant_config` by default.
- Only force the draft model to unquantized mode when the user explicitly passes `--speculative-draft-model-quantization unquant`.
- Add an explicit server-args flag to distinguish "unspecified" from "explicit unquant".

Files:

- `python/sglang/srt/models/glm4_moe_nextn.py`
- `python/sglang/srt/server_args.py`
- `test/registered/unit/models/test_glm4_moe_nextn.py`
- `test/registered/unit/server_args/test_server_args.py`

## Benefit

Same workload, same machine:

- `main@454228e07`: `Accept length 1.00`, `TPOT 35.69 ms`, `E2E 68.74 s`
- this branch: `Accept length 6.83`, `TPOT 5.18 ms`, `E2E 40.84 s`

This is the primary regression fix for the original report.
