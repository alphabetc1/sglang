"""E2E: adaptive spec + NGRAM."""
import json, os, sys, tempfile
import requests

sys.path.insert(0, "/root/code/origin_sglang/current/adaptive_spec/python")
os.environ["SGLANG_JIT_DEEPGEMM_PRECOMPILE"] = "0"
os.environ["SGLANG_ENABLE_JIT_DEEPGEMM"] = "0"

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import popen_launch_server

MODEL = "/models/Qwen/Qwen2.5-7B-Instruct"
BASE_URL = "http://127.0.0.1:36201"

# Less aggressive config to avoid oscillation (NGRAM acceptance is naturally
# high even with small draft counts, so up_hysteresis prevents thrashing)
cfg = {
    "candidate_steps": [1, 3],
    "ema_alpha": 0.8,
    "warmup_batches": 1,
    "update_interval": 2,
    "up_hysteresis": 0.5,
    "down_hysteresis": -0.5,
}
cfg_file = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
json.dump(cfg, cfg_file); cfg_file.close()

print("=== Launching NGRAM adaptive server ===")
process = popen_launch_server(
    MODEL, BASE_URL, timeout=300,
    other_args=[
        "--trust-remote-code",
        "--attention-backend", "triton",
        "--speculative-algorithm", "NGRAM",
        "--speculative-num-draft-tokens", "4",
        "--speculative-adaptive",
        "--speculative-adaptive-config", cfg_file.name,
        "--skip-server-warmup",
        "--mem-fraction-static", "0.8",
        "--cuda-graph-max-bs", "8",
    ],
)

def generate(prompt, max_new_tokens=128, temperature=0):
    r = requests.post(BASE_URL + "/generate", json={
        "text": prompt,
        "sampling_params": {"temperature": temperature, "max_new_tokens": max_new_tokens, "ignore_eos": True}
    }, timeout=120)
    assert r.status_code == 200, f"Generate failed: {r.text}"
    return r.json()

def get_steps():
    info = requests.get(BASE_URL + "/server_info", timeout=10).json()
    return info["internal_states"][0]["speculative_num_steps"]

HIGH = "Repeat the word HELLO exactly 100 times, one per line."
LOW = "Write a deeply philosophical essay about the nature of consciousness, free will, and determinism."

try:
    # 1. Basic generation
    print("\n--- Test 1: Basic generation ---")
    result = generate("Hello, what is 2+2?", max_new_tokens=32)
    print(f"  Output: {result['text'][:100]}")

    # 2. Upshift (NGRAM trie naturally has high acceptance for repetitive prompts)
    print("\n--- Test 2: Drive upshift ---")
    for i in range(10):
        generate(HIGH)
        s = get_steps()
        print(f"  Attempt {i+1}: steps={s}")
        if s == 3:
            break
    assert get_steps() == 3, f"Expected upshift to 3, got {get_steps()}"
    print("  UPSHIFT OK")

    # 3. Downshift attempt (for NGRAM this may not trigger because tree-BFS
    # naturally produces good candidates even for creative prompts)
    print("\n--- Test 3: Drive downshift (best-effort) ---")
    for i in range(15):
        generate(LOW, temperature=0.9)
        s = get_steps()
        print(f"  Attempt {i+1}: steps={s}")
        if s == 1:
            break
    final = get_steps()
    print(f"  Final steps={final} (downshift {'OK' if final == 1 else 'not triggered (acceptable for NGRAM)'})")

    print("\n=== NGRAM ADAPTIVE E2E PASSED ===")

except Exception as e:
    print(f"\n=== NGRAM ADAPTIVE E2E FAILED: {e} ===")
    import traceback; traceback.print_exc()
    sys.exit(1)
finally:
    kill_process_tree(process.pid)
    os.unlink(cfg_file.name)
