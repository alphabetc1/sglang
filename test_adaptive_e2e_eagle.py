"""E2E: adaptive spec + EAGLE (v1, single-layer)."""
import json, os, sys, tempfile
import requests

sys.path.insert(0, "/root/code/origin_sglang/current/adaptive_spec/python")
os.environ["SGLANG_JIT_DEEPGEMM_PRECOMPILE"] = "0"
os.environ["SGLANG_ENABLE_JIT_DEEPGEMM"] = "0"

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import popen_launch_server

TARGET = "/models/meta-llama/Llama-2-7b-chat-hf"
DRAFT = "/models/lmsys/sglang-EAGLE-llama2-chat-7B"
BASE_URL = "http://127.0.0.1:36200"

cfg = {
    "candidate_steps": [1, 3],
    "ema_alpha": 1.0,
    "warmup_batches": 1,
    "update_interval": 1,
    "up_hysteresis": 0.0,
    "down_hysteresis": -0.25,
}
cfg_file = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
json.dump(cfg, cfg_file); cfg_file.close()

print("=== Launching EAGLE adaptive server ===")
process = popen_launch_server(
    TARGET, BASE_URL, timeout=300,
    other_args=[
        "--trust-remote-code",
        "--attention-backend", "triton",
        "--speculative-algorithm", "EAGLE",
        "--speculative-draft-model-path", DRAFT,
        "--speculative-num-steps", "1",
        "--speculative-eagle-topk", "1",
        "--speculative-num-draft-tokens", "2",
        "--speculative-adaptive",
        "--speculative-adaptive-config", cfg_file.name,
        "--skip-server-warmup",
        "--mem-fraction-static", "0.7",
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
    result = generate("What is the capital of France?", max_new_tokens=32)
    print(f"  Output: {result['text'][:100]}")

    # 2. Upshift
    print("\n--- Test 2: Drive upshift ---")
    for i in range(8):
        generate(HIGH)
        s = get_steps()
        print(f"  Attempt {i+1}: steps={s}")
        if s == 3:
            break
    assert get_steps() == 3, f"Expected upshift to 3, got {get_steps()}"
    print("  UPSHIFT OK")

    # 3. Downshift
    print("\n--- Test 3: Drive downshift ---")
    for i in range(10):
        generate(LOW, temperature=0.7)
        s = get_steps()
        print(f"  Attempt {i+1}: steps={s}")
        if s == 1:
            break
    assert get_steps() == 1, f"Expected downshift to 1, got {get_steps()}"
    print("  DOWNSHIFT OK")

    print("\n=== EAGLE ADAPTIVE E2E PASSED ===")

except Exception as e:
    print(f"\n=== EAGLE ADAPTIVE E2E FAILED: {e} ===")
    import traceback; traceback.print_exc()
    sys.exit(1)
finally:
    kill_process_tree(process.pid)
    os.unlink(cfg_file.name)
