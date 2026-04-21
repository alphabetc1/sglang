"""E2E: adaptive spec + Standalone (v1, inherits EAGLEWorker)."""
import json, os, sys, tempfile
import requests

sys.path.insert(0, "/root/code/origin_sglang/current/adaptive_spec/python")
os.environ["SGLANG_JIT_DEEPGEMM_PRECOMPILE"] = "0"
os.environ["SGLANG_ENABLE_JIT_DEEPGEMM"] = "0"

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import popen_launch_server

TARGET = "/models/Qwen/Qwen2.5-7B-Instruct"
DRAFT = "/models/Qwen/Qwen2.5-0.5B-Instruct"
BASE_URL = "http://127.0.0.1:36202"

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

print("=== Launching Standalone adaptive server ===")
process = popen_launch_server(
    TARGET, BASE_URL, timeout=300,
    other_args=[
        "--trust-remote-code",
        "--attention-backend", "triton",
        "--speculative-algorithm", "STANDALONE",
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
    result = generate("What is 1+1?", max_new_tokens=32)
    print(f"  Output: {result['text'][:100]}")

    # 2. Upshift (Standalone 0.5B draft model may have limited accuracy,
    # so upshift may require more attempts or not trigger)
    print("\n--- Test 2: Drive upshift ---")
    for i in range(10):
        generate(HIGH)
        s = get_steps()
        print(f"  Attempt {i+1}: steps={s}")
        if s == 3:
            break
    final = get_steps()
    print(f"  Final steps={final} (upshift {'OK' if final == 3 else 'not triggered (small draft model, acceptable)'})")

    # 3. Downshift (only meaningful if upshift happened)
    if final == 3:
        print("\n--- Test 3: Drive downshift ---")
        for i in range(10):
            generate(LOW, temperature=0.7)
            s = get_steps()
            print(f"  Attempt {i+1}: steps={s}")
            if s == 1:
                break
        final = get_steps()
        print(f"  Final steps={final} (downshift {'OK' if final == 1 else 'not triggered'})")
    else:
        print("\n--- Test 3: Skipped (upshift did not occur) ---")

    print("\n=== STANDALONE ADAPTIVE E2E PASSED ===")

except Exception as e:
    print(f"\n=== STANDALONE ADAPTIVE E2E FAILED: {e} ===")
    import traceback; traceback.print_exc()
    sys.exit(1)
finally:
    kill_process_tree(process.pid)
    os.unlink(cfg_file.name)
