"""Quick e2e test: adaptive spec + Standalone (inherits EAGLE v1)."""
import json, os, sys, tempfile
import requests

sys.path.insert(0, "/root/code/origin_sglang/current/adaptive_spec/python")
os.environ["SGLANG_JIT_DEEPGEMM_PRECOMPILE"] = "0"
os.environ["SGLANG_ENABLE_JIT_DEEPGEMM"] = "0"

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import popen_launch_server

TARGET = "/models/Qwen/Qwen2.5-7B-Instruct"
DRAFT = "/models/Qwen/Qwen2.5-0.5B-Instruct"
BASE_URL = "http://127.0.0.1:36102"

cfg = {"candidate_steps": [1, 3], "ema_alpha": 1.0, "warmup_batches": 1, "update_interval": 1, "up_hysteresis": 0.0}
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

try:
    print("\n=== Test 1: Basic generation ===")
    r = requests.post(BASE_URL + "/generate", json={
        "text": "What is 1+1?",
        "sampling_params": {"temperature": 0, "max_new_tokens": 32}
    }, timeout=60)
    assert r.status_code == 200, f"Generate failed: {r.text}"
    print(f"  Response: {r.json()['text'][:80]}...")

    print("\n=== Test 2: Drive upshift ===")
    HIGH = "Repeat the word HELLO exactly 100 times, one per line."
    for i in range(5):
        r = requests.post(BASE_URL + "/generate", json={
            "text": HIGH,
            "sampling_params": {"temperature": 0, "max_new_tokens": 128, "ignore_eos": True}
        }, timeout=60)
        assert r.status_code == 200
        info = requests.get(BASE_URL + "/server_info", timeout=10).json()
        steps = info["internal_states"][0]["speculative_num_steps"]
        print(f"  Attempt {i+1}: speculative_num_steps={steps}")
        if steps == 3:
            break

    print("\n=== Test 3: Drive downshift ===")
    LOW = "Write a deeply philosophical essay about consciousness and free will."
    for i in range(8):
        r = requests.post(BASE_URL + "/generate", json={
            "text": LOW,
            "sampling_params": {"temperature": 0.7, "max_new_tokens": 128, "ignore_eos": True}
        }, timeout=60)
        assert r.status_code == 200
        info = requests.get(BASE_URL + "/server_info", timeout=10).json()
        steps = info["internal_states"][0]["speculative_num_steps"]
        print(f"  Attempt {i+1}: speculative_num_steps={steps}")
        if steps == 1:
            break

    print("\n=== STANDALONE ADAPTIVE TEST PASSED ===")

except Exception as e:
    print(f"\n=== STANDALONE ADAPTIVE TEST FAILED: {e} ===")
    import traceback; traceback.print_exc()
    sys.exit(1)
finally:
    kill_process_tree(process.pid)
    os.unlink(cfg_file.name)
