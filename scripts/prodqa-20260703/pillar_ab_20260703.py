#!/usr/bin/env python3
"""Pillar A/B: kvcache-v2 candidate vs main baseline (2026-07-03 PM wave).

For each arm: launch serve (Speed 27B, turbo env), measure
  - cold prefill TPS + cold TTFT at 4k/12k/32k (fresh sessions)
  - warm TTFT (same block, new question, same session -> mutated-tail path)
  - decode TPS (3x 512-token generations, thinking off)
then tear down and verify the port is actually free.

Arms alternate A,B,A,B. One summary JSON per pass appended to --output.
"""
import argparse
import json
import subprocess
import time
import urllib.request
import sys
import os

ARMS = {
    "candidate": {
        "root": "/Users/youssof/Projects/MTPLX-release/mtplx-kvcache-v2-20260703",
        "port": 18170,
    },
    "baseline": {
        "root": "/Users/youssof/Projects/MTPLX-release/mtplx",
        "port": 18171,
    },
}
MODEL_PATH = "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed"
MODEL_ID = "mtplx-qwen36-27b-optimized-speed"
SIZES = [4000, 12000, 32000]

sys.path.insert(0, "/Users/youssof/Projects/MTPLX-release/mtplx-kvcache-v2-20260703/scripts")
from kvcache_warm_probe_20260703 import rag_prompt  # noqa: E402


def port_pid(port):
    out = subprocess.run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
                         capture_output=True, text=True)
    pids = [int(x) for x in out.stdout.split()]
    return pids[0] if pids else None


def fans_ok():
    out = subprocess.run(["sudo", "-n", os.path.expanduser("~/.mtplx/bin/thermalforge"), "status"],
                         capture_output=True, text=True)
    try:
        d = json.loads(out.stdout)
        return all(f["actual_rpm"] >= 7000 for f in d["fans"])
    except Exception:
        return False


def launch(root, port, log_path):
    subprocess.Popen(
        f'cd "{root}" && MTPLX_NAX_VERIFY=1 MTPLX_NAX_M4_IMPL=vk_k '
        f'nohup .venv/bin/mtplx serve --model {MODEL_PATH} --port {port} '
        f"--warmup-tokens 16 >> {log_path} 2>&1 &",
        shell=True,
    )


def wait_ready(port, tries=90):
    for _ in range(tries):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2) as r:
                if b'"id"' in r.read(400):
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False


def teardown(port):
    pid = port_pid(port)
    if pid:
        subprocess.run(["kill", str(pid)])
        for _ in range(40):
            if port_pid(port) is None:
                return True
            time.sleep(1)
        stale = port_pid(port)
        if stale:
            subprocess.run(["kill", "-9", str(stale)])
            time.sleep(2)
    return port_pid(port) is None


def chat(port, prompt, session, max_tokens=32, thinking=False):
    body = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.0, "stream": True,
        "enable_thinking": thinking,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json", "X-MTPLX-Session-ID": session},
    )
    started = time.perf_counter()
    ttft = None
    stats = {}
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            payload = json.loads(data)
            for c in payload.get("choices") or []:
                delta = c.get("delta") or {}
                if any(isinstance(delta.get(k), str) and delta.get(k) for k in ("content", "reasoning_content")):
                    if ttft is None:
                        ttft = time.perf_counter() - started
            if isinstance(payload.get("mtplx_stats"), dict):
                stats = payload["mtplx_stats"]
    return {"ttft": ttft, "wall": time.perf_counter() - started, "stats": stats}


def run_pass(arm_name, pass_idx, out):
    arm = ARMS[arm_name]
    port = arm["port"]
    log_path = f"/tmp/ab_{arm_name}_{pass_idx}.server.log"
    assert fans_ok(), "fans not at max before model load"
    launch(arm["root"], port, log_path)
    if not wait_ready(port):
        out.write(json.dumps({"arm": arm_name, "pass": pass_idx, "error": "server never ready"}) + "\n")
        out.flush()
        teardown(port)
        return
    result = {"arm": arm_name, "pass": pass_idx, "ts": time.time(), "sizes": {}}
    for size in SIZES:
        session = f"ab-{arm_name}-{pass_idx}-{size}"
        cold = chat(port, rag_prompt(size, "q1"), session)
        warm = chat(port, rag_prompt(size, "q2"), session)
        s_cold, s_warm = cold["stats"], warm["stats"]
        result["sizes"][size] = {
            "cold_ttft": cold["ttft"],
            "cold_prefill_tps": s_cold.get("prefill_tok_s"),
            "warm_ttft": warm["ttft"],
            "warm_cached": s_warm.get("cached_tokens"),
            "warm_mode": s_warm.get("session_restore_mode"),
            "warm_restore_s": s_warm.get("cache_restore_time_s"),
        }
    decodes = []
    for i in range(3):
        r = chat(port, "Write a detailed essay about the history of navigation at sea.",
                 f"ab-decode-{arm_name}-{pass_idx}-{i}", max_tokens=512)
        decodes.append(r["stats"].get("decode_tok_s"))
    result["decode_tps"] = decodes
    ok = teardown(port)
    result["teardown_port_free"] = ok
    out.write(json.dumps(result) + "\n")
    out.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--order", default="candidate,baseline,candidate,baseline")
    args = ap.parse_args()
    out = open(args.output, "a", encoding="utf-8")
    for idx, arm in enumerate(args.order.split(",")):
        print(f"=== pass {idx} arm {arm} {time.strftime('%H:%M:%S')} ===", flush=True)
        run_pass(arm.strip(), idx, out)
    print("A/B COMPLETE", flush=True)


if __name__ == "__main__":
    main()
