#!/usr/bin/env python3
"""Corrective pass: candidate TRUE-cold prefill/TTFT with a fresh SSD dir.

The 4-pass A/B measured candidate 'cold' against the soak-persisted SSD store
(default-on) — honest product behavior, wrong lane for the engine cold-prefill
pillar. This pass launches the candidate with --ssd-session-cache-dir pointed
at an empty scratch dir so cold is cold, then also records the RAM-tier warm.
"""
import json
import subprocess
import time
import urllib.request
import sys
import os

ROOT = "/Users/youssof/Projects/MTPLX-release/mtplx-kvcache-v2-20260703"
PORT = 18172
SIZES = [4000, 12000, 32000]
MODEL_PATH = "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed"
MODEL_ID = "mtplx-qwen36-27b-optimized-speed"
SCRATCH = os.path.dirname(os.path.abspath(__file__))
FRESH_SSD = os.path.join(SCRATCH, "truecold-ssd")

sys.path.insert(0, os.path.join(ROOT, "scripts"))
from kvcache_warm_probe_20260703 import rag_prompt  # noqa: E402


def port_pid(port):
    out = subprocess.run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
                         capture_output=True, text=True)
    pids = [int(x) for x in out.stdout.split()]
    return pids[0] if pids else None


def chat(prompt, session, max_tokens=32):
    body = {"model": MODEL_ID, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.0, "stream": True,
            "enable_thinking": False}
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json", "X-MTPLX-Session-ID": session})
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
            p = json.loads(data)
            for c in p.get("choices") or []:
                delta = c.get("delta") or {}
                if any(isinstance(delta.get(k), str) and delta.get(k) for k in ("content", "reasoning_content")):
                    if ttft is None:
                        ttft = time.perf_counter() - started
            if isinstance(p.get("mtplx_stats"), dict):
                stats = p["mtplx_stats"]
    return ttft, stats


os.makedirs(FRESH_SSD, exist_ok=True)
subprocess.Popen(
    f'cd "{ROOT}" && MTPLX_NAX_VERIFY=1 MTPLX_NAX_M4_IMPL=vk_k '
    f'nohup .venv/bin/mtplx serve --model {MODEL_PATH} --port {PORT} '
    f'--ssd-session-cache on --ssd-session-cache-dir "{FRESH_SSD}" '
    f"--warmup-tokens 16 >> /tmp/ab_truecold.server.log 2>&1 &",
    shell=True)
for _ in range(90):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=2) as r:
            if b'"id"' in r.read(400):
                break
    except Exception:
        pass
    time.sleep(3)

result = {}
for size in SIZES:
    session = f"truecold-{size}"
    c_ttft, c_stats = chat(rag_prompt(size, "q1"), session)
    w_ttft, w_stats = chat(rag_prompt(size, "q2"), session)
    result[size] = {
        "cold_ttft": round(c_ttft, 3),
        "cold_prefill_tps": round(c_stats.get("prefill_tok_s") or 0, 1),
        "cold_cache": f"{c_stats.get('cache_source')}/{c_stats.get('session_restore_mode')}",
        "warm_ttft": round(w_ttft, 3),
        "warm_mode": w_stats.get("session_restore_mode"),
        "warm_cached": w_stats.get("cached_tokens"),
        "warm_restore_s": w_stats.get("cache_restore_time_s"),
    }
    print(json.dumps({size: result[size]}), flush=True)

pid = port_pid(PORT)
if pid:
    subprocess.run(["kill", str(pid)])
    for _ in range(40):
        if port_pid(PORT) is None:
            break
        time.sleep(1)
print("TEARDOWN port free:", port_pid(PORT) is None, flush=True)
json.dump(result, open(os.path.join(SCRATCH, "truecold_results.json"), "w"), indent=1)
print("TRUECOLD COMPLETE", flush=True)
