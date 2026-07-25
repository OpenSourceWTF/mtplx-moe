#!/usr/bin/env python3
"""Thermal-controlled cold-prefill pair: baseline vs candidate 12k, interleaved
in one window (B,C,B,C), fresh sessions + fresh SSD dir for candidate, no SSD
for baseline (its default). Settles the cold-prefill pillar cleanly."""
import json
import subprocess
import time
import urllib.request
import sys
import os

SCRATCH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/Users/youssof/Projects/MTPLX-release/mtplx-kvcache-v2-20260703/scripts")
from kvcache_warm_probe_20260703 import rag_prompt  # noqa: E402

ARMS = {
    "baseline": ("/Users/youssof/Projects/MTPLX-release/mtplx", 18173, ""),
    "candidate": ("/Users/youssof/Projects/MTPLX-release/mtplx-kvcache-v2-20260703", 18174,
                  f'--ssd-session-cache on --ssd-session-cache-dir "{SCRATCH}/coldpair-ssd"'),
}
MODEL_PATH = "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed"
MODEL_ID = "mtplx-qwen36-27b-optimized-speed"


def port_pid(port):
    out = subprocess.run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
                         capture_output=True, text=True)
    p = [int(x) for x in out.stdout.split()]
    return p[0] if p else None


def one_cold(arm, tag):
    root, port, extra = ARMS[arm]
    # per-tag SSD dir so candidate colds never warm-hit a prior tag's store
    extra = extra.replace("coldpair-ssd", f"coldpair-ssd-{tag}")
    os.makedirs(f"{SCRATCH}/coldpair-ssd-{tag}", exist_ok=True)
    subprocess.Popen(
        f'cd "{root}" && MTPLX_NAX_VERIFY=1 MTPLX_NAX_M4_IMPL=vk_k '
        f'nohup .venv/bin/mtplx serve --model {MODEL_PATH} --port {port} {extra} '
        f"--warmup-tokens 16 >> /tmp/coldpair_{arm}_{tag}.log 2>&1 &", shell=True)
    for _ in range(90):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2) as r:
                if b'"id"' in r.read(400):
                    break
        except Exception:
            pass
        time.sleep(3)
    body = {"model": MODEL_ID, "messages": [{"role": "user", "content": rag_prompt(12000, "q1")}],
            "max_tokens": 24, "temperature": 0.0, "stream": True, "enable_thinking": False}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json",
                                          "X-MTPLX-Session-ID": f"coldpair-{arm}-{tag}"})
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
    pid = port_pid(port)
    if pid:
        subprocess.run(["kill", str(pid)])
        for _ in range(40):
            if port_pid(port) is None:
                break
            time.sleep(1)
    return {"arm": arm, "tag": tag, "ttft": round(ttft, 2),
            "prefill_tps": round(stats.get("prefill_tok_s") or 0, 1),
            "cache": f"{stats.get('cache_source')}/{stats.get('session_restore_mode')}"}


results = []
for tag, arm in enumerate(["baseline", "candidate", "baseline", "candidate"]):
    r = one_cold(arm, str(tag))
    results.append(r)
    print(json.dumps(r), flush=True)
    time.sleep(20)  # equal cool-gap between arms
json.dump(results, open(f"{SCRATCH}/cold_pair_results.json", "w"), indent=1)
print("COLDPAIR COMPLETE", flush=True)
