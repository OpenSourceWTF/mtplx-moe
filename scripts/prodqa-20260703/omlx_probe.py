#!/usr/bin/env python3
"""oMLX head-to-head on OUR Speed artifact: cold/warm at 4k/12k, decode,
then (orchestrated by caller) a restart-warm leg against its SSD cache."""
import json
import sys
import time
import urllib.request

PORT = 18310
KEY = "mtplx-qa"
MODEL = "qwen36-27b-speed"
sys.path.insert(0, "/Users/youssof/Projects/MTPLX-release/mtplx-kvcache-v2-20260703/scripts")
from kvcache_warm_probe_20260703 import rag_prompt  # noqa: E402


def chat(prompt, max_tokens=32):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.0, "stream": True,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    started = time.perf_counter()
    ttft = None
    usage = {}
    ntok = 0
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
                if isinstance(delta.get("content"), str) and delta["content"]:
                    ntok += 1
                    if ttft is None:
                        ttft = time.perf_counter() - started
            if isinstance(p.get("usage"), dict):
                usage = p["usage"]
    wall = time.perf_counter() - started
    return {"ttft": None if ttft is None else round(ttft, 3), "wall": round(wall, 2),
            "usage": usage, "stream_tokens": ntok}


mode = sys.argv[1] if len(sys.argv) > 1 else "full"
if mode == "full":
    out = {}
    for size in (4000, 12000):
        cold = chat(rag_prompt(size, "q1"))
        warm_exact = chat(rag_prompt(size, "q1"))
        warm_rag = chat(rag_prompt(size, "q2"))
        out[size] = {"cold": cold, "warm_exact": warm_exact, "warm_rag": warm_rag}
        print(json.dumps({size: out[size]}), flush=True)
    t0 = time.perf_counter()
    d = chat("Write a detailed essay about the history of navigation at sea.", max_tokens=512)
    d["decode_tok_s_est"] = round(d["stream_tokens"] / (d["wall"] - (d["ttft"] or 0)), 1) if d["wall"] > (d["ttft"] or 0) else None
    print(json.dumps({"decode": d}), flush=True)
else:  # restart-warm leg: repeat the 12k q1 in a fresh process
    r = chat(rag_prompt(12000, "q1"))
    print(json.dumps({"restart_warm_12k": r}), flush=True)
print("OMLX PROBE DONE", flush=True)
