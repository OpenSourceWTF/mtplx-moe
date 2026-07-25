#!/usr/bin/env python3
"""Decompose the OpenCode short-turn decode gap.

Against ONE daemon (caller launches it), fire controlled request shapes and
print the decode decomposition from /v1/mtplx/snapshot latest:
  decode_tok_s, accepted_by_depth, bonus, verify_calls, verify_time_s,
  draft_time_s, per-verify-call ms, accept ratio.

Shapes:
  bare            user: "how are you"                        (thinking on)
  bare_nothink    same, enable_thinking off
  bare_128        "Write a haiku about the sea." max_tokens 128 forced-ish
  octools         OpenCode-like: big system prompt + 11 tools + "how are you"
  octools_128     same shape, longer answer request
"""
import json
import sys
import time
import urllib.request

PORT = int(sys.argv[1])
LABEL = sys.argv[2] if len(sys.argv) > 2 else "daemon"
MODEL = "mtplx-qwen36-27b-optimized-speed"

# A faithful-enough OpenCode-ish system prompt (~2.7k tokens) + tool defs.
OC_SYSTEM = (
    "You are OpenCode, an agentic coding assistant working inside the user's "
    "repository. You take actions via tools. Follow the rules exactly.\n\n"
    + "\n".join(
        f"Rule {i}: " + ("Be precise, minimal, and verify every edit against the file state "
        "before and after. Never fabricate file contents; always read before writing. "
        "Prefer small diffs. Preserve style. Use ripgrep for search. ") * 3
        for i in range(1, 41)
    )
)

def tool(name, desc):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string", "description": "file path"},
                "query": {"type": "string", "description": "search query"},
                "content": {"type": "string", "description": "content"}},
                "required": []}}}

OC_TOOLS = [tool(n, f"{n} tool for the workspace: " + "operates on files and returns structured results. " * 3)
            for n in ("bash", "read", "write", "edit", "grep", "glob", "ls", "webfetch",
                      "task", "todowrite", "todoread")]


def fire(name, messages, tools=None, max_tokens=256, thinking=True):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens,
            "temperature": 0.6, "top_p": 0.95, "stream": True,
            "enable_thinking": thinking}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json",
                 "X-MTPLX-Session-ID": f"gap-{name}-{PORT}"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            pass
    wall = time.perf_counter() - t0
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/mtplx/snapshot", timeout=5) as r:
        lat = json.loads(r.read().decode()).get("latest") or {}
    abd = lat.get("accepted_by_depth") or []
    verify_calls = lat.get("verify_calls") or 0
    verify_t = lat.get("verify_time_s") or 0.0
    draft_t = lat.get("draft_time_s") or 0.0
    comp = lat.get("completion_tokens") or 0
    dec = lat.get("decode_tok_s") or 0.0
    dec_el = lat.get("decode_elapsed_s") or 0.0
    bonus = lat.get("bonus_tokens") or 0
    row = {
        "shape": name, "daemon": LABEL,
        "decode_tok_s": round(dec, 1), "completion": comp,
        "decode_elapsed_s": round(dec_el, 2),
        "verify_calls": verify_calls,
        "tokens_per_verify": round(comp / verify_calls, 2) if verify_calls else None,
        "accepted_by_depth": abd, "bonus": bonus,
        "verify_time_s": round(verify_t, 2),
        "verify_ms_per_call": round(1000 * verify_t / verify_calls, 1) if verify_calls else None,
        "draft_time_s": round(draft_t, 2),
        "draft_share": round(draft_t / dec_el, 2) if dec_el else None,
        "verify_share": round(verify_t / dec_el, 2) if dec_el else None,
        "prompt_tokens": lat.get("prompt_tokens"),
        "ttft": round(lat.get("ttft_s") or 0, 2),
        "thinking": lat.get("request_enable_thinking"),
        "wall": round(wall, 2),
    }
    print(json.dumps(row), flush=True)
    time.sleep(2)
    return row


greet = [{"role": "user", "content": "how are you"}]
haiku = [{"role": "user", "content": "Write six haikus about the sea, numbered."}]
oc_greet = [{"role": "system", "content": OC_SYSTEM}, {"role": "user", "content": "how are you"}]
oc_haiku = [{"role": "system", "content": OC_SYSTEM},
            {"role": "user", "content": "Write six haikus about the sea, numbered."}]

fire("bare_greet", greet, max_tokens=256)
fire("bare_greet_nothink", greet, max_tokens=256, thinking=False)
fire("bare_haiku128", haiku, max_tokens=160, thinking=False)
fire("octools_greet", oc_greet, tools=OC_TOOLS, max_tokens=256)
fire("octools_greet_nothink", oc_greet, tools=OC_TOOLS, max_tokens=256, thinking=False)
fire("octools_haiku128", oc_haiku, tools=OC_TOOLS, max_tokens=160, thinking=False)
fire("ocsys_no_tools_greet", oc_greet, max_tokens=256)
print("MATRIX DONE", flush=True)
