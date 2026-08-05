#!/usr/bin/env python3
"""Escha-W2 native prefill BASELINE benchmark (on-the-fly 2-bit, ~12GB resident).

Loads EschaLabs/Qwen3.6-35B-A3B-Escha-W2 into mlx-lm's qwen3_5_moe architecture:
  - non-expert / shared-expert / embed / lm_head: int8 -> bf16 (per-out-channel scale)
  - MoE experts: eschamoe codes kept PACKED (2-bit resident); each layer's switch_mlp is
    replaced by EschaSwitchGLU which decodes + folds (T128, rin/rout) per forward.

Reports prefill tok/s and PEAK GPU memory across context lengths.
Run under the GPU flock with :8080 paused.
"""
import os, sys, time, glob, json
import numpy as np
import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mtplx.eschamoe import fused_moe_matmul, escha_qmv, t128  # noqa

MODEL_DIR = os.environ.get("ESCHA_DIR") or glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--EschaLabs--Qwen3.6-35B-A3B-Escha-W2/snapshots/*/"))[0]


def _group_layout(ind_np, top_k, E):
    """Vectorized: sort routed slots by expert, pad each expert block to 16 rows.
    Returns tile_expert [Mpad/16], padded_tok [Mpad], erow [Mpad], dst_slot=order [S], valid_prow=ppos [S], S."""
    T = ind_np.shape[0]
    flat_e = ind_np.reshape(-1).astype(np.int64)
    S = flat_e.size
    slot_tok = np.repeat(np.arange(T, dtype=np.int64), top_k)
    order = np.argsort(flat_e, kind="stable")
    se = flat_e[order]
    counts = np.bincount(flat_e, minlength=E)
    ptiles = (counts + 15) // 16
    pstart = np.zeros(E, np.int64)
    if E > 1:
        pstart[1:] = np.cumsum(ptiles[:-1]) * 16
    estart = np.zeros(E, np.int64)
    if E > 1:
        estart[1:] = np.cumsum(counts[:-1])
    ppos = pstart[se] + (np.arange(S, dtype=np.int64) - estart[se])       # padded row per sorted slot
    Mpad = int(ptiles.sum()) * 16
    padded_tok = np.zeros(Mpad, np.int64); padded_tok[ppos] = slot_tok[order]
    erow = np.zeros(Mpad, np.int64); erow[ppos] = se
    tile_expert = np.repeat(np.arange(E, dtype=np.int32), ptiles)          # [Mpad/16]
    return tile_expert, padded_tok, erow, order, ppos, S


class EschaSwitchGLU(nn.Module):
    """Fused eschamoe experts: 2-bit codes stay resident; decode is FUSED into the matmul
    (dense W never formed). Hadamard T128 + rin/rout applied to activations."""
    def __init__(self, gu_code, gu_rin, gu_rout, dn_code, dn_rin, dn_rout, H, I):
        super().__init__()
        self.gu_code, self.gu_rin, self.gu_rout = gu_code, gu_rin, gu_rout
        self.dn_code, self.dn_rin, self.dn_rout = dn_code, dn_rin, dn_rout
        self.H, self.I = H, I

    def __call__(self, x, indices):
        H, I = self.H, self.I
        lead = tuple(indices.shape[:-1]); top_k = indices.shape[-1]
        Tt = 1
        for d in lead:
            Tt *= d
        ind2 = indices.reshape(Tt, top_k)
        x2 = x.reshape(Tt, H)
        S = Tt * top_k
        if S <= 256:
            return self._forward_ondevice(x2, ind2, Tt, top_k, lead)
        ind_np = np.array(ind2)
        tile_expert, padded_tok, erow, dst_slot, valid_prow, S = _group_layout(ind_np, top_k, self.gu_code.shape[0])
        te = mx.array(tile_expert); tok = mx.array(padded_tok); er = mx.array(erow)
        xg = x2[tok]                                                   # [Mpad, H]
        xh = t128(xg * self.gu_rin[er]).astype(mx.float16)
        y_gu = fused_moe_matmul(xh, te, self.gu_code, 2, 2 * I)        # [Mpad, 2I]
        y_gu = t128(y_gu) * self.gu_rout[er]
        gated = (nn.silu(y_gu[:, :I]) * y_gu[:, I:]).astype(mx.float16)
        xhd = t128(gated * self.dn_rin[er]).astype(mx.float16)
        y = fused_moe_matmul(xhd, te, self.dn_code, 3, H)             # [Mpad, H]
        y = (t128(y) * self.dn_rout[er]).astype(mx.float32)
        out = mx.zeros((S, H), mx.float32)
        out[mx.array(dst_slot)] = y[mx.array(valid_prow)]
        return out.reshape(*lead, top_k, H).astype(x.dtype)

    def _forward_ondevice(self, x2, ind2, Tt, top_k, lead):
        """Small S (decode / spec-verify): each routed slot -> its own 16-row tile (row s*16 valid).
        Fully on-device, NO host sync — the decode-tps lever."""
        H, I = self.H, self.I
        flat_e = ind2.reshape(-1)                                  # [S] expert per slot
        flat_tok = mx.repeat(mx.arange(Tt), top_k)                 # [S] slot -> token
        xh = t128(x2[flat_tok], pre=self.gu_rin[flat_e])                     # f32 [S, H]  (rin folded)
        y_gu = escha_qmv(xh, flat_e, self.gu_code, 2, 2 * I)                 # f32 [S, 2I]  QMV
        y_gu = t128(y_gu, post=self.gu_rout[flat_e])                         # rout folded
        gated = nn.silu(y_gu[:, :I]) * y_gu[:, I:]                           # f32 [S, I]
        xhd = t128(gated, pre=self.dn_rin[flat_e])
        y = escha_qmv(xhd, flat_e, self.dn_code, 3, H)                       # f32 [S, H]  QMV
        y = t128(y, post=self.dn_rout[flat_e]).astype(mx.bfloat16)
        return y.reshape(*lead, top_k, H)


def load_model():
    from mlx_lm.models import qwen3_5_moe
    cfg = json.load(open(os.path.join(MODEL_DIR, "config.json")))
    args = qwen3_5_moe.ModelArgs.from_dict(cfg)
    model = qwen3_5_moe.Model(args)
    tcfg = cfg["text_config"]
    H, I, E = tcfg["hidden_size"], tcfg["moe_intermediate_size"], tcfg["num_experts"]

    # gather raw tensors, dequant int8, collect eschamoe per (layer,proj)
    weights, escha = {}, {}
    from safetensors import safe_open
    for shard in sorted(glob.glob(os.path.join(MODEL_DIR, "*.safetensors"))):
        with safe_open(shard, "numpy") as f:
            for k in f.keys():
                if k.startswith("mtp."):
                    continue
                if k.endswith(".weight_scale"):
                    continue
                if k.endswith(".weight_int8"):
                    base = k[: -len(".weight_int8")]
                    w = mx.array(f.get_tensor(k))                              # [out,in] int8
                    s = mx.array(f.get_tensor(base + ".weight_scale"))          # [out]
                    weights[base + ".weight"] = (w.astype(mx.bfloat16) * s[:, None].astype(mx.bfloat16))
                elif ".experts." in k and (".escha_" in k):
                    escha[k] = mx.array(f.get_tensor(k))
                else:
                    weights[k] = mx.array(f.get_tensor(k))
    def g(layer, proj, suf):
        return escha[f"model.language_model.layers.{layer}.mlp.experts.{proj}.escha_{suf}"]

    ONTHEFLY = bool(os.environ.get("ESCHA_ONTHEFLY"))
    if not ONTHEFLY:
        # DENSE (default): decode+fold ALL experts at load -> standard dense expert tensors
        # (fast forward; ~70 GiB resident). Per-layer eval bounds transient decode memory.
        for l in range(tcfg["num_hidden_layers"]):
            pre = f"model.language_model.layers.{l}.mlp.experts"
            Bgu = _fold(decode_expert_weights(g(l, "gate_up_proj", "code"), 2).astype(mx.float32),
                        g(l, "gate_up_proj", "rin"), g(l, "gate_up_proj", "rout"))       # [E,H,2I]
            Bdn = _fold(decode_expert_weights(g(l, "down_proj", "code"), 3).astype(mx.float32),
                        g(l, "down_proj", "rin"), g(l, "down_proj", "rout"))             # [E,I,H]
            weights[f"{pre}.gate_up_proj"] = mx.swapaxes(Bgu, -1, -2).astype(mx.bfloat16)  # [E,2I,H]
            weights[f"{pre}.down_proj"] = mx.swapaxes(Bdn, -1, -2).astype(mx.bfloat16)     # [E,H,I]
            mx.eval(weights[f"{pre}.gate_up_proj"], weights[f"{pre}.down_proj"])

    weights = model.sanitize(weights)   # rename; splits dense experts into switch_mlp
    model.load_weights(list(weights.items()), strict=False)

    if os.environ.get("ESCHA_DRYLOAD"):
        from mlx.utils import tree_flatten
        provided = set(weights.keys())
        params = dict(tree_flatten(model.parameters()))
        missing = [p for p in params if p not in provided and ".switch_mlp." not in p]
        print(f"DRYLOAD: provided={len(provided)} model_params={len(params)} uncovered={len(missing)}")
        for p in missing[:30]:
            print("  MISSING:", p, tuple(params[p].shape))
        raise SystemExit(0)

    if ONTHEFLY:  # keep 2-bit resident, decode per forward (small memory, needs fused kernel to be fast)
        for l in range(tcfg["num_hidden_layers"]):
            model.language_model.model.layers[l].mlp.switch_mlp = EschaSwitchGLU(
                g(l, "gate_up_proj", "code"), g(l, "gate_up_proj", "rin"), g(l, "gate_up_proj", "rout"),
                g(l, "down_proj", "code"), g(l, "down_proj", "rin"), g(l, "down_proj", "rout"), H, I)
    mx.eval(model.parameters())
    return model, args


def main():
    import fcntl
    lockf = open("/tmp/mtplx-gpu-exclusive.lock", "a+")
    try:
        fcntl.flock(lockf.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        print("GPU flock acquired")
    except BlockingIOError:
        print("GPU flock is held by another process — aborting"); sys.exit(1)
    print(f"loading {MODEL_DIR}")
    mx.reset_peak_memory()
    t0 = time.time()
    model, args = load_model()
    model.eval()
    print(f"loaded in {time.time()-t0:.1f}s ; resident peak {mx.get_peak_memory()/2**30:.1f} GiB")

    from pathlib import Path
    from mlx_lm.utils import load_tokenizer
    tok = load_tokenizer(Path(MODEL_DIR))

    # fast 1-forward correctness peek: predict the next token after a real code prompt
    peek = "def is_palindrome(s):\n    \"\"\"Return True if s reads the same forwards and backwards.\"\"\"\n    s = "
    pids = mx.array(tok.encode(peek))[None]
    lg = model(pids, cache=model.make_cache())
    mx.eval(lg)
    top = mx.argsort(-lg[0, -1])[:5].tolist()
    print("correctness peek — prompt tail:", repr(peek[-24:]))
    print("  top-5 next tokens:", [repr(tok.decode([t])) for t in top])

    if os.environ.get("ESCHA_SAMPLE"):  # full generation (slow in baseline: re-decodes all experts/step)
        from mlx_lm import generate
        prompt = ("Write a Python function is_palindrome(s) that returns True if the string s is a "
                  "palindrome, ignoring case, spaces, and punctuation. Include a short docstring.")
        inp = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                      add_generation_prompt=True, tokenize=False)
        ntok = int(os.environ.get("ESCHA_SAMPLE_TOKENS", "64"))
        print("\n===== SAMPLE GENERATION =====\nPROMPT:\n", prompt)
        t0 = time.time()
        text = generate(model, tok, prompt=inp, max_tokens=ntok, verbose=False)
        print(f"\nOUTPUT ({ntok} tok in {time.time()-t0:.1f}s):\n{text}\n")

    if os.environ.get("ESCHA_LENGTHS"):
        lengths = [int(x) for x in os.environ["ESCHA_LENGTHS"].split(",")]
    else:
        lengths = [1024] + list(range(16384, 131072 + 1, 16384))
    CH = int(os.environ.get("ESCHA_CHUNK", "2048"))   # chunked prefill (head_dim 256 -> no flash; bound attn)
    print(f"\n{'ctx':>8} {'prefill_tok/s':>14} {'peak_GiB':>10} {'ms':>10}  (chunk={CH})")
    for L in lengths:
        try:
            x = mx.random.randint(0, args.text_config["vocab_size"], (1, L))
            cache = model.make_cache()
            mx.reset_peak_memory()
            mx.eval(x)
            t0 = time.time()
            for i in range(0, L, CH):
                out = model(x[:, i:i + CH], cache=cache)
                mx.eval(out)
            dt = time.time() - t0
            print(f"{L:>8} {L/dt:>14.1f} {mx.get_peak_memory()/2**30:>10.1f} {dt*1000:>10.1f}", flush=True)
            del cache, out
            mx.clear_cache()
        except Exception as ex:
            print(f"{L:>8}  ERROR: {type(ex).__name__}: {str(ex)[:80]}  (memory wall)", flush=True)
            break

    # decode (autoregressive) tok/s + peak, via mlx-lm's async-pipelined generate_step
    from mlx_lm.generate import generate_step
    prompt = mx.array(tok.encode("def fibonacci(n):\n    "))
    NG = int(os.environ.get("ESCHA_DECODE_TOK", "32"))
    WARM = 16
    it = generate_step(prompt, model, max_tokens=NG + WARM + 1)
    last, _ = next(it)                      # prefill + first token
    for _ in range(WARM):                   # warm up (report: warm-up matters)
        last, _ = next(it)
    mx.eval(last)
    mx.reset_peak_memory()
    t0 = time.time(); n = 0
    for t_, _ in it:
        last = t_; n += 1
        if n >= NG:
            break
    mx.eval(last)
    dt = time.time() - t0
    print(f"\ndecode (AR, pipelined): {n/dt:.1f} tok/s   peak {mx.get_peak_memory()/2**30:.1f} GiB")


if __name__ == "__main__":
    main()
