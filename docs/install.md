# Install

See [INSTALL.md](../INSTALL.md) for the short path.

MTPLX v0.1 is Apple-Silicon-first:

- macOS 14.0 or newer
- native arm64 Python 3.10 or newer
- `python3 -m pip install mlx` in that same environment
- enough unified memory and disk for the selected model/profile, checked by `mtplx doctor`

The first-run default model is `Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed`. The quantized 27B flagships (Optimized-Speed, Optimized-Quality, and the legacy Optimized hybrid) launch on the Turbo profile by default — the same NAX verify-kernel + compiled-verify fast path the macOS app uses; every other model defaults to Sustained (`--profile sustained`). `stable` remains available as the conservative compatibility alias, and Burst is available explicitly as `--profile performance-cold --max` for short-context benchmark runs.

Do not install model weights into the source checkout. Use the MTPLX model cache or a Hugging Face cache.
