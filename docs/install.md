# Install

See [INSTALL.md](../INSTALL.md) for the short path.

MTPLX is Apple-Silicon-first:

- macOS 14.0 or newer
- native arm64 Python 3.11 or newer
- the `mlx` and `mlx-lm` versions declared by the installed MTPLX package
- enough unified memory and disk for the selected model/profile, checked by `mtplx doctor`

The first-run default model is `Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed`. The quantized 27B
flagships select the Turbo product profile; other models normally select Sustained
(`--profile sustained`). `stable` remains a conservative compatibility value, while
`performance-cold` is the short-context Burst lane. See [Profiles](profiles.md).

Do not install model weights into the source checkout. Use the MTPLX model cache or a Hugging Face cache.
