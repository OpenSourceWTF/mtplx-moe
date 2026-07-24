# Architectures

Architecture support is registry-driven and may differ by installed release. Inspect the current
matrix without loading a model:

```bash
mtplx model architectures
mtplx model architectures --json
```

`mtplx inspect MODEL --json` classifies an individual artifact as verified,
architecture-compatible-but-unverified, incompatible, or lacking MTP heads. The registry reports the
backend lifecycle, supported capabilities, and an actionable reason when a model cannot run; do not
maintain a static family list in an integration.
