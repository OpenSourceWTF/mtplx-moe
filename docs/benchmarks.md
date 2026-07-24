# Benchmarks

## Realistic programming prompt ladder

`mtplx bench prefill-ladder` defaults to exact formatted prompt sizes of
1,024, 2,048, 4,096, 8,192, and 16,384 tokens. The `coding-agent` policy
assembles deterministic Python repository context from source, tests,
configuration, documentation, diagnostics, and review notes, then preserves a
complete implementation request at the tail.

Exact size is measured with the active tokenizer after applying the selected
`--prompt-format` (`chat` by default). The result metadata records policy
`realistic_programming_v1`, content hashes, requested and actual token counts,
and tail preservation. Results using this policy are not prompt-identical to
historical `coding_agent_tail_v2` runs. Use explicit `--contexts` for other
sizes or `--prompt-style legacy-repeat` only for historical diagnostics.

Every benchmark claim should record:

- hardware and RAM
- macOS version
- model and quantization
- sampler settings
- prompt suite
- token count
- profile
- fan mode
- date and commit

Separate cold headline runs from sustained no-fan runs and fan-controlled diagnostics.

```bash
mtplx bench run --suite cold-long-code-192 --max-tokens 192 --strict-cold
mtplx bench run --suite flappy --max-tokens 10000 --no-fanmax
```
