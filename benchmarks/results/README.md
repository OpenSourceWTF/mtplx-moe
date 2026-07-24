# Benchmark Results

This directory contains curated, human-readable benchmark summaries. Put all
raw machine-generated artifacts, regardless of size or file type, under the
ignored `benchmarks/raw/<benchmark>/<run-id>/` tree. Never force-add raw
artifacts. The JSON files already tracked here are legacy summaries, not a
precedent for adding new raw output.

Each curated summary must include enough reproducibility metadata to identify:

- hardware and RAM
- macOS version
- model and quantization
- sampler settings
- prompt suite
- token count
- profile
- fan mode
- UTC run date and commit SHA
- benchmark command and relevant environment/configuration values
- repeat count and the aggregation method used

Summaries must stand on their own: do not link to files under the ignored
`benchmarks/raw/` tree. If raw evidence must be shared, publish it in a durable
external location and link to that location instead.
