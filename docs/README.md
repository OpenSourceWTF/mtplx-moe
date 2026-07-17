# MTPLX Docs

## Configuration source of truth

Use `mtplx settings list --json` to discover the catalog installed on a machine. The generated
[settings reference](reference/settings.md) is the complete checked-in human-readable catalog; the
generated [migration tables](migration-settings.md) map supported compatibility flags and reviewed
environment aliases. Command operands and mechanical flags remain in [CLI help](cli.md), while
internal and experiment-only switches are classified in the generated
[experiment inventory](experiments/inventory.md).

Do not maintain a second copied flag list in an integration or agent prompt.

## Start

- [Getting started](getting-started.md)
- [Install details](install.md)
- [Quickstart reference](quickstart.md)

## Configure

- [Settings](settings.md)
- [CLI guide](cli.md)
- [Migrating flags and environment variables](migration-settings.md)
- [Generated settings reference](reference/settings.md)

## Operate

- [Model compatibility](model-compatibility.md)
- [Profiles](profiles.md)
- [Server](server.md)
- [API](api.md)
- [Troubleshooting](troubleshooting.md)
- [Experimental SSD-streamed MoE](advanced/ssd-streamed-moe.md)

## Experiment

- [Experiment bundles and lifecycle](experiments.md)
- [Generated experiment inventory](experiments/inventory.md)
- [Benchmarks](benchmarks.md)

## Develop and reference

- [Architecture](architecture.md)
- [Runtime contract](runtime-contract.md)
- [Development](development.md)
- [Research note](research/native-mtp-on-mlx.md)
- [Settings JSON](reference/settings.json)
