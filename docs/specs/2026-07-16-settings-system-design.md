# Hierarchical Settings System Design

Date: 2026-07-16
Parent: [Repository Settings Cleanup](2026-07-16-repository-settings-cleanup-design.md)

## Purpose

Create one typed, inspectable settings system for reusable MTPLX behavior.
Individual configuration flags and `MTPLX_*` variables become adapters rather
than independent sources of parsing and defaults.

## Public Interface

```bash
# Effective settings for the next local run.
mtplx settings show
mtplx settings explain runtime.profile
mtplx settings list --group generation

# Persistent user settings in ~/.mtplx/config.toml.
mtplx settings user show
mtplx settings user set runtime.profile=sustained
mtplx settings user unset runtime.profile

# Running-daemon settings.
mtplx settings live show
mtplx settings live set generation.temperature=0.7

# One process only.
mtplx start --set runtime.profile=sustained \
  --set generation.temperature=0.7
mtplx start --settings ./deployment.toml
```

`--set` may repeat. `--settings` may repeat, with later bundles overriding
earlier bundles. Neither mutates persistent state.

## Schema

The no-MLX `mtplx.settings` package contains:

- `SettingSpec`: canonical name, value type, default provider, domain,
  visibility, lifecycle, description, validation, secrecy, restart behavior,
  and ownership.
- `SettingAlias`: legacy CLI names, environment names, config keys, conversion,
  and deprecation metadata.
- `SettingCatalog`: immutable indexed collection with alias lookup and typo
  suggestions.
- `SettingsResolver`: ordered source merge, validation, constraint application,
  and provenance.
- `ResolvedSettings`: immutable typed values and source records.

Visibility and lifecycle are independent:

| Dimension | Values |
|---|---|
| Visibility | `public`, `advanced`, `experimental`, `internal` |
| Lifecycle | `active`, `deprecated`, `compatibility`, `retired` |

Initial domains are `model`, `runtime`, `generation`, `server`, `memory`,
`cache`, `attention`, `verify`, `streaming`, `integration`, `thermal`,
`benchmark`, and `diagnostics`.

Example canonical names:

```text
model.ref
model.cache_dir
runtime.profile
runtime.mtp.enabled
runtime.mtp.depth
generation.temperature
generation.top_p
generation.top_k
server.host
server.port
cache.session.ssd.enabled
cache.session.ssd.max_size
streaming.experts.memory_limit
verify.compiled.mode
```

## Resolution

For settings-native commands, the value order from strongest to weakest is:

1. Hard safety and model-compatibility constraints.
2. Repeated `--set key=value` entries.
3. Legacy individual CLI configuration flags.
4. Explicit `--settings` bundles, last bundle wins.
5. Legacy `MTPLX_*` environment aliases.
6. Persistent user settings.
7. Model-specific defaults.
8. Selected product profile.
9. Built-in defaults.

Hard constraints do not silently replace a requested value. The resolver keeps
the requested value, effective value, constraint, and explanation in
provenance. Commands fail when a request is unsafe or nonsensical unless the
existing product contract explicitly defines a safe fallback.

Legacy-only invocations first pass through a compatibility adapter that
preserves their characterized behavior. Mixing native and legacy sources uses
the order above and is visible in `settings explain`.

## Persistence and Scopes

User settings remain TOML at `~/.mtplx/config.toml`. Existing flat keys are read
as compatibility aliases. New writes use canonical dotted names grouped into
TOML tables.

```toml
[runtime]
profile = "sustained"

[generation]
temperature = 0.6
top_p = 0.95

[cache.session.ssd]
enabled = true
max_size = "32GiB"
```

Writes are atomic: validate the complete candidate document, write a temporary
file with user-only permissions, flush, and replace. Unknown keys, invalid
types, and settings outside the user-writable visibility/lifecycle set fail
without modifying the file.

Live daemon mutation is separate. `settings live` uses the existing API and
schema metadata marks which values are live-mutable versus restart-required.

## Secrets

Secret values are never displayed, serialized by `settings export`, included in
bundles, or accepted through normal user TOML. The supported representation is
a secret-file reference such as `server.api_key_file`; legacy direct secret
flags and environment values are accepted only by compatibility adapters and
redacted in provenance.

## Migration Boundary

Migration proceeds by domain. A runtime consumer is considered migrated only
when it receives typed settings or uses a catalog-backed accessor. Existing
module-local environment helpers remain temporarily inside an explicit
compatibility allowlist.

Automated checks enforce:

- Every literal `MTPLX_*` production name is registered as a setting alias or
  classified internal process state.
- New direct reads of registered settings outside compatibility code fail.
- Canonical names and aliases are unique.
- Defaults are not duplicated in generated documentation.

## Error Handling

- Unknown key: exit 2 with up to three catalog suggestions.
- Invalid value: exit 2 with canonical type, accepted values, and source.
- Conflicting aliases at the same precedence: exit 2 and name both inputs.
- Read-only/internal setting mutation: exit 2 with its visibility and owner.
- Restart-required live mutation: preserve the existing API rejection and point
  to `settings user set` plus restart.
- Unreadable bundle or user config: fail before model import or process launch.

## Tests

- Catalog uniqueness, parsing, aliases, validation, and redaction.
- Resolution matrix covering every source and mixed-source conflicts.
- Atomic TOML writes and round trips from legacy flat config.
- No-MLX imports for all settings inspection commands.
- Characterization tests for current config/profile/environment precedence.
- Live settings compatibility and restart-required behavior.
- Static audit for unregistered names and unauthorized direct reads.

## Acceptance Criteria

- Runtime configuration for `start`, `ask`, `run`, `chat`, `serve`, and product
  benchmark actions can be expressed through canonical settings.
- Primary help for those commands uses `--set` and `--settings`, not a wall of
  individual reusable configuration flags.
- Existing flags and environment names still produce characterized results.
- Effective values and their sources are inspectable before MLX loads.
- Invalid settings cannot partially mutate user or live state.
