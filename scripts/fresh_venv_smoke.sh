#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMPDIR="${TMPDIR:-/tmp}"
WORKDIR="$(mktemp -d "$TMPDIR/mtplx-fresh-venv.XXXXXX")"
VENV="$WORKDIR/.venv"
MODEL_DIR="$WORKDIR/non-mtp-model"

cleanup() {
  rm -rf "$WORKDIR"
}
trap cleanup EXIT

python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip >/dev/null

shopt -s nullglob
wheels=("$ROOT"/dist/*.whl)
shopt -u nullglob

if [[ "${#wheels[@]}" -eq 0 ]]; then
  echo "fresh_venv_smoke: no wheel found in $ROOT/dist" >&2
  echo "Run: python -m build" >&2
  exit 2
fi

"$VENV/bin/python" -m pip install --no-deps "${wheels[0]}" >/dev/null

mkdir -p "$MODEL_DIR"
printf '{"model_type":"llama"}\n' > "$MODEL_DIR/config.json"

"$VENV/bin/mtplx" --help >/dev/null
"$VENV/bin/mtplx" doctor --json >/dev/null
set +e
INSPECT_JSON="$("$VENV/bin/mtplx" inspect "$MODEL_DIR" --json)"
INSPECT_STATUS=$?
set -e
# 2.10.0 contract: constructable models always run, MTP is an accelerator
# and not a gate. In this no-deps venv there is no bundled mlx-lm module
# either, so a plain llama config classifies as a capability gap (exit 4),
# not the old hard refusal (exit 2).
if [ "$INSPECT_STATUS" -ne 4 ]; then
  echo "expected no-MTP inspect in a no-mlx-lm venv to exit 4, got $INSPECT_STATUS" >&2
  exit 1
fi
if ! printf '%s' "$INSPECT_JSON" | grep -q '"exit_code": 4'; then
  echo "inspect payload does not carry the capability-gap exit_code" >&2
  exit 1
fi
"$VENV/bin/mtplx" init --dry-run --json --config "$WORKDIR/config.toml" >/dev/null

echo "fresh_venv_smoke: passed no-MLX CLI survival checks"
