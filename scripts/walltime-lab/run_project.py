#!/usr/bin/env python3
"""Wall-time campaign runner: drive OpenCode against an MTPLX daemon on a
fresh copy of a project spec, and produce a per-run report.

One run = one fresh project dir + one `opencode run` invocation with the
a realistic user prompt shape, timed end-to-end, followed by telemetry collection
(server snapshot + optional --request-log-jsonl slice), an acceptance check
(the spec's `node --test tests/` gate), and a session_forensics report.

Usage:
  python3 run_project.py --project pomodoro-cli --arm F \
      [--port 8001] [--request-log PATH] [--label note]

Outputs under runs/<ts>-<project>-arm<ARM>/:
  workspace/        the project dir OpenCode built
  events.jsonl      raw `opencode run --format json` event stream
  snapshot.json     server snapshot taken at completion
  requests.jsonl    slice of the server request log covering the run
  forensics.txt     session_forensics output for the run window
  summary.json      wall time, request stats, acceptance verdict
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

LAB = Path(__file__).resolve().parent
FORENSICS = LAB.parent / "session_forensics.py"
PROMPT = (
    "i want you to read {spec} in this workspace and build it all now. "
    "adhere to it well."
)


def _fetch(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=15) as response:
        return json.loads(response.read().decode())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--request-log", default=None)
    parser.add_argument("--model", default="mtplx/v2exp-s-w8head")
    parser.add_argument("--label", default="")
    parser.add_argument("--timeout-s", type=int, default=3600)
    args = parser.parse_args()

    spec_source = LAB / "specs" / f"{args.project}-spec.md"
    if not spec_source.exists():
        print(f"no spec: {spec_source}", file=sys.stderr)
        return 2
    stamp = time.strftime("%H%M%S")
    run_dir = LAB / "runs" / f"{stamp}-{args.project}-arm{args.arm}"
    run_dir.mkdir(parents=True)
    # The workspace is a STANDALONE dir directly under ~/Projects — the
    # the real-world usage shape. Nested lab-subdir workspaces made
    # OpenCode's resolver adopt the lab root (badrun-01/02), and a commitless
    # `git init` did not stop it: the workspace needs a resolvable HEAD.
    workspace = Path.home() / "Projects" / f"WTLab-{args.project}-arm{args.arm}-{stamp}"
    workspace.mkdir(parents=True)
    (run_dir / "workspace-path.txt").write_text(str(workspace) + "\n")
    spec_name = spec_source.name
    shutil.copy(spec_source, workspace / spec_name)
    subprocess.run(
        ["git", "init", "-q"], cwd=workspace, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "add", "-A"], cwd=workspace, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "-c", "user.email=lab@mtplx.local", "-c", "user.name=WTLab",
         "commit", "-q", "-m", "spec"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )

    log_offset = 0
    if args.request_log and Path(args.request_log).exists():
        log_offset = len(
            Path(args.request_log).read_text(encoding="utf-8").splitlines()
        )

    prompt = PROMPT.format(spec=spec_name)
    events_path = run_dir / "events.jsonl"
    started = time.time()
    timed_out = False
    process = None
    with open(events_path, "w", encoding="utf-8") as sink:
        try:
            process = subprocess.run(
                [
                    "opencode",
                    "run",
                    prompt,
                    "-m",
                    args.model,
                    "--format",
                    "json",
                    "--title",
                    f"walltime-{args.project}-arm{args.arm}",
                ],
                cwd=workspace,
                # OpenCode trusts $PWD over the real cwd — subprocess cwd=
                # does NOT update the inherited PWD env var, which bound
                # badrun-03's session to the harness shell's directory.
                env={**os.environ, "PWD": str(workspace)},
                stdout=sink,
                stderr=subprocess.STDOUT,
                timeout=args.timeout_s,
            )
        except subprocess.TimeoutExpired:
            # A timed-out run is still a datapoint (DNF): keep collecting
            # telemetry, acceptance, and forensics below.
            timed_out = True
    wall_s = time.time() - started

    snapshot = {}
    try:
        snapshot = _fetch(f"http://127.0.0.1:{args.port}/v1/mtplx/snapshot")
        (run_dir / "snapshot.json").write_text(json.dumps(snapshot, indent=1))
    except Exception as exc:
        print(f"snapshot fetch failed: {exc}", file=sys.stderr)

    request_rows: list[dict] = []
    if args.request_log and Path(args.request_log).exists():
        lines = Path(args.request_log).read_text(encoding="utf-8").splitlines()
        slice_lines = lines[log_offset:]
        (run_dir / "requests.jsonl").write_text("\n".join(slice_lines) + "\n")
        for line in slice_lines:
            try:
                request_rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    tests_dir = workspace / "tests"
    acceptance = {"ran": False, "exit_code": None, "output_tail": ""}
    if tests_dir.exists():
        try:
            test_files = sorted(str(p.relative_to(workspace)) for p in tests_dir.glob("*.mjs"))
            test_run = subprocess.run(
                ["node", "--test", *test_files],
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=180,
            )
            acceptance = {
                "ran": True,
                "exit_code": test_run.returncode,
                "output_tail": (test_run.stdout + test_run.stderr)[-2000:],
            }
        except Exception as exc:
            acceptance = {"ran": True, "exit_code": -1, "output_tail": str(exc)}

    forensics_text = ""
    if FORENSICS.exists():
        try:
            forensics_run = subprocess.run(
                [
                    sys.executable,
                    str(FORENSICS),
                    "--server",
                    f"http://127.0.0.1:{args.port}",
                    *(
                        ["--request-log", str(run_dir / "requests.jsonl")]
                        if request_rows
                        else []
                    ),
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            forensics_text = forensics_run.stdout
            (run_dir / "forensics.txt").write_text(forensics_text)
        except Exception as exc:
            forensics_text = f"forensics failed: {exc}"

    completions = sum(int(r.get("completion_tokens") or 0) for r in request_rows)
    guard_hits = [
        r.get("thinking_guard")
        for r in request_rows
        if (r.get("thinking_guard") or {}).get("engaged")
    ]
    think_tokens = sum(
        int((r.get("thinking_guard") or {}).get("think_tokens") or 0)
        for r in request_rows
    )
    files_created = [
        str(p.relative_to(workspace))
        for p in sorted(workspace.rglob("*"))
        if p.is_file() and p.name != spec_name and ".git" not in p.parts
    ]
    summary = {
        "project": args.project,
        "arm": args.arm,
        "label": args.label,
        "wall_s": round(wall_s, 1),
        "timed_out": timed_out,
        "opencode_exit": process.returncode if process is not None else None,
        "requests": len(request_rows),
        "completion_tokens": completions,
        "think_tokens_tracked": think_tokens,
        "guard_engagements": len(guard_hits),
        "acceptance": acceptance,
        "files_created": files_created,
        "run_dir": str(run_dir),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
