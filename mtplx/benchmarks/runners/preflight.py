"""Benchmark preflight checks for local MTPLX runs."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


BENCH_RE = re.compile(
    r"(python|mlx|mtplx|benchmark|mtp-depth|mtp1|verify-ratio|verify-profile|hf download)",
    re.IGNORECASE,
)
# /opt/sovereign/: the Pulse deep-tier operator prep keeps a long-lived
# low-CPU python keeper (keep-amfi.py) resident; it is system prep, not a
# benchmark, and flagging it deadlocks every deep-tier gated cell.
SELF_EXCLUDE_RE = re.compile(
    r"(rg -i|Codex|Electron|python -m server|bench-preflight|hermes_cli|/opt/sovereign/)"
)
# The Claude desktop app is the founder's permanent control plane on this
# machine (founder ruling 2026-08-27): it is ALWAYS open, it renders the
# very session that launches gated cells, and its UI spikes track the
# controller's own output — so a heavy-CPU flag on it deadlocks every cell
# forever (the app can never be quiet while the controller exists). It is
# part of the baseline environment measurements ship into, not a rogue
# hog. Physics gates (die temp, fans, power) still stand.
CONTROL_PLANE_RE = re.compile(r"(Claude Helper|Claude\b|claude\b)")


def _run(args: list[str]) -> tuple[int, str]:
    try:
        proc = subprocess.run(args, check=False, text=True, capture_output=True)
    except FileNotFoundError as exc:
        return 127, str(exc)
    return proc.returncode, (proc.stdout or proc.stderr).strip()


def _top_processes(limit: int) -> list[dict[str, Any]]:
    code, out = _run(["ps", "-axo", "pid,pcpu,pmem,etime,command"])
    if code != 0:
        return []
    rows = []
    for line in out.splitlines()[1:]:
        parts = line.strip().split(None, 4)
        if len(parts) < 5:
            continue
        pid, pcpu, pmem, etime, command = parts
        try:
            rows.append(
                {
                    "pid": int(pid),
                    "pcpu": float(pcpu),
                    "pmem": float(pmem),
                    "etime": etime,
                    "command": command,
                }
            )
        except ValueError:
            continue
    rows.sort(key=lambda item: item["pcpu"], reverse=True)
    return rows[:limit]


def _active_bench_processes() -> list[dict[str, Any]]:
    current_pid = os.getpid()
    matches = []
    for row in _top_processes(200):
        if row["pid"] == current_pid:
            continue
        command = row["command"]
        if BENCH_RE.search(command) and not SELF_EXCLUDE_RE.search(command):
            matches.append(row)
    return matches


def _power_state() -> dict[str, Any]:
    _, batt = _run(["pmset", "-g", "batt"])
    _, therm = _run(["pmset", "-g", "therm"])
    return {
        "battery": batt,
        "thermal": therm,
        "ac_power": "AC Power" in batt,
        "thermal_warning": "warning level has been recorded" in therm
        and "No thermal warning" not in therm,
        "performance_warning": "performance warning level has been recorded" in therm
        and "No performance warning" not in therm,
    }


def run_preflight(
    project_root: Path | str = ".",
    *,
    top_limit: int = 12,
    cpu_threshold: float = 25.0,
    min_free_gib: float = 25.0,
) -> dict[str, Any]:
    root = Path(project_root)
    usage = shutil.disk_usage(root)
    free_gib = usage.free / (1024**3)
    top = _top_processes(top_limit)
    current_pid = os.getpid()
    heavy = [
        row
        for row in top
        if row["pid"] != current_pid
        and row["pcpu"] >= cpu_threshold
        and not CONTROL_PLANE_RE.search(row["command"])
    ]
    active_bench = _active_bench_processes()
    power = _power_state()

    issues: list[str] = []
    if active_bench:
        issues.append("active_benchmark_process")
    if heavy:
        issues.append("heavy_background_process")
    if not power["ac_power"]:
        issues.append("not_on_ac_power")
    if power["thermal_warning"] or power["performance_warning"]:
        issues.append("thermal_or_performance_warning")
    if free_gib < min_free_gib:
        issues.append("low_disk_free")

    return {
        "project_root": str(root.resolve()),
        "clean": not issues,
        "issues": issues,
        "thresholds": {
            "cpu_threshold": cpu_threshold,
            "min_free_gib": min_free_gib,
        },
        "disk": {
            "total_gib": usage.total / (1024**3),
            "used_gib": usage.used / (1024**3),
            "free_gib": free_gib,
        },
        "power": power,
        "active_benchmark_processes": active_bench,
        "top_processes": top,
    }


def write_preflight(path: Path | str, result: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True))
