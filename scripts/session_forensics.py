#!/usr/bin/env python3
"""Session forensics: correlate MTPLX per-request telemetry with an OpenCode
conversation and flag the wall-clock pathologies found in the 2026-07-20
2026-07-20 live agent-session investigation.

Detectors:
  REWIND           cached tokens fell below the previous request's context —
                   something mutated committed history (compaction flip, edit
                   rewrite, client-side truncation) and forced a re-prefill.
  PREFILL_STALL    time-to-first-token above threshold with a big re-prefill.
  THINK_MARATHON   a turn whose reasoning output exceeds the threshold
                   (chars from OpenCode parts, tokens from thinking_guard
                   telemetry when present).
  DOUBLE_EMISSION  an assistant text part contains the same content as a
                   write/edit tool argument — the model wrote the file twice.
  TOOL_ERROR       any tool part that ended in status=error.
  ID_ROTATION      the server-side session id changed mid-conversation
                   (prefix-match identity broke; multiplies the session bank).
  USAGE_MISMATCH   client-visible cache_read=0 while the server internally
                   reused a prefix (trust-corroding usage misreport).
  GUARD_EVENT      thinking_guard engaged/closed on a request.

Sources (all optional, best effort — use what exists):
  --server URL             live server; pulls /v1/mtplx/snapshot and
                           /v1/mtplx/prefill_history (default
                           http://127.0.0.1:8001; pass "" to skip)
  --request-log FILE       --request-log-jsonl trail (full history; preferred)
  --snapshot FILE          saved snapshot.json instead of --server
  --prefill-history FILE   saved prefill_history.json instead of --server
  --opencode-db FILE       OpenCode sqlite (default auto-locate)
  --opencode-session ID    session id (default: most recently created)

Output: human-readable timeline + summary (default), or --json.
Stdlib only; the DB is copied to a temp file and opened read-only.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import shutil
import sqlite3
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

REWIND_TOLERANCE_TOKENS = 64
PREFILL_STALL_TTFT_S = 5.0
THINK_MARATHON_CHARS = 8000
THINK_MARATHON_TOKENS = 2600
DOUBLE_EMISSION_PROBE_CHARS = 200
USAGE_MISMATCH_MIN_TOKENS = 1024


def _fetch_json(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def _load_requests(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Best-available request records, oldest first, deduped by request_id."""
    records: list[dict[str, Any]] = []
    if args.request_log:
        for line in Path(args.request_log).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    snapshot = None
    if args.snapshot:
        snapshot = json.loads(Path(args.snapshot).read_text(encoding="utf-8"))
    elif args.server:
        try:
            snapshot = _fetch_json(args.server.rstrip("/") + "/v1/mtplx/snapshot")
        except Exception as exc:
            print(f"note: snapshot fetch failed ({exc})", file=sys.stderr)
    if snapshot:
        records.extend(snapshot.get("recent") or [])
    prefill = None
    if args.prefill_history:
        prefill = json.loads(Path(args.prefill_history).read_text(encoding="utf-8"))
    elif args.server:
        try:
            prefill = _fetch_json(
                args.server.rstrip("/") + "/v1/mtplx/prefill_history"
            )
        except Exception:
            prefill = None
    prefill_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    if prefill:
        for entry in prefill.get("history") or []:
            key = (
                int(entry.get("prompt_tokens") or 0),
                int(entry.get("cached_tokens") or 0),
            )
            prefill_by_key[key] = entry
    seen: set[str] = set()
    merged: list[dict[str, Any]] = []
    for record in records:
        request_id = str(record.get("request_id") or id(record))
        if request_id in seen:
            continue
        seen.add(request_id)
        if "t" not in record:
            key = (
                int(record.get("prompt_tokens") or 0),
                int(record.get("cached_tokens") or 0),
            )
            hit = prefill_by_key.get(key)
            if hit:
                record = {**record, "t": hit.get("t")}
        merged.append(record)
    merged.sort(key=lambda r: (r.get("t") is None, r.get("t") or 0.0))
    return merged


def _default_opencode_db() -> Path | None:
    candidate = Path.home() / ".local/share/opencode/opencode.db"
    return candidate if candidate.exists() else None


def _load_opencode(
    db_path: Path, session_id: str | None
) -> tuple[str | None, list[dict[str, Any]]]:
    """Copy the DB (plus WAL) and return (session_id, message dicts)."""
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "opencode.db"
        shutil.copy(db_path, copy)
        for suffix in ("-wal", "-shm"):
            side = Path(str(db_path) + suffix)
            if side.exists():
                shutil.copy(side, Path(str(copy) + suffix))
        connection = sqlite3.connect(copy)
        connection.row_factory = sqlite3.Row
        if session_id is None:
            row = connection.execute(
                "SELECT id FROM session ORDER BY time_created DESC LIMIT 1"
            ).fetchone()
            if row is None:
                return None, []
            session_id = row["id"]
        messages: list[dict[str, Any]] = []
        for row in connection.execute(
            "SELECT id, time_created, data FROM message"
            " WHERE session_id=? ORDER BY time_created",
            (session_id,),
        ):
            data = json.loads(row["data"])
            parts = []
            for part_row in connection.execute(
                "SELECT data FROM part WHERE message_id=? ORDER BY id",
                (row["id"],),
            ):
                parts.append(json.loads(part_row["data"]))
            messages.append(
                {
                    "id": row["id"],
                    "t": row["time_created"] / 1000.0,
                    "role": data.get("role"),
                    "tokens": data.get("tokens") or {},
                    "parts": parts,
                }
            )
        connection.close()
        return session_id, messages


def _message_features(message: dict[str, Any]) -> dict[str, Any]:
    think_chars = 0
    text_chars = 0
    text_blobs: list[str] = []
    tool_payloads: list[tuple[str, str]] = []
    tool_errors: list[str] = []
    for part in message["parts"]:
        kind = part.get("type")
        if kind == "reasoning":
            think_chars += len(part.get("text") or "")
        elif kind == "text":
            blob = part.get("text") or ""
            text_chars += len(blob)
            text_blobs.append(blob)
        elif kind == "tool":
            state = part.get("state") or {}
            tool = str(part.get("tool") or "?")
            if state.get("status") == "error":
                tool_errors.append(f"{tool}: {str(state.get('error'))[:120]}")
            arguments = state.get("input") or {}
            payload = arguments.get("content") or arguments.get("newString") or ""
            if isinstance(payload, str) and len(payload) >= DOUBLE_EMISSION_PROBE_CHARS:
                tool_payloads.append((tool, payload))
    double_emission = False
    for _tool, payload in tool_payloads:
        probe = payload[:DOUBLE_EMISSION_PROBE_CHARS]
        if any(probe in blob for blob in text_blobs):
            double_emission = True
            break
    return {
        "think_chars": think_chars,
        "text_chars": text_chars,
        "tool_errors": tool_errors,
        "double_emission": double_emission,
        "cache_read": (message["tokens"].get("cache") or {}).get("read"),
        "tok_in": message["tokens"].get("input"),
        "tok_out": message["tokens"].get("output"),
    }


def _nearest_message(
    messages: list[dict[str, Any]], t: float | None, used: set[str]
) -> dict[str, Any] | None:
    if t is None:
        return None
    best = None
    best_delta = 90.0  # requests start within seconds of their message
    for message in messages:
        if message["role"] != "assistant" or message["id"] in used:
            continue
        delta = abs(message["t"] - t)
        if delta < best_delta:
            best = message
            best_delta = delta
    return best


def _clock(t: float | None) -> str:
    if t is None:
        return "--:--:--"
    return _dt.datetime.fromtimestamp(t).strftime("%H:%M:%S")


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    requests = _load_requests(args)
    db_path = (
        Path(args.opencode_db) if args.opencode_db else _default_opencode_db()
    )
    session_id, messages = (
        _load_opencode(db_path, args.opencode_session) if db_path else (None, [])
    )
    findings: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    used_messages: set[str] = set()
    previous_context: int | None = None
    previous_session: str | None = None
    totals = {
        "requests": len(requests),
        "reprefill_tokens": 0,
        "stall_seconds": 0.0,
        "think_chars": 0,
        "completion_tokens": 0,
        "decode_seconds": 0.0,
    }

    def flag(kind: str, t: float | None, detail: str) -> None:
        findings.append({"kind": kind, "t": t, "detail": detail})

    for record in requests:
        t = record.get("t") or record.get("logged_at_s")
        prompt = int(record.get("prompt_tokens") or 0)
        cached = int(record.get("cached_tokens") or 0)
        new_prefill = int(record.get("new_prefill_tokens") or 0)
        ttft = float(record.get("ttft_s") or 0.0)
        completion = int(record.get("completion_tokens") or 0)
        decode_s = float(record.get("decode_elapsed_s") or 0.0)
        context = record.get("context_len")
        session = record.get("session_id")
        guard = record.get("thinking_guard") or {}
        totals["reprefill_tokens"] += new_prefill
        totals["completion_tokens"] += completion
        totals["decode_seconds"] += decode_s
        message = _nearest_message(messages, t, used_messages)
        features = _message_features(message) if message else {}
        if message:
            used_messages.add(message["id"])
            totals["think_chars"] += features["think_chars"]

        if (
            previous_context is not None
            and prompt >= cached
            and cached + REWIND_TOLERANCE_TOKENS < min(previous_context, prompt)
        ):
            flag(
                "REWIND",
                t,
                f"cached {cached} fell {previous_context - cached} below prev"
                f" context {previous_context} — mid-history mutation forced a"
                f" {new_prefill}-token re-prefill",
            )
        if ttft >= PREFILL_STALL_TTFT_S and new_prefill >= 1024:
            totals["stall_seconds"] += ttft
            flag(
                "PREFILL_STALL",
                t,
                f"{ttft:.1f}s TTFT re-prefilling {new_prefill} tokens"
                f" ({record.get('prefill_tok_s') or '?'} tok/s)",
            )
        if previous_session is not None and session != previous_session:
            flag(
                "ID_ROTATION",
                t,
                f"server session id rotated {previous_session} -> {session}"
                " (prefix-match identity broke)",
            )
        think_tokens = guard.get("think_tokens")
        if think_tokens is not None and int(think_tokens) >= THINK_MARATHON_TOKENS:
            flag(
                "THINK_MARATHON",
                t,
                f"{think_tokens} reasoning tokens"
                + (
                    f" (guard engaged: {guard.get('engaged')})"
                    if guard.get("engaged")
                    else " (guard not engaged)"
                ),
            )
        elif features.get("think_chars", 0) >= THINK_MARATHON_CHARS:
            flag(
                "THINK_MARATHON",
                t,
                f"{features['think_chars']} reasoning chars in the paired"
                " OpenCode turn",
            )
        if guard.get("engaged"):
            flag(
                "GUARD_EVENT",
                t,
                f"thinking_guard {guard.get('engaged')} closed_at="
                f"{guard.get('closed_at')} think_tokens={guard.get('think_tokens')}",
            )
        if features.get("double_emission"):
            flag(
                "DOUBLE_EMISSION",
                t,
                f"text part duplicates tool payload ({features['text_chars']}"
                " chars of visible text)",
            )
        for error in features.get("tool_errors", []):
            flag("TOOL_ERROR", t, error)
        client_cached = features.get("cache_read")
        if (
            client_cached in (0, None)
            and cached >= USAGE_MISMATCH_MIN_TOKENS
            and message is not None
        ):
            flag(
                "USAGE_MISMATCH",
                t,
                f"client saw cache_read={client_cached} but server reused"
                f" {cached} tokens",
            )
        rows.append(
            {
                "t": t,
                "clock": _clock(t),
                "session": session,
                "prompt": prompt,
                "cached": cached,
                "new_prefill": new_prefill,
                "ttft_s": round(ttft, 2),
                "completion": completion,
                "decode_tok_s": record.get("decode_tok_s"),
                "think": features.get("think_chars"),
                "guard": guard.get("engaged"),
            }
        )
        previous_context = int(context) if context else prompt + completion
        previous_session = session

    wall_s = None
    stamped = [row["t"] for row in rows if row["t"]]
    if len(stamped) >= 2:
        wall_s = max(stamped) - min(stamped)
    return {
        "opencode_session": session_id,
        "requests": rows,
        "findings": findings,
        "summary": {
            **totals,
            "wall_seconds": wall_s,
            "decode_share": (
                round(totals["decode_seconds"] / wall_s, 3) if wall_s else None
            ),
            "finding_counts": {
                kind: sum(1 for f in findings if f["kind"] == kind)
                for kind in sorted({f["kind"] for f in findings})
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--server", default="http://127.0.0.1:8001")
    parser.add_argument("--request-log", default=None)
    parser.add_argument("--snapshot", default=None)
    parser.add_argument("--prefill-history", default=None)
    parser.add_argument("--opencode-db", default=None)
    parser.add_argument("--opencode-session", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = analyze(args)
    if args.json:
        print(json.dumps(report, indent=2, default=str))
        return 0
    print(f"OpenCode session: {report['opencode_session']}")
    print(
        f"{'time':<9}{'sess':<22}{'prompt':>8}{'cached':>8}{'newpf':>7}"
        f"{'ttft':>7}{'comp':>7}{'dec':>7}{'think':>8}{'guard':>9}"
    )
    for row in report["requests"]:
        decode = row["decode_tok_s"]
        print(
            f"{row['clock']:<9}{str(row['session'])[:20]:<22}{row['prompt']:>8}"
            f"{row['cached']:>8}{row['new_prefill']:>7}{row['ttft_s']:>7}"
            f"{row['completion']:>7}"
            f"{(round(decode, 1) if decode else '-'):>7}"
            f"{(row['think'] if row['think'] is not None else '-'):>8}"
            f"{(row['guard'] or '-'):>9}"
        )
    print("\nFindings:")
    if not report["findings"]:
        print("  (none)")
    for finding in report["findings"]:
        print(f"  [{_clock(finding['t'])}] {finding['kind']:<16} {finding['detail']}")
    print("\nSummary:")
    for key, value in report["summary"].items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
