#!/usr/bin/env python3
"""Drive a REAL Hermes chat session through its production launcher over a pty.

Spawns ~/.mtplx/open-hermes.command exactly as the app's Terminal lane would,
sends real user turns, waits for the agent to finish each one (output-idle
heuristic), then exits the client cleanly. Full raw transcript saved.
"""
import os
import pty
import select
import signal
import sys
import time
import re

LAUNCHER = os.path.expanduser("~/.mtplx/open-hermes.command")
LOG = sys.argv[1] if len(sys.argv) > 1 else "/tmp/hermes_session.log"
TURNS = [
    ("Create a file called hermes_qa_kvcache_20260703.txt in the current folder "
     "with exactly three lines: line 1 the output of the date command, line 2 the "
     "count of .md files in this folder, line 3 the word DONE. Then print the file back to me.",
     300),
    ("Now append one more line to that same file containing the word WARM, "
     "then show me the final file contents.", 300),
]
ANSI = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]|\x1b\][^\x07]*\x07|\x1b[=>]|\r")

def main():
    pid, fd = pty.fork()
    if pid == 0:
        os.environ["TERM"] = "xterm-256color"
        os.environ["COLUMNS"] = "200"
        os.environ["LINES"] = "50"
        os.execvp("/bin/zsh", ["/bin/zsh", LAUNCHER])
    log = open(LOG, "wb")
    buf = b""

    def pump(timeout):
        nonlocal buf
        r, _, _ = select.select([fd], [], [], timeout)
        if fd in r:
            try:
                chunk = os.read(fd, 65536)
            except OSError:
                return None
            if not chunk:
                return None
            log.write(chunk)
            log.flush()
            buf += chunk
            return chunk
        return b""

    def wait_idle(min_activity_bytes, idle_s, cap_s):
        """Wait until we've seen min bytes AND then idle_s of silence."""
        start = time.time()
        seen = 0
        last_data = time.time()
        while time.time() - start < cap_s:
            chunk = pump(1.0)
            if chunk is None:
                return False
            if chunk:
                seen += len(chunk)
                last_data = time.time()
            elif seen >= min_activity_bytes and time.time() - last_data >= idle_s:
                return True
        return True

    # startup: gateway checks + client boot + first paint
    wait_idle(min_activity_bytes=400, idle_s=6.0, cap_s=90)
    print(f"[driver] client booted ({len(buf)} bytes)", flush=True)

    for i, (text, cap) in enumerate(TURNS, 1):
        mark = len(buf)
        os.write(fd, text.encode() + b"\r")
        t0 = time.time()
        wait_idle(min_activity_bytes=200, idle_s=12.0, cap_s=cap)
        turn_out = ANSI.sub("", buf[mark:].decode("utf-8", "replace"))
        print(f"[driver] turn {i} done in {time.time()-t0:.1f}s, {len(turn_out)} chars", flush=True)

    # exit: Ctrl+C then Ctrl+D
    os.write(fd, b"\x03")
    time.sleep(1.5)
    os.write(fd, b"\x04")
    time.sleep(1.5)
    pump(2.0)
    try:
        os.kill(pid, 0)
        os.kill(pid, signal.SIGTERM)
        time.sleep(2)
    except ProcessLookupError:
        pass
    _, status = os.waitpid(pid, os.WNOHANG)
    print(f"[driver] session closed (status {status})", flush=True)
    log.close()

if __name__ == "__main__":
    main()
