#!/usr/bin/env python3
"""Drive a REAL Pi session through its production launcher over a pty."""
import os
import pty
import select
import signal
import sys
import time
import re

LAUNCHER = os.path.expanduser("~/.mtplx/open-pi.command")
LOG = sys.argv[1] if len(sys.argv) > 1 else "/tmp/pi_session.log"
TURNS = [
    ("Create a folder pi_qa_kvcache_20260703 in the current directory. Inside it write "
     "fizzbuzz.py implementing classic fizzbuzz for 1..30, run it with bash, and show me "
     "the last 5 lines of its output.", 360),
    ("Now add a unit test file test_fizzbuzz.py in the same folder using plain asserts, "
     "run it, and tell me pass or fail.", 360),
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

    wait_idle(min_activity_bytes=200, idle_s=5.0, cap_s=60)
    print(f"[driver] pi booted ({len(buf)} bytes)", flush=True)

    for i, (text, cap) in enumerate(TURNS, 1):
        mark = len(buf)
        os.write(fd, text.encode() + b"\r")
        t0 = time.time()
        wait_idle(min_activity_bytes=200, idle_s=15.0, cap_s=cap)
        print(f"[driver] turn {i} done in {time.time()-t0:.1f}s ({len(buf)-mark} bytes)", flush=True)

    os.write(fd, b"\x03")
    time.sleep(1.5)
    os.write(fd, b"\x04")
    time.sleep(1.5)
    pump(2.0)
    try:
        os.kill(pid, signal.SIGTERM)
        time.sleep(2)
    except ProcessLookupError:
        pass
    try:
        os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        pass
    print("[driver] pi session closed", flush=True)

if __name__ == "__main__":
    main()
