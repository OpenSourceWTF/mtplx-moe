"""Repro: A3B MTP speculative depth defaults to the contract CEILING (mtp_depth_max=3).

Authors' published Qwen3.6-35B-A3B benchmark:
    AR baseline   94.46 tok/s
    D1           138.39 tok/s   <-- best mode
    D2           135.66 tok/s
    D3           107.67 tok/s   <-- what MTPLX selects as the default

Run:
    PYTHONPATH=. python repro_depth_pin.py
"""
import json
import sys
from pathlib import Path

from mtplx.commands.public import _model_contract_depth

MODEL = Path("/Users/davidtai/.mtplx/models/Youssofal--Qwen3.6-35B-A3B-MTPLX-Optimized-Speed")
contract = json.loads((MODEL / "mtplx_runtime.json").read_text())


class Profile:
    """Stand-in for the resolved runtime profile; A3B recommends 'sustained'."""

    def __init__(self, name):
        self.name = name


print("contract mtp_depth_max   :", contract["mtp_depth_max"], "  (a CEILING)")
print("contract recommended     :", contract["recommended_profile"])

inspection = {"compatibility": {"runtime_contract": contract}}
profile = Profile(contract["recommended_profile"])

# fallback=1 makes the pin unambiguous: any 3 below comes from the contract,
# not from the caller's default.
resolved = _model_contract_depth(inspection, profile=profile, fallback=1)
print("resolved default depth   :", resolved)
print()

if resolved == contract["mtp_depth_max"] == 3:
    print("STUCK AT 3: the ceiling is returned as the default depth.")
    print("Authors' data: D3 107.67 vs D1 138.39  ->  -22.2% throughput out of the box.")
    sys.exit(0)

print("not pinned; resolved", resolved)
sys.exit(1)
