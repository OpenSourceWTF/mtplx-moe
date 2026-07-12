"""Qwen3-Next native MTP backend.

v0.1 ships only this backend.  The heavy runtime imports remain lazy so the
registry and CLI inspection path stay usable without MLX installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import DraftTokens, ModelState, MTPBackend, VerifyOutput
from mtplx.profiles import DEFAULT_PROFILE_NAME, get_profile


class Qwen3NextMTPBackend(MTPBackend):
    arch_id = "qwen3-next-mtp"

    def load(self, model_path: Path) -> ModelState:
        from mtplx.mtp_patch import MTPContract
        from mtplx.runtime import load

        runtime = load(model_path, mtp=True, contract=MTPContract())
        return ModelState(
            model_path=Path(model_path),
            runtime=runtime,
            metadata={"arch_id": self.arch_id},
        )

    def verify(self, state: ModelState, draft_tokens: DraftTokens, hidden: Any) -> VerifyOutput:
        raise NotImplementedError("Qwen3NextMTPBackend.verify is wired through generation.py in v0.1")

    def propose(self, state: ModelState, hidden: Any) -> DraftTokens:
        raise NotImplementedError("Qwen3NextMTPBackend.propose is wired through generation.py in v0.1")

    def recommended_profile(self) -> str:
        return DEFAULT_PROFILE_NAME

    def health(self) -> dict[str, Any]:
        profile = get_profile("performance-cold")
        return {
            "arch_id": self.arch_id,
            "runtime_path": "mtplx.runtime + mtplx.generation",
            "performance_cold_requirements": {
                "env": profile.env_dict(),
                "draft_lm_head": (
                    None
                    if profile.draft_lm_head is None
                    else {
                        "bits": profile.draft_lm_head.bits,
                        "group_size": profile.draft_lm_head.group_size,
                        "mode": profile.draft_lm_head.mode,
                    }
                ),
            },
        }
