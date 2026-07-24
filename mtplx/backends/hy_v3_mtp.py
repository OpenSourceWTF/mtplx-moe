"""Tencent Hy3 (hy_v3) native MTP backend facade.

Hy3 ships one appended MTP layer (num_nextn_predict_layers=1): a full MoE
decoder layer fed concat[RMSNorm(next-token embedding), RMSNorm(trunk
post-final-norm hidden state)] through an eh_proj down-projection, sharing the
trunk's embeddings and lm_head. The MLX reference implementation exposes it as
``Model.predict_next_tokens(hidden, token_ids, cache)`` with
``return_hidden_states=True`` on the trunk forward (see
mlx-lm ``models/hy_v3.py``, MTP revision).

Like the GLM/DeepSeek facades, drafting and verification are wired through the
shared speculative sampler in ``generation.py``; this facade gates execution
behind the verified runtime contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import DraftTokens, ModelState, MTPBackend, VerifyOutput
from mtplx.profiles import DEFAULT_PROFILE_NAME


class HyV3MTPBackend(MTPBackend):
    arch_id = "hy-v3-mtp"

    def load(self, model_path: Path) -> ModelState:
        from mtplx.mtp_patch import MTPContract
        from mtplx.runtime import load

        runtime = load(model_path, mtp=True, contract=MTPContract())
        return ModelState(
            model_path=Path(model_path),
            runtime=runtime,
            metadata={"arch_id": self.arch_id, "contract_gated": True},
        )

    def verify(self, state: ModelState, draft_tokens: DraftTokens, hidden: Any) -> VerifyOutput:
        raise NotImplementedError("HyV3MTPBackend.verify is wired through generation.py")

    def propose(self, state: ModelState, hidden: Any) -> DraftTokens:
        raise NotImplementedError("HyV3MTPBackend.propose is wired through generation.py")

    def recommended_profile(self) -> str:
        return DEFAULT_PROFILE_NAME

    def health(self) -> dict[str, Any]:
        return {
            "arch_id": self.arch_id,
            "runtime_path": "mtplx.runtime + mtplx.hy_v3_mtp_patch + mtplx.generation",
            "support_level": "experimental-native-contract-gated",
            "contract_required": True,
            "supported_model_types": ["hy_v3"],
            "mtp_depth_max": 1,
            "notes": (
                "Single appended NextN layer with its own 192-expert MoE MLP, "
                "sigmoid top-8 routing with expert bias, eh_proj over "
                "concat[enorm(embedding), hnorm(hidden)], shared embeddings "
                "and head. Draft layer consumes the trunk POST-final-norm "
                "hidden state (measured 0.773 vs 0.387 pre-norm). Verification is exact rejection sampling in "
                "generation.py; the MLX reference (mlx-lm hy_v3 MTP revision) "
                "verifies greedily and is temp-0 exact."
            ),
            "references": [
                "REFERENCES:TOOLS/vllm-official-main/vllm/model_executor/models/hy_v3_mtp.py",
                "REFERENCES:TOOLS/mlx-lm/mlx_lm/models/hy_v3.py",
            ],
        }
