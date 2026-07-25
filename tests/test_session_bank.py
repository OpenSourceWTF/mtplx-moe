from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from mtplx.cache_state import CacheSnapshot
from mtplx.session_bank import SessionBank


class DenseMaterializingCache:
    @property
    def state(self):
        raise RuntimeError("Paged KV cache attempted to materialize active K/V arrays")


class TrimmableLiveCache:
    def __init__(self, offset: int):
        self.offset = offset
        self.trimmed: list[int] = []

    @property
    def state(self):
        raise RuntimeError("Paged KV cache attempted to materialize active K/V arrays")

    def is_trimmable(self) -> bool:
        # Models a real attention KV container; without this the bank's
        # conservative recurrent detection treats it as untrimmable state and
        # boundary-true restores fail closed.
        return True

    def trim(self, n: int) -> int:
        self.trimmed.append(int(n))
        self.offset -= int(n)
        return int(n)


class RuntimeWithCaches:
    model_path = Path("models/example")
    mtp_enabled = True

    def make_cache(self):
        return [TrimmableLiveCache(0)]

    def make_mtp_cache(self):
        return [TrimmableLiveCache(0)]


class AROnlyRuntime:
    model_path = Path("models/example")
    mtp_enabled = False

    def __init__(self):
        self.make_mtp_cache_calls = 0

    def make_cache(self):
        return []

    def make_mtp_cache(self):
        self.make_mtp_cache_calls += 1
        raise AssertionError("AR-only SessionBank restore must not build MTP cache")


_EMPTY_SNAPSHOT = CacheSnapshot(states=(), meta_states=())


@pytest.mark.parametrize(
    "auxiliary",
    [
        {"hidden": "stale-hidden"},
        {"hidden_variant": "post_norm"},
        {"mtp_history_snapshot": _EMPTY_SNAPSHOT},
        {"mtp_history_cache_ref": []},
        {"mtp_snapshot_epoch": 3},
        {"gdn_boundaries": [(2, _EMPTY_SNAPSHOT, "stale-boundary-hidden")]},
    ],
    ids=[
        "hidden",
        "hidden-variant",
        "mtp-snapshot",
        "mtp-live-ref",
        "mtp-epoch",
        "boundary-hidden",
    ],
)
def test_session_bank_put_rejects_explicit_none_auxiliary_state(auxiliary):
    bank = SessionBank()
    kwargs = {
        "runtime": AROnlyRuntime(),
        "token_ids": [1, 2, 3],
        "cache": [],
        "logits": "logits",
        "hidden": None,
        "mtp_history_policy": "none",
        "snapshot_epoch": 3,
        **auxiliary,
    }

    with pytest.raises(ValueError, match="mtp_history_policy='none'"):
        bank.put(**kwargs)


@pytest.mark.parametrize(
    "auxiliary",
    [
        {"hidden": "stale-hidden"},
        {"hidden_variant": "post_norm"},
        {"mtp_history_snapshot": _EMPTY_SNAPSHOT},
        {"mtp_snapshot_epoch": 3},
    ],
    ids=["hidden", "hidden-variant", "mtp-snapshot", "mtp-epoch"],
)
def test_session_bank_put_snapshot_rejects_explicit_none_auxiliary_state(
    auxiliary,
):
    bank = SessionBank()
    kwargs = {
        "runtime": AROnlyRuntime(),
        "token_ids": [1, 2, 3],
        "cache_snapshot": _EMPTY_SNAPSHOT,
        "logits": "logits",
        "hidden": None,
        "mtp_history_policy": "none",
        "snapshot_epoch": 3,
        **auxiliary,
    }

    with pytest.raises(ValueError, match="mtp_history_policy='none'"):
        bank.put_snapshot(**kwargs)


def test_session_bank_exact_none_restore_ignores_stale_auxiliary_state():
    bank = SessionBank()
    runtime = AROnlyRuntime()
    entry = bank.put(
        runtime=runtime,
        token_ids=[1, 2, 3],
        cache=[],
        logits="logits",
        hidden=None,
        hidden_variant=None,
        mtp_history_policy="none",
        snapshot_epoch=3,
    )
    assert entry is not None
    # Models a pre-fix RAM entry or corrupt persisted payload.
    entry.hidden = "stale-hidden"
    entry.hidden_variant = "post_norm"
    entry.mtp_history_snapshot = _EMPTY_SNAPSHOT
    entry.mtp_history_cache_ref = [TrimmableLiveCache(offset=2)]
    entry.mtp_snapshot_epoch = 999

    restored = bank.restore(
        runtime,
        [1, 2, 3, 4],
        hidden_variant=None,
        mtp_history_policy="none",
    )

    assert restored is not None
    assert runtime.make_mtp_cache_calls == 0
    assert restored.hidden is None
    assert restored.mtp_history_snapshot is None
    assert restored.mtp_history_cache is None
    assert restored.entry.hidden_variant is None
    assert restored.entry.mtp_snapshot_epoch is None


def test_session_bank_near_none_restore_ignores_stale_live_mtp_reference():
    bank = SessionBank()
    runtime = AROnlyRuntime()
    entry = bank.put(
        runtime=runtime,
        token_ids=[1, 2, 3, 4, 5, 6],
        cache=[],
        logits="logits",
        hidden=None,
        hidden_variant=None,
        mtp_history_policy="none",
        snapshot_epoch=6,
    )
    assert entry is not None
    entry.cache_ref = [TrimmableLiveCache(offset=5)]
    entry.mtp_history_cache_ref = [TrimmableLiveCache(offset=5)]
    entry.hidden = "stale-hidden"

    restored = bank.restore_entry_prefix_cache(
        runtime,
        entry,
        5,
        mode="reference",
    )

    assert restored is not None
    cache, mtp_cache, restore_mode, restore_point, boundary_hidden = restored
    assert cache is not None
    assert mtp_cache is None
    assert restore_mode == "reference_lease"
    assert restore_point == 5
    assert boundary_hidden is None
    assert runtime.make_mtp_cache_calls == 0
    assert entry.hidden is None
    assert entry.mtp_history_cache_ref is None


def test_session_bank_cold_none_restore_ignores_stale_auxiliary_state():
    runtime = AROnlyRuntime()
    record = SimpleNamespace(
        token_ids=(1, 2, 3),
        cache_snapshot=_EMPTY_SNAPSHOT,
        logits="logits",
        hidden="stale-hidden",
        mtp_history_snapshot=_EMPTY_SNAPSHOT,
        nbytes=128,
        restore_s=0.01,
        metadata={
            "model_path": str(runtime.model_path),
            "mtp_enabled": False,
            "hidden_variant": "post_norm",
            "mtp_history_policy": "none",
            "snapshot_epoch": 3,
            "mtp_snapshot_epoch": 999,
        },
    )

    class ColdTier:
        def lookup(self, *_args, **_kwargs):
            return record

    bank = SessionBank(cold_tier=ColdTier())
    restored = bank.restore(
        runtime,
        [1, 2, 3, 4],
        hidden_variant=None,
        mtp_history_policy="none",
    )

    assert restored is not None
    assert restored.cache_source == "ssd"
    assert runtime.make_mtp_cache_calls == 0
    assert restored.hidden is None
    assert restored.mtp_history_snapshot is None
    assert restored.mtp_history_cache is None
    assert restored.entry.hidden_variant is None
    assert restored.entry.mtp_snapshot_epoch is None


def test_session_bank_committed_restore_keeps_hidden_and_mtp_snapshot():
    bank = SessionBank()
    runtime = RuntimeWithCaches()
    entry = bank.put(
        runtime=runtime,
        token_ids=[1, 2, 3],
        cache=[],
        logits="logits",
        hidden="committed-hidden",
        hidden_variant="post_norm",
        mtp_history_policy="committed",
        mtp_history_snapshot=_EMPTY_SNAPSHOT,
        snapshot_epoch=3,
        mtp_snapshot_epoch=3,
    )
    assert entry is not None

    incompatible = bank.restore(
        runtime,
        [1, 2, 3, 4],
        hidden_variant=None,
        mtp_history_policy="none",
    )

    assert incompatible is None
    assert entry.hidden == "committed-hidden"
    assert entry.mtp_history_snapshot is not None

    restored = bank.restore(
        runtime,
        [1, 2, 3, 4],
        hidden_variant="post_norm",
        mtp_history_policy="committed",
    )

    assert restored is not None
    assert restored.hidden == "committed-hidden"
    assert restored.mtp_history_snapshot is not None
    assert restored.mtp_history_cache is not None


@pytest.mark.parametrize(
    "ar_tokens",
    [
        [1, 2, 3],
        [1, 2, 3, 4],
    ],
    ids=["same-key", "prefix-donor"],
)
def test_session_bank_none_put_strips_inherited_boundary_hidden(ar_tokens):
    bank = SessionBank()
    boundary_snapshot = CacheSnapshot(states=("trunk-boundary",), meta_states=(None,))
    committed = bank.put(
        runtime=RuntimeWithCaches(),
        token_ids=[1, 2, 3],
        cache=[],
        logits="logits",
        hidden="committed-hidden",
        hidden_variant="post_norm",
        mtp_history_policy="committed",
        snapshot_epoch=3,
        gdn_boundaries=[(2, boundary_snapshot, "committed-boundary-hidden")],
    )
    assert committed is not None

    ar_entry = bank.put(
        runtime=AROnlyRuntime(),
        token_ids=ar_tokens,
        cache=[],
        logits="ar-logits",
        hidden=None,
        hidden_variant=None,
        mtp_history_policy="none",
        snapshot_epoch=len(ar_tokens),
    )

    assert ar_entry is not None
    assert len(ar_entry.gdn_boundaries) == 1
    boundary, restored_snapshot, boundary_hidden = ar_entry.gdn_boundaries[0]
    assert boundary == 2
    assert restored_snapshot.states == ("trunk-boundary",)
    assert boundary_hidden is None


def test_session_bank_none_put_sanitizes_loader_boundaries_before_ssd_enqueue():
    enqueued: list[tuple[list[tuple], list[str]]] = []

    class ColdTier:
        def put_entry(self, entry, *, capabilities):
            enqueued.append((list(entry.gdn_boundaries), list(capabilities)))

    bank = SessionBank(cold_tier=ColdTier())
    committed = bank.put(
        runtime=RuntimeWithCaches(),
        token_ids=[1, 2, 3],
        cache=[],
        logits="logits",
        hidden="committed-hidden",
        hidden_variant="post_norm",
        mtp_history_policy="committed",
        snapshot_epoch=3,
    )
    assert committed is not None
    enqueued.clear()
    committed.gdn_boundary_loader = lambda: [
        (
            2,
            CacheSnapshot(states=("lazy-trunk-boundary",), meta_states=(None,)),
            "lazy-committed-hidden",
        )
    ]

    ar_entry = bank.put(
        runtime=AROnlyRuntime(),
        token_ids=[1, 2, 3, 4],
        cache=[],
        logits="ar-logits",
        hidden=None,
        hidden_variant=None,
        mtp_history_policy="none",
        snapshot_epoch=4,
    )

    assert ar_entry is not None
    assert ar_entry.gdn_boundary_loader is None
    assert len(ar_entry.gdn_boundaries) == 1
    assert ar_entry.gdn_boundaries[0][1].states == ("lazy-trunk-boundary",)
    assert ar_entry.gdn_boundaries[0][2] is None
    assert len(enqueued) == 1
    persisted_boundaries, capabilities = enqueued[0]
    assert persisted_boundaries[0][1].states == ("lazy-trunk-boundary",)
    assert persisted_boundaries[0][2] is None
    assert capabilities == ["ar_insert"]


def test_session_bank_skips_single_oversized_snapshot_before_insert():
    bank = SessionBank(max_entries=4, max_bytes=1024, per_session_max_bytes=512)
    runtime = SimpleNamespace(model_path=Path("models/example"), mtp_enabled=True)

    entry = bank.put(
        runtime=runtime,
        token_ids=[1, 2, 3],
        cache=[],
        logits=None,
        hidden=None,
        session_id="session-1",
        nbytes_override=2048,
    )

    assert entry is None
    assert len(bank) == 0
    assert bank.last_put_nbytes == 2048
    assert bank.last_put_skipped_oversized_snapshot is True
    assert bank.eviction_log[-1]["reason"] == "skipped_oversized_snapshot"


def test_session_bank_skips_dense_materializing_snapshot():
    bank = SessionBank(max_entries=4, max_bytes=1024, per_session_max_bytes=512)
    runtime = SimpleNamespace(model_path=Path("models/example"), mtp_enabled=True)

    entry = bank.put(
        runtime=runtime,
        token_ids=[1, 2, 3],
        cache=[DenseMaterializingCache()],
        logits=None,
        hidden=None,
        session_id="session-1",
    )

    assert entry is None
    assert len(bank) == 0
    assert bank.last_put_skipped_oversized_snapshot is True
    assert bank.eviction_log[-1]["reason"] == "skipped_dense_materializing_snapshot"


def test_session_bank_oversized_prompt_prefix_can_use_live_reference_lease():
    bank = SessionBank(max_entries=4, max_bytes=1024, per_session_max_bytes=512)
    runtime = RuntimeWithCaches()
    cache = [TrimmableLiveCache(offset=11)]
    mtp_cache = [TrimmableLiveCache(offset=11)]

    entry = bank.put(
        runtime=runtime,
        token_ids=list(range(10)),
        cache=cache,
        logits="logits",
        hidden="hidden",
        keep_live_ref=True,
        session_id="session-1",
        mtp_history_policy="committed",
        mtp_history_cache_ref=mtp_cache,
        snapshot_epoch=10,
        mtp_snapshot_epoch=10,
        nbytes_override=2048,
    )

    assert entry is not None
    assert entry.live_ref_only is True
    assert entry.cache_ref is cache
    assert entry.mtp_history_cache_ref is mtp_cache
    assert bank.eviction_log[-1]["fallback"] == "live_reference_lease"

    restored = bank.restore(
        runtime,
        list(range(10)),
        mode="reference",
        session_id="session-1",
        mtp_history_policy="committed",
    )

    assert restored is not None
    assert restored.restore_mode == "reference_lease"
    assert restored.cache is cache
    assert restored.mtp_history_cache is mtp_cache
    assert cache[0].offset == 9
    assert mtp_cache[0].offset == 9
    assert entry.cache_ref is None
    assert entry.mtp_history_cache_ref is None


def test_session_bank_clone_restore_can_use_custom_cache_factory():
    bank = SessionBank(max_entries=4, max_bytes=1024, per_session_max_bytes=512)
    runtime = SimpleNamespace(model_path=Path("models/example"), mtp_enabled=True)
    custom_cache = ["prefill-layout-cache"]

    entry = bank.put(
        runtime=runtime,
        token_ids=[1, 2, 3],
        cache=[],
        logits="logits",
        hidden="hidden",
        session_id="session-1",
        nbytes_override=128,
    )
    assert entry is not None

    restored = bank.restore(
        runtime,
        [1, 2, 3, 4],
        mode="clone",
        cache_factory=lambda: custom_cache,
    )

    assert restored is not None
    assert restored.cache is custom_cache
    assert restored.restore_mode == "clone"


def test_session_bank_live_reference_can_restore_block_prefix_boundary():
    bank = SessionBank(max_entries=4, max_bytes=1024, per_session_max_bytes=512)
    runtime = RuntimeWithCaches()
    cache = [TrimmableLiveCache(offset=1199)]
    mtp_cache = [TrimmableLiveCache(offset=1199)]

    entry = bank.put(
        runtime=runtime,
        token_ids=list(range(1200)),
        cache=cache,
        logits="logits",
        hidden="hidden",
        keep_live_ref=True,
        session_id="session-1",
        mtp_history_policy="committed",
        mtp_history_cache_ref=mtp_cache,
        snapshot_epoch=1200,
        mtp_snapshot_epoch=1200,
        nbytes_override=2048,
    )

    assert entry is not None
    assert entry.live_ref_only is True

    restored = bank.restore_entry_prefix_cache(
        runtime,
        entry,
        1024,
        mode="reference",
    )

    assert restored is not None
    restored_cache, restored_mtp_cache, restore_mode, restore_point, boundary_hidden = (
        restored
    )
    assert restored_cache is cache
    assert restored_mtp_cache is mtp_cache
    assert restore_mode == "reference_lease"
    assert restore_point == 1024
    assert boundary_hidden is None
    assert cache[0].offset == 1023
    assert mtp_cache[0].offset == 1023
    assert entry.cache_ref is None
    assert entry.mtp_history_cache_ref is None


def test_session_bank_near_prefix_trims_mtp_history_by_gap_not_absolute_offset():
    bank = SessionBank(max_entries=4, max_bytes=1024, per_session_max_bytes=512)
    runtime = RuntimeWithCaches()
    cache = [TrimmableLiveCache(offset=1199)]
    mtp_cache = [TrimmableLiveCache(offset=127)]

    entry = bank.put(
        runtime=runtime,
        token_ids=list(range(1200)),
        cache=cache,
        logits="logits",
        hidden="hidden",
        keep_live_ref=True,
        session_id="session-1",
        mtp_history_policy="last_window",
        mtp_history_cache_ref=mtp_cache,
        snapshot_epoch=1200,
        mtp_snapshot_epoch=1200,
        nbytes_override=2048,
    )

    assert entry is not None

    restored = bank.restore_entry_prefix_cache(
        runtime,
        entry,
        1199,
        mode="reference",
    )

    assert restored is not None
    assert cache[0].trimmed == [1]
    assert mtp_cache[0].trimmed == [1]
    assert cache[0].offset == 1198
    assert mtp_cache[0].offset == 126


def test_session_bank_near_prefix_candidates_only_accept_boundary_drift():
    bank = SessionBank(max_entries=4, max_bytes=1024, per_session_max_bytes=512)
    runtime = SimpleNamespace(model_path=Path("models/example"), mtp_enabled=True)
    entry = bank.put(
        runtime=runtime,
        token_ids=list(range(200)),
        cache=[],
        logits=None,
        hidden=None,
        session_id="session-1",
        nbytes_override=128,
    )
    assert entry is not None

    near = list(range(197)) + [10_001, 10_002, 10_003, 10_004]
    far = list(range(120)) + [20_001, 20_002]

    candidates = bank.near_prefix_candidates(
        near,
        max_token_gap=8,
        min_matched_tokens=64,
    )

    assert candidates == [(entry, 197)]
    assert (
        bank.near_prefix_candidates(
            far,
            max_token_gap=8,
            min_matched_tokens=64,
            allow_block_prefix=False,
        )
        == []
    )


def test_session_bank_near_prefix_rejects_prompt_inside_longer_completion():
    bank = SessionBank(max_entries=4, max_bytes=1024, per_session_max_bytes=512)
    runtime = SimpleNamespace(model_path=Path("models/example"), mtp_enabled=True)
    entry = bank.put(
        runtime=runtime,
        token_ids=list(range(70)) + [90_001, 90_002],
        cache=[],
        logits=None,
        hidden=None,
        session_id="session-1",
        nbytes_override=128,
    )
    assert entry is not None

    prompt_only = list(range(70))

    assert (
        bank.near_prefix_candidates(
            prompt_only,
            max_token_gap=8,
            min_matched_tokens=64,
            allow_block_prefix=True,
        )
        == []
    )


def test_session_bank_contained_long_prompt_uses_block_prefix_not_answer_tail():
    bank = SessionBank(max_entries=4, max_bytes=1024, per_session_max_bytes=512)
    runtime = SimpleNamespace(model_path=Path("models/example"), mtp_enabled=True)
    entry = bank.put(
        runtime=runtime,
        token_ids=list(range(1200)),
        cache=[],
        logits=None,
        hidden=None,
        session_id="session-1",
        nbytes_override=128,
    )
    assert entry is not None

    prompt_inside_completion = list(range(1197))
    candidates = bank.near_prefix_candidates(
        prompt_inside_completion,
        max_token_gap=8,
        min_matched_tokens=64,
        block_size=256,
        block_min_matched_tokens=512,
        allow_block_prefix=True,
    )

    # kvcache-v2: matches are token-exact (no block quantization) for entries
    # that can restore at any offset. A contained prompt restores at its own
    # full length; the trim + seed-forward make that state cold-identical, so
    # the pre-v2 "back off to the last block edge" conservatism is obsolete.
    assert candidates == [(entry, 1197)]
    assert bank.last_prefix_diagnostic is not None
    assert bank.last_prefix_diagnostic["restore_kind"] == "near_boundary"


def test_session_bank_block_prefix_candidates_restore_large_agent_overlap():
    bank = SessionBank(max_entries=4, max_bytes=1024, per_session_max_bytes=512)
    runtime = SimpleNamespace(model_path=Path("models/example"), mtp_enabled=True)
    entry = bank.put(
        runtime=runtime,
        token_ids=list(range(1200)),
        cache=[],
        logits=None,
        hidden=None,
        session_id="session-1",
        nbytes_override=128,
    )
    assert entry is not None

    followup = list(range(1050)) + [99_001, 99_002, 99_003]
    candidates = bank.near_prefix_candidates(
        followup,
        max_token_gap=8,
        min_matched_tokens=64,
        block_size=256,
        block_min_matched_tokens=512,
        allow_block_prefix=True,
    )

    # kvcache-v2 token-granularity: the agent follow-up diverges at 1050, so
    # the candidate matches exactly there instead of backing off to 1024.
    assert candidates == [(entry, 1050)]
    assert bank.last_prefix_diagnostic is not None
    assert bank.last_prefix_diagnostic["restore_kind"] == "block_prefix"
    assert bank.last_prefix_diagnostic["new_prefill_tokens"] == len(followup) - 1050


# --- prefix-supersede (2026-07-04 multitask capacity fix) --------------------
# One busy OpenCode conversation banked 13/16 RAM entries (20.6 of 24 GB),
# a third of them strict prefixes of a newer entry; multitasking across
# projects then churned every other project out of RAM. A newer entry that
# extends an older one dominates it for every restore shape, so the bank
# drops the contained entry at put() time.


def test_session_bank_put_supersedes_contained_prefixes():
    bank = SessionBank(max_entries=8, max_bytes=4096, per_session_max_bytes=2048)
    runtime = SimpleNamespace(model_path=Path("models/example"), mtp_enabled=True)

    short = bank.put(
        runtime=runtime,
        token_ids=[1, 2, 3, 4],
        cache=[],
        logits=None,
        hidden=None,
        session_id="round-1",
        nbytes_override=64,
    )
    assert short is not None

    longer = bank.put(
        runtime=runtime,
        token_ids=[1, 2, 3, 4, 5, 6],
        cache=[],
        logits=None,
        hidden=None,
        session_id="round-2",
        nbytes_override=64,
    )
    assert longer is not None

    assert len(bank) == 1
    assert bank.longest_prefix([1, 2, 3, 4, 5, 6, 7]) is longer
    assert bank.eviction_log[-1]["reason"] == "superseded_by_longer_prefix"
    assert bank.eviction_log[-1]["prefix_len"] == 4


def test_session_bank_put_keeps_divergent_and_policy_mismatched_entries():
    bank = SessionBank(max_entries=8, max_bytes=4096, per_session_max_bytes=2048)
    runtime = SimpleNamespace(model_path=Path("models/example"), mtp_enabled=True)

    divergent = bank.put(
        runtime=runtime,
        token_ids=[1, 2, 9, 9],
        cache=[],
        logits=None,
        hidden=None,
        session_id="other-project",
        nbytes_override=64,
    )
    policy_mismatch = bank.put(
        runtime=runtime,
        token_ids=[1, 2, 3],
        cache=[],
        logits=None,
        hidden=None,
        session_id="old-policy",
        policy_fingerprint="policy-A",
        nbytes_override=64,
    )
    container = bank.put(
        runtime=runtime,
        token_ids=[1, 2, 3, 4, 5],
        cache=[],
        logits=None,
        hidden=None,
        session_id="round-2",
        policy_fingerprint="policy-B",
        nbytes_override=64,
    )

    assert divergent is not None
    assert policy_mismatch is not None
    assert container is not None
    # The divergent prefix is not contained; the contained entry carries a
    # different policy fingerprint and can serve requests the container
    # cannot. Both must survive.
    assert len(bank) == 3


def test_session_bank_recurrent_container_without_boundaries_does_not_supersede():
    bank = SessionBank(max_entries=8, max_bytes=4096, per_session_max_bytes=2048)
    runtime = RuntimeWithCaches()

    class RecurrentCache:
        state = None

        def is_trimmable(self) -> bool:
            return False

    short = bank.put(
        runtime=runtime,
        token_ids=[1, 2, 3, 4],
        cache=[RecurrentCache()],
        logits=None,
        hidden=None,
        session_id="round-1",
        nbytes_override=64,
    )
    longer = bank.put(
        runtime=runtime,
        token_ids=[1, 2, 3, 4, 5, 6],
        cache=[RecurrentCache()],
        logits=None,
        hidden=None,
        session_id="round-2",
        nbytes_override=64,
    )

    assert short is not None
    assert longer is not None
    # A recurrent container with no interior boundaries fails closed on
    # sub-prefix restores, so the shorter exact frontier still adds coverage.
    assert len(bank) == 2


def test_eviction_log_is_bounded_for_daemon_lifetime():
    # The log is appended on every eviction/skip forever while health
    # snapshots read only the newest entries: an unbounded list is pure
    # retention on a long-running agent daemon (external review F5).
    bank = SessionBank(max_entries=4, max_bytes=1024, per_session_max_bytes=512)
    runtime = SimpleNamespace(model_path=Path("models/example"), mtp_enabled=True)
    assert bank.eviction_log.maxlen == 256
    for index in range(300):
        bank.put(
            runtime=runtime,
            token_ids=[1, 2, index],
            cache=[],
            logits=None,
            hidden=None,
            session_id=f"session-{index}",
            nbytes_override=2048,
        )
    assert len(bank.eviction_log) == 256
    # Newest entry survives at the tail; the oldest 44 fell off the front.
    assert bank.eviction_log[-1]["reason"] == "skipped_oversized_snapshot"
