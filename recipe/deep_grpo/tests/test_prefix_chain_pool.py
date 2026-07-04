"""Unit tests for PrefixChainPool. Mirrors the simulator tests but with real
FailedTrajectoryEntry types.

Run directly:
    .venv/bin/python recipe/deep_grpo/tests/test_prefix_chain_pool.py

Or via pytest:
    .venv/bin/python -m pytest recipe/deep_grpo/tests/test_prefix_chain_pool.py -v
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from recipe.deep_grpo.pools.prefix_chain_pool import (
    ChainNode,
    ChainState,
    PrefixChain,
    PrefixChainPool,
)


# NOTE: The pool treats these entries as opaque objects (stored and passed
# through), so a structural duck-type with the relevant fields is sufficient
# for behavioural tests and keeps this file free of package imports.
@dataclass
class FakeFailedTrajectoryEntry:
    prompt_ids: List[int]
    response_ids: List[int]
    response_mask: List[int]
    data_instance: Dict[str, Any]
    tree_id: str
    num_turns: float
    agent_name: str = ""
    branch_points: Optional[List[Any]] = None


# ----------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------


def _mk_rollout(
    tree_id: str,
    prompt_ids: List[int],
    response_ids: List[int],
    agent_name: str = "test_agent",
) -> FakeFailedTrajectoryEntry:
    return FakeFailedTrajectoryEntry(
        prompt_ids=list(prompt_ids),
        response_ids=list(response_ids),
        response_mask=[1] * len(response_ids),
        data_instance={"answer": "42"},
        tree_id=tree_id,
        num_turns=1.0,
        agent_name=agent_name,
    )


def _mk_batch(
    tree_id: str,
    prompt_ids: List[int],
    n: int = 8,
    response_len: int = 50,
) -> List[FakeFailedTrajectoryEntry]:
    """Create n failed rollouts sharing a tree_id (same group)."""
    return [
        _mk_rollout(
            tree_id=tree_id,
            prompt_ids=prompt_ids,
            # Make each rollout slightly different so len-sorting has signal.
            response_ids=list(range(1000 + i, 1000 + i + response_len)),
        )
        for i in range(n)
    ]


def _check_invariants(pool: PrefixChainPool, ctx: str = ""):
    """Verify INV-1..5 on the pool's current state."""
    seen_keys = set()
    with pool._lock:
        for pk, chain in pool._chains.items():
            assert pk not in seen_keys, f"[{ctx}] INV-5 violation: duplicate key {pk}"
            seen_keys.add(pk)

            depths = [n.depth for n in chain.nodes]
            for i in range(1, len(depths)):
                assert depths[i] > depths[i - 1], (
                    f"[{ctx}] INV-2 violation in chain {pk}: depths {depths}"
                )

            if chain.state == ChainState.LEARNING:
                assert 0 <= chain.active_idx < len(chain.nodes), (
                    f"[{ctx}] INV-1 violation: LEARNING with invalid active_idx"
                )
                active = chain.active_node
                assert not active.is_mastered, (
                    f"[{ctx}] INV-1 violation: LEARNING active is mastered"
                )
                if active.last_k_succ is not None:
                    assert active.last_k_succ != 0, (
                        f"[{ctx}] INV-3 violation: LEARNING active has last_k_succ=0"
                    )
            for node in chain.nodes:
                assert not node.is_mastered, (
                    f"[{ctx}] INV-4 violation: chain {pk} has mastered node"
                )


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------


def test_happy_path():
    """Full lifecycle: main fail → DEEPENING → LEARNING → retreat → COMPLETED."""
    pool = PrefixChainPool()
    prompt_ids = [1, 2, 3, 4]

    # Main pool failure creates chain directly in DEEPENING_REQUESTED.
    rollouts = _mk_batch("main_tid_1", prompt_ids)
    key = pool.on_main_failure(
        original_prompt_ids=prompt_ids,
        data_instance={"answer": "42"},
        failed_rollouts=rollouts,
        current_step=0,
    )
    assert key is not None
    _check_invariants(pool, "after on_main_failure")
    chain = pool._chains[key]
    assert chain.state == ChainState.DEEPENING_REQUESTED
    assert len(chain.nodes) == 1
    assert chain.nodes[0].depth == 0
    assert len(chain.nodes[0].last_failed_rollouts) == 8

    # Teacher pulls pending requests.
    pending = pool.pending_teacher_requests()
    assert len(pending) == 1
    assert pending[0][0] == key
    assert len(pending[0][1]) == 8

    # Teacher succeeds: append N₁ at depth 100.
    pool.on_teacher_response(
        prompt_key=key,
        new_augmented_ids=list(prompt_ids) + list(range(9000, 9100)),
        data_instance={"answer": "42"},
        agent_name="test_agent",
        current_step=5,
        success=True,
    )
    _check_invariants(pool, "after teacher response")
    assert chain.state == ChainState.LEARNING
    assert len(chain.nodes) == 2
    assert chain.nodes[1].depth == 100
    assert chain.active_idx == 1

    # N₁ gets partial success — stay LEARNING.
    pool.record_observation(key, k_succ=3, k_total=8, current_step=10)
    _check_invariants(pool, "after partial")
    assert chain.state == ChainState.LEARNING
    assert chain.active_idx == 1

    # N₁ masters — retreat to N₀.
    pool.record_observation(key, k_succ=8, k_total=8, current_step=20)
    _check_invariants(pool, "after master N1")
    assert chain.state == ChainState.LEARNING
    assert chain.active_idx == 0
    assert len(chain.nodes) == 1  # N₁ popped
    assert chain.nodes[0].observations == 0  # reset

    # N₀ masters — COMPLETED.
    pool.record_observation(key, k_succ=8, k_total=8, current_step=30)
    _check_invariants(pool, "after master N0")
    assert chain.state == ChainState.COMPLETED
    assert len(chain.nodes) == 0
    print("test_happy_path PASSED")


def test_duplicate_on_main_failure_ignored():
    """Second main failure on same prompt is a no-op."""
    pool = PrefixChainPool()
    prompt_ids = [1, 2, 3]

    key1 = pool.on_main_failure(
        original_prompt_ids=prompt_ids,
        data_instance={"answer": "1"},
        failed_rollouts=_mk_batch("t1", prompt_ids),
        current_step=0,
    )
    assert key1 is not None
    assert len(pool) == 1

    key2 = pool.on_main_failure(
        original_prompt_ids=prompt_ids,
        data_instance={"answer": "1"},
        failed_rollouts=_mk_batch("t2", prompt_ids),
        current_step=5,
    )
    assert key2 is None  # no new chain created
    assert len(pool) == 1
    print("test_duplicate_on_main_failure_ignored PASSED")


def test_teacher_failure_abandons():
    pool = PrefixChainPool()
    prompt_ids = [1, 2, 3]
    key = pool.on_main_failure(
        original_prompt_ids=prompt_ids,
        data_instance={"answer": "1"},
        failed_rollouts=_mk_batch("t1", prompt_ids),
        current_step=0,
    )
    pool.pending_teacher_requests()  # drain queue
    pool.on_teacher_response(
        prompt_key=key,
        new_augmented_ids=None,
        data_instance=None,
        agent_name="",
        current_step=5,
        success=False,
    )
    _check_invariants(pool, "teacher_failure")
    assert pool._chains[key].state == ChainState.ABANDONED
    print("test_teacher_failure_abandons PASSED")


def test_teacher_bad_prefix_abandons():
    """Teacher returns augmented_prompt that doesn't start with original → ABANDONED.

    Catches upstream routing bugs (e.g., wrong prompt_key lookup sending a
    different chain's annotation here).
    """
    pool = PrefixChainPool()
    prompt_ids = [10, 20, 30]
    key = pool.on_main_failure(
        original_prompt_ids=prompt_ids,
        data_instance={"answer": "1"},
        failed_rollouts=_mk_batch("t1", prompt_ids),
        current_step=0,
    )
    pool.pending_teacher_requests()
    # Bad response: doesn't start with [10, 20, 30]
    pool.on_teacher_response(
        prompt_key=key,
        new_augmented_ids=[99, 98, 97, 50, 51, 52],
        data_instance={"answer": "1"},
        agent_name="",
        current_step=5,
    )
    _check_invariants(pool, "bad_prefix")
    assert pool._chains[key].state == ChainState.ABANDONED
    # And the metric counter advanced (verify it's the prefix one).
    assert pool._teacher_invalid_prefix_count == 1
    assert pool._teacher_invalid_depth_count == 0
    print("test_teacher_bad_prefix_abandons PASSED")


def test_teacher_short_response_abandons():
    """Teacher returns an augmented_prompt SHORTER than the original → ABANDONED."""
    pool = PrefixChainPool()
    prompt_ids = [10, 20, 30, 40, 50]
    key = pool.on_main_failure(
        original_prompt_ids=prompt_ids,
        data_instance={"answer": "1"},
        failed_rollouts=_mk_batch("t1", prompt_ids),
        current_step=0,
    )
    pool.pending_teacher_requests()
    pool.on_teacher_response(
        prompt_key=key,
        new_augmented_ids=[10, 20],  # shorter than original
        data_instance={"answer": "1"},
        agent_name="",
        current_step=5,
    )
    _check_invariants(pool, "short_response")
    assert pool._chains[key].state == ChainState.ABANDONED
    assert pool._teacher_invalid_prefix_count == 1
    print("test_teacher_short_response_abandons PASSED")


def test_teacher_invalid_depth_abandons():
    """Teacher returns depth ≤ existing last → ABANDONED."""
    pool = PrefixChainPool()
    prompt_ids = [1, 2, 3]
    key = pool.on_main_failure(
        original_prompt_ids=prompt_ids,
        data_instance={"answer": "1"},
        failed_rollouts=_mk_batch("t1", prompt_ids),
        current_step=0,
    )
    pool.pending_teacher_requests()
    # First deepening succeeds at depth 100.
    pool.on_teacher_response(
        prompt_key=key,
        new_augmented_ids=list(prompt_ids) + list(range(100)),
        data_instance={"answer": "1"},
        agent_name="",
        current_step=5,
    )
    # Drive back to DEEPENING by observing k=0 on the new active (N₁).
    pool.record_observation(key, k_succ=0, k_total=8, current_step=10,
                            failed_rollouts=_mk_batch("t2", prompt_ids))
    pool.pending_teacher_requests()
    # Teacher returns same depth (invalid).
    pool.on_teacher_response(
        prompt_key=key,
        new_augmented_ids=list(prompt_ids) + list(range(100)),  # same length → same depth
        data_instance={"answer": "1"},
        agent_name="",
        current_step=15,
    )
    _check_invariants(pool, "invalid_depth")
    assert pool._chains[key].state == ChainState.ABANDONED
    print("test_teacher_invalid_depth_abandons PASSED")


def test_deepening_requested_not_sampled():
    """Chain in DEEPENING_REQUESTED must not appear in sample()."""
    pool = PrefixChainPool()
    prompt_ids = [1, 2, 3]
    key = pool.on_main_failure(
        original_prompt_ids=prompt_ids,
        data_instance={"answer": "1"},
        failed_rollouts=_mk_batch("t1", prompt_ids),
        current_step=0,
    )
    assert pool._chains[key].state == ChainState.DEEPENING_REQUESTED
    sampled = pool.sample(10, current_step=1)
    assert sampled == []
    print("test_deepening_requested_not_sampled PASSED")


def test_partial_success_stays_learning():
    pool = PrefixChainPool()
    prompt_ids = [1, 2, 3]
    key = pool.on_main_failure(
        original_prompt_ids=prompt_ids,
        data_instance={"answer": "1"},
        failed_rollouts=_mk_batch("t1", prompt_ids),
        current_step=0,
    )
    pool.pending_teacher_requests()
    pool.on_teacher_response(
        prompt_key=key,
        new_augmented_ids=list(prompt_ids) + list(range(50)),
        data_instance={"answer": "1"},
        agent_name="",
        current_step=5,
    )
    # Active is N₁ at depth 50. Several partial observations.
    for step in range(10, 40, 5):
        pool.record_observation(key, k_succ=3, k_total=8, current_step=step)
    _check_invariants(pool, "partial_success")
    assert pool._chains[key].state == ChainState.LEARNING
    assert pool._chains[key].active_idx == 1
    print("test_partial_success_stays_learning PASSED")


def test_immediate_master_completes():
    """If the first chain observation masters N₀ directly, chain is COMPLETED."""
    pool = PrefixChainPool()
    prompt_ids = [1, 2, 3]
    key = pool.on_main_failure(
        original_prompt_ids=prompt_ids,
        data_instance={"answer": "1"},
        failed_rollouts=_mk_batch("t1", prompt_ids),
        current_step=0,
    )
    pool.pending_teacher_requests()
    pool.on_teacher_response(
        prompt_key=key,
        new_augmented_ids=list(prompt_ids) + list(range(50)),
        data_instance={"answer": "1"},
        agent_name="",
        current_step=5,
    )
    # N₁ masters immediately — retreat to N₀.
    pool.record_observation(key, k_succ=8, k_total=8, current_step=10)
    assert pool._chains[key].active_idx == 0
    # N₀ masters immediately — COMPLETED.
    pool.record_observation(key, k_succ=8, k_total=8, current_step=15)
    _check_invariants(pool, "immediate_master")
    assert pool._chains[key].state == ChainState.COMPLETED
    print("test_immediate_master_completes PASSED")


def test_stuck_triggers_immediate_deepening():
    """Single k_succ=0 observation in LEARNING → DEEPENING_REQUESTED."""
    pool = PrefixChainPool()
    prompt_ids = [1, 2, 3]
    key = pool.on_main_failure(
        original_prompt_ids=prompt_ids,
        data_instance={"answer": "1"},
        failed_rollouts=_mk_batch("t1", prompt_ids),
        current_step=0,
    )
    pool.pending_teacher_requests()
    pool.on_teacher_response(
        prompt_key=key,
        new_augmented_ids=list(prompt_ids) + list(range(50)),
        data_instance={"answer": "1"},
        agent_name="",
        current_step=5,
    )
    assert pool._chains[key].state == ChainState.LEARNING
    pool.record_observation(
        key, k_succ=0, k_total=8, current_step=10,
        failed_rollouts=_mk_batch("t2", list(prompt_ids) + list(range(50))),
    )
    _check_invariants(pool, "stuck_immediate")
    assert pool._chains[key].state == ChainState.DEEPENING_REQUESTED
    assert len(pool._chains[key].nodes[-1].last_failed_rollouts) == 8
    print("test_stuck_triggers_immediate_deepening PASSED")


def test_sample_weights_untried_and_partial():
    """Untried nodes get weight 1.0; partial success gets 4p(1-p)."""
    pool = PrefixChainPool()
    # Build two chains both in LEARNING.
    for i, pid in enumerate([[10], [20], [30]]):
        key = pool.on_main_failure(
            original_prompt_ids=pid,
            data_instance={"answer": str(i)},
            failed_rollouts=_mk_batch(f"t{i}", pid),
            current_step=0,
        )
        pool.pending_teacher_requests()
        pool.on_teacher_response(
            prompt_key=key,
            new_augmented_ids=list(pid) + list(range(50)),
            data_instance={"answer": str(i)},
            agent_name="",
            current_step=1,
        )
    # Chain 0: leave untried → weight 1.0
    # Chain 1: set partial p=0.5 → weight 1.0
    # Chain 2: set partial p=0.125 → weight 0.4375
    chains = list(pool._chains.values())
    # chain 0 stays untried
    chains[1].active_node.last_k_succ = 4
    chains[1].active_node.last_k_total = 8
    chains[2].active_node.last_k_succ = 1
    chains[2].active_node.last_k_total = 8

    w0 = pool._node_weight(chains[0].active_node)
    w1 = pool._node_weight(chains[1].active_node)
    w2 = pool._node_weight(chains[2].active_node)
    assert abs(w0 - 1.0) < 1e-9
    assert abs(w1 - 1.0) < 1e-9
    assert abs(w2 - 4 * 0.125 * 0.875) < 1e-9
    print("test_sample_weights_untried_and_partial PASSED")


def test_sample_returns_active_nodes():
    pool = PrefixChainPool()
    for i, pid in enumerate([[10], [20], [30]]):
        key = pool.on_main_failure(
            original_prompt_ids=pid,
            data_instance={"answer": str(i)},
            failed_rollouts=_mk_batch(f"t{i}", pid),
            current_step=0,
        )
        pool.pending_teacher_requests()
        pool.on_teacher_response(
            prompt_key=key,
            new_augmented_ids=list(pid) + list(range(20)),
            data_instance={"answer": str(i)},
            agent_name="",
            current_step=1,
        )
    # All three in LEARNING now.
    assert pool.active_chains_count() == 3
    sampled = pool.sample(2, current_step=5)
    assert len(sampled) == 2
    for pk, node in sampled:
        assert pk in pool._chains
        assert node is pool._chains[pk].active_node
    print("test_sample_returns_active_nodes PASSED")


def test_state_dict_roundtrip():
    pool = PrefixChainPool()
    prompt_ids = [1, 2, 3]
    pool.on_main_failure(
        original_prompt_ids=prompt_ids,
        data_instance={"answer": "1"},
        failed_rollouts=_mk_batch("t1", prompt_ids),
        current_step=0,
    )
    state = pool.state_dict()

    pool2 = PrefixChainPool()
    pool2.load_state_dict(state)
    assert len(pool2) == 1
    _check_invariants(pool2, "after_load")
    print("test_state_dict_roundtrip PASSED")


def test_load_reenqueues_orphaned_deepening():
    """If a DEEPENING chain is missing from the queue (teacher crashed after
    popping it), load_state_dict must re-enqueue it on resume.
    """
    pool = PrefixChainPool()
    prompt_ids = [1, 2, 3]
    pool.on_main_failure(
        original_prompt_ids=prompt_ids,
        data_instance={"answer": "1"},
        failed_rollouts=_mk_batch("t1", prompt_ids),
        current_step=0,
    )
    # Simulate teacher popping from queue without completing: drain once,
    # then persist state with queue empty but chain still in DEEPENING.
    pool.pending_teacher_requests()  # drains queue
    assert len(pool._deepening_queue) == 0
    chain_key = next(iter(pool._chains.keys()))
    assert pool._chains[chain_key].state == ChainState.DEEPENING_REQUESTED
    state = pool.state_dict()
    assert state["deepening_queue"] == []

    # Load into fresh pool — safety net must re-enqueue.
    pool2 = PrefixChainPool()
    pool2.load_state_dict(state)
    assert chain_key in pool2._deepening_queue
    # And pending_teacher_requests should now see it again.
    pending = pool2.pending_teacher_requests()
    assert len(pending) == 1
    assert pending[0][0] == chain_key
    print("test_load_reenqueues_orphaned_deepening PASSED")


def run_all():
    tests = [
        test_happy_path,
        test_duplicate_on_main_failure_ignored,
        test_teacher_failure_abandons,
        test_teacher_bad_prefix_abandons,
        test_teacher_short_response_abandons,
        test_teacher_invalid_depth_abandons,
        test_deepening_requested_not_sampled,
        test_partial_success_stays_learning,
        test_immediate_master_completes,
        test_stuck_triggers_immediate_deepening,
        test_sample_weights_untried_and_partial,
        test_sample_returns_active_nodes,
        test_state_dict_roundtrip,
        test_load_reenqueues_orphaned_deepening,
    ]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} unit tests PASSED.")


if __name__ == "__main__":
    run_all()
