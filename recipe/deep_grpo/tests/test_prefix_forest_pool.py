"""Unit tests for hard-state PrefixForestPool.

Run directly:
    PYTHONPATH=. python3 recipe/deep_grpo/tests/test_prefix_forest_pool.py
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from recipe.deep_grpo.pools.prefix_forest_pool import PrefixForestPool


@dataclass
class FakeRewardInfo:
    reward: float
    completed: int
    finished: int = 0


@dataclass
class FakeTeacherSuffix:
    suffix_ids: List[int]
    suffix_mask: List[int]
    reward: float
    reward_info: FakeRewardInfo
    original_failed_suffix_ids: List[int]


@dataclass
class FakeFailedTrajectoryEntry:
    prompt_ids: List[int]
    response_ids: List[int]
    response_mask: List[int]
    data_instance: Dict[str, Any]
    tree_id: str
    num_turns: float
    agent_name: str = ""


@dataclass
class FakeBranchPointEntry:
    prompt_ids: List[int]
    response_ids: List[int]
    response_mask: List[int]
    data_instance: Dict[str, Any]
    num_turns: float
    tree_id: str
    branch_chain_root_index: int
    chain_total_length: int
    agent_name: str = ""
    teacher_suffix: Optional[FakeTeacherSuffix] = None


def _failed(prompt, response, tree_id="t", agent_name="agent"):
    return FakeFailedTrajectoryEntry(
        prompt_ids=list(prompt),
        response_ids=list(response),
        response_mask=[1] * len(response),
        data_instance={"extra_info": {"answer": "x"}},
        tree_id=tree_id,
        num_turns=1.0,
        agent_name=agent_name,
    )


def _suffix(ids):
    return FakeTeacherSuffix(
        suffix_ids=list(ids),
        suffix_mask=[1] * len(ids),
        reward=1.0,
        reward_info=FakeRewardInfo(reward=1.0, completed=1, finished=1),
        original_failed_suffix_ids=[99],
    )


def _annotated(prompt, response, suffix_ids=(50,)):
    return FakeBranchPointEntry(
        prompt_ids=list(prompt),
        response_ids=list(response),
        response_mask=[1] * len(response),
        data_instance={"extra_info": {"answer": "x"}},
        num_turns=1.0,
        tree_id="teacher",
        branch_chain_root_index=0,
        chain_total_length=1,
        agent_name="agent",
        teacher_suffix=_suffix(suffix_ids),
    )


def _tree(pool, prompt):
    return pool._trees[pool.compute_tree_key(prompt)]


def _root(pool, prompt):
    tree = _tree(pool, prompt)
    return tree, tree.root_id, tree.nodes[tree.root_id]


def _create_root_events(pool, prompt, current_step=0, k_succ=0, k_total=2):
    failures = [
        _failed(prompt, [10], tree_id=f"{prompt}-a"),
        _failed(prompt, [11], tree_id=f"{prompt}-b"),
    ][: max(0, k_total - k_succ)]
    pool.record_root_observation(
        original_prompt_ids=list(prompt),
        data_instance={"prompt": list(prompt)},
        agent_name="agent",
        k_succ=k_succ,
        k_total=k_total,
        current_step=current_step,
        failed_rollouts=failures,
    )
    return failures


def _create_child(pool, prompt, response, step=1, suffix_ids=(50,)):
    _create_root_events(pool, prompt, current_step=step - 1, k_succ=0, k_total=1)
    event = pool.pending_teacher_requests(max_items=1, current_step=step)[0]
    pool.on_teacher_response(
        event_id=event.event_id,
        annotated_entry=_annotated(prompt, response, suffix_ids=suffix_ids),
        current_step=step + 1,
        success=True,
    )
    tree = _tree(pool, prompt)
    child_ids = [
        node_id
        for node_id, node in tree.nodes.items()
        if node.parent_id == tree.root_id
    ]
    return tree, child_ids[-1], tree.nodes[child_ids[-1]]


def test_all_fail_and_partial_success_create_teacher_events():
    pool = PrefixForestPool(max_model_len=1024)
    _create_root_events(pool, [1, 2], current_step=1, k_succ=0, k_total=2)
    _create_root_events(pool, [3, 4], current_step=1, k_succ=1, k_total=3)

    assert pool.stats["teacher/events_queued"] == 4
    assert pool.stats["forest/num_trees"] == 2
    print("test_all_fail_and_partial_success_create_teacher_events PASSED")


def test_root_partial_success_without_failed_rollout_does_not_create_orphan_tree():
    pool = PrefixForestPool(max_model_len=1024)
    update = pool.record_root_observation(
        original_prompt_ids=[1, 2],
        data_instance={},
        agent_name="agent",
        k_succ=1,
        k_total=2,
        current_step=1,
        failed_rollouts=[],
    )

    assert update["tree_created"] == 0
    assert pool.stats["forest/num_trees"] == 0
    assert pool.stats["teacher/events_queued"] == 0
    print("test_root_partial_success_without_failed_rollout_does_not_create_orphan_tree PASSED")


def test_all_success_root_deactivates_root_without_deleting_children():
    pool = PrefixForestPool(max_model_len=1024)
    prompt = [1, 2]
    tree, child_id, child = _create_child(pool, prompt, [10])

    update = pool.record_root_observation(
        original_prompt_ids=prompt,
        data_instance={},
        agent_name="",
        k_succ=2,
        k_total=2,
        current_step=3,
        failed_rollouts=None,
    )

    assert update["root_deactivated"] == 1
    assert pool.compute_tree_key(prompt) in pool._trees
    assert not tree.nodes[tree.root_id].active
    assert child_id in tree.nodes
    assert child.active
    assert pool.sample(10, current_step=4)[0][1] == child_id
    print("test_all_success_root_deactivates_root_without_deleting_children PASSED")


def test_all_success_nonroot_deactivates_only_current_node():
    pool = PrefixForestPool(max_model_len=1024)
    prompt = [1, 2]
    tree, child_id, _child = _create_child(pool, prompt, [10], step=1)

    # Create an active grandchild under child.
    pool.record_observation(
        tree_key=tree.tree_key,
        node_id=child_id,
        k_succ=0,
        k_total=1,
        current_step=3,
        failed_rollouts=[_failed(prompt + [10], [20])],
    )
    event = pool.pending_teacher_requests(max_items=1, current_step=4)[0]
    pool.on_teacher_response(
        event.event_id,
        _annotated(prompt + [10], [20], suffix_ids=(60,)),
        current_step=5,
        success=True,
    )
    grandchild_id = [
        node_id
        for node_id, node in tree.nodes.items()
        if node.parent_id == child_id
    ][0]

    update = pool.record_observation(
        tree.tree_key,
        child_id,
        k_succ=1,
        k_total=1,
        current_step=6,
    )

    assert update["node_deactivated"] == 1
    assert not tree.nodes[child_id].active
    assert grandchild_id in tree.nodes
    assert tree.nodes[grandchild_id].active
    print("test_all_success_nonroot_deactivates_only_current_node PASSED")


def test_inactive_node_not_sampled_for_rollout_sft_or_teacher_dispatch():
    pool = PrefixForestPool(max_model_len=1024)
    prompt = [1, 2]
    tree, child_id, _child = _create_child(pool, prompt, [10])

    pool.record_observation(
        tree.tree_key,
        child_id,
        k_succ=0,
        k_total=1,
        current_step=3,
        failed_rollouts=[_failed(prompt + [10], [30])],
    )
    pending_before = pool.stats["teacher/events_queued"]
    pool.record_observation(
        tree.tree_key,
        child_id,
        k_succ=1,
        k_total=1,
        current_step=4,
    )

    assert pending_before == 1
    assert pool.sample(10, current_step=5) == []
    assert pool.sample_suffix_sft(10, current_step=5) == []
    assert pool.pending_teacher_requests(max_items=10, current_step=5) == []
    print("test_inactive_node_not_sampled_for_rollout_sft_or_teacher_dispatch PASSED")


def test_later_success_does_not_delete_pending_or_inflight_events():
    pool = PrefixForestPool(max_model_len=1024)
    prompt = [1, 2]
    _create_root_events(pool, prompt, current_step=1, k_succ=0, k_total=2)
    assert pool.stats["teacher/events_queued"] == 2

    pool.record_root_observation(
        prompt,
        data_instance={},
        agent_name="",
        k_succ=1,
        k_total=1,
        current_step=2,
    )
    assert pool.stats["teacher/events_queued"] == 2
    assert pool.pending_teacher_requests(max_items=10, current_step=3) == []

    # In-flight events are also retained until the teacher response arrives.
    prompt2 = [3, 4]
    _create_root_events(pool, prompt2, current_step=1, k_succ=0, k_total=1)
    event = pool.pending_teacher_requests(max_items=1, current_step=2)[0]
    pool.record_root_observation(
        prompt2,
        data_instance={},
        agent_name="",
        k_succ=1,
        k_total=1,
        current_step=3,
    )
    assert pool.stats["teacher/events_in_flight"] == 1
    pool.on_teacher_response(
        event.event_id,
        _annotated(prompt2, [10]),
        current_step=4,
        success=True,
    )
    assert pool.stats["teacher/events_stale_parent_inactive"] == 1
    print("test_later_success_does_not_delete_pending_or_inflight_events PASSED")


def test_teacher_dispatch_lru_uses_ascending_step_with_never_used_first():
    pool = PrefixForestPool(max_model_len=1024)
    p1 = [1]
    p2 = [2]
    _create_root_events(pool, p1, current_step=1, k_succ=0, k_total=1)
    first = pool.pending_teacher_requests(max_items=1, current_step=10)[0]
    pool.on_teacher_response(first.event_id, None, current_step=10, success=False)
    _create_root_events(pool, p1, current_step=11, k_succ=0, k_total=1)
    _create_root_events(pool, p2, current_step=11, k_succ=0, k_total=1)

    event = pool.pending_teacher_requests(max_items=1, current_step=12)[0]
    assert event.tree_key == pool.compute_tree_key(p2)
    print("test_teacher_dispatch_lru_uses_ascending_step_with_never_used_first PASSED")


def test_same_parent_dispatches_latest_pending_event():
    pool = PrefixForestPool(max_model_len=1024)
    prompt = [1]
    _create_root_events(pool, prompt, current_step=1, k_succ=0, k_total=1)
    _create_root_events(pool, prompt, current_step=2, k_succ=0, k_total=1)

    event = pool.pending_teacher_requests(max_items=1, current_step=3)[0]
    assert event.created_step == 2
    print("test_same_parent_dispatches_latest_pending_event PASSED")


def test_teacher_dispatch_cleans_stale_tree_queue_ids():
    pool = PrefixForestPool(max_model_len=1024)
    prompt = [1]
    _create_root_events(pool, prompt, current_step=1, k_succ=0, k_total=1)
    _create_root_events(pool, prompt, current_step=2, k_succ=0, k_total=1)
    tree_key = pool.compute_tree_key(prompt)
    pool._failed_by_tree[tree_key].appendleft("missing-event-id")

    event = pool.pending_teacher_requests(max_items=1, current_step=3)[0]

    assert event.created_step == 2
    assert "missing-event-id" not in pool._failed_by_tree[tree_key]
    print("test_teacher_dispatch_cleans_stale_tree_queue_ids PASSED")


def test_teacher_success_can_create_multiple_children_for_same_parent():
    pool = PrefixForestPool(max_model_len=1024)
    prompt = [1]
    _create_root_events(pool, prompt, current_step=1, k_succ=0, k_total=1)
    _create_root_events(pool, prompt, current_step=2, k_succ=0, k_total=1)

    event1 = pool.pending_teacher_requests(max_items=1, current_step=3)[0]
    pool.on_teacher_response(
        event1.event_id,
        _annotated(prompt, [10], suffix_ids=(50,)),
        current_step=4,
        success=True,
    )
    event2 = pool.pending_teacher_requests(max_items=1, current_step=5)[0]
    pool.on_teacher_response(
        event2.event_id,
        _annotated(prompt, [11], suffix_ids=(51,)),
        current_step=6,
        success=True,
    )

    tree, root_id, _root_node = _root(pool, prompt)
    children = [tree.nodes[child_id].augmented_prompt_ids for child_id in tree.nodes[root_id].children]
    assert [1, 10] in children
    assert [1, 11] in children
    assert pool.stats["teacher/events_succeeded"] == 2
    print("test_teacher_success_can_create_multiple_children_for_same_parent PASSED")


def test_duplicate_locked_prefix_is_dropped():
    pool = PrefixForestPool(max_model_len=1024)
    prompt = [1]
    _create_root_events(pool, prompt, current_step=1, k_succ=0, k_total=1)
    _create_root_events(pool, prompt, current_step=2, k_succ=0, k_total=1)

    event1 = pool.pending_teacher_requests(max_items=1, current_step=3)[0]
    pool.on_teacher_response(
        event1.event_id,
        _annotated(prompt, [10], suffix_ids=(50,)),
        current_step=4,
        success=True,
    )
    event2 = pool.pending_teacher_requests(max_items=1, current_step=5)[0]
    pool.on_teacher_response(
        event2.event_id,
        _annotated(prompt, [10], suffix_ids=(51,)),
        current_step=6,
        success=True,
    )

    tree, root_id, _root_node = _root(pool, prompt)
    assert len(tree.nodes[root_id].children) == 1
    assert pool.stats["teacher/events_duplicate_dropped"] == 1
    print("test_duplicate_locked_prefix_is_dropped PASSED")


def test_teacher_success_without_suffix_is_invalid():
    pool = PrefixForestPool(max_model_len=1024)
    prompt = [1]
    _create_root_events(pool, prompt, current_step=1, k_succ=0, k_total=1)
    event = pool.pending_teacher_requests(max_items=1, current_step=2)[0]

    annotated = _annotated(prompt, [10])
    annotated.teacher_suffix = None
    pool.on_teacher_response(
        event.event_id,
        annotated,
        current_step=3,
        success=True,
    )

    tree, root_id, _root_node = _root(pool, prompt)
    assert tree.nodes[root_id].children == []
    assert pool.stats["teacher/events_invalid"] == 1
    assert pool.stats["forest/teacher_suffix_nodes"] == 0
    print("test_teacher_success_without_suffix_is_invalid PASSED")


def test_prefix_injection_samples_at_most_one_node_per_tree():
    pool = PrefixForestPool(max_model_len=1024)
    p1 = [1]
    p2 = [2]
    tree1, _child1, _ = _create_child(pool, p1, [10], step=1)
    _create_child(pool, p1, [11], step=3)
    _create_child(pool, p2, [20], step=5)

    sampled = pool.sample(10, current_step=7)
    sampled_tree_keys = [tree_key for tree_key, _node_id, _node in sampled]
    assert len(sampled) == 2
    assert sampled_tree_keys.count(tree1.tree_key) == 1
    print("test_prefix_injection_samples_at_most_one_node_per_tree PASSED")


def test_prefix_injection_node_lru_uses_never_used_first():
    pool = PrefixForestPool(max_model_len=1024)
    prompt = [1]
    _tree1, child1_id, _ = _create_child(pool, prompt, [10], step=1)
    _create_child(pool, prompt, [11], step=3)

    first = pool.sample(1, current_step=10)
    second = pool.sample(1, current_step=11)

    assert first[0][1] == child1_id
    assert second[0][1] != child1_id
    print("test_prefix_injection_node_lru_uses_never_used_first PASSED")


def test_prefix_injection_does_not_resample_tree_in_same_step():
    pool = PrefixForestPool(max_model_len=1024)
    prompt = [1]
    _create_child(pool, prompt, [10], step=1)
    _create_child(pool, prompt, [11], step=3)

    first = pool.sample(1, current_step=10)
    second = pool.sample(1, current_step=10)
    third = pool.sample(1, current_step=11)

    assert len(first) == 1
    assert second == []
    assert len(third) == 1
    print("test_prefix_injection_does_not_resample_tree_in_same_step PASSED")


def test_sft_replay_samples_at_most_one_suffix_node_per_tree():
    pool = PrefixForestPool(max_model_len=1024)
    p1 = [1]
    p2 = [2]
    tree1, _child1, _ = _create_child(pool, p1, [10], step=1)
    _create_child(pool, p1, [11], step=3)
    _create_child(pool, p2, [20], step=5)

    sampled = pool.sample_suffix_sft(10, current_step=7)
    sampled_tree_keys = [tree_key for tree_key, _node_id, _node in sampled]
    assert len(sampled) == 2
    assert sampled_tree_keys.count(tree1.tree_key) == 1
    print("test_sft_replay_samples_at_most_one_suffix_node_per_tree PASSED")


def test_sft_sampling_after_deactivation_uses_current_active_set():
    pool = PrefixForestPool(max_model_len=1024)
    child_routes = []
    for i in range(65):
        prompt = [1000 + i]
        tree, child_id, _child = _create_child(
            pool,
            prompt,
            [2000 + i],
            step=2 * i + 1,
        )
        child_routes.append((tree.tree_key, child_id))

    stale_tree_key, stale_child_id = child_routes[0]
    pool.record_observation(
        stale_tree_key,
        stale_child_id,
        k_succ=1,
        k_total=1,
        current_step=200,
    )

    ready_trees, ready_nodes = pool.suffix_sft_ready_counts(max_nodes_per_tree=1)
    sampled = pool.sample_suffix_sft(64, current_step=201, max_nodes_per_tree=1)
    sampled_routes = {(tree_key, node_id) for tree_key, node_id, _node in sampled}

    assert ready_trees == 64
    assert ready_nodes == 64
    assert len(sampled) == 64
    assert len({tree_key for tree_key, _node_id, _node in sampled}) == 64
    assert (stale_tree_key, stale_child_id) not in sampled_routes
    print("test_sft_sampling_after_deactivation_uses_current_active_set PASSED")


def test_epoch_suffix_freeze_filters_window_and_clears_teacher_events():
    pool = PrefixForestPool(max_model_len=1024)

    old_tree, old_child_id, _old_child = _create_child(
        pool,
        [1],
        [10],
        step=1,
    )
    new_tree, new_child_id, _new_child = _create_child(
        pool,
        [2],
        [20],
        step=11,
    )
    inactive_tree, inactive_child_id, _inactive_child = _create_child(
        pool,
        [3],
        [30],
        step=11,
    )
    pool.record_observation(
        inactive_tree.tree_key,
        inactive_child_id,
        k_succ=1,
        k_total=1,
        current_step=13,
    )

    _create_root_events(pool, [4], current_step=12, k_succ=0, k_total=1)
    _create_root_events(pool, [5], current_step=12, k_succ=0, k_total=1)
    in_flight = pool.pending_teacher_requests(max_items=1, current_step=13)[0]

    snapshot, clear_stats = (
        pool.freeze_suffix_sft_epoch_and_clear_teacher_events(
            min_created_step=10,
            max_created_step=12,
        )
    )
    routes = {(tree_key, node_id) for tree_key, node_id, _node in snapshot}

    assert (new_tree.tree_key, new_child_id) in routes
    assert (old_tree.tree_key, old_child_id) not in routes
    assert (inactive_tree.tree_key, inactive_child_id) not in routes
    assert len(snapshot) == 1
    assert clear_stats["teacher_events_cleared"] == 2
    assert clear_stats["teacher_events_cleared_pending"] == 1
    assert clear_stats["teacher_events_cleared_in_flight"] == 1
    assert pool.stats["teacher/events_queued"] == 0
    assert pool.stats["teacher/events_in_flight"] == 0
    assert pool.pending_teacher_requests(max_items=10, current_step=14) == []

    pool.on_teacher_response(
        in_flight.event_id,
        _annotated([5], [50]),
        current_step=14,
        success=True,
    )
    stale_tree = _tree(pool, [5])
    assert stale_tree.nodes[stale_tree.root_id].children == []
    assert pool.stats["teacher/events_stale"] == 1
    print("test_epoch_suffix_freeze_filters_window_and_clears_teacher_events PASSED")


def test_mark_suffix_sft_sampled_updates_lru_without_maturing():
    pool = PrefixForestPool(max_model_len=1024)
    tree, child_id, _child = _create_child(pool, [1], [10])

    sampled = pool.mark_suffix_sft_sampled(
        [(tree.tree_key, child_id)],
        current_step=7,
    )

    assert sampled == 1
    assert tree.last_sft_sampled_step == 7
    assert tree.nodes[child_id].last_sft_step == 7
    assert pool.stats["sft/sampled_nodes"] == 1
    assert pool.suffix_sft_trainable(tree.tree_key, child_id)
    print("test_mark_suffix_sft_sampled_updates_lru_without_maturing PASSED")


def test_sft_node_lru_uses_never_used_first():
    pool = PrefixForestPool(max_model_len=1024)
    prompt = [1]
    _tree1, child1_id, _ = _create_child(pool, prompt, [10], step=1)
    _create_child(pool, prompt, [11], step=3)

    first = pool.sample_suffix_sft(1, current_step=10)
    second = pool.sample_suffix_sft(1, current_step=11)

    assert first[0][1] == child1_id
    assert second[0][1] != child1_id
    print("test_sft_node_lru_uses_never_used_first PASSED")


def test_teacher_dispatch_parent_lru_uses_never_dispatched_first():
    pool = PrefixForestPool(max_model_len=1024)
    prompt = [1]
    tree, child_id, _ = _create_child(pool, prompt, [10], step=1)

    pool.record_root_observation(
        original_prompt_ids=prompt,
        data_instance={"prompt": prompt},
        agent_name="agent",
        k_succ=0,
        k_total=1,
        current_step=4,
        failed_rollouts=[_failed(prompt, [12])],
    )
    pool.record_observation(
        tree.tree_key,
        child_id,
        k_succ=0,
        k_total=1,
        current_step=4,
        failed_rollouts=[_failed(prompt + [10], [20])],
    )

    event = pool.pending_teacher_requests(max_items=1, current_step=5)[0]
    assert event.parent_node_id == child_id
    print("test_teacher_dispatch_parent_lru_uses_never_dispatched_first PASSED")


def test_sft_update_does_not_mature_node_out_of_eligibility():
    pool = PrefixForestPool(max_model_len=1024)
    prompt = [1]
    tree, child_id, _child = _create_child(pool, prompt, [10])

    sampled = pool.sample_suffix_sft(1, current_step=3)
    assert sampled[0][1] == child_id
    update = pool.record_suffix_sft(tree.tree_key, child_id, current_step=4)
    assert update["suffix_sft_recorded"] == 1
    assert update["suffix_sft_matured"] == 0
    assert pool.suffix_sft_trainable(tree.tree_key, child_id)
    assert pool.sample_suffix_sft(1, current_step=5)[0][1] == child_id
    print("test_sft_update_does_not_mature_node_out_of_eligibility PASSED")


def test_cleanup_only_removes_tree_without_active_nodes_or_events():
    pool = PrefixForestPool(max_model_len=1024)
    prompt = [1]
    _create_root_events(pool, prompt, current_step=1, k_succ=0, k_total=1)
    event_id = next(iter(pool._events))

    pool.record_root_observation(
        prompt,
        data_instance={},
        agent_name="",
        k_succ=1,
        k_total=1,
        current_step=2,
    )
    assert pool.cleanup_inactive_trees() == 0

    pool.on_teacher_response(event_id, None, current_step=3, success=False)
    assert pool.cleanup_inactive_trees() == 1
    assert pool.stats["forest/num_trees"] == 0
    print("test_cleanup_only_removes_tree_without_active_nodes_or_events PASSED")


def test_state_dict_roundtrip_requeues_inflight_events():
    pool = PrefixForestPool(max_model_len=1024)
    prompt = [1]
    _create_root_events(pool, prompt, current_step=1, k_succ=0, k_total=1)
    dispatched = pool.pending_teacher_requests(max_items=1, current_step=5)[0]
    assert pool.stats["teacher/events_in_flight"] == 1

    state = pool.state_dict()
    restored = PrefixForestPool(max_model_len=1024)
    restored.load_state_dict(state)

    assert restored.stats["teacher/events_in_flight"] == 0
    assert restored.stats["teacher/events_queued"] == 1
    retry = restored.pending_teacher_requests(max_items=1, current_step=6)
    assert len(retry) == 1
    assert retry[0].event_id == dispatched.event_id
    tree = _tree(restored, prompt)
    assert tree.last_teacher_dispatched_step == 6
    print("test_state_dict_roundtrip_requeues_inflight_events PASSED")


def test_load_state_dict_drops_legacy_solved_trees():
    pool = PrefixForestPool(max_model_len=1024)
    prompt = [1]
    _create_root_events(pool, prompt, current_step=1, k_succ=0, k_total=1)

    state = pool.state_dict()
    for tree in state["trees"].values():
        tree.solved = True

    restored = PrefixForestPool(max_model_len=1024)
    restored.load_state_dict(state)

    assert restored.stats["forest/num_trees"] == 0
    assert restored.stats["teacher/events_queued"] == 0
    assert restored.pending_teacher_requests(max_items=10, current_step=2) == []
    print("test_load_state_dict_drops_legacy_solved_trees PASSED")


def _install_teacher_worker_import_stubs():
    import sys
    import types

    saved_modules = {}
    stubs = {
        "recipe.deep_grpo.pools.failed_trajectory_pool": {"FailedTrajectoryPool": object},
        "recipe.deep_grpo.pools.teacher_annotated_pool": {"TeacherAnnotatedPool": object},
        "recipe.deep_grpo.pools.synthetic_prompt_pool": {"SyntheticPromptPool": object},
        "recipe.deep_grpo.pools.prefix_chain_pool": {"PrefixChainPool": object},
    }
    for module_name, attrs in stubs.items():
        saved_modules[module_name] = sys.modules.get(module_name)
        if module_name in sys.modules:
            continue
        module = types.ModuleType(module_name)
        for attr_name, value in attrs.items():
            setattr(module, attr_name, value)
        sys.modules[module_name] = module
    return saved_modules


def _restore_modules(saved_modules):
    import sys

    for module_name, module in saved_modules.items():
        if module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = module


def test_teacher_worker_forest_fetch_dispatches_one_event_per_permit():
    import importlib
    import sys

    saved_modules = _install_teacher_worker_import_stubs()
    saved_teacher_worker = sys.modules.pop("recipe.deep_grpo.teacher_worker", None)
    try:
        teacher_worker = importlib.import_module("recipe.deep_grpo.teacher_worker")
        TeacherAnnotationWorker = teacher_worker.TeacherAnnotationWorker

        pool = PrefixForestPool(max_model_len=1024)
        _create_root_events(pool, [1], current_step=1, k_succ=0, k_total=1)
        _create_root_events(pool, [2], current_step=1, k_succ=0, k_total=1)

        worker = TeacherAnnotationWorker(
            config={},
            tokenizer=None,
            failed_pool=None,
            annotated_pool=None,
            agent_loop_class=object,
            agent_loop_config={},
            forest_pool=pool,
        )
        worker.update_current_step(2)

        work1 = worker._fetch_next_work()
        assert work1 is not None
        assert work1[0] == "forest"
        assert pool.stats["teacher/events_in_flight"] == 1
        assert pool.stats["teacher/events_queued"] == 1

        work2 = worker._fetch_next_work()
        assert work2 is not None
        assert work2[0] == "forest"
        assert pool.stats["teacher/events_in_flight"] == 2
        assert pool.stats["teacher/events_queued"] == 0
    finally:
        sys.modules.pop("recipe.deep_grpo.teacher_worker", None)
        if saved_teacher_worker is not None:
            sys.modules["recipe.deep_grpo.teacher_worker"] = saved_teacher_worker
        _restore_modules(saved_modules)
    print("test_teacher_worker_forest_fetch_dispatches_one_event_per_permit PASSED")


def run_all():
    tests = [
        test_all_fail_and_partial_success_create_teacher_events,
        test_root_partial_success_without_failed_rollout_does_not_create_orphan_tree,
        test_all_success_root_deactivates_root_without_deleting_children,
        test_all_success_nonroot_deactivates_only_current_node,
        test_inactive_node_not_sampled_for_rollout_sft_or_teacher_dispatch,
        test_later_success_does_not_delete_pending_or_inflight_events,
        test_teacher_dispatch_lru_uses_ascending_step_with_never_used_first,
        test_same_parent_dispatches_latest_pending_event,
        test_teacher_dispatch_cleans_stale_tree_queue_ids,
        test_teacher_success_can_create_multiple_children_for_same_parent,
        test_duplicate_locked_prefix_is_dropped,
        test_teacher_success_without_suffix_is_invalid,
        test_prefix_injection_samples_at_most_one_node_per_tree,
        test_prefix_injection_node_lru_uses_never_used_first,
        test_prefix_injection_does_not_resample_tree_in_same_step,
        test_sft_replay_samples_at_most_one_suffix_node_per_tree,
        test_sft_sampling_after_deactivation_uses_current_active_set,
        test_epoch_suffix_freeze_filters_window_and_clears_teacher_events,
        test_mark_suffix_sft_sampled_updates_lru_without_maturing,
        test_sft_node_lru_uses_never_used_first,
        test_teacher_dispatch_parent_lru_uses_never_dispatched_first,
        test_sft_update_does_not_mature_node_out_of_eligibility,
        test_cleanup_only_removes_tree_without_active_nodes_or_events,
        test_state_dict_roundtrip_requeues_inflight_events,
        test_load_state_dict_drops_legacy_solved_trees,
        test_teacher_worker_forest_fetch_dispatches_one_event_per_permit,
    ]
    for test in tests:
        test()
    print(f"\nAll {len(tests)} hard-state PrefixForestPool tests PASSED.")


if __name__ == "__main__":
    run_all()
