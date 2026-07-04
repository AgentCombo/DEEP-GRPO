"""Hard-state prefix forest for prefix injection.

The forest is a replay buffer of concrete prompt states:

  failed trajectory -> teacher event -> child node
  all-success rollout -> deactivate the current node only

There is no solved tree state, success-rate frontier, pruning, cooldown, or
suffix-SFT maturation. Rollout injection, teacher dispatch, and suffix-SFT
replay are all tree-balanced LRU samplers.
"""

from __future__ import annotations

import copy
import threading
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Deque, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from recipe.deep_grpo.protocol import FailedTrajectoryEntry

TreeKey = Tuple[int, ...]


@dataclass
class ForestNode:
    """One concrete state: original prompt plus locked prefix."""

    node_id: str
    augmented_prompt_ids: List[int]
    data_instance: Dict[str, Any]
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    active: bool = True
    agent_name: str = ""
    created_step: int = 0

    observations: int = 0
    last_k_succ: Optional[int] = None
    last_k_total: Optional[int] = None
    last_rollout_step: int = -1
    last_used_step: int = -1  # legacy/debug alias for last_rollout_step
    last_success_step: int = -1
    last_fail_step: int = -1

    last_teacher_dispatched_step: int = -1

    teacher_suffix_ids: Optional[List[int]] = None
    teacher_suffix_mask: Optional[List[int]] = None
    teacher_suffix_reward: Optional[float] = None
    teacher_suffix_reward_info: Optional[Any] = None
    teacher_original_failed_suffix_ids: Optional[List[int]] = None

    sft_updates: int = 0
    last_sft_step: int = -1
    # Legacy/debug aliases kept so trainer debug code and older checkpoints stay
    # readable while the semantic names above drive the hard-state logic.
    suffix_sft_updates: int = 0
    last_suffix_sft_step: int = -1

    num_turns: float = 1.0


@dataclass
class PrefixTree:
    """All concrete prefix states for one original prompt."""

    tree_key: TreeKey
    root_id: str
    original_prompt_ids: List[int]
    nodes: Dict[str, ForestNode]
    created_step: int = 0
    last_updated_step: int = 0
    last_rollout_sampled_step: int = -1
    last_teacher_dispatched_step: int = -1
    last_sft_sampled_step: int = -1


@dataclass
class FailedPrefixEvent:
    """One failed rollout awaiting teacher expansion."""

    event_id: str
    tree_key: TreeKey
    parent_node_id: str
    failed_entry: "FailedTrajectoryEntry"
    created_step: int
    in_flight: bool = False
    step: int = -1

    def __post_init__(self) -> None:
        if self.step < 0:
            self.step = self.created_step
        elif self.created_step < 0:
            self.created_step = self.step


class PrefixForestPool:
    """Thread-safe hard-state prefix forest."""

    def __init__(
        self,
        max_model_len: Optional[int] = None,
        min_remaining_tokens: int = 32,
        suffix_sft_maturation_steps: int = 0,
    ):
        assert min_remaining_tokens >= 0, (
            f"min_remaining_tokens must be >= 0, got {min_remaining_tokens}"
        )
        if max_model_len is not None:
            assert max_model_len > min_remaining_tokens, (
                f"max_model_len={max_model_len} must be greater than "
                f"min_remaining_tokens={min_remaining_tokens}"
            )
        self.max_model_len = max_model_len
        self.min_remaining_tokens = min_remaining_tokens
        # Kept as a compatibility knob. Hard-state suffix replay never matures
        # nodes; enabling replay is controlled by the trainer.
        self.suffix_sft_maturation_steps = max(
            0, int(suffix_sft_maturation_steps)
        )

        self._trees: Dict[TreeKey, PrefixTree] = {}
        self._failed_by_tree: Dict[TreeKey, Deque[str]] = defaultdict(deque)
        self._events: Dict[str, FailedPrefixEvent] = {}
        self._lock = threading.RLock()

        self._event_enqueued_count = 0
        self._event_dispatched_count = 0
        self._teacher_success_count = 0
        self._teacher_failed_count = 0
        self._teacher_stale_parent_inactive_count = 0
        self._teacher_stale_count = 0
        self._duplicate_child_count = 0
        self._invalid_child_count = 0
        self._deactivated_node_count = 0
        self._suffix_sft_sampled_count = 0
        self._teacher_cleared_count = 0

    # ------------------------------------------------------------------ keys
    @staticmethod
    def compute_tree_key(original_prompt_ids: List[int]) -> TreeKey:
        """Exact prompt identity shared by trainer and pool."""
        return tuple(int(t) for t in original_prompt_ids)

    # ---------------------------------------------------------------- sampling
    def active_trees_count(self) -> int:
        """Return trees with at least one active non-root rollout node."""
        return self.prefix_injection_ready_trees_count()

    def prefix_injection_ready_trees_count(self) -> int:
        with self._lock:
            return sum(
                1 for tree in self._trees.values()
                if self._rollout_candidate_ids_locked(tree)
            )

    def sample(
        self,
        n: int,
        current_step: int,
        require_suffix_sft_matured: bool = False,
        tree_cooldown_steps: int = 0,
    ) -> List[Tuple[TreeKey, str, ForestNode]]:
        """Sample active non-root nodes by tree LRU then node LRU.

        Legacy arguments are accepted but ignored: hard-state sampling has no
        cooldown and no suffix maturation gate.
        """
        del require_suffix_sft_matured, tree_cooldown_steps
        if n <= 0:
            return []
        with self._lock:
            eligible = [
                (self._lru_key(tree.last_rollout_sampled_step, tree.created_step, key), key)
                for key, tree in self._trees.items()
                if (
                    tree.last_rollout_sampled_step != current_step
                    and self._rollout_candidate_ids_locked(tree)
                )
            ]
            eligible.sort(key=lambda item: item[0])

            out: List[Tuple[TreeKey, str, ForestNode]] = []
            for _sort_key, key in eligible[:n]:
                tree = self._trees[key]
                node_id = self._rollout_candidate_ids_locked(tree)[0]
                node = tree.nodes[node_id]
                tree.last_rollout_sampled_step = current_step
                node.last_rollout_step = current_step
                node.last_used_step = current_step
                out.append((key, node_id, copy.deepcopy(node)))
            return out

    def suffix_sft_ready_trees_count(self) -> int:
        with self._lock:
            return self._suffix_sft_ready_counts_locked(max_nodes_per_tree=1)[0]

    def suffix_sft_ready_nodes_count(self) -> int:
        with self._lock:
            return self._suffix_sft_ready_counts_locked()[1]

    def suffix_sft_ready_counts(
        self,
        max_nodes_per_tree: Optional[int] = None,
    ) -> Tuple[int, int]:
        with self._lock:
            return self._suffix_sft_ready_counts_locked(max_nodes_per_tree)

    def sample_suffix_sft(
        self,
        n: int,
        current_step: int,
        max_nodes_per_tree: Optional[int] = None,
    ) -> List[Tuple[TreeKey, str, ForestNode]]:
        """Sample active non-root teacher-suffix nodes by tree/node LRU.

        A single mini-batch contributes at most one node per tree, matching the
        hard-state distinct-tree SFT rule. ``max_nodes_per_tree`` is accepted
        for API compatibility and capped to 1 for a single call.
        """
        if n <= 0:
            return []
        if max_nodes_per_tree is not None:
            max_nodes_per_tree = int(max_nodes_per_tree)
            assert max_nodes_per_tree > 0, (
                f"max_nodes_per_tree must be > 0, got {max_nodes_per_tree}"
            )
        with self._lock:
            eligible = [
                (self._lru_key(tree.last_sft_sampled_step, tree.created_step, key), key)
                for key, tree in self._trees.items()
                if self._suffix_sft_candidate_ids_locked(tree)
            ]
            eligible.sort(key=lambda item: item[0])

            out: List[Tuple[TreeKey, str, ForestNode]] = []
            for _sort_key, key in eligible[:n]:
                tree = self._trees[key]
                node_id = self._suffix_sft_candidate_ids_locked(tree)[0]
                node = tree.nodes[node_id]
                tree.last_sft_sampled_step = current_step
                node.last_sft_step = current_step
                node.last_suffix_sft_step = current_step
                out.append((key, node_id, copy.deepcopy(node)))
            self._suffix_sft_sampled_count += len(out)
            return out

    def mark_suffix_sft_sampled(
        self,
        routes: List[Tuple[TreeKey, str]],
        current_step: int,
    ) -> int:
        """Mark already-selected suffix-SFT routes as sampled.

        Epoch-local SFT freezes a dataset first, then trains it in full
        teacher mini-batches. This method applies the same LRU/sample counters
        as ``sample_suffix_sft`` without re-sampling from the global pool.
        """
        if not routes:
            return 0
        sampled = 0
        with self._lock:
            for tree_key, node_id in routes:
                tree = self._trees.get(tree_key)
                if tree is None or node_id not in tree.nodes:
                    continue
                node = tree.nodes[node_id]
                if not self._is_sft_candidate_locked(tree, node):
                    continue
                tree.last_sft_sampled_step = current_step
                node.last_sft_step = current_step
                node.last_suffix_sft_step = current_step
                sampled += 1
            self._suffix_sft_sampled_count += sampled
        return sampled

    def freeze_suffix_sft_epoch_and_clear_teacher_events(
        self,
        min_created_step: int,
        max_created_step: int,
    ) -> Tuple[List[Tuple[TreeKey, str, ForestNode]], Dict[str, int]]:
        """Freeze active suffix nodes from one PPO epoch and clear teacher work.

        The snapshot and queue clear happen under one lock. Any teacher response
        for an event that was pending or in-flight at the boundary will later be
        treated as stale because its event id has been removed.
        """
        if max_created_step < min_created_step:
            min_created_step, max_created_step = max_created_step, min_created_step

        with self._lock:
            candidates: List[Tuple[Tuple[int, int, TreeKey], TreeKey, str]] = []
            for key, tree in self._trees.items():
                node_ids = [
                    node_id
                    for node_id in self._suffix_sft_candidate_ids_locked(tree)
                    if (
                        min_created_step
                        <= int(tree.nodes[node_id].created_step)
                        <= max_created_step
                    )
                ]
                if not node_ids:
                    continue
                tree_key = self._lru_key(
                    tree.last_sft_sampled_step,
                    tree.created_step,
                    key,
                )
                for node_id in node_ids:
                    candidates.append((tree_key, key, node_id))
            candidates.sort(
                key=lambda item: (
                    item[0],
                    self._trees[item[1]].nodes[item[2]].last_sft_step,
                    self._trees[item[1]].nodes[item[2]].created_step,
                    item[2],
                )
            )
            snapshot = [
                (key, node_id, copy.deepcopy(self._trees[key].nodes[node_id]))
                for _sort_key, key, node_id in candidates
            ]

            pending_events = sum(
                1 for event in self._events.values() if not event.in_flight
            )
            in_flight_events = sum(
                1 for event in self._events.values() if event.in_flight
            )
            total_events = len(self._events)
            self._events.clear()
            self._failed_by_tree.clear()
            self._teacher_cleared_count += total_events

            return snapshot, {
                "teacher_events_cleared": total_events,
                "teacher_events_cleared_pending": pending_events,
                "teacher_events_cleared_in_flight": in_flight_events,
            }

    def record_suffix_sft(
        self,
        tree_key: TreeKey,
        node_id: str,
        current_step: int,
        count: int = 1,
    ) -> Dict[str, int]:
        """Record a successful suffix-SFT optimizer step for one node."""
        result = {"suffix_sft_recorded": 0, "suffix_sft_matured": 0}
        if count <= 0:
            return result
        with self._lock:
            tree = self._trees.get(tree_key)
            if tree is None or node_id not in tree.nodes:
                return result
            node = tree.nodes[node_id]
            if not self._is_sft_candidate_locked(tree, node):
                return result
            node.sft_updates += int(count)
            node.suffix_sft_updates = node.sft_updates
            tree.last_updated_step = current_step
            result["suffix_sft_recorded"] = int(count)
            return result

    def suffix_sft_trainable(self, tree_key: TreeKey, node_id: str) -> bool:
        with self._lock:
            tree = self._trees.get(tree_key)
            if tree is None or node_id not in tree.nodes:
                return False
            return self._is_sft_candidate_locked(tree, tree.nodes[node_id])

    # -------------------------------------------------------------- observation
    def record_root_observation(
        self,
        original_prompt_ids: List[int],
        data_instance: Dict[str, Any],
        agent_name: str,
        k_succ: int,
        k_total: int,
        current_step: int,
        failed_rollouts: Optional["List[FailedTrajectoryEntry]"] = None,
    ) -> Dict[str, int]:
        """Record normal dataloader root rollout outcomes into the forest."""
        if not original_prompt_ids or k_total <= 0:
            return _empty_result()
        key = self.compute_tree_key(original_prompt_ids)
        with self._lock:
            tree = self._trees.get(key)
            should_create = tree is None and bool(failed_rollouts)
            if tree is None and not should_create:
                return _empty_result()

            result = _empty_result()
            if tree is None:
                tree = self._create_tree(
                    original_prompt_ids=original_prompt_ids,
                    data_instance=data_instance,
                    agent_name=agent_name,
                    current_step=current_step,
                )
                result["tree_created"] = 1

            update = self._record_observation_locked(
                tree=tree,
                node_id=tree.root_id,
                k_succ=k_succ,
                k_total=k_total,
                current_step=current_step,
                failed_rollouts=failed_rollouts,
                allow_reactivate=True,
            )
            _merge_counts(result, update)
            return result

    def record_observation(
        self,
        tree_key: TreeKey,
        node_id: str,
        k_succ: int,
        k_total: int,
        current_step: int,
        failed_rollouts: Optional["List[FailedTrajectoryEntry]"] = None,
    ) -> Dict[str, int]:
        """Record K-rollout outcome for an injected active non-root node."""
        with self._lock:
            tree = self._trees.get(tree_key)
            if tree is None or node_id not in tree.nodes:
                return _empty_result()
            return self._record_observation_locked(
                tree=tree,
                node_id=node_id,
                k_succ=k_succ,
                k_total=k_total,
                current_step=current_step,
                failed_rollouts=failed_rollouts,
                allow_reactivate=False,
            )

    def _record_observation_locked(
        self,
        tree: PrefixTree,
        node_id: str,
        k_succ: int,
        k_total: int,
        current_step: int,
        failed_rollouts: Optional["List[FailedTrajectoryEntry]"],
        allow_reactivate: bool,
    ) -> Dict[str, int]:
        result = _empty_result()
        node = tree.nodes.get(node_id)
        if node is None or k_total <= 0:
            return result
        if not node.active and not allow_reactivate:
            return result

        node.observations += 1
        node.last_k_succ = int(k_succ)
        node.last_k_total = int(k_total)
        node.last_rollout_step = current_step
        node.last_used_step = current_step
        tree.last_updated_step = current_step
        result["observations_recorded"] = 1

        if k_succ > 0:
            node.last_success_step = current_step
        if k_succ < k_total:
            node.last_fail_step = current_step
            if not node.active:
                node.active = True
                result["node_reactivated"] = 1
            for entry in failed_rollouts or []:
                self._enqueue_event_locked(
                    tree.tree_key,
                    node_id,
                    entry,
                    current_step,
                )
                result["events_added"] += 1
            return result

        if node.active:
            node.active = False
            self._deactivated_node_count += 1
            result["node_deactivated"] = 1
            if node_id != tree.root_id:
                result["node_retired"] = 1  # legacy metric alias
            else:
                result["root_deactivated"] = 1
        return result

    # ------------------------------------------------------------- teacher side
    def pending_teacher_requests(
        self,
        max_items: Optional[int] = None,
        current_step: Optional[int] = None,
    ) -> List[FailedPrefixEvent]:
        """Dispatch eligible pending events by tree LRU then parent-node LRU."""
        if max_items is not None and max_items <= 0:
            return []
        if current_step is None:
            current_step = -1

        out: List[FailedPrefixEvent] = []
        with self._lock:
            candidates: List[Tuple[Tuple[int, int, TreeKey], TreeKey, str]] = []
            for key in list(self._failed_by_tree.keys()):
                tree = self._trees.get(key)
                if tree is None:
                    continue
                event_id = self._select_teacher_event_for_tree_locked(tree)
                if event_id is None:
                    continue
                candidates.append(
                    (
                        self._lru_key(
                            tree.last_teacher_dispatched_step,
                            tree.created_step,
                            key,
                        ),
                        key,
                        event_id,
                    )
                )
            candidates.sort(key=lambda item: item[0])

            limit = len(candidates) if max_items is None else min(max_items, len(candidates))
            for _sort_key, key, event_id in candidates[:limit]:
                event = self._events.get(event_id)
                tree = self._trees.get(key)
                if event is None or tree is None:
                    continue
                parent = tree.nodes.get(event.parent_node_id)
                if parent is None or not parent.active or event.in_flight:
                    continue
                event.in_flight = True
                tree.last_teacher_dispatched_step = current_step
                parent.last_teacher_dispatched_step = current_step
                self._event_dispatched_count += 1
                out.append(copy.deepcopy(event))
            return out

    def on_teacher_response(
        self,
        event_id: str,
        annotated_entry: Optional[Any],
        current_step: int,
        success: bool = True,
    ) -> None:
        """Handle teacher completion: create one child or drop the event."""
        with self._lock:
            event = self._events.pop(event_id, None)
            if event is None:
                self._teacher_stale_count += 1
                return

            self._remove_event_from_tree_queue_locked(event)
            tree = self._trees.get(event.tree_key)
            parent = tree.nodes.get(event.parent_node_id) if tree is not None else None
            if tree is None or parent is None or not parent.active:
                self._teacher_stale_parent_inactive_count += 1
                self._teacher_stale_count += 1
                return

            if not success or annotated_entry is None:
                self._teacher_failed_count += 1
                return

            child_augmented_ids = (
                list(annotated_entry.prompt_ids)
                + list(annotated_entry.response_ids)
            )
            if not self._valid_child(parent.augmented_prompt_ids, child_augmented_ids):
                self._invalid_child_count += 1
                return

            if self._find_node_by_augmented_ids_locked(tree, child_augmented_ids):
                self._duplicate_child_count += 1
                return

            teacher_suffix_metadata = self._teacher_suffix_metadata(annotated_entry)
            if not teacher_suffix_metadata.get("teacher_suffix_ids"):
                self._invalid_child_count += 1
                return

            child_id = uuid.uuid4().hex
            child = ForestNode(
                node_id=child_id,
                augmented_prompt_ids=child_augmented_ids,
                data_instance=copy.deepcopy(annotated_entry.data_instance),
                parent_id=event.parent_node_id,
                active=True,
                agent_name=getattr(annotated_entry, "agent_name", parent.agent_name),
                created_step=current_step,
                num_turns=float(getattr(annotated_entry, "num_turns", parent.num_turns)),
                **teacher_suffix_metadata,
            )
            tree.nodes[child_id] = child
            parent.children.append(child_id)
            tree.last_updated_step = current_step
            self._teacher_success_count += 1

    # --------------------------------------------------------------- cleanup
    def cleanup_inactive_trees(self) -> int:
        """Delete trees with no active nodes and no pending/in-flight events."""
        with self._lock:
            event_tree_keys = {event.tree_key for event in self._events.values()}
            removable = [
                key for key, tree in self._trees.items()
                if (
                    not any(node.active for node in tree.nodes.values())
                    and key not in event_tree_keys
                )
            ]
            for key in removable:
                self._trees.pop(key, None)
                self._failed_by_tree.pop(key, None)
            return len(removable)

    # ---------------------------------------------------------------- debug
    def debug_node_context(
        self,
        tree_key: TreeKey,
        node_id: str,
        max_children: int = 8,
        max_lineage: int = 32,
    ) -> Dict[str, Any]:
        with self._lock:
            tree = self._trees.get(tree_key)
            if tree is None or node_id not in tree.nodes:
                return {}
            return self._debug_node_context_locked(
                tree,
                node_id,
                max_children=max_children,
                max_lineage=max_lineage,
            )

    def debug_deeper_nodes(
        self,
        max_nodes: int,
        max_children: int = 8,
        max_lineage: int = 32,
    ) -> List[Tuple[TreeKey, str, ForestNode, Dict[str, Any]]]:
        if max_nodes <= 0:
            return []
        with self._lock:
            candidates = []
            for tree_key, tree in self._trees.items():
                for node_id, node in tree.nodes.items():
                    if node.parent_id is None:
                        continue
                    candidates.append(
                        (
                            self._node_depth_edges_locked(tree, node_id),
                            len(node.children),
                            node.observations,
                            tree_key,
                            node_id,
                        )
                    )
            candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
            out: List[Tuple[TreeKey, str, ForestNode, Dict[str, Any]]] = []
            for _depth, _child_count, _obs, tree_key, node_id in candidates[:max_nodes]:
                tree = self._trees[tree_key]
                context = self._debug_node_context_locked(
                    tree,
                    node_id,
                    max_children=max_children,
                    max_lineage=max_lineage,
                )
                out.append((tree_key, node_id, copy.deepcopy(tree.nodes[node_id]), context))
            return out

    # ---------------------------------------------------------------- internals
    @staticmethod
    def _lru_key(last_step: int, created_step: int, key: TreeKey) -> Tuple[int, int, TreeKey]:
        # -1 means never used and must sort before any non-negative step.
        return (int(last_step), int(created_step), key)

    def _create_tree(
        self,
        original_prompt_ids: List[int],
        data_instance: Dict[str, Any],
        agent_name: str,
        current_step: int,
    ) -> PrefixTree:
        key = self.compute_tree_key(original_prompt_ids)
        root_id = uuid.uuid4().hex
        root = ForestNode(
            node_id=root_id,
            augmented_prompt_ids=list(original_prompt_ids),
            data_instance=copy.deepcopy(data_instance),
            parent_id=None,
            active=True,
            agent_name=agent_name,
            created_step=current_step,
        )
        tree = PrefixTree(
            tree_key=key,
            root_id=root_id,
            original_prompt_ids=list(original_prompt_ids),
            nodes={root_id: root},
            created_step=current_step,
            last_updated_step=current_step,
        )
        self._trees[key] = tree
        return tree

    def _enqueue_event_locked(
        self,
        tree_key: TreeKey,
        parent_node_id: str,
        failed_entry: "FailedTrajectoryEntry",
        step: int,
    ) -> None:
        event_id = uuid.uuid4().hex
        event = FailedPrefixEvent(
            event_id=event_id,
            tree_key=tree_key,
            parent_node_id=parent_node_id,
            failed_entry=failed_entry,
            created_step=step,
            in_flight=False,
            step=step,
        )
        self._events[event_id] = event
        self._failed_by_tree[tree_key].append(event_id)
        self._event_enqueued_count += 1

    def _remove_event_from_tree_queue_locked(self, event: FailedPrefixEvent) -> None:
        dq = self._failed_by_tree.get(event.tree_key)
        if not dq:
            return
        kept = deque(event_id for event_id in dq if event_id != event.event_id)
        if kept:
            self._failed_by_tree[event.tree_key] = kept
        else:
            self._failed_by_tree.pop(event.tree_key, None)

    def _select_teacher_event_for_tree_locked(self, tree: PrefixTree) -> Optional[str]:
        by_parent: Dict[str, List[FailedPrefixEvent]] = defaultdict(list)
        event_ids = self._failed_by_tree.get(tree.tree_key)
        if not event_ids:
            return None

        kept_event_ids: Deque[str] = deque()
        for event_id in event_ids:
            event = self._events.get(event_id)
            if event is None:
                continue
            if event.tree_key != tree.tree_key:
                continue
            kept_event_ids.append(event_id)
            if event.in_flight:
                continue
            parent = tree.nodes.get(event.parent_node_id)
            if parent is None or not parent.active:
                continue
            by_parent[event.parent_node_id].append(event)

        if len(kept_event_ids) != len(event_ids):
            if kept_event_ids:
                self._failed_by_tree[tree.tree_key] = kept_event_ids
            else:
                self._failed_by_tree.pop(tree.tree_key, None)

        if not by_parent:
            return None

        parent_items = []
        for parent_id, events in by_parent.items():
            parent = tree.nodes[parent_id]
            latest_event = max(events, key=lambda event: (event.created_step, event.event_id))
            parent_items.append(
                (
                    (
                        parent.last_teacher_dispatched_step,
                        parent.created_step,
                        parent_id,
                    ),
                    latest_event.event_id,
                )
            )
        parent_items.sort(key=lambda item: item[0])
        return parent_items[0][1]

    def _rollout_candidate_ids_locked(self, tree: PrefixTree) -> List[str]:
        candidates = [
            node_id for node_id, node in tree.nodes.items()
            if node.active and node.parent_id is not None
        ]
        candidates.sort(
            key=lambda node_id: (
                tree.nodes[node_id].last_rollout_step,
                tree.nodes[node_id].created_step,
                node_id,
            )
        )
        return candidates

    def _suffix_sft_candidate_ids_locked(self, tree: PrefixTree) -> List[str]:
        candidates = [
            node_id for node_id, node in tree.nodes.items()
            if self._is_sft_candidate_locked(tree, node)
        ]
        candidates.sort(
            key=lambda node_id: (
                tree.nodes[node_id].last_sft_step,
                tree.nodes[node_id].created_step,
                node_id,
            )
        )
        return candidates

    def _suffix_sft_ready_counts_locked(
        self,
        max_nodes_per_tree: Optional[int] = None,
    ) -> Tuple[int, int]:
        if max_nodes_per_tree is not None:
            max_nodes_per_tree = int(max_nodes_per_tree)
            assert max_nodes_per_tree > 0, (
                f"max_nodes_per_tree must be > 0, got {max_nodes_per_tree}"
            )
        ready_trees = 0
        ready_nodes = 0
        for tree in self._trees.values():
            count = len(self._suffix_sft_candidate_ids_locked(tree))
            if max_nodes_per_tree is not None:
                count = min(count, max_nodes_per_tree)
            if count:
                ready_trees += 1
                ready_nodes += count
        return ready_trees, ready_nodes

    def _is_sft_candidate_locked(self, tree: PrefixTree, node: ForestNode) -> bool:
        return (
            node.active
            and node.parent_id is not None
            and bool(node.teacher_suffix_ids)
            and node.node_id in tree.nodes
        )

    @staticmethod
    def _teacher_suffix_metadata(annotated_entry: Any) -> Dict[str, Any]:
        teacher_suffix = getattr(annotated_entry, "teacher_suffix", None)
        if teacher_suffix is None:
            return {}
        suffix_ids = list(getattr(teacher_suffix, "suffix_ids", []) or [])
        if not suffix_ids:
            return {}
        return {
            "teacher_suffix_ids": suffix_ids,
            "teacher_suffix_mask": list(
                getattr(teacher_suffix, "suffix_mask", []) or []
            ),
            "teacher_suffix_reward": getattr(teacher_suffix, "reward", None),
            "teacher_suffix_reward_info": copy.deepcopy(
                getattr(teacher_suffix, "reward_info", None)
            ),
            "teacher_original_failed_suffix_ids": list(
                getattr(teacher_suffix, "original_failed_suffix_ids", []) or []
            ),
        }

    def _valid_child(self, parent_ids: List[int], child_ids: List[int]) -> bool:
        if not _is_strict_prefix(parent_ids, child_ids):
            return False
        if self.max_model_len is None:
            return True
        return len(child_ids) <= self.max_model_len - self.min_remaining_tokens

    @staticmethod
    def _find_node_by_augmented_ids_locked(
        tree: PrefixTree,
        augmented_ids: List[int],
    ) -> Optional[str]:
        target = list(augmented_ids)
        for node_id, node in tree.nodes.items():
            if list(node.augmented_prompt_ids) == target:
                return node_id
        return None

    @staticmethod
    def _has_teacher_suffix_locked(node: ForestNode) -> bool:
        return bool(getattr(node, "teacher_suffix_ids", None))

    def _node_depth_edges_locked(self, tree: PrefixTree, node_id: str) -> int:
        depth = 0
        seen = set()
        cur = node_id
        while cur is not None and cur not in seen:
            seen.add(cur)
            node = tree.nodes.get(cur)
            if node is None or node.parent_id is None:
                break
            depth += 1
            cur = node.parent_id
        return depth

    def _debug_node_summary_locked(
        self,
        tree: PrefixTree,
        node_id: str,
    ) -> Dict[str, Any]:
        node = tree.nodes[node_id]
        root_len = len(tree.original_prompt_ids)
        k_succ = node.last_k_succ
        k_total = node.last_k_total
        success_rate = (
            float(k_succ or 0) / float(k_total)
            if k_total is not None and k_total > 0 else None
        )
        locked_prefix_ids = list(node.augmented_prompt_ids[root_len:])
        return {
            "node_id": node_id,
            "parent_id": node.parent_id,
            "active": node.active,
            "depth_edges": self._node_depth_edges_locked(tree, node_id),
            "augmented_prompt_token_len": len(node.augmented_prompt_ids),
            "locked_prefix_token_len": len(locked_prefix_ids),
            "child_count": len(node.children),
            "observations": node.observations,
            "last_k_succ": k_succ,
            "last_k_total": k_total,
            "success_rate": success_rate,
            "last_rollout_step": node.last_rollout_step,
            "last_used_step": node.last_used_step,
            "last_teacher_dispatched_step": node.last_teacher_dispatched_step,
            "last_sft_step": node.last_sft_step,
            "last_success_step": node.last_success_step,
            "last_fail_step": node.last_fail_step,
            "created_step": node.created_step,
            "teacher_suffix_token_len": len(node.teacher_suffix_ids or []),
            "sft_updates": getattr(node, "sft_updates", 0),
            "suffix_sft_updates": getattr(node, "suffix_sft_updates", 0),
            "last_suffix_sft_step": getattr(node, "last_suffix_sft_step", -1),
            "rollout_retired": not node.active,
            "retired_step": node.last_rollout_step if not node.active else -1,
            "retired_reason": "all_success" if not node.active else "",
            "augmented_prompt_ids": list(node.augmented_prompt_ids),
            "locked_prefix_ids": locked_prefix_ids,
            "teacher_suffix_ids": list(node.teacher_suffix_ids or []),
        }

    def _debug_node_context_locked(
        self,
        tree: PrefixTree,
        node_id: str,
        max_children: int,
        max_lineage: int,
    ) -> Dict[str, Any]:
        max_children = max(0, int(max_children))
        max_lineage = max(0, int(max_lineage))
        node = tree.nodes[node_id]
        lineage_ids = []
        seen = set()
        cur = node_id
        while cur is not None and cur not in seen:
            seen.add(cur)
            lineage_ids.append(cur)
            parent = tree.nodes.get(cur).parent_id if cur in tree.nodes else None
            cur = parent
        lineage_ids.reverse()
        lineage_truncated = max(0, len(lineage_ids) - max_lineage)
        if len(lineage_ids) > max_lineage:
            lineage_ids = lineage_ids[-max_lineage:] if max_lineage > 0 else []

        child_ids = list(node.children)
        children_truncated = max(0, len(child_ids) - max_children)
        child_ids = child_ids[:max_children]

        descendant_count = 0
        deepest_descendant_depth = self._node_depth_edges_locked(tree, node_id)
        stack = list(node.children)
        seen_descendants = set()
        while stack:
            child_id = stack.pop()
            if child_id in seen_descendants or child_id not in tree.nodes:
                continue
            seen_descendants.add(child_id)
            descendant_count += 1
            deepest_descendant_depth = max(
                deepest_descendant_depth,
                self._node_depth_edges_locked(tree, child_id),
            )
            stack.extend(tree.nodes[child_id].children)

        root_len = len(tree.original_prompt_ids)
        return {
            "node_depth_edges": self._node_depth_edges_locked(tree, node_id),
            "node_depth_tokens": len(node.augmented_prompt_ids) - root_len,
            "descendant_count": descendant_count,
            "deepest_descendant_depth_edges": deepest_descendant_depth,
            "has_deeper_descendants": descendant_count > 0,
            "debug_lineage": [
                self._debug_node_summary_locked(tree, cur_id)
                for cur_id in lineage_ids
                if cur_id in tree.nodes
            ],
            "debug_lineage_truncated": lineage_truncated,
            "debug_children": [
                self._debug_node_summary_locked(tree, child_id)
                for child_id in child_ids
                if child_id in tree.nodes
            ],
            "debug_children_truncated": children_truncated,
        }

    # --------------------------------------------------------------- state/stat
    def __len__(self) -> int:
        with self._lock:
            return len(self._trees)

    @property
    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total_nodes = sum(len(tree.nodes) for tree in self._trees.values())
            active_nodes = 0
            active_nonroot_nodes = 0
            teacher_suffix_nodes = 0
            deactivated_nodes = 0
            observed_root = 0
            observed_nonroot = 0
            root_any_success = 0
            prefix_any_success = 0

            for tree in self._trees.values():
                for node in tree.nodes.values():
                    if node.active:
                        active_nodes += 1
                        if node.parent_id is not None:
                            active_nonroot_nodes += 1
                    else:
                        deactivated_nodes += 1
                    if self._has_teacher_suffix_locked(node):
                        teacher_suffix_nodes += 1
                    if node.last_k_total is not None and node.last_k_total > 0:
                        any_success = int((node.last_k_succ or 0) > 0)
                        if node.parent_id is None:
                            observed_root += 1
                            root_any_success += any_success
                        else:
                            observed_nonroot += 1
                            prefix_any_success += any_success

            pending_events = sum(1 for event in self._events.values() if not event.in_flight)
            in_flight_events = sum(1 for event in self._events.values() if event.in_flight)
            sft_eligible_trees, sft_eligible_nodes = self._suffix_sft_ready_counts_locked()

            stats = {
                "forest/num_trees": len(self._trees),
                "forest/num_nodes": total_nodes,
                "forest/active_nodes": active_nodes,
                "forest/active_nonroot_nodes": active_nonroot_nodes,
                "forest/teacher_suffix_nodes": teacher_suffix_nodes,
                "forest/deactivated_nodes": deactivated_nodes,
                "teacher/events_queued": pending_events,
                "teacher/events_dispatched": self._event_dispatched_count,
                "teacher/events_in_flight": in_flight_events,
                "teacher/events_succeeded": self._teacher_success_count,
                "teacher/events_duplicate_dropped": self._duplicate_child_count,
                "teacher/events_invalid": self._invalid_child_count,
                "teacher/events_stale_parent_inactive": (
                    self._teacher_stale_parent_inactive_count
                ),
                "teacher/events_failed": self._teacher_failed_count,
                "teacher/events_stale": self._teacher_stale_count,
                "teacher/events_cleared": self._teacher_cleared_count,
                "sft/eligible_trees": sft_eligible_trees,
                "sft/eligible_nodes": sft_eligible_nodes,
                "sft/sampled_nodes": self._suffix_sft_sampled_count,
                "sft/optimizer_steps": 0,
                "sft/recorded_node_updates": sum(
                    int(getattr(node, "sft_updates", 0))
                    for tree in self._trees.values()
                    for node in tree.nodes.values()
                ),
                "prefix_any_success_rate": (
                    prefix_any_success / observed_nonroot if observed_nonroot else 0.0
                ),
                "root_any_success_rate": (
                    root_any_success / observed_root if observed_root else 0.0
                ),
            }

            # Compatibility aliases for existing trainer dashboards.
            stats.update(
                {
                    "forest_pool/trees": stats["forest/num_trees"],
                    "forest_pool/active_trees": self.prefix_injection_ready_trees_count(),
                    "forest_pool/solved_trees": 0,
                    "forest_pool/total_nodes": stats["forest/num_nodes"],
                    "forest_pool/observed_root_nodes": observed_root,
                    "forest_pool/observed_nonroot_nodes": observed_nonroot,
                    "forest_pool/pending_events": len(self._events),
                    "forest_pool/suffix_sft_maturation_steps": 0,
                    "forest_pool/suffix_sft_nodes": teacher_suffix_nodes,
                    "forest_pool/suffix_sft_candidate_nodes": sft_eligible_nodes,
                    "forest_pool/suffix_sft_matured_nodes": 0,
                    "forest_pool/suffix_sft_updates_total": stats["sft/recorded_node_updates"],
                    "forest_pool/suffix_sft_sampled_cumulative": (
                        self._suffix_sft_sampled_count
                    ),
                    "forest_pool/suffix_sft_matured_cumulative": 0,
                    "forest_pool/rollout_retired_nodes": deactivated_nodes,
                    "forest_pool/events_enqueued": self._event_enqueued_count,
                    "forest_pool/events_dispatched": self._event_dispatched_count,
                    "forest_pool/events_coalesced": 0,
                    "forest_pool/teacher_successes": self._teacher_success_count,
                    "forest_pool/teacher_failed": self._teacher_failed_count,
                    "forest_pool/teacher_stale": self._teacher_stale_count,
                    "forest_pool/teacher_cleared": self._teacher_cleared_count,
                    "forest_pool/duplicate_child": self._duplicate_child_count,
                    "forest_pool/invalid_child": self._invalid_child_count,
                    "forest_pool/pruned_nodes": 0,
                    "forest_pool/solved_cumulative": 0,
                    "forest_pool/reactivated_cumulative": 0,
                    "forest_pool/rr_tree_queue_size": 0,
                }
            )
            return stats

    def state_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "trees": copy.deepcopy(self._trees),
                "failed_by_tree": {
                    key: list(value) for key, value in self._failed_by_tree.items()
                },
                "events": copy.deepcopy(self._events),
                "event_enqueued_count": self._event_enqueued_count,
                "event_dispatched_count": self._event_dispatched_count,
                "teacher_success_count": self._teacher_success_count,
                "teacher_failed_count": self._teacher_failed_count,
                "teacher_stale_parent_inactive_count": (
                    self._teacher_stale_parent_inactive_count
                ),
                "teacher_stale_count": self._teacher_stale_count,
                "duplicate_child_count": self._duplicate_child_count,
                "invalid_child_count": self._invalid_child_count,
                "deactivated_node_count": self._deactivated_node_count,
                "suffix_sft_sampled_count": self._suffix_sft_sampled_count,
                "teacher_cleared_count": self._teacher_cleared_count,
            }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        with self._lock:
            raw_trees = dict(state.get("trees", {}))
            key_map: Dict[Any, TreeKey] = {}
            dropped_legacy_solved_keys = set()
            self._trees = {}
            for old_key, tree in raw_trees.items():
                new_key = self.compute_tree_key(tree.original_prompt_ids)
                tree.tree_key = new_key
                key_map[old_key] = new_key
                if bool(getattr(tree, "solved", False)):
                    # Older checkpoints used a global solved bit that removed
                    # trees from sampling. Do not resurrect those states as
                    # active hard-state nodes when migrating.
                    dropped_legacy_solved_keys.add(new_key)
                    continue
                self._ensure_tree_optional_fields_locked(tree)
                for node in tree.nodes.values():
                    self._ensure_node_optional_fields_locked(node)
                self._trees[new_key] = tree

            self._failed_by_tree = defaultdict(deque)
            for key, values in state.get("failed_by_tree", {}).items():
                mapped_key = key_map.get(key, key)
                if mapped_key in dropped_legacy_solved_keys:
                    continue
                self._failed_by_tree[mapped_key].extend(values)

            self._events = {}
            for event_id, event in state.get("events", {}).items():
                self._ensure_event_optional_fields_locked(event)
                event.tree_key = key_map.get(event.tree_key, event.tree_key)
                if (
                    event.tree_key in dropped_legacy_solved_keys
                    or event.tree_key not in self._trees
                ):
                    continue
                # Checkpoint restore cannot preserve the background teacher
                # task that owned an in-flight event, so make every restored
                # event dispatchable again.
                event.in_flight = False
                self._events[event_id] = event

            self._event_enqueued_count = state.get("event_enqueued_count", 0)
            self._event_dispatched_count = state.get("event_dispatched_count", 0)
            self._teacher_success_count = state.get("teacher_success_count", 0)
            self._teacher_failed_count = state.get("teacher_failed_count", 0)
            self._teacher_stale_parent_inactive_count = state.get(
                "teacher_stale_parent_inactive_count",
                state.get("teacher_stale_count", 0),
            )
            self._teacher_stale_count = state.get("teacher_stale_count", 0)
            self._duplicate_child_count = state.get("duplicate_child_count", 0)
            self._invalid_child_count = state.get("invalid_child_count", 0)
            self._deactivated_node_count = state.get("deactivated_node_count", 0)
            self._suffix_sft_sampled_count = state.get(
                "suffix_sft_sampled_count", 0
            )
            self._teacher_cleared_count = state.get("teacher_cleared_count", 0)
            self._repair_event_queues_locked()

    @staticmethod
    def _ensure_tree_optional_fields_locked(tree: PrefixTree) -> None:
        for attr in (
            "last_rollout_sampled_step",
            "last_teacher_dispatched_step",
            "last_sft_sampled_step",
        ):
            if not hasattr(tree, attr):
                setattr(tree, attr, -1)

    @staticmethod
    def _ensure_node_optional_fields_locked(node: ForestNode) -> None:
        if not hasattr(node, "active"):
            node.active = not bool(getattr(node, "rollout_retired", False))
        for attr, default in (
            ("last_rollout_step", getattr(node, "last_used_step", -1)),
            ("last_used_step", getattr(node, "last_rollout_step", -1)),
            ("last_teacher_dispatched_step", -1),
            ("teacher_suffix_ids", None),
            ("teacher_suffix_mask", None),
            ("teacher_suffix_reward", None),
            ("teacher_suffix_reward_info", None),
            ("teacher_original_failed_suffix_ids", None),
            ("sft_updates", getattr(node, "suffix_sft_updates", 0)),
            ("last_sft_step", getattr(node, "last_suffix_sft_step", -1)),
            ("suffix_sft_updates", getattr(node, "sft_updates", 0)),
            ("last_suffix_sft_step", getattr(node, "last_sft_step", -1)),
            ("num_turns", 1.0),
        ):
            if not hasattr(node, attr):
                setattr(node, attr, default)

    @staticmethod
    def _ensure_event_optional_fields_locked(event: FailedPrefixEvent) -> None:
        if not hasattr(event, "created_step"):
            event.created_step = int(getattr(event, "step", -1))
        if not hasattr(event, "step") or int(getattr(event, "step", -1)) < 0:
            event.step = int(getattr(event, "created_step", -1))
        if not hasattr(event, "in_flight"):
            event.in_flight = False

    def _repair_event_queues_locked(self) -> None:
        queued = {
            event_id
            for dq in self._failed_by_tree.values()
            for event_id in dq
        }
        for event_id, event in self._events.items():
            if event_id not in queued:
                self._failed_by_tree[event.tree_key].append(event_id)
        for key in list(self._failed_by_tree.keys()):
            kept = deque(
                event_id
                for event_id in self._failed_by_tree[key]
                if event_id in self._events
            )
            if kept:
                self._failed_by_tree[key] = kept
            else:
                self._failed_by_tree.pop(key, None)


def _is_prefix(prefix: List[int], seq: List[int]) -> bool:
    return len(prefix) <= len(seq) and list(seq[:len(prefix)]) == list(prefix)


def _is_strict_prefix(prefix: List[int], seq: List[int]) -> bool:
    return len(prefix) < len(seq) and _is_prefix(prefix, seq)


def _empty_result() -> Dict[str, int]:
    return {
        "tree_created": 0,
        "tree_reactivated": 0,
        "tree_solved": 0,
        "root_deactivated": 0,
        "node_reactivated": 0,
        "node_deactivated": 0,
        "events_added": 0,
        "events_coalesced": 0,
        "nodes_pruned": 0,
        "node_retired": 0,
        "observations_recorded": 0,
    }


def _merge_counts(dst: Dict[str, int], src: Dict[str, int]) -> None:
    for key, value in src.items():
        dst[key] = dst.get(key, 0) + value
