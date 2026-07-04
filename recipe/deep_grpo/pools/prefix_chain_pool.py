"""Chain-based prefix injection pool for the prefix_inject_mode (chain variant).

Each hard prompt gets a `PrefixChain` — a linear list of nodes at increasing prefix
depths. N₀ is always the original prompt (depth 0). Deeper nodes are produced by the
teacher worker when a shallower node gets stuck.

This replaces the flat `SyntheticPromptPool` design, which had three structural
failures validated by live metrics and simulation:

  1. Pool dilution: each entry averaged 0.09 training events over 500 steps.
  2. Retreat was emergent-only and never observed in production runs.
  3. Curriculum sampler needed alongside it, which killed main-pool diversity.

The chain design addresses all three by organizing entries per-prompt (bounded
pool size), making retreat an explicit state-machine transition, and leaving the
main pool to vanilla GRPO uniform sampling.

State machine (see plan file rippling-honking-anchor.md for full diagram):

  - Chain created by `on_main_failure(P, rollouts)` → state = DEEPENING_REQUESTED
    immediately. N₀ is seeded with main's failed rollouts for teacher retry.
  - Teacher pulls via `pending_teacher_requests()`, tries each rollout in sequence,
    calls `on_teacher_response(..., success=True|False)`:
      * success → append new deeper node Nᵢ₊₁ to chain, state = LEARNING
      * failure → state = ABANDONED (terminal)
  - While LEARNING, chain samples its active node via `sample(n)`. On rollout
    completion, `record_observation(k_succ, ...)`:
      * k_succ == 0        → store rollouts, state = DEEPENING_REQUESTED
      * 0 < k_succ < k_tot → stay LEARNING (partial success, active training)
      * k_succ == k_tot    → mastered: pop active, retreat to previous node
        (obs reset) or COMPLETED if active_idx was 0 (terminal)

Invariants (verified by unit tests):
  INV-1: state == LEARNING ⟺ valid active_idx and active node not mastered
  INV-2: nodes depths strictly increasing
  INV-3: LEARNING active node's last_k_succ is never 0
  INV-4: no mastered node remains in `nodes` (compacted on master)
  INV-5: at most one chain per prompt_key

Sampling weight is zero-knob:
    untried node → 1.0
    partial     → 4p(1-p)   (p = last_k_succ / last_k_total)
"""

import logging
import math
import random
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    # `FailedTrajectoryEntry` is referenced only in type annotations (string
    # form below); the pool treats entries as opaque objects.
    from recipe.deep_grpo.protocol import FailedTrajectoryEntry

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------------


class ChainState(Enum):
    LEARNING = "learning"
    DEEPENING_REQUESTED = "deepening_requested"
    COMPLETED = "completed"       # terminal: N₀ mastered, prompt graduated
    ABANDONED = "abandoned"       # terminal: teacher exhausted on all rollouts


@dataclass
class ChainNode:
    """One node in a chain: prompt + some (possibly zero) locked prefix."""

    # Full prompt fed to the model: original + locked_prefix.
    # depth == len(augmented_prompt_ids) - len(chain.original_prompt_ids)
    augmented_prompt_ids: List[int]
    data_instance: Dict[str, Any]
    depth: int
    agent_name: str = ""

    # Observation state (overwritten on each record_observation).
    observations: int = 0
    last_k_succ: Optional[int] = None
    last_k_total: Optional[int] = None
    first_observed_step: int = -1
    last_used_step: int = -1

    # The last "all-fail" rollout batch. Populated when the node is being queued
    # for teacher deepening; cleared on retreat / chain reset.
    # String annotation to avoid a runtime import of verl-dependent types.
    last_failed_rollouts: "List[FailedTrajectoryEntry]" = field(default_factory=list)

    @property
    def is_mastered(self) -> bool:
        return (
            self.last_k_total is not None
            and self.last_k_total > 0
            and self.last_k_succ == self.last_k_total
        )


@dataclass
class PrefixChain:
    """All nodes for one hard prompt, ordered by depth."""

    prompt_key: int
    original_prompt_ids: List[int]
    data_instance: Dict[str, Any]

    nodes: List[ChainNode] = field(default_factory=list)
    active_idx: int = -1
    state: ChainState = ChainState.LEARNING

    created_step: int = 0
    last_updated_step: int = 0

    # Audit trail for debugging / metrics. Keep bounded.
    transitions: List[Tuple[int, ChainState, ChainState]] = field(default_factory=list)

    @property
    def active_node(self) -> Optional[ChainNode]:
        if 0 <= self.active_idx < len(self.nodes):
            return self.nodes[self.active_idx]
        return None

    def _transition_to(self, new_state: ChainState, step: int):
        if new_state == self.state:
            return
        self.transitions.append((step, self.state, new_state))
        # Bound transition log to avoid unbounded growth over very long training.
        if len(self.transitions) > 64:
            self.transitions = self.transitions[-64:]
        self.state = new_state
        self.last_updated_step = step


# ----------------------------------------------------------------------------
# Pool
# ----------------------------------------------------------------------------


class PrefixChainPool:
    """Thread-safe chain-based prefix pool.

    Producer: trainer (`on_main_failure`) + teacher worker (`on_teacher_response`).
    Consumer: trainer (`sample`), teacher worker (`pending_teacher_requests`).

    The pool itself has no tunable knobs. All dynamics are determined by observable
    events on the nodes.
    """

    def __init__(self):
        self._chains: Dict[int, PrefixChain] = {}
        # FIFO of prompt_keys awaiting teacher processing.
        self._deepening_queue: deque[int] = deque()
        self._lock = threading.RLock()

        # Cumulative event counts for logging.
        self._retreat_count: int = 0
        self._master_count: int = 0
        self._completed_count: int = 0
        self._abandoned_count: int = 0
        self._deepening_request_count: int = 0
        self._teacher_success_count: int = 0
        self._teacher_invalid_depth_count: int = 0
        self._teacher_invalid_prefix_count: int = 0

    # ------------------------------------------------------------------------
    # Public API — trainer side
    # ------------------------------------------------------------------------

    @staticmethod
    def compute_prompt_key(prompt_ids: List[int]) -> int:
        """Hash function used across the trainer/teacher boundary for lookup."""
        return hash(tuple(prompt_ids))

    def on_main_failure(
        self,
        original_prompt_ids: List[int],
        data_instance: Dict[str, Any],
        failed_rollouts: "List[FailedTrajectoryEntry]",
        current_step: int,
        agent_name: str = "",
    ) -> Optional[int]:
        """Create a chain for a prompt the main pool just observed k_succ=0 on.

        No-op if a chain already exists for this prompt (any state). Returns the
        `prompt_key` if a new chain was created, else None.
        """
        if not original_prompt_ids:
            return None
        key = self.compute_prompt_key(original_prompt_ids)
        with self._lock:
            if key in self._chains:
                return None
            n0 = ChainNode(
                augmented_prompt_ids=list(original_prompt_ids),
                data_instance=data_instance,
                depth=0,
                agent_name=agent_name,
                last_failed_rollouts=list(failed_rollouts),
            )
            chain = PrefixChain(
                prompt_key=key,
                original_prompt_ids=list(original_prompt_ids),
                data_instance=data_instance,
                nodes=[n0],
                active_idx=0,
                state=ChainState.DEEPENING_REQUESTED,
                created_step=current_step,
                last_updated_step=current_step,
            )
            # Start the audit trail with the initial transition so it shows up
            # in debugging dumps.
            chain.transitions.append(
                (current_step, ChainState.LEARNING, ChainState.DEEPENING_REQUESTED)
            )
            self._chains[key] = chain
            self._deepening_queue.append(key)
            self._deepening_request_count += 1
            return key

    def sample(
        self,
        n: int,
        current_step: int,
    ) -> List[Tuple[int, ChainNode]]:
        """Return up to n (prompt_key, active_node) pairs from LEARNING chains.

        Selection is Efraimidis-Spirakis weighted sampling without replacement
        using `_node_weight`. Untried nodes get weight 1.0; partial nodes get
        4p(1-p). Mastered and floor states are impossible in LEARNING by state
        machine design.
        """
        if n <= 0:
            return []
        with self._lock:
            active_chains = [
                c for c in self._chains.values() if c.state == ChainState.LEARNING
            ]
            if not active_chains:
                return []

            weights = [self._node_weight(c.active_node) for c in active_chains]
            if sum(weights) <= 0:
                return []

            n_actual = min(n, len(active_chains))
            selected = _weighted_sample_without_replacement(
                active_chains, weights, n_actual,
            )

            out: List[Tuple[int, ChainNode]] = []
            for chain in selected:
                node = chain.active_node
                assert node is not None  # INV-1
                node.last_used_step = current_step
                out.append((chain.prompt_key, node))
            return out

    def record_observation(
        self,
        prompt_key: int,
        k_succ: int,
        k_total: int,
        current_step: int,
        failed_rollouts: "Optional[List[FailedTrajectoryEntry]]" = None,
    ) -> None:
        """Record the outcome of a chain rollout; drive state transitions.

        `failed_rollouts` (all k_total failed trajectories from this sample)
        should be passed when `k_succ == 0`; they're stored on the node for the
        teacher to retry later. For other outcomes, the argument is ignored.
        """
        with self._lock:
            chain = self._chains.get(prompt_key)
            if chain is None:
                logger.warning(
                    f"record_observation for unknown prompt_key {prompt_key}"
                )
                return
            if chain.state != ChainState.LEARNING:
                # Possible race: the chain moved out of LEARNING (e.g., master
                # + retreat, or stuck → DEEPENING) between sample() and here.
                # Skip — the node's state is already updated.
                logger.warning(
                    f"record_observation on chain[{prompt_key}] in state "
                    f"{chain.state} (expected LEARNING); ignoring"
                )
                return

            node = chain.active_node
            assert node is not None  # INV-1

            node.observations += 1
            node.last_k_succ = k_succ
            node.last_k_total = k_total
            node.last_used_step = current_step
            if node.first_observed_step == -1:
                node.first_observed_step = current_step

            if k_total > 0 and k_succ == k_total:
                self._on_master(chain, current_step)
            elif k_succ == 0 and k_total > 0:
                node.last_failed_rollouts = (
                    list(failed_rollouts) if failed_rollouts else []
                )
                chain._transition_to(ChainState.DEEPENING_REQUESTED, current_step)
                self._deepening_queue.append(prompt_key)
                self._deepening_request_count += 1
            # else: partial success → stay LEARNING, keep training

    # ------------------------------------------------------------------------
    # Public API — teacher side
    # ------------------------------------------------------------------------

    def pending_teacher_requests(
        self,
    ) -> "List[Tuple[int, List[FailedTrajectoryEntry]]]":
        """Drain the deepening queue; return work items for the teacher worker.

        Each item is `(prompt_key, list_of_failed_rollouts)`. The teacher should
        try each rollout in turn — first successful annotation wins; all failed
        → `on_teacher_response(success=False)` → chain ABANDONED.
        """
        out: List[Tuple[int, List[FailedTrajectoryEntry]]] = []
        with self._lock:
            while self._deepening_queue:
                key = self._deepening_queue.popleft()
                chain = self._chains.get(key)
                if chain is None or chain.state != ChainState.DEEPENING_REQUESTED:
                    # Dropped chain or state changed; skip silently.
                    continue
                node = chain.active_node
                if node is None:
                    # Defensive: shouldn't happen if INV-1 holds during DEEPENING.
                    continue
                rollouts = list(node.last_failed_rollouts)
                if not rollouts:
                    # Nothing to annotate — mark as ABANDONED directly.
                    logger.warning(
                        f"chain[{key}] DEEPENING with no failed rollouts; "
                        "abandoning"
                    )
                    chain._transition_to(
                        ChainState.ABANDONED,
                        chain.last_updated_step,
                    )
                    self._abandoned_count += 1
                    continue
                out.append((key, rollouts))
            return out

    def on_teacher_response(
        self,
        prompt_key: int,
        new_augmented_ids: Optional[List[int]],
        data_instance: Optional[Dict[str, Any]],
        agent_name: str,
        current_step: int,
        success: bool = True,
    ) -> None:
        """Handle teacher's deepening response.

        On success: append a new deeper node, chain → LEARNING active = new node.
        On failure (or invalid depth): chain → ABANDONED.
        """
        with self._lock:
            chain = self._chains.get(prompt_key)
            if chain is None:
                return
            if chain.state != ChainState.DEEPENING_REQUESTED:
                # Late teacher response for a chain that already moved on
                # (e.g., concurrent retreat). Safe to ignore — chain already in
                # a valid state.
                return

            if not success:
                chain._transition_to(ChainState.ABANDONED, current_step)
                self._abandoned_count += 1
                return

            if not new_augmented_ids:
                chain._transition_to(ChainState.ABANDONED, current_step)
                self._abandoned_count += 1
                return

            # Validate the teacher's augmented prompt actually extends the
            # original — catches upstream routing bugs (wrong prompt_key →
            # wrong chain) early rather than silently corrupting the chain.
            orig_len = len(chain.original_prompt_ids)
            if (
                len(new_augmented_ids) < orig_len
                or list(new_augmented_ids[:orig_len])
                != list(chain.original_prompt_ids)
            ):
                logger.warning(
                    f"chain[{prompt_key}] teacher response does not extend "
                    f"original prompt (got len={len(new_augmented_ids)}, "
                    f"orig len={orig_len}); abandoning"
                )
                chain._transition_to(ChainState.ABANDONED, current_step)
                self._abandoned_count += 1
                self._teacher_invalid_prefix_count += 1
                return

            new_depth = len(new_augmented_ids) - orig_len
            # Invariant INV-2: depths strictly increasing.
            current_last_depth = chain.nodes[-1].depth if chain.nodes else -1
            if new_depth <= current_last_depth:
                logger.warning(
                    f"chain[{prompt_key}] teacher returned non-monotonic depth "
                    f"{new_depth} <= {current_last_depth}; abandoning"
                )
                chain._transition_to(ChainState.ABANDONED, current_step)
                self._abandoned_count += 1
                self._teacher_invalid_depth_count += 1
                return

            new_node = ChainNode(
                augmented_prompt_ids=list(new_augmented_ids),
                data_instance=data_instance or chain.data_instance,
                depth=new_depth,
                agent_name=agent_name,
            )
            chain.nodes.append(new_node)
            chain.active_idx = len(chain.nodes) - 1
            chain._transition_to(ChainState.LEARNING, current_step)
            self._teacher_success_count += 1

    # ------------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------------

    @staticmethod
    def _node_weight(node: ChainNode) -> float:
        """Sampling weight for an active node in a LEARNING chain."""
        if node.last_k_succ is None:
            return 1.0  # untried → max priority
        # INV-3: in LEARNING, last_k_succ > 0.
        # INV-4: in LEARNING, active node not mastered.
        # So the only case here is partial success.
        if not node.last_k_total:
            return 1.0  # defensive
        p = node.last_k_succ / node.last_k_total
        return 4.0 * p * (1.0 - p)

    def _on_master(self, chain: PrefixChain, step: int):
        """Active node just mastered. Compact mastered nodes from the end and
        retreat, or mark COMPLETED if we popped everything.
        """
        self._master_count += 1
        # Pop mastered nodes from the end (contiguous run). In practice this is
        # only the just-mastered active (INV-4 forbids stale mastered
        # elsewhere), but the loop tolerates rare paths where retreat happens
        # to land on a previously mastered node.
        while chain.active_idx >= 0 and chain.active_node.is_mastered:
            chain.nodes.pop(chain.active_idx)
            chain.active_idx -= 1

        if chain.active_idx < 0:
            chain._transition_to(ChainState.COMPLETED, step)
            self._completed_count += 1
            return

        # Retreat target: reset its observation state so it gets re-evaluated
        # afresh by the next sampling pass.
        new_active = chain.active_node
        new_active.observations = 0
        new_active.last_k_succ = None
        new_active.last_k_total = None
        new_active.first_observed_step = -1
        new_active.last_used_step = -1
        new_active.last_failed_rollouts = []
        chain.last_updated_step = step
        self._retreat_count += 1
        # State stays LEARNING.

    # ------------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------------

    def __len__(self) -> int:
        with self._lock:
            return len(self._chains)

    def __contains__(self, prompt_key: int) -> bool:
        with self._lock:
            return prompt_key in self._chains

    def active_chains_count(self) -> int:
        with self._lock:
            return sum(
                1 for c in self._chains.values() if c.state == ChainState.LEARNING
            )

    @property
    def stats(self) -> Dict[str, Any]:
        with self._lock:
            by_state: Dict[str, int] = defaultdict(int)
            total_nodes = 0
            depths: List[int] = []
            chain_lens: List[int] = []
            aug_lens: List[int] = []
            for c in self._chains.values():
                by_state[c.state.value] += 1
                chain_lens.append(len(c.nodes))
                total_nodes += len(c.nodes)
                for n in c.nodes:
                    depths.append(n.depth)
                    aug_lens.append(len(n.augmented_prompt_ids))

            def _mean(xs):
                return sum(xs) / len(xs) if xs else 0.0

            return {
                "chain_pool/size": len(self._chains),
                "chain_pool/learning": by_state.get("learning", 0),
                "chain_pool/deepening": by_state.get("deepening_requested", 0),
                "chain_pool/completed": by_state.get("completed", 0),
                "chain_pool/abandoned": by_state.get("abandoned", 0),
                "chain_pool/total_nodes": total_nodes,
                "chain_pool/avg_chain_len": _mean(chain_lens),
                "chain_pool/max_chain_len": max(chain_lens) if chain_lens else 0,
                "chain_pool/avg_depth": _mean(depths),
                "chain_pool/max_depth": max(depths) if depths else 0,
                "chain_pool/avg_aug_prompt_len": _mean(aug_lens),
                "chain_pool/max_aug_prompt_len": max(aug_lens) if aug_lens else 0,
                "chain_pool/retreats": self._retreat_count,
                "chain_pool/masters": self._master_count,
                "chain_pool/completed_cumulative": self._completed_count,
                "chain_pool/abandoned_cumulative": self._abandoned_count,
                "chain_pool/deepening_requests": self._deepening_request_count,
                "chain_pool/teacher_successes": self._teacher_success_count,
                "chain_pool/teacher_invalid_depth": self._teacher_invalid_depth_count,
                "chain_pool/teacher_invalid_prefix": self._teacher_invalid_prefix_count,
                "chain_pool/deepening_queue_size": len(self._deepening_queue),
            }

    def state_dict(self) -> Dict[str, Any]:
        # Deep-copy the chains under the lock so the returned snapshot is a
        # consistent point-in-time view. Without this, callers that pickle
        # asynchronously (after state_dict returns but before the pickle
        # completes) could race with on_teacher_response / record_observation
        # and serialise a torn chain mid-mutation.
        import copy as _copy
        with self._lock:
            return {
                "chains": _copy.deepcopy(self._chains),
                "deepening_queue": list(self._deepening_queue),
                "retreat_count": self._retreat_count,
                "master_count": self._master_count,
                "completed_count": self._completed_count,
                "abandoned_count": self._abandoned_count,
                "deepening_request_count": self._deepening_request_count,
                "teacher_success_count": self._teacher_success_count,
                "teacher_invalid_depth_count": self._teacher_invalid_depth_count,
                "teacher_invalid_prefix_count": self._teacher_invalid_prefix_count,
            }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        with self._lock:
            self._chains = dict(state.get("chains", {}))
            self._deepening_queue = deque(state.get("deepening_queue", []))
            self._retreat_count = state.get("retreat_count", 0)
            self._master_count = state.get("master_count", 0)
            self._completed_count = state.get("completed_count", 0)
            self._abandoned_count = state.get("abandoned_count", 0)
            self._deepening_request_count = state.get("deepening_request_count", 0)
            self._teacher_success_count = state.get("teacher_success_count", 0)
            self._teacher_invalid_depth_count = state.get(
                "teacher_invalid_depth_count", 0
            )
            self._teacher_invalid_prefix_count = state.get(
                "teacher_invalid_prefix_count", 0
            )
            # Safety net: if the teacher thread popped chains from the queue
            # but crashed before calling on_teacher_response, those chains sit
            # in DEEPENING_REQUESTED state without any queue entry. Re-enqueue
            # to ensure they get re-processed after resume.
            queued = set(self._deepening_queue)
            for key, chain in self._chains.items():
                if chain.state == ChainState.DEEPENING_REQUESTED and key not in queued:
                    self._deepening_queue.append(key)
                    queued.add(key)


# ----------------------------------------------------------------------------
# Sampling helper (shared implementation with SyntheticPromptPool)
# ----------------------------------------------------------------------------


def _weighted_sample_without_replacement(
    items: List[Any],
    weights: List[float],
    k: int,
) -> List[Any]:
    """Efraimidis-Spirakis reservoir-style weighted sampling without replacement.

    Uses key = log(random()) / weight; take top-k by key (highest key wins).

    For w > 0 and u in (0, 1], log(u)/w is in (-inf, 0]; larger w → key closer
    to 0 → higher chance of being in top-k. Zero-weight items get -inf so they
    are never selected unless they are the only options left.
    """
    if k >= len(items):
        return list(items)
    keyed: List[Tuple[float, Any]] = []
    for item, w in zip(items, weights):
        if w <= 0:
            key = float("-inf")
        else:
            u = random.random()
            if u <= 0:
                u = 1e-12
            key = math.log(u) / w
        keyed.append((key, item))
    keyed.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in keyed[:k]]
