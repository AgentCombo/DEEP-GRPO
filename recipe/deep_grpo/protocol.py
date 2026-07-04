"""DEEP-GRPO data structures.

Rollout-side node/reward types plus the teacher-annotation pipeline entries
(failed student trajectories, teacher-verified suffixes, branch points, and
synthetic prefix-augmented prompts).

This module is kept import-light on purpose (no verl/torch imports), so the
pool unit tests under recipe/deep_grpo/tests can run without the training
stack installed. The agent-loop output types that extend verl's
AgentLoopOutput live in recipe/deep_grpo/agent_loop/outputs.py instead.
"""

from typing import Any, Dict, List, Optional
from enum import Enum, auto

from dataclasses import dataclass


class FINISH_REASON(Enum):
    EXCEED_LENGTH = auto()
    STOP = auto()
    COMPLETED = auto()


@dataclass
class RewardInfo:
   reward: float
   completed: int
   finished: int = 0
   judgement_reply: Optional[str] = None


@dataclass
class Node:
  node_id: str
  prompt_ids: List[int]
  response_ids: List[int]
  response_mask: List[int]
  data_instance: Dict[str, Any]
  finish_reason: Optional[FINISH_REASON] = None
  num_turns: Optional[float] = None
  children: Optional[List["Node"]] = None
  reward: Optional[float] = None
  reward_info: Optional[RewardInfo] = None
  advantage: Optional[float] = None
  log_probs: Optional[List[float]] = None


@dataclass
class FailedTrajectoryEntry:
    """Main chain trajectory entry for teacher analysis or branch point selection.

    In teacher mode (expand_branch_chain=False): only failed chains, branch_points=None.
    In regular mode (expand_branch_chain=True): failed chains with pre-selected branch_points.
    When expand_only_on_low_quality=False, may also include non-failed chains that carry branch_points.
    """
    prompt_ids: List[int]
    response_ids: List[int]
    response_mask: List[int]
    data_instance: Dict[str, Any]
    tree_id: str
    num_turns: float
    agent_name: str = ""
    # Pre-selected branch points (filled when expand_branch_chain=True, None otherwise)
    branch_points: Optional[List["BranchPointEntry"]] = None


@dataclass
class TeacherSuffix:
    """Pre-computed teacher-synthesized correct suffix for a branch point."""
    suffix_ids: List[int]
    suffix_mask: List[int]
    reward: float
    reward_info: RewardInfo
    original_failed_suffix_ids: List[int]


@dataclass
class SyntheticPromptEntry:
    """A prompt-level inject entry for the prefix_inject_mode.

    Carries an augmented prompt (= original_prompt + accumulated locked_prefix)
    that the trainer injects as an extra rollout target. All rollout / reward /
    GRPO logic sees it as an ordinary prompt -- the trainer mints a fresh
    tree_id per injection for GRPO grouping.

    Selection state uses **latest-observation only** semantics (no EMA, no
    cumulative aggregation): `last_k_succ` and `last_k_total` hold the most
    recent rollout group's outcome, overwritten each time the entry is used.
    `initial_p_hat` is the informed prior used before the first observation.
    """
    augmented_prompt_ids: List[int]
    # Preserved from the failed trajectory for reward scoring (extra_info.answer,
    # reward_model, etc.) and rollout bookkeeping.
    data_instance: Dict[str, Any]
    agent_name: str = ""

    # Informed prior: used as p_hat before the first rollout observation.
    # Computed as: prefix_frac + (1 - prefix_frac) * parent_p_hat  (teacher_worker).
    initial_p_hat: float = 0.5

    # Latest-observation state (overwritten by SyntheticPromptPool.record_usage)
    last_k_succ: Optional[int] = None   # None = untried
    last_k_total: Optional[int] = None
    last_used_step: int = -1

    @property
    def prompt_key(self) -> int:
        """Hash key for per-prompt grouping in selection."""
        return hash(tuple(self.augmented_prompt_ids))


@dataclass
class BranchPointEntry:
    """A branch point: the state needed to generate alternative continuations."""
    # Node state (for reconstructing a Node via to_node())
    prompt_ids: List[int]          # Full context up to this point
    response_ids: List[int]        # This node's response segment
    response_mask: List[int]       # Mask for the response segment
    data_instance: Dict[str, Any]  # For reward scoring of continuations
    num_turns: float               # Turn count at this point

    # Chain context
    tree_id: str                   # Which prompt this came from
    branch_chain_root_index: int   # Position in the original chain
    chain_total_length: int        # Length of the original chain

    # Prompt-level attribute (set by AgentLoopWorker, not by TSAgentLoop)
    agent_name: str = ""

    # Teacher-synthesized suffix (set by TeacherAnnotationWorker)
    teacher_suffix: Optional["TeacherSuffix"] = None

    def to_node(self) -> "Node":
        """Reconstruct a Node suitable for _expand()."""
        from uuid import uuid4
        return Node(
            node_id=uuid4().hex,
            prompt_ids=list(self.prompt_ids),
            response_ids=list(self.response_ids),
            response_mask=list(self.response_mask),
            data_instance=self.data_instance,
            num_turns=self.num_turns,
            finish_reason=FINISH_REASON.STOP,  # Expandable nodes are always STOP
        )

    @property
    def total_token_length(self) -> int:
        return len(self.prompt_ids) + len(self.response_ids)

    @property
    def position_ratio(self) -> float:
        return (self.branch_chain_root_index + 1) / self.chain_total_length
