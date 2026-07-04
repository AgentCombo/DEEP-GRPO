import random
import logging
from typing import List, Dict, Any, Tuple
from collections import defaultdict

from recipe.deep_grpo.protocol import BranchPointEntry

logger = logging.getLogger(__name__)


class BranchPointBuffer:
    """Buffer that stores branch points from previous steps for one-stage generation.

    Key properties:
    - FIFO with max_size cap (oldest entries evicted first)
    - Entries expire after max_age steps
    - Sampling can be stratified by tree_id for diversity
    - Entries that leave too little room for continuation are filtered at insertion
    - step_collected is managed internally, not stored in BranchPointEntry
    """

    def __init__(
        self,
        max_size: int = 10000,
        max_age: int = 3,
        max_model_len: int = 4096,
        min_remaining_ratio: float = 0.1,
    ):
        self.max_size = max_size
        self.max_age = max_age
        self.max_model_len = max_model_len
        self.min_remaining_tokens = int(max_model_len * min_remaining_ratio)
        self._entries: List[BranchPointEntry] = []
        self._steps: List[int] = []  # parallel list: step when each entry was collected

    def add(self, entries: List[BranchPointEntry], current_step: int):
        """Add new branch points, filtering those that leave too little room for continuation."""
        for entry in entries:
            remaining = self.max_model_len - entry.total_token_length
            if remaining < self.min_remaining_tokens:
                continue
            self._entries.append(entry)
            self._steps.append(current_step)

        # Evict by age
        keep = [(e, s) for e, s in zip(self._entries, self._steps)
                if (current_step - s) <= self.max_age]
        if keep:
            self._entries, self._steps = map(list, zip(*keep))
        else:
            self._entries, self._steps = [], []

        # Evict by size (keep newest)
        if len(self._entries) > self.max_size:
            self._entries = self._entries[-self.max_size:]
            self._steps = self._steps[-self.max_size:]

    def sample(self, n: int, strategy: str = "stratified") -> List[BranchPointEntry]:
        """Sample n entries from the buffer.

        Args:
            n: Number of entries to sample.
            strategy: "uniform" for uniform random, "stratified" for
                      stratified by tree_id (ensures diversity across prompts).

        Returns:
            List of sampled BranchPointEntry (may be fewer than n if buffer is small).
        """
        if len(self._entries) == 0:
            return []

        n = min(n, len(self._entries))

        if strategy == "uniform":
            return random.sample(self._entries, n)

        elif strategy == "stratified":
            by_tree: Dict[str, List[BranchPointEntry]] = defaultdict(list)
            for entry in self._entries:
                by_tree[entry.tree_id].append(entry)

            tree_ids = list(by_tree.keys())
            random.shuffle(tree_ids)

            sampled = []
            idx = 0
            while len(sampled) < n:
                tree_id = tree_ids[idx % len(tree_ids)]
                pool = by_tree[tree_id]
                if pool:
                    sampled.append(pool.pop(random.randrange(len(pool))))
                idx += 1
            return sampled

        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")

    def state_dict(self) -> dict:
        """Serialize buffer state for checkpointing (pickle-friendly)."""
        return {
            "entries": list(self._entries),
            "steps": list(self._steps),
        }

    def load_state_dict(self, state: dict):
        """Restore buffer state from checkpoint."""
        self._entries = list(state["entries"])
        self._steps = list(state["steps"])

    def __len__(self):
        return len(self._entries)

    @property
    def stats(self) -> Dict[str, Any]:
        """Return buffer statistics for logging."""
        if not self._entries:
            return {"buffer/size": 0}
        unique_steps = set(self._steps)
        unique_trees = set(e.tree_id for e in self._entries)
        return {
            "buffer/size": len(self._entries),
            "buffer/unique_trees": len(unique_trees),
            "buffer/unique_steps": len(unique_steps),
            "buffer/avg_position_ratio": sum(e.position_ratio for e in self._entries) / len(self._entries),
        }
