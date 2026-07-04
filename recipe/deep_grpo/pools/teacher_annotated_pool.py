import random
import logging
import threading
from typing import List, Dict, Any
from collections import defaultdict

from recipe.deep_grpo.protocol import BranchPointEntry

logger = logging.getLogger(__name__)


class TeacherAnnotatedPool:
    """Thread-safe pool of teacher-annotated BranchPointEntry objects ready for branch generation.

    All entries in this pool have teacher_suffix set (guaranteed).
    The background TeacherAnnotationWorker produces entries.
    The training loop consumes entries when enough have accumulated.
    """

    def __init__(
        self,
        max_size: int = 10000,
    ):
        self.max_size = max_size
        self._entries: List[BranchPointEntry] = []
        self._lock = threading.Lock()

    def add(self, entries: List[BranchPointEntry]):
        """Add teacher-annotated entries (called by background worker).

        No length filtering here — synthesize_teacher_suffix already ensures
        sufficient remaining room for model branch generation.
        """
        with self._lock:
            for entry in entries:
                assert entry.teacher_suffix is not None, (
                    "TeacherAnnotatedPool only accepts entries with teacher_suffix"
                )
                self._entries.append(entry)

            # Evict by size (keep newest)
            if len(self._entries) > self.max_size:
                self._entries = self._entries[-self.max_size:]

    def has_enough(self, threshold: int) -> bool:
        """Check if pool has enough entries for branch training."""
        with self._lock:
            return len(self._entries) >= threshold

    def sample(self, n: int, strategy: str = "stratified") -> List[BranchPointEntry]:
        """Sample n entries for branch generation (called by training loop).

        Sampled entries are removed from the pool.
        """
        with self._lock:
            if len(self._entries) == 0:
                return []
            n = min(n, len(self._entries))

            if strategy == "uniform":
                indices = random.sample(range(len(self._entries)), n)
            elif strategy == "stratified":
                by_tree: Dict[str, List[int]] = defaultdict(list)
                for i, entry in enumerate(self._entries):
                    by_tree[entry.tree_id].append(i)

                tree_ids = list(by_tree.keys())
                random.shuffle(tree_ids)

                indices = []
                idx = 0
                while len(indices) < n:
                    tree_id = tree_ids[idx % len(tree_ids)]
                    pool = by_tree[tree_id]
                    if pool:
                        chosen = pool.pop(random.randrange(len(pool)))
                        indices.append(chosen)
                    idx += 1
            else:
                raise ValueError(f"Unknown sampling strategy: {strategy}")

            # Remove sampled entries (sort descending to pop without index shift)
            indices_sorted = sorted(indices, reverse=True)
            sampled = [None] * len(indices)
            for pos, idx in enumerate(indices):
                sampled[pos] = self._entries[idx]
            for idx in indices_sorted:
                self._entries.pop(idx)
            return sampled

    def state_dict(self) -> dict:
        with self._lock:
            return {"entries": list(self._entries)}

    def load_state_dict(self, state: dict):
        with self._lock:
            self._entries = list(state["entries"])

    def __len__(self):
        with self._lock:
            return len(self._entries)

    @property
    def stats(self) -> Dict[str, Any]:
        with self._lock:
            if not self._entries:
                return {"teacher_pool/size": 0}
            unique_trees = set(e.tree_id for e in self._entries)
            return {
                "teacher_pool/size": len(self._entries),
                "teacher_pool/unique_trees": len(unique_trees),
            }
