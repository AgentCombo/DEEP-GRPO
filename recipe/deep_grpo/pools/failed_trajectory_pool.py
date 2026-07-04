import random
import logging
import threading
from collections import deque
from typing import Any, Deque, Dict, List, Tuple

from recipe.deep_grpo.protocol import FailedTrajectoryEntry

logger = logging.getLogger(__name__)


class FailedTrajectoryPool:
    """Thread-safe pool of complete failed trajectories awaiting teacher analysis.

    Two modes:

    1. Legacy mode (prefix_inject_mode disabled): FIFO-ish list with age-based
       eviction and uniform random sampling. Behavior preserved bit-identical
       to the original implementation for backward compatibility.

    2. Priority mode (prefix_inject_mode enabled): Dict[prompt_key, Deque[(step, entry)]]
       with round-robin sampling across prompt keys, ordered each lap by the
       deque's right-end step (= freshness of the newest remaining entry in
       that deque). LIFO within a deque. See plan doc for rationale.
    """

    def __init__(
        self,
        max_size: int = 10000,
        max_age: int = 5,
        priority_mode: bool = False,
    ):
        self.max_size = max_size
        self.max_age = max_age
        self.priority_mode = priority_mode

        # Legacy-mode state
        self._entries: List[FailedTrajectoryEntry] = []
        self._steps: List[int] = []

        # Priority-mode state
        self._priority_pool: Dict[int, Deque[Tuple[int, FailedTrajectoryEntry]]] = {}

        self._lock = threading.Lock()

    # ------------------------------------------------------------------ add
    def add(self, entries: List[FailedTrajectoryEntry], current_step: int):
        """Add new failed trajectories (called by training loop)."""
        with self._lock:
            if self.priority_mode:
                self._add_priority(entries, current_step)
            else:
                self._add_legacy(entries, current_step)

    def _add_legacy(self, entries: List[FailedTrajectoryEntry], current_step: int):
        for entry in entries:
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

    def _add_priority(self, entries: List[FailedTrajectoryEntry], current_step: int):
        """Append entries without eviction.

        Old entries naturally sit at the bottom of each deque and are
        never sampled (LIFO from the right end), so keeping them does
        not affect sampling. Bounding memory is left to natural dynamics.
        """
        for entry in entries:
            key = hash(tuple(entry.prompt_ids))
            if key not in self._priority_pool:
                self._priority_pool[key] = deque()
            self._priority_pool[key].append((current_step, entry))

    # ------------------------------------------------------- sample_and_remove
    def sample_and_remove(self, n: int) -> List[FailedTrajectoryEntry]:
        """Sample and remove up to n entries.

        Legacy mode: uniform random (matches original behavior).
        Priority mode: freshness-ordered round-robin (see class docstring).
        """
        with self._lock:
            if self.priority_mode:
                return self._sample_priority(n)
            else:
                return self._sample_legacy(n)

    def _sample_legacy(self, n: int) -> List[FailedTrajectoryEntry]:
        if len(self._entries) == 0:
            return []
        n = min(n, len(self._entries))
        indices = random.sample(range(len(self._entries)), n)
        indices.sort(reverse=True)

        # Collect in original random order, then remove by reverse index
        sampled = [self._entries[idx] for idx in indices]
        for idx in sorted(indices, reverse=True):
            self._entries.pop(idx)
            self._steps.pop(idx)
        return sampled

    def _sample_priority(self, n: int) -> List[FailedTrajectoryEntry]:
        if n <= 0 or not self._priority_pool:
            return []

        result: List[FailedTrajectoryEntry] = []
        while len(result) < n and self._priority_pool:
            # Re-sort keys at the start of each lap by the step of the deque's
            # rightmost (newest) remaining entry. This reflects the true
            # freshness of what's left in each deque, independent of past pops.
            keys_sorted = sorted(
                self._priority_pool.keys(),
                key=lambda k: self._priority_pool[k][-1][0],
                reverse=True,
            )
            lap_took_any = False
            for key in keys_sorted:
                if len(result) >= n:
                    break
                dq = self._priority_pool.get(key)
                if dq:
                    _, entry = dq.pop()
                    result.append(entry)
                    lap_took_any = True
                    if not dq:
                        del self._priority_pool[key]
            if not lap_took_any:
                break
        return result

    # ------------------------------------------------------------- state dict
    def state_dict(self) -> dict:
        with self._lock:
            if self.priority_mode:
                # Flatten priority pool to (step, entry) list
                flat: List[Tuple[int, FailedTrajectoryEntry]] = []
                for dq in self._priority_pool.values():
                    flat.extend(dq)
                return {"priority_entries": flat, "priority_mode": True}
            return {
                "entries": list(self._entries),
                "steps": list(self._steps),
                "priority_mode": False,
            }

    def load_state_dict(self, state: dict):
        with self._lock:
            saved_priority = state.get("priority_mode", False)
            if saved_priority and self.priority_mode:
                self._priority_pool = {}
                for step, entry in state["priority_entries"]:
                    key = hash(tuple(entry.prompt_ids))
                    if key not in self._priority_pool:
                        self._priority_pool[key] = deque()
                    self._priority_pool[key].append((step, entry))
            elif not saved_priority and not self.priority_mode:
                self._entries = list(state["entries"])
                self._steps = list(state["steps"])
            else:
                # Mode mismatch: reset to empty rather than corrupt state.
                logger.warning(
                    "FailedTrajectoryPool.load_state_dict: mode mismatch "
                    "(saved priority=%s, current priority=%s); starting empty.",
                    saved_priority, self.priority_mode,
                )
                if self.priority_mode:
                    self._priority_pool = {}
                else:
                    self._entries, self._steps = [], []

    def __len__(self):
        with self._lock:
            if self.priority_mode:
                return sum(len(dq) for dq in self._priority_pool.values())
            return len(self._entries)

    @property
    def stats(self) -> Dict[str, Any]:
        with self._lock:
            if self.priority_mode:
                total = sum(len(dq) for dq in self._priority_pool.values())
                unique = len(self._priority_pool)
                max_depth = (
                    max((len(dq) for dq in self._priority_pool.values()), default=0)
                )
                return {
                    "failed_pool/size": total,
                    "failed_pool/unique_keys": unique,
                    "failed_pool/max_deque_depth": max_depth,
                }
            return {"failed_pool/size": len(self._entries)}
