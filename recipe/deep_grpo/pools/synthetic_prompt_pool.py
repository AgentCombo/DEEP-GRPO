"""Pool of synthetic prompt entries for the prefix_inject_mode.

A SyntheticPromptPool holds SyntheticPromptEntry objects produced by the
TeacherAnnotationWorker in prefix_inject_mode. Each entry carries an
augmented prompt (= original_prompt + locked_prefix) that the trainer samples
from each step and injects into the regular dataloader batch.

Design principles (as of the unified curriculum redesign):

  - No eviction (neither observation-based nor size-based). Entries live
    forever until a future opt-in policy removes them. This preserves
    shallow-depth entries long enough for compositional lift to activate
    them (the "retreat" mechanism).
  - Latest-observation p_hat (no EMA, no cumulative succ/total).
  - Informed prior initialises untried entries.
  - Asymmetric floor: k_succ=0 -> w_floor_hard, k_succ=k_total ->
    w_floor_mastered, otherwise 4*p*(1-p).

Selection logic (single-phase):
  Because informed prior gives every untried entry a concrete weight
  (via `initial_p_hat`), tried and untried entries are treated uniformly:

    1. Group entries by `prompt_key`. For each group, keep the top
       `max_per_prompt` entries by weight (random tiebreak). This
       enforces per-prompt diversity.
    2. Efraimidis-Spirakis weighted sampling without replacement across
       the deduplicated candidate set.

  Weight formula (`_entry_weight`) unifies both cases:
    - Untried:              4 * initial_p_hat * (1 - initial_p_hat)
    - k_succ == 0:          w_floor_hard
    - k_succ == k_total:    w_floor_mastered
    - Otherwise:            4 * p * (1 - p)

Writeback:
  After a sampled entry's rollout completes, call
  record_usage(entry, k_successes, k_total, current_step) to overwrite
  the entry's latest-observation fields.
"""

import logging
import threading
from typing import Any, Dict, List
from collections import defaultdict

import random

from recipe.deep_grpo.protocol import SyntheticPromptEntry

logger = logging.getLogger(__name__)


class SyntheticPromptPool:
    """Thread-safe pool of SyntheticPromptEntry for prefix_inject_mode.

    Producer: TeacherAnnotationWorker (via add)
    Consumer: trainer fit loop (via sample + record_usage)

    No eviction is performed. Entries are added monotonically.
    """

    def __init__(
        self,
        max_per_prompt: int = 3,
        w_floor_hard: float = 0.2,
        w_floor_mastered: float = 0.01,
    ):
        self.max_per_prompt = max_per_prompt
        self.w_floor_hard = w_floor_hard
        self.w_floor_mastered = w_floor_mastered
        self._entries: List[SyntheticPromptEntry] = []
        self._lock = threading.Lock()

    def add(self, entries: List[SyntheticPromptEntry]) -> None:
        """Accept teacher-produced entries. No eviction."""
        with self._lock:
            self._entries.extend(entries)

    def _entry_weight(self, entry: SyntheticPromptEntry) -> float:
        """Asymmetric-floor variance weight.

        - Untried (or k_total unset/zero): use informed prior.
        - k_succ == 0 and k_total > 0:     w_floor_hard      (probe for lift).
        - k_succ == k_total > 0:            w_floor_mastered (near-skip).
        - Otherwise:                        4 * p * (1 - p)  (natural variance).
        """
        # Treat entries with missing/zero k_total as untried (defensive: no
        # valid probability can be computed without positive k_total).
        if entry.last_k_succ is None or not entry.last_k_total:
            p = entry.initial_p_hat
            return max(1e-5, 4.0 * p * (1.0 - p))
        if entry.last_k_succ == 0:
            return self.w_floor_hard
        if entry.last_k_succ == entry.last_k_total:
            return self.w_floor_mastered
        p = entry.last_k_succ / entry.last_k_total
        return 4.0 * p * (1.0 - p)

    def sample(self, n: int) -> List[SyntheticPromptEntry]:
        """Select up to `n` entries for the next batch. Entries stay in pool.

        Single-phase variance-weighted sampling. Untried and tried entries
        are treated uniformly through `_entry_weight` (informed prior for
        untried, observed p for tried). Per-prompt diversity is enforced
        by keeping at most `max_per_prompt` top-weight entries per prompt.
        """
        if n <= 0:
            return []
        with self._lock:
            if not self._entries:
                return []

            # Per-prompt top-K by weight, with random tiebreak.
            by_prompt: Dict[int, List[SyntheticPromptEntry]] = defaultdict(list)
            for e in self._entries:
                by_prompt[e.prompt_key].append(e)

            candidates: List[SyntheticPromptEntry] = []
            for group in by_prompt.values():
                if len(group) <= self.max_per_prompt:
                    candidates.extend(group)
                else:
                    ranked = sorted(
                        group,
                        key=lambda e: (self._entry_weight(e), random.random()),
                        reverse=True,
                    )
                    candidates.extend(ranked[:self.max_per_prompt])

            n_pick = min(n, len(candidates))
            if n_pick == 0:
                return []
            weights = [self._entry_weight(e) for e in candidates]
            if sum(weights) > 0:
                return _weighted_sample_without_replacement(candidates, weights, n_pick)
            return random.sample(candidates, n_pick)

    def record_usage(
        self,
        entry: SyntheticPromptEntry,
        k_successes: int,
        k_total: int,
        current_step: int,
    ) -> None:
        """Overwrite the entry's latest-observation fields.

        No accumulation: this is pure overwrite (matches the curriculum
        sampler's latest-observation semantics).
        """
        with self._lock:
            entry.last_k_succ = k_successes
            entry.last_k_total = k_total
            entry.last_used_step = current_step

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def state_dict(self) -> dict:
        with self._lock:
            return {"entries": list(self._entries)}

    def load_state_dict(self, state: dict) -> None:
        with self._lock:
            self._entries = list(state["entries"])

    @property
    def stats(self) -> Dict[str, Any]:
        with self._lock:
            if not self._entries:
                return {
                    "synthetic_pool/size": 0,
                    "synthetic_pool/unique_prompts": 0,
                    "synthetic_pool/used_entries": 0,
                }
            unique_prompts = len({e.prompt_key for e in self._entries})
            tried = [e for e in self._entries if e.last_k_succ is not None]
            used = len(tried)
            # Only count entries with valid k_total for the mean; divide by
            # the same filtered count (matched numerator / denominator).
            p_values = [
                e.last_k_succ / e.last_k_total
                for e in tried if e.last_k_total
            ]
            avg_p_hat = sum(p_values) / max(len(p_values), 1)
            aug_lens = [len(e.augmented_prompt_ids) for e in self._entries]
            return {
                "synthetic_pool/size": len(self._entries),
                "synthetic_pool/unique_prompts": unique_prompts,
                "synthetic_pool/used_entries": used,
                "synthetic_pool/avg_p_hat_tried": avg_p_hat,
                "synthetic_pool/avg_aug_prompt_len": sum(aug_lens) / len(aug_lens),
                "synthetic_pool/max_aug_prompt_len": max(aug_lens),
                "synthetic_pool/min_aug_prompt_len": min(aug_lens),
            }


def _weighted_sample_without_replacement(
    items: List[Any],
    weights: List[float],
    k: int,
) -> List[Any]:
    """Efraimidis-Spirakis reservoir-style weighted sampling without replacement.

    Uses key = log(random())/weight; take top-k by key (highest key wins).

    For w > 0 and u in (0, 1], log(u)/w is in (-inf, 0]; larger w → key closer
    to 0 → higher chance of being in top-k. Zero-weight items get -inf so they
    are never selected unless they are the only options left.
    """
    import math

    if k >= len(items):
        return list(items)
    keyed = []
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
