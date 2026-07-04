from abc import ABC, abstractmethod
from typing import List, Tuple


class PartitionStrategy(ABC):
    """Strategy for partitioning a complete response into segments for branch point selection."""

    @abstractmethod
    def partition(self, response_ids: List[int], response_mask: List[int]) -> List[Tuple[int, int]]:
        """Partition response into contiguous, non-overlapping segments.

        Args:
            response_ids: Full response token IDs.
            response_mask: Mask for the response tokens.

        Returns:
            List of (start, end) index pairs. Segments cover [0, len(response_ids)).
        """
        raise NotImplementedError


class SentencePartitionStrategy(PartitionStrategy):
    """Split at sentence-ending token boundaries (e.g., period tokens).

    Equivalent to the old reasoning_agent_loop_sp behavior, but applied post-hoc
    on an already-generated response instead of stopping generation at each boundary.
    """

    def __init__(self, stop_token_ids: List[int]):
        self.stop_token_ids = set(stop_token_ids)

    def partition(self, response_ids: List[int], response_mask: List[int]) -> List[Tuple[int, int]]:
        if not response_ids:
            return [(0, 0)]

        segments = []
        start = 0
        for i, tid in enumerate(response_ids):
            if tid in self.stop_token_ids:
                segments.append((start, i + 1))
                start = i + 1
        # Remaining tokens after the last stop token
        if start < len(response_ids):
            segments.append((start, len(response_ids)))
        return segments


class TokenCountPartitionStrategy(PartitionStrategy):
    """Split at fixed token count intervals.

    Equivalent to the old reasoning_agent_loop_tp behavior, but applied post-hoc.
    """

    def __init__(self, tokens_per_segment: int):
        assert tokens_per_segment > 0
        self.tokens_per_segment = tokens_per_segment

    def partition(self, response_ids: List[int], response_mask: List[int]) -> List[Tuple[int, int]]:
        if not response_ids:
            return [(0, 0)]

        segments = []
        n = len(response_ids)
        for start in range(0, n, self.tokens_per_segment):
            end = min(start + self.tokens_per_segment, n)
            segments.append((start, end))
        return segments


class FixedCountPartitionStrategy(PartitionStrategy):
    """Split into a fixed number of approximately equal segments.

    This is a new partition mode ("fp" = fixed point count).
    If n_segments > len(response_ids), segments will equal len(response_ids).
    """

    def __init__(self, n_segments: int):
        assert n_segments > 0
        self.n_segments = n_segments

    def partition(self, response_ids: List[int], response_mask: List[int]) -> List[Tuple[int, int]]:
        n = len(response_ids)
        if n == 0:
            return [(0, 0)]

        k = min(self.n_segments, n)
        seg_size = n // k
        remainder = n % k

        segments = []
        start = 0
        for i in range(k):
            end = start + seg_size + (1 if i < remainder else 0)
            segments.append((start, end))
            start = end
        return segments
