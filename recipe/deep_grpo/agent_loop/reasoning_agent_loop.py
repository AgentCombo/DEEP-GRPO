"""Unified Reasoning Agent Loop with one-shot generation + post-hoc partitioning.

Instead of generating one sentence/chunk per LLM call (multi-step), this generates
the FULL response in a single LLM call and then partitions it into branch candidate
points using a configurable PartitionStrategy. This dramatically reduces inference
time for reasoning tasks (no environment feedback needed between steps).

Replaces the old reasoning_agent_loop_sp.py and reasoning_agent_loop_tp.py.
"""

import logging
import os
from typing import Any, Dict, List
from uuid import uuid4

from recipe.deep_grpo.protocol import FINISH_REASON, Node
from recipe.deep_grpo.agent_loop.deep_grpo_agent_loop import DeepGRPOAgentLoop
from recipe.deep_grpo.partition_strategy import (
    SentencePartitionStrategy,
    TokenCountPartitionStrategy,
    FixedCountPartitionStrategy,
)


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ReasoningAgentLoop(DeepGRPOAgentLoop):
    def __init__(self, config, server_manager, tokenizer):
        super().__init__(config, server_manager, tokenizer)

    @classmethod
    def init_class(cls, config, tokenizer):
        if cls._class_initialized:
            return

        super().init_class(config, tokenizer)

        segment_method = config.actor_rollout_ref.rollout.deep_grpo.segment_method
        if segment_method == "sp":
            stop_words = ["."]
            stop_token_ids = [tokenizer.convert_tokens_to_ids(w) for w in stop_words]
            cls.partition_strategy = SentencePartitionStrategy(stop_token_ids)
        elif segment_method == "tp":
            tokens_per_node = config.actor_rollout_ref.rollout.deep_grpo.tokens_per_node
            cls.partition_strategy = TokenCountPartitionStrategy(tokens_per_node)
        elif segment_method == "fp":
            n_segments = config.actor_rollout_ref.rollout.deep_grpo.n_segments
            cls.partition_strategy = FixedCountPartitionStrategy(n_segments)
        else:
            raise ValueError(
                f"Unknown segment_method: {segment_method}. "
                f"Supported: 'sp' (sentence), 'tp' (token count), 'fp' (fixed count)."
            )

    async def _step(self, parent_node: Node, sampling_params: Dict[str, Any], request_id: str) -> Node:
        """Generate FULL response in one LLM call (no stop words, no per-step token limit)."""
        prompt_ids = parent_node.prompt_ids + parent_node.response_ids

        remaining_len = self.max_model_len - len(prompt_ids)
        if remaining_len <= 0:
            return Node(
                node_id=uuid4().hex,
                prompt_ids=prompt_ids,
                response_ids=[],
                response_mask=[],
                data_instance=parent_node.data_instance,
                finish_reason=FINISH_REASON.EXCEED_LENGTH,
                num_turns=parent_node.num_turns + 1,
            )

        # No stop_token_ids, no per-step max_tokens — generate until EOS or max_model_len
        assert len(prompt_ids) < self.max_model_len
        result = await self.server_manager.generate_with_finish_reason(
            request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params
        )

        response_ids = result["token_ids"]
        response_mask = [1] * len(response_ids)

        if result["finish_reason"] == "length":
            finish_reason = FINISH_REASON.EXCEED_LENGTH
        elif result["finish_reason"] == "stop":
            if len(prompt_ids) + len(response_ids) >= self.max_model_len:
                finish_reason = FINISH_REASON.EXCEED_LENGTH
            else:
                finish_reason = FINISH_REASON.COMPLETED
        else:
            finish_reason = FINISH_REASON.COMPLETED

        total_len = len(prompt_ids) + len(response_ids)
        assert total_len <= self.max_model_len, (
            f"node length {total_len} > max_model_len {self.max_model_len}"
        )

        return Node(
            node_id=uuid4().hex,
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            data_instance=parent_node.data_instance,
            finish_reason=finish_reason,
            num_turns=parent_node.num_turns + 1,
        )

    async def _rollout(self, node: Node, sampling_params: Dict[str, Any], request_id: str) -> Node:
        """One-shot generation + post-hoc partitioning into branch candidate chain."""
        # 1. Generate full response in one LLM call
        full_node = await self._step(node, sampling_params, request_id)

        # 2. Score the complete response
        await self._score_node(full_node)

        # 3. Partition into chain of Nodes for branch point selection
        chain = self._partition_response_to_chain(full_node, self.partition_strategy)

        # 4. Link chain via .children pointers
        for i in range(len(chain) - 1):
            chain[i].children = [chain[i + 1]]

        return chain[0]
