from abc import abstractmethod
from typing import Any, Dict, List, Union, Tuple, Optional

import asyncio
import logging
import os
from uuid import uuid4

import numpy as np

from verl.experimental.agent_loop.agent_loop import AgentLoopBase
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

from recipe.deep_grpo.protocol import Node, FINISH_REASON
from recipe.deep_grpo.agent_loop.outputs import TSTrainAgentLoopOutput, TSValAgentLoopOutput
from recipe.deep_grpo.reward.reward_manager import score_node

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class TSAgentLoop(AgentLoopBase):
    """Generic tree-search rollout core.

    Owns the generic tree/chain machinery: step-wise rollout of a chain of
    Nodes, parallel expansion, chain compression, reward scoring, and packing
    nodes into train/val outputs with group-baseline advantages.

    Method extensions (e.g. DEEP-GRPO's branch expansion, teacher suffix
    synthesis, and prefix injection) subclass this without modifying it —
    see recipe/deep_grpo/agent_loop.py.
    """

    def __init__(self, config, server_manager, tokenizer):
        super().__init__(config, server_manager, tokenizer)

    @classmethod
    def init_class(cls, config, tokenizer):
        if cls._class_initialized:
            return
        cls._class_initialized = True

        cls.tokenizer = tokenizer

        cls.max_model_len = config.actor_rollout_ref.rollout.max_model_len

        cls.rollout_n = config.actor_rollout_ref.rollout.n
        assert cls.rollout_n > 1

    async def _score_node(self, node: Node):
        result = await score_node(node=node, tokenizer=self.tokenizer, is_validation=getattr(self, '_is_validation', False))
        node.reward = result.reward
        node.reward_info = result

    @abstractmethod
    async def _step(self, parent_node: Node, sampling_params: Dict[str, Any], request_id: str) -> Node:
        raise NotImplementedError

    async def _rollout(self, node: Node, sampling_params, request_id: str) -> Node:
        current_node = node
        generated_chain: List[Node] = []
        while True:
            new_node = await self._step(current_node, sampling_params, request_id)
            generated_chain.append(new_node)
            if new_node.finish_reason == FINISH_REASON.COMPLETED or new_node.finish_reason == FINISH_REASON.EXCEED_LENGTH:
                break
            current_node = new_node
        assert len(generated_chain) > 0
        last_node = generated_chain[-1]
        await self._score_node(last_node)
        for i in range(len(generated_chain) - 1):
            generated_chain[i].children = [generated_chain[i + 1]]
        return generated_chain[0]

    async def _expand(self, node: Node, sampling_params: Dict[str, Any], rollout_n: int):
        assert node.finish_reason is None or node.finish_reason == FINISH_REASON.STOP
        tasks = []
        for rollout_idx in range(rollout_n):
            request_id = f"{node.node_id}:{rollout_idx}"
            tasks.append(asyncio.create_task(self._rollout(node, sampling_params, request_id)))
        new_children = await asyncio.gather(*tasks)
        if node.children is not None:
            node.children.extend(new_children)
        else:
            node.children = new_children

    def _get_chain_from_start_node(self, start_node: Node) -> List[Node]:
        chain = []
        current_node = start_node
        while current_node:
            chain.append(current_node)
            if current_node.children and len(current_node.children) > 0:
                current_node = current_node.children[0]
            else:
                break
        return chain

    def _node_to_val_output(self, node: Node, metrics: Dict[str, Any]) -> TSValAgentLoopOutput:
        assert len(node.prompt_ids) > 0
        assert len(node.response_ids) > 0
        assert len(node.response_ids) == len(node.response_mask)
        assert node.num_turns is not None
        assert node.reward is not None
        assert node.reward_info is not None

        return TSValAgentLoopOutput(
            prompt_ids=node.prompt_ids,
            response_ids=node.response_ids,
            response_mask=node.response_mask,
            num_turns=node.num_turns,
            reward=node.reward,
            reward_info=node.reward_info,
            metrics=metrics
        )

    def _node_to_train_output(self, node: Node, tree_id: str, metrics: Dict[str, Any]) -> TSTrainAgentLoopOutput:
        val_output = self._node_to_val_output(node, metrics)

        assert node.advantage is not None

        return TSTrainAgentLoopOutput(
            **val_output.model_dump(),
            tree_id=tree_id,
            node_id=node.node_id,
            advantage=node.advantage,
        )

    def _compress_chain(self, chain: List[Node]) -> Node:
        head = chain[0]
        last = chain[-1]

        prompt_ids = head.prompt_ids
        data_instance = head.data_instance

        finish_reason = last.finish_reason
        reward = last.reward
        reward_info = last.reward_info
        num_turns = last.num_turns

        response_ids = []
        response_mask = []

        for node in chain:
            response_ids.extend(node.response_ids)
            response_mask.extend(node.response_mask)

        return Node(node_id=uuid4().hex,
                    prompt_ids=prompt_ids,
                    response_ids=response_ids,
                    response_mask=response_mask,
                    data_instance=data_instance,
                    finish_reason=finish_reason,
                    num_turns=num_turns,
                    reward=reward,
                    reward_info=reward_info
                )

    def _collect_train_outputs(
        self,
        nodes: List[Node],
        tree_id: str,
        timing_metrics: Optional[List[Dict[str, Any]]] = None
    ) -> List[TSTrainAgentLoopOutput]:
        rewards = [n.reward for n in nodes]
        baseline = np.mean(rewards)

        outputs = []
        for i, node in enumerate(nodes):
            node.advantage = float(node.reward) - float(baseline)
            if timing_metrics is not None:
                metrics = timing_metrics[i]
            else:
                metrics = {}
            outputs.append(self._node_to_train_output(node, tree_id=tree_id, metrics=metrics))

        return outputs

    async def _process_single_main_chain(
        self,
        prompt_ids: List[int],
        sampling_params: Dict[str, Any],
        data_instance: Dict[str, Any],
    ) -> Tuple[Node, Dict[str, float]]:
        """Generate one main chain and compress it into a single node."""
        timing_metrics = {}
        root = Node(
            node_id=uuid4().hex,
            prompt_ids=prompt_ids,
            response_ids=[],
            response_mask=[],
            data_instance=data_instance,
            num_turns=0
        )

        with simple_timer("generate_main_chain", timing_metrics):
            await self._expand(root, sampling_params, rollout_n=1)

        main_chain = [root] + self._get_chain_from_start_node(root.children[0])
        main_chain_node = self._compress_chain(main_chain)

        return main_chain_node, timing_metrics

    @rollout_trace_op
    async def _run_train(self,
                        messages: List[Dict[str, Any]],
                        sampling_params: Dict[str, Any],
                        data_instance: Dict[str, Any]) -> Tuple[List[TSTrainAgentLoopOutput], List[TSTrainAgentLoopOutput]]:
        """Group rollout: rollout_n main chains with group-baseline advantages."""
        self._is_validation = False
        tree_id = data_instance["uid"]

        prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=getattr(self, 'add_generation_prompt', True), tokenize=True
            ),
        )

        tasks = []
        for _ in range(self.rollout_n):
            task = asyncio.create_task(
                self._process_single_main_chain(
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                    data_instance=data_instance
                )
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        main_chain_nodes = []
        main_chain_timing_metrics = []
        for main_chain_node, timing_metrics in results:
            main_chain_nodes.append(main_chain_node)
            main_chain_timing_metrics.append(timing_metrics)

        main_chain_outputs = self._collect_train_outputs(
            nodes=main_chain_nodes,
            tree_id=tree_id,
            timing_metrics=main_chain_timing_metrics
        )

        return main_chain_outputs, []

    @rollout_trace_op
    async def _run_val(
        self,
        messages: List[Dict[str, Any]],
        sampling_params: Dict[str, Any],
        data_instance: Dict[str, Any],
    ) -> TSValAgentLoopOutput:
        self._is_validation = True
        metrics = {}

        prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=getattr(self, 'add_generation_prompt', True), tokenize=True
            ),
        )

        root = Node(
            node_id=uuid4().hex,
            prompt_ids=prompt_ids,
            response_ids=[],
            response_mask=[],
            data_instance=data_instance,
            num_turns=0
        )

        with simple_timer("generate_main_chain", metrics):
            await self._expand(root, sampling_params, rollout_n=1)

        node = self._compress_chain(self._get_chain_from_start_node(root.children[0]))

        return self._node_to_val_output(node, metrics)

    async def run(
        self,
        mode: str,
        messages: List[Dict[str, Any]],
        sampling_params: Dict[str, Any],
        data_instance: Dict[str, Any],
    ) -> Union[TSValAgentLoopOutput, Tuple[List[TSTrainAgentLoopOutput], List[TSTrainAgentLoopOutput]], List[Dict[str, Any]]]:
        if mode == "train":
            return await self._run_train(messages, sampling_params, data_instance)
        elif mode == "validate":
            return await self._run_val(messages, sampling_params, data_instance)
        else:
            raise ValueError(f"Unknown mode: {mode}")
