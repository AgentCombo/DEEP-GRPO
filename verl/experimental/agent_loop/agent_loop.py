# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import heapq
import logging
import os
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Union, Tuple

import numpy as np
import ray
import torch
from cachetools import LRUCache
from omegaconf import DictConfig
from pydantic import BaseModel
from tensordict import TensorDict
from transformers import AutoTokenizer

from verl.protocol import DataProto
from verl.single_controller.ray.base import RayWorkerGroup
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.rollout_trace import RolloutTraceConfig, rollout_trace_attr, rollout_trace_op
from verl.workers.rollout.async_server import async_server_class

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def flatten(lst):
    for item in lst:
        if isinstance(item, (list, tuple)):
            yield from flatten(item)
        else:
            yield item


class AsyncLLMServerManager:
    """
    A class to manage multiple OpenAI compatible LLM servers. This class provides
    - Load balance: least requests load balancing
    - Sticky session: send multi-turn chat completions to same server for automatic prefix caching
    """

    def __init__(self, config: DictConfig, server_handles: List[ray.actor.ActorHandle], max_cache_size: int = 10000):
        """Initialize the AsyncLLMServerManager.

        Args:
            config (DictConfig): YAML config.
            server_handles (List[ray.actor.ActorHandle]): OpenAI compatible LLM server actor handles.
            max_cache_size (int, optional): max cache size for request_id to server mapping. Defaults to 10000.
        """
        self.config = config
        self.server_handles = server_handles
        random.shuffle(self.server_handles)

        # Least requests load balancing
        self.weighted_serveres = [[0, (hash(server), server)] for server in server_handles]
        heapq.heapify(self.weighted_serveres)

        # LRU cache to map request_id to server
        self.request_id_to_server = LRUCache(maxsize=max_cache_size)

    def _choose_server(self, request_id: str) -> ray.actor.ActorHandle:
        # TODO: implement server pressure awareness load balancing
        if request_id in self.request_id_to_server:
            return self.request_id_to_server[request_id]

        server = self.weighted_serveres[0][1][1]
        self.weighted_serveres[0][0] += 1
        heapq.heapreplace(self.weighted_serveres, self.weighted_serveres[0])
        self.request_id_to_server[request_id] = server
        return server

    @rollout_trace_op
    async def generate(
        self,
        request_id,
        *,
        prompt_ids: List[int],
        sampling_params: Dict[str, Any],
    ) -> List[int]:
        """Generate tokens from prompt ids.

        Args:
            request_id (str): request id for sticky session.
            prompt_ids (List[int]): List of prompt token ids.
            sampling_params (Dict[str, Any]): Sampling parameters for the chat completion.

        Returns:
            List[int]: List of generated token ids.
        """
        server = self._choose_server(request_id)
        output = await server.generate.remote(
            request_id=request_id,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
        )
        return output
    
    @rollout_trace_op
    async def generate_with_finish_reason(
        self,
        request_id,
        *,
        prompt_ids: List[int],
        sampling_params: Dict[str, Any],
    ) -> List[int]:
        """Generate tokens from prompt ids.

        Args:
            request_id (str): request id for sticky session.
            prompt_ids (List[int]): List of prompt token ids.
            sampling_params (Dict[str, Any]): Sampling parameters for the chat completion.

        Returns:
            List[int]: List of generated token ids.
        """
        server = self._choose_server(request_id)
        output = await server.generate_with_finish_reason.remote(
            request_id=request_id,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
        )
        return output


class AgentLoopMetrics(BaseModel):
    """Agent loop performance metrics."""
    generate_main_chain: float = 0.0
    generate_branch_chain: float = 0.0


class AgentLoopOutput(BaseModel):
    """Agent loop output."""

    prompt_ids: List[int]
    response_ids: List[int]
    response_mask: List[int]
    num_turns: float = 0
    metrics: AgentLoopMetrics


class AgentLoopBase(ABC):
    """An agent loop takes a input message, chat with OpenAI compatible LLM server and interact with various
    environments."""

    _class_initialized = False

    def __init__(self, config: DictConfig, server_manager: AsyncLLMServerManager, tokenizer: AutoTokenizer):
        """Initialize agent loop.

        Args:
            config (DictConfig): YAML config.
            server_manager (AsyncLLMServerManager): OpenAI compatible LLM server manager.
            tokenizer (AutoTokenizer): Tokenizer for tokenize messages.
        """
        self.config = config
        self.server_manager = server_manager
        self.tokenizer = tokenizer
        self.loop = asyncio.get_running_loop()
        self.init_class(config, tokenizer)

    @classmethod
    def init_class(cls, config: DictConfig, tokenizer: AutoTokenizer):
        """Initialize class state shared across all instances."""
        if cls._class_initialized:
            return
        cls._class_initialized = True

    @abstractmethod
    async def run(self, mode: str, messages: List[Dict[str, Any]], sampling_params: Dict[str, Any], data_instance: Dict[str, Any]) -> Union[AgentLoopOutput, Tuple[List[AgentLoopOutput], List[AgentLoopOutput]]]:
        """Run agent loop to interact with LLM server and environment.

        Args:
            messages (List[Dict[str, Any]]): Input messages.
            sampling_params (Dict[str, Any]): LLM sampling params.

        Returns:
            AgentLoopOutput: Agent loop output.
        """
        raise NotImplementedError


@ray.remote
class AgentLoopWorker:
    """Agent loop worker takes a batch of messages and run each message in an agent loop."""

    def __init__(self, config: DictConfig, server_handles: List[ray.actor.ActorHandle]):
        """Initialize agent loop manager.

        Args:
            config (DictConfig): YAML config.
            server_handles (List[ray.actor.ActorHandle]): OpenAI compatible LLM server actor handles.
        """
        self.config = config
        self.server_manager = AsyncLLMServerManager(config, server_handles)

        model_path = config.actor_rollout_ref.model.path
        self.model_name = "/".join(model_path.split("/")[-2:])
        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=True)

        trace_config = config.trainer.get("rollout_trace", {})

        RolloutTraceConfig.init(
            config.trainer.project_name,
            config.trainer.experiment_name,
            trace_config.get("backend"),
            trace_config.get("token2text", False),
        )

    async def generate_sequences(self, batch: DataProto) -> DataProto:
        """Generate sequences from agent loop.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=1.0,
        )

        # override sampling params for validation
        validate = batch.meta_info.get("validate", False)
        mode = "validate" if validate else "train"
        if validate:
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["temperature"] = config.val_kwargs.temperature

        if "agent_name" not in batch.non_tensor_batch:
            default_agent = self.config.actor_rollout_ref.rollout.deep_grpo.get("default_agent_name", "reasoning_agent_loop")
            batch.non_tensor_batch["agent_name"] = np.array([default_agent] * len(batch), dtype=object)
                
        tasks = []
        agent_names = batch.non_tensor_batch["agent_name"]
        raw_prompts = batch.non_tensor_batch["raw_prompt"]
        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(len(raw_prompts))

        trajectory_info = await get_trajectory_info(batch.meta_info.get("global_steps", -1), index)

        for i in range(len(raw_prompts)):
            agent_name = agent_names[i]
            messages = raw_prompts[i]
            trajectory = trajectory_info[i]
            data_instance = batch[i].non_tensor_batch
            tasks.append(
                asyncio.create_task(
                    self._run_agent_loop(mode, agent_name, messages.tolist(), sampling_params, trajectory, data_instance)
                )
            )

        outputs = await asyncio.gather(*tasks)

        outputs = list(flatten(outputs))

        output = self._postprocess(outputs)
        return output
    
    async def ts_generate_sequences(self, batch: DataProto) -> Tuple[DataProto, DataProto]:
        """Generate sequences from agent loop.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
                from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
                and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=1.0,
        )

        # override sampling params for validation
        validate = batch.meta_info.get("validate", False)
        mode = "validate" if validate else "train"
        if validate:
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["temperature"] = config.val_kwargs.temperature

        if "agent_name" not in batch.non_tensor_batch:
            default_agent = self.config.actor_rollout_ref.rollout.deep_grpo.get("default_agent_name", "reasoning_agent_loop")
            batch.non_tensor_batch["agent_name"] = np.array([default_agent] * len(batch), dtype=object)
                
        tasks = []
        agent_names = batch.non_tensor_batch["agent_name"]
        raw_prompts = batch.non_tensor_batch["raw_prompt"]
        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(len(raw_prompts))

        trajectory_info = await get_trajectory_info(batch.meta_info.get("global_steps", -1), index)

        # breakpoint()

        for i in range(len(raw_prompts)):
            agent_name = agent_names[i]
            messages = raw_prompts[i]
            trajectory = trajectory_info[i]
            data_instance = batch[i].non_tensor_batch
            tasks.append(
                asyncio.create_task(
                    self._run_agent_loop(mode, agent_name, messages.tolist(), sampling_params, trajectory, data_instance)
                )
            )
        outputs = await asyncio.gather(*tasks)
        main_chain_outputs, branch_chain_outputs = zip(*outputs)

        main_chain_outputs = list(flatten(main_chain_outputs))
        branch_chain_outputs = list(flatten(branch_chain_outputs))

        main_chain_output = self._postprocess(main_chain_outputs)
        branch_chain_output = self._postprocess(branch_chain_outputs) if branch_chain_outputs else DataProto()

        return main_chain_output, branch_chain_output

    async def ts_generate_one_stage(
        self,
        batch: DataProto,
        buffer_entries: List,
        branches_per_entry: int,
        synthetic_entries: Optional[List[Tuple[str, Any]]] = None,
    ) -> Tuple[DataProto, DataProto, DataProto, List, Dict]:
        """One-stage generation: main chains + buffer branches in parallel.

        Args:
            batch: Input batch with prompts for main chain generation.
            buffer_entries: List of BranchPointEntry objects from buffer.
            branches_per_entry: How many continuations per buffer entry.
            synthetic_entries: Optional list of (tree_id, SyntheticPromptEntry)
                tuples for prefix_inject_mode. Each gets K rollouts from its
                augmented_prompt_ids; outputs are merged into main_chain_output
                and failed trajectories feed back through all_failed_trajectories.

        Returns:
            main_chain_output: DataProto of main chain training data.
            branch_chain_output: DataProto of model branch training data.
            teacher_chain_output: DataProto of teacher suffix training data.
            all_failed_trajectories: List of failed trajectories for teacher annotation.
            one_stage_timing: Dict of wall-clock timing for main/buffer generation.
        """
        config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=1.0,
        )

        if "agent_name" not in batch.non_tensor_batch:
            default_agent = self.config.actor_rollout_ref.rollout.deep_grpo.get(
                "default_agent_name", "reasoning_agent_loop"
            )
            batch.non_tensor_batch["agent_name"] = np.array(
                [default_agent] * len(batch), dtype=object
            )

        # ---- Task Group 1: Main chain generation (one-stage mode) ----
        agent_names = batch.non_tensor_batch["agent_name"]
        raw_prompts = batch.non_tensor_batch["raw_prompt"]
        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(len(raw_prompts))

        trajectory_info = await get_trajectory_info(
            batch.meta_info.get("global_steps", -1), index
        )

        main_tasks = []
        for i in range(len(raw_prompts)):
            agent_name = agent_names[i]
            messages = raw_prompts[i]
            trajectory = trajectory_info[i]
            data_instance = batch[i].non_tensor_batch
            main_tasks.append(
                asyncio.create_task(
                    self._run_agent_loop(
                        "train_one_stage", agent_name, messages.tolist(),
                        sampling_params, trajectory, data_instance
                    )
                )
            )

        # ---- Task Group 2: Buffer branch generation ----
        buffer_tasks = []
        for entry in buffer_entries:
            buffer_tasks.append(
                asyncio.create_task(
                    self._run_buffer_branch(entry, sampling_params, branches_per_entry)
                )
            )

        # ---- Task Group 3: Synthetic prompt rollouts (prefix_inject_mode) ----
        # Each synthetic entry goes through _run_synthetic_entry which does K
        # rollouts from its augmented_prompt_ids. Outputs are format-compatible
        # with main chain outputs (same TSTrainAgentLoopOutput shape, same
        # failed-trajectory format), so they get merged into all_main_chain_outputs
        # and all_failed_trajectories below without any special handling.
        synthetic_tasks = []
        if synthetic_entries:
            for tree_id, entry in synthetic_entries:
                synthetic_tasks.append(
                    asyncio.create_task(
                        self._run_synthetic_rollout(entry, tree_id, sampling_params)
                    )
                )

        # Run all three groups concurrently. _timed_gather handles empty task
        # lists correctly (returns ([], ~0 elapsed)), so we don't need to
        # branch on presence/absence.
        import time

        async def _timed_gather(tasks):
            t0 = time.monotonic()
            results = await asyncio.gather(*tasks)
            return results, time.monotonic() - t0

        (main_results, t_main), (buffer_results, t_buffer), (synth_results, t_synth) = (
            await asyncio.gather(
                _timed_gather(main_tasks),
                _timed_gather(buffer_tasks),
                _timed_gather(synthetic_tasks),
            )
        )

        # ---- Process main chain results ----
        # main_results[i] corresponds to agent_names[i] (asyncio.gather preserves order)
        all_main_chain_outputs = []
        all_failed_trajectories = []
        for i, (main_chain_outputs, failed_trajectories) in enumerate(main_results):
            all_main_chain_outputs.extend(main_chain_outputs)
            for ft in failed_trajectories:
                ft.agent_name = str(agent_names[i])
                if ft.branch_points:
                    for bp in ft.branch_points:
                        bp.agent_name = str(agent_names[i])
            all_failed_trajectories.extend(failed_trajectories)

        # ---- Process buffer results (three streams) ----
        all_branch_chain_outputs = []
        all_teacher_chain_outputs = []
        for branch_outputs, teacher_outputs in buffer_results:
            all_branch_chain_outputs.extend(branch_outputs)
            all_teacher_chain_outputs.extend(teacher_outputs)

        # ---- Process synthetic results (merge into main stream) ----
        # Synthetic rollouts are "just more main chain samples" as far as the
        # training pipeline is concerned -- same tree_id grouping, same reward,
        # same advantage computation, same loss. Tag agent_name on any failed
        # trajectories so downstream code can attribute them.
        if synthetic_entries:
            synth_default_agent = self.config.actor_rollout_ref.rollout.deep_grpo.get(
                "default_agent_name", "reasoning_agent_loop"
            )
            for (main_chain_outputs, failed_trajectories), (_, entry) in zip(
                synth_results, synthetic_entries
            ):
                all_main_chain_outputs.extend(main_chain_outputs)
                agent_name = entry.agent_name or synth_default_agent
                for ft in failed_trajectories:
                    ft.agent_name = str(agent_name)
                all_failed_trajectories.extend(failed_trajectories)

        # ---- Postprocess to DataProto ----
        main_chain_output = self._postprocess(all_main_chain_outputs)
        branch_chain_output = self._postprocess(all_branch_chain_outputs) if all_branch_chain_outputs else DataProto()
        teacher_chain_output = self._postprocess(all_teacher_chain_outputs) if all_teacher_chain_outputs else DataProto()

        one_stage_timing = {
            "one_stage/main_chain_wall": t_main,
            "one_stage/buffer_branch_wall": t_buffer,
            "one_stage/buffer_branch_count": len(buffer_entries),
        }
        # Only add synthetic timing keys when prefix_inject_mode is active,
        # so legacy runs have exactly the same metric schema as pre-patch.
        if synthetic_entries:
            one_stage_timing["one_stage/synthetic_wall"] = t_synth
            one_stage_timing["one_stage/synthetic_count"] = len(synthetic_entries)

        return main_chain_output, branch_chain_output, teacher_chain_output, all_failed_trajectories, one_stage_timing

    async def recoverability_eval_sequences(self, batch: DataProto) -> List[Dict[str, Any]]:
        """Run recoverability evaluation on a batch.

        For each prompt, generates a main chain; if it fails, branches at every
        position and records per-position recoverability.

        Returns a flat list of result dicts (one per evaluated position).
        """
        # Use TRAINING temperature/top_p, not validation (which is typically 0).
        # Recoverability eval needs diverse branches via sampling.
        config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=1.0,
        )

        if "agent_name" not in batch.non_tensor_batch:
            default_agent = self.config.actor_rollout_ref.rollout.deep_grpo.get("default_agent_name", "reasoning_agent_loop")
            batch.non_tensor_batch["agent_name"] = np.array([default_agent] * len(batch), dtype=object)

        agent_names = batch.non_tensor_batch["agent_name"]
        raw_prompts = batch.non_tensor_batch["raw_prompt"]
        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(len(raw_prompts))

        trajectory_info = await get_trajectory_info(batch.meta_info.get("global_steps", -1), index)

        tasks = []
        for i in range(len(raw_prompts)):
            agent_name = agent_names[i]
            messages = raw_prompts[i]
            trajectory = trajectory_info[i]
            data_instance = batch[i].non_tensor_batch
            tasks.append(
                asyncio.create_task(
                    self._run_agent_loop("recoverability_eval", agent_name, messages.tolist(), sampling_params, trajectory, data_instance)
                )
            )

        all_results = await asyncio.gather(*tasks)

        # Flatten
        records = []
        for result_list in all_results:
            if isinstance(result_list, list):
                records.extend(result_list)
        return records

    async def _run_buffer_branch(
        self,
        entry,
        sampling_params: Dict[str, Any],
        branches_per_entry: int,
    ) -> Tuple[List, List]:
        """Generate branches from a single buffer entry.

        Dispatches to run_from_teacher_entry if the entry has a teacher suffix,
        otherwise uses the regular run_from_buffer_entry.
        """
        agent_name = entry.agent_name or self.config.actor_rollout_ref.rollout.deep_grpo.get(
            "default_agent_name", "reasoning_agent_loop"
        )
        agent_loop_class = self.get_agent_loop_class(agent_name)
        agent_loop = agent_loop_class(
            self.config, self.server_manager, self.tokenizer
        )
        if getattr(entry, 'teacher_suffix', None) is not None:
            return await agent_loop.run_from_teacher_entry(
                entry, sampling_params, branches_per_entry
            )
        return await agent_loop.run_from_buffer_entry(
            entry, sampling_params, branches_per_entry
        )

    async def _run_synthetic_rollout(
        self,
        entry,
        tree_id: str,
        sampling_params: Dict[str, Any],
    ) -> Tuple[List, List]:
        """Run K rollouts from a SyntheticPromptEntry's augmented prompt.

        Dispatches to the specific agent loop class's _run_synthetic_entry
        which takes the pre-tokenized augmented_prompt_ids directly (skipping
        apply_chat_template).

        Returns (main_chain_outputs, failed_trajectories) in the same format
        as _run_train_one_stage, so outputs can be merged into
        all_main_chain_outputs and failed_trajectories into all_failed_trajectories.
        """
        agent_name = entry.agent_name or self.config.actor_rollout_ref.rollout.deep_grpo.get(
            "default_agent_name", "reasoning_agent_loop"
        )
        agent_loop_class = self.get_agent_loop_class(agent_name)
        agent_loop = agent_loop_class(
            self.config, self.server_manager, self.tokenizer
        )
        return await agent_loop._run_synthetic_entry(
            entry, tree_id, sampling_params
        )

    async def _run_agent_loop(
        self,
        mode: str,
        agent_name: str,
        messages: List[Dict[str, Any]],
        sampling_params: Dict[str, Any],
        trajectory: Dict[str, Any],
        data_instance: Dict[str, Any],
    ) -> Union[AgentLoopOutput, Tuple[List[AgentLoopOutput], List[AgentLoopOutput]]]:
        with rollout_trace_attr(
            step=trajectory["step"], sample_index=trajectory["sample_index"], rollout_n=trajectory["rollout_n"]
        ):
            agent_loop_class = self.get_agent_loop_class(agent_name)
            agent_loop = agent_loop_class(self.config, self.server_manager, self.tokenizer)
            output = await agent_loop.run(mode, messages, sampling_params, data_instance)
            return output

    def get_agent_loop_class(self, agent_name: str) -> Type[AgentLoopBase]:
        """Get the appropriate agent loop class based on agent name.

        Factory method that returns the correct agent loop class implementation
        for the specified agent type.

        Args:
            agent_name (str): Name of the agent type ('single_turn_agent' or 'tool_agent').

        Returns:
            Type[AgentLoopBase]: Agent loop class corresponding to the agent name.

        Raises:
            ValueError: If the agent_name is not recognized.
        """
        # TODO: add tool agent registrary
        from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
        from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
        from recipe.deep_grpo.agent_loop.deep_analyze_agent_loop import DeepAnalyzeAgentLoop
        from recipe.deep_grpo.agent_loop.search_agent_loop import SearchAgentLoop

        if agent_name == "single_turn_agent":
            return SingleTurnAgentLoop
        elif agent_name == "tool_agent":
            return ToolAgentLoop
        elif agent_name == "deep_analyze_agent_loop":
            return DeepAnalyzeAgentLoop
        elif agent_name == "search_agent":
            return SearchAgentLoop
        elif agent_name == "reasoning_agent_loop":
            from recipe.deep_grpo.agent_loop.reasoning_agent_loop import ReasoningAgentLoop
            return ReasoningAgentLoop
        elif agent_name == "treerl_agent_loop":
            from recipe.deep_grpo.agent_loop.treerl_agent_loop import TreeRLAgentLoop
            return TreeRLAgentLoop
        raise ValueError(f"Unknown agent_name: {agent_name}")

    def _postprocess(self, inputs: List[AgentLoopOutput]) -> DataProto:
        # NOTE: consistent with batch version of generate_sequences in vllm_rollout_spmd.py
        # prompts: left pad
        # responses: right pad
        # input_ids: prompt + response
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]

        # prompts
        self.tokenizer.padding_side = "left"
        outputs = self.tokenizer.pad(
            [{"input_ids": input.prompt_ids} for input in inputs],
            padding="max_length",
            max_length=self.config.actor_rollout_ref.rollout.max_model_len, 
            # padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        )
        prompt_ids, prompt_attention_mask = outputs["input_ids"], outputs["attention_mask"]

        # responses
        self.tokenizer.padding_side = "right"
        outputs = self.tokenizer.pad(
            [{"input_ids": input.response_ids} for input in inputs],
            padding="max_length",
            max_length=self.config.actor_rollout_ref.rollout.max_model_len,
            # padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        )
        response_ids, response_attention_mask = outputs["input_ids"], outputs["attention_mask"]

        # response_mask
        outputs = self.tokenizer.pad(
            [{"input_ids": input.response_mask} for input in inputs],
            padding="max_length",
            max_length=self.config.actor_rollout_ref.rollout.max_model_len,
            # padding="longest",
            return_tensors="pt",
            return_attention_mask=False,
        )
        response_mask = outputs["input_ids"]
        assert response_ids.shape == response_mask.shape, (
            f"mismatch in response_ids and response_mask shape: {response_ids.shape} vs {response_mask.shape}"
        )
        response_mask = response_mask * response_attention_mask

        input_ids = torch.cat([prompt_ids, response_ids], dim=1)
        attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=1)
        position_ids = (attention_mask.cumsum(dim=1) - 1) * attention_mask

        batch = TensorDict(
            {
                "prompts": prompt_ids,  # [bsz, prompt_length]
                "responses": response_ids,  # [bsz, response_length]
                "response_mask": response_mask,  # [bsz, response_length]
                "input_ids": input_ids,  # [bsz, prompt_length + response_length]
                "attention_mask": attention_mask,  # [bsz, prompt_length + response_length]
                "position_ids": position_ids,  # [bsz, prompt_length + response_length]
            },
            batch_size=len(input_ids),
        )

        # score_tensor and reward_tensor
        if hasattr(inputs[0], "reward"):
            reward_tensor = torch.zeros_like(response_ids, dtype=torch.float32)
            prompt_length = prompt_ids.shape[-1]

            for i in range(len(inputs)):
                valid_response_length = attention_mask[i, prompt_length:].sum()
                reward_tensor[i, valid_response_length - 1] = inputs[i].reward
            
            batch["token_level_scores"] = reward_tensor
            batch["token_level_rewards"] = reward_tensor

        if hasattr(inputs[0], "token_level_advantages") and inputs[0].token_level_advantages is not None:
            # TreeRL: per-token process rewards
            advantage_tensor = torch.zeros_like(response_ids, dtype=torch.float32)
            for i, inp in enumerate(inputs):
                advs = inp.token_level_advantages
                valid_len = min(len(advs), int(response_attention_mask[i].sum().item()))
                advantage_tensor[i, :valid_len] = torch.tensor(advs[:valid_len], dtype=torch.float32)
            advantage_tensor = advantage_tensor * response_mask
            batch["advantages"] = advantage_tensor
            batch["returns"] = advantage_tensor
        elif hasattr(inputs[0], "advantage"):
            advantage_tensor = torch.tensor([i.advantage for i in inputs])
            advantage_tensor = advantage_tensor.unsqueeze(-1) * response_mask

            batch["advantages"] = advantage_tensor
            batch["returns"] = advantage_tensor

        num_turns = np.array([input.num_turns for input in inputs], dtype=np.int32)
        non_tensor_batch = {"__num_turns__": num_turns}

        if hasattr(inputs[0], "tree_id"):
            tree_ids = np.array([i.tree_id for i in inputs])
            non_tensor_batch["__tree_ids__"] = tree_ids

        if hasattr(inputs[0], "node_id"):
            node_ids = np.array([i.node_id for i in inputs])
            non_tensor_batch["__node_ids__"] = node_ids

        if hasattr(inputs[0], "parent_node_id"):
            parent_node_ids = np.array([i.parent_node_id for i in inputs])
            non_tensor_batch["__parent_node_ids__"] = parent_node_ids

        if hasattr(inputs[0], "reward_info"):
            reward_infos = np.array([i.reward_info for i in inputs])
            non_tensor_batch["__reward_infos__"] = reward_infos
        
        if hasattr(inputs[0], "is_main_chain_end"):
            is_main_chain_ends = np.array([i.is_main_chain_end for i in inputs])
            non_tensor_batch["__is_main_chain_ends__"] = is_main_chain_ends

        if hasattr(inputs[0], "is_branch_chain_end"):
            is_branch_chain_ends = np.array([i.is_branch_chain_end for i in inputs])
            non_tensor_batch["__is_branch_chain_ends__"] = is_branch_chain_ends

        metrics = [input.metrics.model_dump() for input in inputs]
        return DataProto(batch=batch,
                         non_tensor_batch=non_tensor_batch, 
                         meta_info={"metrics": metrics})


async def get_trajectory_info(step, index):
    """Get the trajectory info (step, sample_index, rollout_n) asynchrously"""
    trajectory_info = []
    rollout_n = 0
    for i in range(len(index)):
        if i > 0 and index[i - 1] == index[i]:
            rollout_n += 1
        else:
            rollout_n = 0
        trajectory_info.append({"step": step, "sample_index": index[i], "rollout_n": rollout_n})
    return trajectory_info


class AgentLoopManager:
    """Agent loop manager that manages a group of agent loop workers."""

    def __init__(self, config: DictConfig, worker_group: RayWorkerGroup):
        """Initialize agent loop manager.

        Args:
            config (DictConfig): trainer config.
            worker_group (RayWorkerGroup): ActorRolloutRef worker group.
        """
        self.config = config
        self.worker_group = worker_group

        self._initialize_llm_servers()
        self._init_agent_loop_workers()

        # Initially we're in sleep mode.
        self.sleep()

    def _initialize_llm_servers(self):
        self.rollout_tp_size = self.config.actor_rollout_ref.rollout.tensor_model_parallel_size
        self.rollout_dp_size = self.worker_group.world_size // self.rollout_tp_size

        register_center = ray.get_actor(f"{self.worker_group.name_prefix}_register_center")
        workers_info = ray.get(register_center.get_worker_info.remote())
        assert len(workers_info) == self.worker_group.world_size

        self.async_llm_servers = [None] * self.rollout_dp_size
        self.server_addresses = [None] * self.rollout_dp_size

        if self.config.actor_rollout_ref.rollout.agent.custom_async_server:
            server_class = async_server_class(
                rollout_backend=self.config.actor_rollout_ref.rollout.name,
                rollout_backend_module=self.config.actor_rollout_ref.rollout.agent.custom_async_server.path,
                rollout_backend_class=self.config.actor_rollout_ref.rollout.agent.custom_async_server.name,
            )
        else:
            server_class = async_server_class(rollout_backend=self.config.actor_rollout_ref.rollout.name)

        # Start all server instances, restart if address already in use.
        unready_dp_ranks = set(range(self.rollout_dp_size))
        while len(unready_dp_ranks) > 0:
            servers = {
                rollout_dp_rank: server_class.options(
                    # make sure AsyncvLLMServer colocates with its corresponding workers
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=workers_info[rollout_dp_rank * self.rollout_tp_size],
                        soft=False,
                    ),
                    name=f"async_llm_server_{rollout_dp_rank}",
                ).remote(self.config, self.rollout_dp_size, rollout_dp_rank, self.worker_group.name_prefix)
                for rollout_dp_rank in unready_dp_ranks
            }

            for rollout_dp_rank, server in servers.items():
                try:
                    address = ray.get(server.get_server_address.remote())
                    self.server_addresses[rollout_dp_rank] = address
                    self.async_llm_servers[rollout_dp_rank] = server
                    unready_dp_ranks.remove(rollout_dp_rank)
                except Exception:
                    ray.kill(server)
                    print(f"rollout server {rollout_dp_rank} failed, maybe address already in use, restarting...")

        # All server instances are ready, init AsyncLLM engine.
        ray.get([server.init_engine.remote() for server in self.async_llm_servers])

    def _init_agent_loop_workers(self):
        assert self.config.actor_rollout_ref.rollout.agent.num_workers == 1, "Now only one agent loop worker is supported"
        self.agent_loop_workers = []
        for i in range(self.config.actor_rollout_ref.rollout.agent.num_workers):
            self.agent_loop_workers.append(
                AgentLoopWorker.options(
                    name=f"agent_loop_worker_{i}",
                ).remote(self.config, self.async_llm_servers)
            )
    
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Split input batch and dispatch to agent loop workers.

        Args:
            prompts (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
        """
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.wake_up()
        chunkes = prompts.chunk(len(self.agent_loop_workers))
        outputs = ray.get(
            [worker.generate_sequences.remote(chunk) for worker, chunk in zip(self.agent_loop_workers, chunkes)]
        )
        output = DataProto.concat(outputs)
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.sleep()

        # calculate performance metrics
        metrics = [output.meta_info["metrics"] for output in outputs]  # List[List[Dict[str, str]]]
        timing = self._performance_metrics(metrics, output)

        output.meta_info = {"timing": timing}
        return output

    def ts_generate_sequences(self, prompts: DataProto) -> Tuple[DataProto, DataProto]:
        """Split input batch and dispatch to agent loop workers.

        Args:
            prompts (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
        """
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.wake_up()
        chunkes = prompts.chunk(len(self.agent_loop_workers))
        outputs = ray.get(
            [worker.ts_generate_sequences.remote(chunk) for worker, chunk in zip(self.agent_loop_workers, chunkes)]
        )

        # Currently only 1 worker supported (assert in _init_agent_loop_workers)
        main_chain_output, branch_chain_output = outputs[0]

        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.sleep()

        # calculate performance metrics
        main_chain_metrics = main_chain_output.meta_info.get("metrics", [])
        timing = {}
        if main_chain_metrics:
            timing.update(self._performance_metrics([main_chain_metrics], main_chain_output))
        main_chain_output.meta_info = {"timing": timing}

        return main_chain_output, branch_chain_output
    
    def ts_generate_one_stage(
        self,
        prompts: DataProto,
        buffer_entries: List,
        branches_per_entry: int,
        synthetic_entries: Optional[List[Tuple[str, Any]]] = None,
    ) -> Tuple[DataProto, DataProto, DataProto, List]:
        """One-stage dispatch: main chains + buffer branches + teacher.

        Args:
            prompts: Batch of prompts for main chain generation.
            buffer_entries: List of BranchPointEntry objects from buffer.
            branches_per_entry: Continuations per buffer entry.
            synthetic_entries: Optional list of (tree_id, SyntheticPromptEntry)
                tuples for prefix_inject_mode. Distributed across workers.

        Returns:
            main_chain_output, branch_chain_output, teacher_chain_output, failed_trajectories
        """
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.wake_up()

        # Currently only 1 worker supported (assert in _init_agent_loop_workers)
        chunks = prompts.chunk(len(self.agent_loop_workers))

        # Split buffer entries across workers proportionally
        n_workers = len(self.agent_loop_workers)
        buffer_chunks = [
            buffer_entries[i::n_workers] for i in range(n_workers)
        ]

        # Split synthetic entries across workers proportionally
        if synthetic_entries:
            synthetic_chunks = [
                synthetic_entries[i::n_workers] for i in range(n_workers)
            ]
        else:
            synthetic_chunks = [None] * n_workers

        outputs = ray.get([
            worker.ts_generate_one_stage.remote(
                chunk, buf_chunk, branches_per_entry, synth_chunk
            )
            for worker, chunk, buf_chunk, synth_chunk in zip(
                self.agent_loop_workers, chunks, buffer_chunks, synthetic_chunks
            )
        ])

        # Currently only 1 worker (asserted in _init_agent_loop_workers)
        main_chain_output, branch_chain_output, teacher_chain_output, failed_trajectories, one_stage_timing = outputs[0]

        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.sleep()

        # Performance metrics for main chains
        main_chain_metrics = main_chain_output.meta_info.get("metrics", [])
        timing = {}
        if main_chain_metrics:
            timing.update(self._performance_metrics([main_chain_metrics], main_chain_output))

        timing.update(one_stage_timing)
        main_chain_output.meta_info = {"timing": timing}

        return main_chain_output, branch_chain_output, teacher_chain_output, failed_trajectories

    def recoverability_eval_sequences(self, prompts: DataProto) -> List[Dict[str, Any]]:
        """Run recoverability evaluation and return results."""
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.wake_up()
        chunks = prompts.chunk(len(self.agent_loop_workers))
        worker_results = ray.get([
            worker.recoverability_eval_sequences.remote(chunk)
            for worker, chunk in zip(self.agent_loop_workers, chunks)
        ])
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.sleep()
        # Flatten results from all workers
        records = []
        for result_list in worker_results:
            records.extend(result_list)
        return records

    def _performance_metrics(self, metrics: List[List[Dict[str, str]]], output: DataProto) -> Dict[str, float]:
        timing = {}
        all_metrics = [metric for chunk in metrics for metric in chunk]
        assert len(all_metrics) > 0
        assert len(all_metrics) == len(output)

        t_generate_main_chain = np.array([float(metric["generate_main_chain"]) for metric in all_metrics])
        t_generate_branch_chain = np.array([float(metric["generate_branch_chain"]) for metric in all_metrics])

        timing["agent_loop/generate_main_chain/min"] = t_generate_main_chain.min()
        timing["agent_loop/generate_main_chain/max"] = t_generate_main_chain.max()
        timing["agent_loop/generate_main_chain/mean"] = t_generate_main_chain.mean()

        timing["agent_loop/generate_branch_chain/min"] = t_generate_branch_chain.min()
        timing["agent_loop/generate_branch_chain/max"] = t_generate_branch_chain.max()
        timing["agent_loop/generate_branch_chain/mean"] = t_generate_branch_chain.mean()

        prompt_length = output.batch["prompts"].shape[1]

        generate_main_chain_slowest = np.argmax(t_generate_main_chain)
        attention_mask = output.batch["attention_mask"][generate_main_chain_slowest]
        timing["agent_loop/slowest/generate_main_chain"] = t_generate_main_chain[generate_main_chain_slowest]
        timing["agent_loop/slowest/generate_main_chain/prompt_length"] = attention_mask[:prompt_length].sum().item()
        timing["agent_loop/slowest/generate_main_chain/response_length"] = attention_mask[prompt_length:].sum().item()

        generate_branch_chain_slowest = np.argmax(t_generate_branch_chain)
        attention_mask = output.batch["attention_mask"][generate_branch_chain_slowest]
        timing["agent_loop/slowest/generate_branch_chain"] = t_generate_branch_chain[generate_branch_chain_slowest]
        timing["agent_loop/slowest/generate_branch_chain/prompt_length"] = attention_mask[:prompt_length].sum().item()
        timing["agent_loop/slowest/generate_branch_chain/response_length"] = attention_mask[prompt_length:].sum().item()

        return timing

    def wake_up(self):
        """Wake up all rollout server instances."""
        ray.get([server.wake_up.remote() for server in self.async_llm_servers])

    def sleep(self):
        """Sleep all rollout server instances."""
        ray.get([server.sleep.remote() for server in self.async_llm_servers])
