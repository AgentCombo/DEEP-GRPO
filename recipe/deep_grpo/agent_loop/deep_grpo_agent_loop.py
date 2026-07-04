"""DEEP-GRPO agent-loop extension.

DeepGRPOAgentLoop extends the generic tree-search core (TSAgentLoop) with the
rollout logic of every DEEP-GRPO variant, without modifying the base class:

- Branch expansion (earliest form): pick branch points on failed chains
  (random / utility / teacher selection) and generate alternative
  continuations, either inline (`train` mode) or decoupled through the
  branch point buffer (`train_one_stage` mode).
- Teacher suffix synthesis: analyze failed trajectories with a teacher
  model, locate the first error via longest-common-prefix matching, and
  produce reward-verified teacher suffixes (`synthesize_teacher_suffix`,
  called by the background TeacherAnnotationWorker).
- Prefix injection (current form): roll out from augmented prompts
  (= original prompt + verified prefix) injected by the trainer
  (`_run_synthetic_entry`), plus teacher-continuation rows
  (`run_from_teacher_entry`).

Concrete loops (ReasoningAgentLoop, TreeRLAgentLoop) subclass this class
and provide `_step` / `_rollout`.
"""

from typing import Any, Dict, List, Union, Tuple, Optional

import asyncio
import logging
import os
from uuid import uuid4
import re
import json

import numpy as np

from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

from recipe.deep_grpo.protocol import Node, FINISH_REASON
from recipe.deep_grpo.agent_loop.outputs import TSTrainAgentLoopOutput, TSValAgentLoopOutput
from recipe.deep_grpo.agent_loop.tree_search_agent_loop import TSAgentLoop
from recipe.deep_grpo.utils import call_teacher_with_retry
from recipe.deep_grpo.protocol import BranchPointEntry, FailedTrajectoryEntry, TeacherSuffix, SyntheticPromptEntry
from recipe.deep_grpo.teacher_suffix_utils import append_eos_if_missing
from recipe.deep_grpo.prompts import (
    TEACHER_SELECTION_PROMPT_TEMPLATE,
    TEACHER_SUFFIX_SYNTHESIS_PROMPT_TEMPLATE,
    TEACHER_SUFFIX_SYNTHESIS_PROMPT_TEMPLATE_V2,
)
from recipe.deep_grpo.branching_strategy import RandomBranchingStrategy, UtilitySamplingStrategy, BranchingSelection, BranchingFeedback
from recipe.deep_grpo.partition_strategy import PartitionStrategy

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _parse_teacher_selection_reply(reply: str) -> int:
    matches = re.findall(r"```(?:\w+)?\s*(\{.*?\})\s*```", reply, re.DOTALL)
    if not matches:
        matches = re.findall(r"```(?:\w+)?\s*(.*?)```", reply, re.DOTALL)
    if not matches:
        matches = [reply]
    for json_str in reversed(matches):
        try:
            cleaned_str = json_str.strip()
            data = json.loads(cleaned_str)
            required_key = "first_error_step_index"
            if required_key in data:
                return int(data[required_key])

        except (json.JSONDecodeError, ValueError, TypeError):
            continue

    raise ValueError(f"Failed to parse teacher selection JSON from reply. Reply:\n{reply}")


def _parse_teacher_synthesis_reply(reply: str) -> str:
    """Parse teacher synthesis reply, extracting the correct_solution section.

    Expects the reply to contain a [CORRECT_SOLUTION] section header followed by
    the solution text. No JSON parsing needed — avoids LaTeX/JSON escape conflicts.
    """
    marker = "[CORRECT_SOLUTION]"
    idx = reply.find(marker)
    if idx == -1:
        raise ValueError(f"Missing {marker} section in reply. Reply:\n{reply}")

    solution = reply[idx + len(marker):].strip()
    if not solution:
        raise ValueError(f"Empty solution after {marker}. Reply:\n{reply}")

    return solution


class DeepGRPOAgentLoop(TSAgentLoop):

    @classmethod
    def init_class(cls, config, tokenizer):
        if cls._class_initialized:
            return

        super().init_class(config, tokenizer)

        prefix_inject_cfg = (
            config.actor_rollout_ref.rollout.deep_grpo.get("prefix_inject_mode", {})
            or {}
        )
        prefix_rollout_n = prefix_inject_cfg.get("rollout_n", None)
        cls.prefix_rollout_n = (
            cls.rollout_n if prefix_rollout_n is None else int(prefix_rollout_n)
        )
        assert cls.prefix_rollout_n > 1

        cls.expand_branch_chain = config.actor_rollout_ref.rollout.deep_grpo.expand_branch_chain
        cls.expand_only_on_low_quality = config.actor_rollout_ref.rollout.deep_grpo.expand_only_on_low_quality
        cls.low_quality_trajectory_reward_threshold = config.actor_rollout_ref.rollout.deep_grpo.low_quality_trajectory_reward_threshold
        cls.branches_per_point = config.actor_rollout_ref.rollout.deep_grpo.branches_per_point
        cls.pick_branch_chain_root_method = config.actor_rollout_ref.rollout.deep_grpo.pick_branch_chain_root_method
        cls.n_branch_points = config.actor_rollout_ref.rollout.deep_grpo.n_branch_points
        assert cls.pick_branch_chain_root_method in ["random", "utility"], f"Unknown pick_branch_chain_root_method: {cls.pick_branch_chain_root_method}"
        if cls.pick_branch_chain_root_method == "random":
            cls.branching_strategy = RandomBranchingStrategy(max_model_len=cls.max_model_len)
        elif cls.pick_branch_chain_root_method == "utility":
            cls.branching_strategy = UtilitySamplingStrategy(max_model_len=cls.max_model_len,
                                                             prob_model_type=config.actor_rollout_ref.rollout.deep_grpo.utility_sampling.prob_model_type,
                                                             window_size=config.actor_rollout_ref.rollout.deep_grpo.utility_sampling.window_size,
                                                             position_bias=config.actor_rollout_ref.rollout.deep_grpo.utility_sampling.position_bias)

    def _partition_response_to_chain(self, full_node: Node, partition_strategy: PartitionStrategy) -> List[Node]:
        """Split a full-response Node into a chain of Nodes for branch point selection.

        Each intermediate node gets finish_reason=STOP (expandable for branching).
        The last node inherits the original finish_reason, reward, and reward_info.
        """
        n = len(full_node.response_ids)
        assert n == len(full_node.response_mask), (
            f"response_ids length {n} != response_mask length {len(full_node.response_mask)}"
        )

        segments = partition_strategy.partition(full_node.response_ids, full_node.response_mask)

        if len(segments) <= 1:
            return [full_node]

        # Validate segment boundaries
        for i, (start, end) in enumerate(segments):
            assert 0 <= start <= end <= n, (
                f"Segment {i} out of bounds: ({start}, {end}), response length {n}"
            )
            if i > 0:
                assert start == segments[i - 1][1], (
                    f"Segment {i} not contiguous: starts at {start}, previous ends at {segments[i - 1][1]}"
                )
        assert segments[0][0] == 0, f"First segment doesn't start at 0: {segments[0][0]}"
        assert segments[-1][1] == n, f"Last segment doesn't end at {n}: {segments[-1][1]}"

        chain = []
        cumulative_prompt = list(full_node.prompt_ids)

        for seg_idx, (start, end) in enumerate(segments):
            seg_response_ids = full_node.response_ids[start:end]
            seg_response_mask = full_node.response_mask[start:end]
            is_last = (seg_idx == len(segments) - 1)

            node = Node(
                node_id=uuid4().hex,
                prompt_ids=list(cumulative_prompt),
                response_ids=seg_response_ids,
                response_mask=seg_response_mask,
                data_instance=full_node.data_instance,
                finish_reason=full_node.finish_reason if is_last else FINISH_REASON.STOP,
                num_turns=full_node.num_turns + seg_idx,
                reward=full_node.reward if is_last else None,
                reward_info=full_node.reward_info if is_last else None,
            )
            chain.append(node)
            cumulative_prompt = cumulative_prompt + seg_response_ids

        return chain

    async def _try_teacher_selection(self, chain: List[Node]) -> Optional[Node]:
        prompt_ids = chain[0].prompt_ids
        all_response_ids = [node.response_ids for node in chain]

        instruction, segments = await asyncio.gather(
            self.loop.run_in_executor(None, lambda: self.tokenizer.decode(prompt_ids, skip_special_tokens=True)),
            self.loop.run_in_executor(None, lambda: self.tokenizer.batch_decode(all_response_ids, skip_special_tokens=True))
        )

        steps_text = "\n\n".join([f"Step {i}:\n{seg}" for i, seg in enumerate(segments)])
        prompt = TEACHER_SELECTION_PROMPT_TEMPLATE.format(
            instruction=instruction,
            reference=chain[0].data_instance["extra_info"]["answer"], # TODO: here assert reference answer is in [extra_info][answer]
            steps=steps_text
        )

        idx, reply = await call_teacher_with_retry(
            message=prompt,
            parse_fn=_parse_teacher_selection_reply,
            temperature_schedule=(0.0,), # only try once
            log_prefix="LLM Judge (teacher_selection)",
        )

        if idx is None:
            logger.warning(
                "Teacher selection did not return a index. "
                f"Raw reply: {reply}"
            )
            return None

        if idx > 0 and idx < len(chain):
            node = chain[idx - 1]
            if node.finish_reason == FINISH_REASON.STOP and \
                (len(node.prompt_ids) + len(node.response_ids) < self.max_model_len):
                return node

        return None

    async def _create_branching_selection(self, chain: List[Node], n_points: int) -> List[BranchingSelection]:
        target_nodes = self.branching_strategy.select_node(chain, n_points)
        selections = []
        for target_node in target_nodes:
            branch_chain_root_index = chain.index(target_node)

            target_node = Node(
                node_id=uuid4().hex,
                prompt_ids=target_node.prompt_ids,
                response_ids=target_node.response_ids,
                response_mask=target_node.response_mask,
                data_instance=target_node.data_instance,
                finish_reason=target_node.finish_reason,
                num_turns=target_node.num_turns
            )

            selections.append(BranchingSelection(
                branch_chain_root=target_node,
                branch_chain_root_index=branch_chain_root_index,
                total_length=len(chain)
            ))

        return selections


    async def _process_one_branch_selection(
        self,
        selection: BranchingSelection,
        sampling_params: Dict[str, Any]
    ) -> Tuple[List[Node], BranchingFeedback]:
        branch_chain_root = selection.branch_chain_root
        await self._expand(branch_chain_root, sampling_params, rollout_n=self.branches_per_point)

        branch_nodes = []
        for child in branch_chain_root.children:
            branch_nodes.append(self._compress_chain(self._get_chain_from_start_node(child)))

        any_branch_success = any(node.reward is not None and node.reward > self.low_quality_trajectory_reward_threshold for node in branch_nodes)

        feedback = BranchingFeedback(
            total_length=selection.total_length,
            branch_chain_root_index=selection.branch_chain_root_index,
            is_success=any_branch_success
        )

        return branch_nodes, feedback

    async def _process_single_chain(self,
                                    prompt_ids: List[int],
                                    sampling_params: Dict[str, Any],
                                    data_instance: Dict[str, Any]) -> Tuple[Node, List[List[Node]], Dict[str, float]]:
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

        with simple_timer("generate_branch_chain", timing_metrics):
            branch_chain_node_groups: List[List[Node]] = []

            is_low_quality = (
                main_chain[-1].reward is not None and
                main_chain[-1].reward <= self.low_quality_trajectory_reward_threshold
            )

            do_branch = (
                self.expand_branch_chain and
                (not self.expand_only_on_low_quality or is_low_quality)
            )

            if do_branch:
                branching_selections = await self._create_branching_selection(main_chain, n_points=self.n_branch_points)

                tasks = []
                for selection in branching_selections:
                    task = asyncio.create_task(
                        self._process_one_branch_selection(selection, sampling_params)
                    )
                    tasks.append(task)

                results = await asyncio.gather(*tasks)

                for branch_nodes, feedback in results:
                    branch_chain_node_groups.append(branch_nodes)
                    self.branching_strategy.update(feedback)

        return main_chain_node, branch_chain_node_groups, timing_metrics

    async def _process_single_chain_main_only(
        self,
        prompt_ids: List[int],
        sampling_params: Dict[str, Any],
        data_instance: Dict[str, Any],
    ) -> Tuple[Node, List[BranchingSelection], Dict[str, float]]:
        """Generate a single main chain and collect branch points (no branch generation).

        Returns:
            main_chain_node: Compressed main chain node.
            branch_selections: Selected branch points as BranchingSelection objects.
            timing_metrics: Timing info.
        """
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

        # Collect branch points using the existing branching strategy (Random/Utility).
        # The strategy learns which positions are promising via feedback from buffer
        # generation results. Same logic as two-stage, but we only collect — not generate.
        branch_selections = []
        if self.expand_branch_chain:
            is_low_quality = (
                main_chain[-1].reward is not None and
                main_chain[-1].reward <= self.low_quality_trajectory_reward_threshold
            )
            if not self.expand_only_on_low_quality or is_low_quality:
                branch_selections = await self._create_branching_selection(
                    main_chain, n_points=self.n_branch_points
                )

        return main_chain_node, branch_selections, timing_metrics

    @rollout_trace_op
    async def _run_train(self,
                        messages: List[Dict[str, Any]],
                        sampling_params: Dict[str, Any],
                        data_instance: Dict[str, Any]) -> Tuple[List[TSTrainAgentLoopOutput], List[TSTrainAgentLoopOutput]]:
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
                self._process_single_chain(
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                    data_instance=data_instance
                )
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        main_chain_nodes = []
        branch_chain_outputs = []
        main_chain_timing_metrics = []

        for main_chain_node, branch_chain_node_groups, timing_metrics in results:
            main_chain_nodes.append(main_chain_node)
            main_chain_timing_metrics.append(timing_metrics)

            for branch_chain_nodes in branch_chain_node_groups:
                branch_chain_outputs.extend(self._collect_train_outputs(
                    nodes=branch_chain_nodes,
                    tree_id=tree_id
                ))

        main_chain_outputs = self._collect_train_outputs(
            nodes=main_chain_nodes,
            tree_id=tree_id,
            timing_metrics=main_chain_timing_metrics
        )

        return main_chain_outputs, branch_chain_outputs

    @rollout_trace_op
    async def _run_train_one_stage(
        self,
        messages: List[Dict[str, Any]],
        sampling_params: Dict[str, Any],
        data_instance: Dict[str, Any],
    ) -> Tuple[List[TSTrainAgentLoopOutput], List[FailedTrajectoryEntry]]:
        """One-stage main chain generation.

        Returns:
            main_chain_outputs: Training outputs for main chains.
            failed_trajectories: Failed main chains with optional pre-selected branch points.
                The trainer decides how to use them (branch point buffer or teacher pool).
        """
        self._is_validation = False
        tree_id = data_instance["uid"]

        prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=getattr(self, 'add_generation_prompt', True), tokenize=True
            ),
        )

        tasks = []
        for _ in range(self.prefix_rollout_n):
            task = asyncio.create_task(
                self._process_single_chain_main_only(
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                    data_instance=data_instance
                )
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        main_chain_nodes = []
        failed_trajectories = []
        main_chain_timing_metrics = []

        for main_chain_node, branch_selections, timing_metrics in results:
            main_chain_nodes.append(main_chain_node)
            main_chain_timing_metrics.append(timing_metrics)

            is_failed = (
                main_chain_node.reward is not None and
                main_chain_node.reward <= self.low_quality_trajectory_reward_threshold
            )

            # Collect entry if: chain is failed (for teacher) OR has branch_selections (for buffer).
            # When expand_only_on_low_quality=True (default), these are the same set.
            # When expand_only_on_low_quality=False, non-failed chains may also have branch_selections.
            if is_failed or branch_selections:
                bp_list = None
                if branch_selections:
                    bp_list = []
                    for sel in branch_selections:
                        node = sel.branch_chain_root
                        bp_list.append(BranchPointEntry(
                            prompt_ids=node.prompt_ids,
                            response_ids=node.response_ids,
                            response_mask=node.response_mask,
                            data_instance=node.data_instance,
                            num_turns=node.num_turns,
                            tree_id=tree_id,
                            branch_chain_root_index=sel.branch_chain_root_index,
                            chain_total_length=sel.total_length,
                        ))

                failed_trajectories.append(FailedTrajectoryEntry(
                    prompt_ids=list(main_chain_node.prompt_ids),
                    response_ids=list(main_chain_node.response_ids),
                    response_mask=list(main_chain_node.response_mask),
                    data_instance=data_instance,
                    tree_id=tree_id,
                    num_turns=main_chain_node.num_turns,
                    branch_points=bp_list,
                ))

        main_chain_outputs = self._collect_train_outputs(
            nodes=main_chain_nodes,
            tree_id=tree_id,
            timing_metrics=main_chain_timing_metrics
        )

        return main_chain_outputs, failed_trajectories

    @rollout_trace_op
    async def _run_synthetic_entry(
        self,
        entry: SyntheticPromptEntry,
        tree_id: str,
        sampling_params: Dict[str, Any],
    ) -> Tuple[List[TSTrainAgentLoopOutput], List[FailedTrajectoryEntry]]:
        """Run K rollouts from a SyntheticPromptEntry's augmented prompt.

        Used in prefix_inject_mode. Mirrors the structure of _run_train_one_stage
        but takes pre-tokenized augmented_prompt_ids directly (skipping
        apply_chat_template, which would recover only the original prompt and
        drop the locked_prefix). The tree_id is supplied by the trainer so it
        can match back to the source SyntheticPromptEntry for record_usage.

        Outputs are format-compatible with main chain outputs and get merged
        into all_main_chain_outputs by the caller. Failed rollouts are
        collected as FailedTrajectoryEntry and feed back to failed_pool
        through the same path as main chain failures.
        """
        self._is_validation = False
        prompt_ids = list(entry.augmented_prompt_ids)
        data_instance = entry.data_instance

        tasks = []
        for _ in range(self.rollout_n):
            task = asyncio.create_task(
                self._process_single_chain_main_only(
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                    data_instance=data_instance,
                )
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        main_chain_nodes = []
        failed_trajectories = []
        main_chain_timing_metrics = []

        for main_chain_node, branch_selections, timing_metrics in results:
            main_chain_nodes.append(main_chain_node)
            main_chain_timing_metrics.append(timing_metrics)

            is_failed = (
                main_chain_node.reward is not None
                and main_chain_node.reward <= self.low_quality_trajectory_reward_threshold
            )
            if is_failed:
                failed_trajectories.append(FailedTrajectoryEntry(
                    prompt_ids=list(main_chain_node.prompt_ids),
                    response_ids=list(main_chain_node.response_ids),
                    response_mask=list(main_chain_node.response_mask),
                    data_instance=data_instance,
                    tree_id=tree_id,
                    num_turns=main_chain_node.num_turns,
                    branch_points=None,  # synthetic rollouts don't pre-select branch points
                ))

        main_chain_outputs = self._collect_train_outputs(
            nodes=main_chain_nodes,
            tree_id=tree_id,
            timing_metrics=main_chain_timing_metrics,
        )

        return main_chain_outputs, failed_trajectories

    async def run_from_buffer_entry(
        self,
        entry: BranchPointEntry,
        sampling_params: Dict[str, Any],
        branches_per_entry: int,
    ) -> Tuple[List[TSTrainAgentLoopOutput], List[TSTrainAgentLoopOutput]]:
        """Generate multiple continuations from a buffered branch point.

        Returns:
            branch_outputs: Training outputs with advantages computed across continuations.
            teacher_outputs: Empty list (no teacher data from buffer entries).
        """
        self._is_validation = False
        branch_node = entry.to_node()

        await self._expand(branch_node, sampling_params, rollout_n=branches_per_entry)

        branch_chain_nodes = []
        for child in branch_node.children:
            compressed = self._compress_chain(self._get_chain_from_start_node(child))
            branch_chain_nodes.append(compressed)

        # Provide branching strategy feedback so it can learn which positions
        # are promising. The feedback is delayed (from previous steps' entries),
        # but UtilitySamplingStrategy's position-based model is robust to this.
        any_success = any(
            node.reward is not None and node.reward > self.low_quality_trajectory_reward_threshold
            for node in branch_chain_nodes
        )
        feedback = BranchingFeedback(
            total_length=entry.chain_total_length,
            branch_chain_root_index=entry.branch_chain_root_index,
            is_success=any_success,
        )
        self.branching_strategy.update(feedback)

        # Compute advantages: reward - mean(rewards of all continuations)
        branch_outputs = self._collect_train_outputs(
            nodes=branch_chain_nodes,
            tree_id=entry.tree_id,
        )

        return branch_outputs, []

    async def synthesize_teacher_suffix(
        self,
        entry: FailedTrajectoryEntry,
        min_prefix_match_tokens: int = 10,
        min_prefix_match_ratio: float = 0.05,
    ) -> Optional[BranchPointEntry]:
        """Analyze a failed trajectory with teacher model and create an annotated branch entry.

        Called by the background TeacherAnnotationWorker (in its own event loop).

        Returns:
            BranchPointEntry with teacher_suffix set, or None if annotation fails.
        """
        # 1. Decode prompt (instruction) and response for teacher
        instruction, student_response = await asyncio.gather(
            self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(entry.prompt_ids, skip_special_tokens=True)
            ),
            self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(entry.response_ids, skip_special_tokens=True)
            ),
        )

        # 2. Call teacher. In prefix_inject_mode, the instruction may contain
        #    the original problem plus a previously-verified locked_prefix,
        #    so we use V2 template which is explicit about treating the Context
        #    as given background and starting the output from the Student's
        #    Failed Trajectory. The reference answer is intentionally hidden
        #    from the teacher; the reward scorer remains the verifier below.
        prefix_inject_enabled = self.config.actor_rollout_ref.rollout.deep_grpo.get(
            "prefix_inject_mode", {}
        ).get("enabled", False)
        template = (
            TEACHER_SUFFIX_SYNTHESIS_PROMPT_TEMPLATE_V2
            if prefix_inject_enabled
            else TEACHER_SUFFIX_SYNTHESIS_PROMPT_TEMPLATE
        )
        prompt = template.format(
            instruction=instruction,
            student_response=student_response,
        )

        correct_solution, raw_reply = await call_teacher_with_retry(
            message=prompt,
            parse_fn=_parse_teacher_synthesis_reply,
            temperature_schedule=(0.3, 0.7),
            log_prefix="TeacherSynthesis",
        )

        if correct_solution is None:
            logger.debug("Teacher synthesis failed to produce a correct solution.")
            return None

        # 3. Tokenize full teacher output with student tokenizer
        teacher_response_ids = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.encode(correct_solution, add_special_tokens=False)
        )

        if not teacher_response_ids:
            logger.debug("Teacher solution tokenized to empty sequence.")
            return None

        # 4. Prefix match against original response_ids
        match_len = 0
        max_check = min(len(entry.response_ids), len(teacher_response_ids))
        for i in range(max_check):
            if entry.response_ids[i] == teacher_response_ids[i]:
                match_len = i + 1
            else:
                break

        # 5. Validate prefix match
        if match_len < min_prefix_match_tokens:
            logger.debug(
                f"Teacher prefix match too short: {match_len} tokens "
                f"(min={min_prefix_match_tokens})"
            )
            return None

        if len(entry.response_ids) > 0 and match_len / len(entry.response_ids) < min_prefix_match_ratio:
            logger.debug(
                f"Teacher prefix match ratio too low: {match_len}/{len(entry.response_ids)} "
                f"= {match_len / len(entry.response_ids):.2%} (min={min_prefix_match_ratio:.0%})"
            )
            return None

        # 6. Split
        prefix_response_ids = list(entry.response_ids[:match_len])
        teacher_suffix_text_ids = list(teacher_response_ids[match_len:])
        original_failed_suffix_ids = list(entry.response_ids[match_len:])

        if not teacher_suffix_text_ids:
            logger.debug("Teacher suffix is empty after prefix matching.")
            return None

        min_teacher_suffix_len = self.config.actor_rollout_ref.rollout.deep_grpo.get(
            "teacher_suffix_synthesis", {}
        ).get("min_suffix_len", 0)
        if len(teacher_suffix_text_ids) < min_teacher_suffix_len:
            logger.debug(
                f"Teacher suffix too short: {len(teacher_suffix_text_ids)} < {min_teacher_suffix_len}"
            )
            return None

        teacher_suffix_ids, teacher_suffix_mask, eos_appended = (
            append_eos_if_missing(
                teacher_suffix_text_ids,
                [1] * len(teacher_suffix_text_ids),
                self.tokenizer,
            )
        )
        if eos_appended:
            logger.debug("Appended EOS token to teacher suffix SFT target.")

        # 7. Length check: teacher trajectory must fit, AND leave room for model branches
        total_len = len(entry.prompt_ids) + len(prefix_response_ids) + len(teacher_suffix_ids)
        if total_len > self.max_model_len:
            logger.debug(
                f"Teacher trajectory exceeds max_model_len: {total_len} > {self.max_model_len}"
            )
            return None

        branch_context_len = len(entry.prompt_ids) + len(prefix_response_ids)
        remaining_for_branches = self.max_model_len - branch_context_len
        min_branch_tokens = 32
        if remaining_for_branches < min_branch_tokens:
            logger.debug(
                f"Too little room for model branches: context={branch_context_len}, "
                f"remaining={remaining_for_branches}, min_required={min_branch_tokens}"
            )
            return None

        # 8. Score teacher trajectory
        teacher_full_response_ids = prefix_response_ids + teacher_suffix_text_ids
        teacher_full_response_mask = [1] * len(teacher_full_response_ids)

        score_node_obj = Node(
            node_id=uuid4().hex,
            prompt_ids=list(entry.prompt_ids),
            response_ids=teacher_full_response_ids,
            response_mask=teacher_full_response_mask,
            data_instance=entry.data_instance,
            finish_reason=FINISH_REASON.COMPLETED,
            num_turns=entry.num_turns,
        )
        await self._score_node(score_node_obj)

        # 9. Validate reward
        if (
            score_node_obj.reward is None
            or score_node_obj.reward <= self.low_quality_trajectory_reward_threshold
        ):
            logger.debug(
                f"Teacher trajectory scored reward={score_node_obj.reward}, discarding."
            )
            return None

        # 10. Create annotated BranchPointEntry
        teacher_suffix = TeacherSuffix(
            suffix_ids=teacher_suffix_ids,
            suffix_mask=teacher_suffix_mask,
            reward=score_node_obj.reward,
            reward_info=score_node_obj.reward_info,
            original_failed_suffix_ids=original_failed_suffix_ids,
        )

        annotated_entry = BranchPointEntry(
            prompt_ids=list(entry.prompt_ids),
            response_ids=prefix_response_ids,
            response_mask=[1] * len(prefix_response_ids),
            data_instance=entry.data_instance,
            num_turns=entry.num_turns,
            tree_id=entry.tree_id,
            branch_chain_root_index=0,
            chain_total_length=1,
            agent_name=entry.agent_name,
            teacher_suffix=teacher_suffix,
        )

        return annotated_entry

    async def run_from_teacher_entry(
        self,
        entry: BranchPointEntry,
        sampling_params: Dict[str, Any],
        branches_per_entry: int,
    ) -> Tuple[List[TSTrainAgentLoopOutput], List[TSTrainAgentLoopOutput]]:
        """Generate branches from a teacher-annotated entry.

        Returns:
            branch_outputs: Model branch training outputs (empty if branches_per_entry=0).
            teacher_outputs: Single-element list with the teacher suffix output.
        """
        assert entry.teacher_suffix is not None, "Entry must have teacher_suffix"
        self._is_validation = False

        # Build teacher node (always needed)
        ts = entry.teacher_suffix
        teacher_node = Node(
            node_id=uuid4().hex,
            prompt_ids=list(entry.prompt_ids) + list(entry.response_ids),
            response_ids=list(ts.suffix_ids),
            response_mask=list(ts.suffix_mask),
            data_instance=entry.data_instance,
            finish_reason=FINISH_REASON.COMPLETED,
            num_turns=entry.num_turns + 1,
            reward=ts.reward,
            reward_info=ts.reward_info,
        )

        if branches_per_entry == 0:
            # Teacher-only: skip branch generation.
            # Advantage is dummy (SFT ignores it), but non-zero to pass zero-advantage filter.
            teacher_node.advantage = 1.0
            teacher_output = self._node_to_train_output(teacher_node, tree_id=entry.tree_id, metrics={})
            return [], [teacher_output]

        # Generate K model branches from the branch point
        branch_node = entry.to_node()
        await self._expand(branch_node, sampling_params, rollout_n=branches_per_entry)

        model_nodes = []
        for child in branch_node.children:
            compressed = self._compress_chain(self._get_chain_from_start_node(child))
            model_nodes.append(compressed)

        # Compute advantages
        teacher_loss_type = self.config.actor_rollout_ref.actor.deep_grpo.get("teacher_loss_type", None)
        if teacher_loss_type == "sft":
            # SFT: baseline from model branches only
            model_outputs = self._collect_train_outputs(
                nodes=model_nodes,
                tree_id=entry.tree_id,
            )
            model_baseline = np.mean([n.reward for n in model_nodes])
            teacher_node.advantage = float(teacher_node.reward) - float(model_baseline)
            teacher_output = self._node_to_train_output(teacher_node, tree_id=entry.tree_id, metrics={})
        else:
            # LUFFY/PPO: include teacher in advantage group
            all_nodes = model_nodes + [teacher_node]
            all_outputs = self._collect_train_outputs(
                nodes=all_nodes,
                tree_id=entry.tree_id,
            )
            model_outputs = all_outputs[:-1]
            teacher_output = all_outputs[-1]

        return model_outputs, [teacher_output]

    @rollout_trace_op
    async def _run_recoverability_eval(
        self,
        messages: List[Dict[str, Any]],
        sampling_params: Dict[str, Any],
        data_instance: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Evaluate recoverability at every position of a failed trajectory.

        For each failed main chain:
        1. Partition the response into segments.
        2. At every expandable position, branch branches_per_point times.
        3. Record success rate per position.

        Returns a list of dicts, one per position evaluated.
        Returns empty list if the main chain succeeds.
        """
        self._is_validation = True

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
            num_turns=0,
        )
        await self._expand(root, sampling_params, rollout_n=1)

        # _rollout already scores the last node; _compress_chain propagates reward.
        # For ReasoningAgentLoop, _rollout also partitions internally, so the
        # chain returned by _get_chain_from_start_node is already partitioned.
        # Include root in chain to match training code's position_ratio convention
        # (training's _create_branching_selection uses main_chain which includes root).
        chain = [root] + self._get_chain_from_start_node(root.children[0])
        main_chain_reward = chain[-1].reward

        # Only evaluate failed trajectories
        if main_chain_reward is not None and main_chain_reward > self.low_quality_trajectory_reward_threshold:
            return []

        # Branch at every expandable position (in parallel)
        branch_roots = []
        tasks = []
        for i, node in enumerate(chain):
            if not self.branching_strategy._is_expandable(node):
                continue
            branch_root = Node(
                node_id=uuid4().hex,
                prompt_ids=list(node.prompt_ids),
                response_ids=list(node.response_ids),
                response_mask=list(node.response_mask),
                data_instance=node.data_instance,
                finish_reason=node.finish_reason,
                num_turns=node.num_turns,
            )
            branch_roots.append((i, branch_root))
            tasks.append(asyncio.create_task(
                self._expand(branch_root, sampling_params, rollout_n=self.branches_per_point)
            ))

        if tasks:
            await asyncio.gather(*tasks)

        # Collect results
        results = []
        for pos_idx, branch_root in branch_roots:
            branch_rewards = []
            for child in branch_root.children:
                compressed = self._compress_chain(self._get_chain_from_start_node(child))
                r = compressed.reward if compressed.reward is not None else 0.0
                branch_rewards.append(float(r))

            n_success = sum(1 for r in branch_rewards if r > self.low_quality_trajectory_reward_threshold)
            results.append({
                "uid": data_instance.get("uid", ""),
                "position_index": pos_idx,
                "chain_length": len(chain),
                "position_ratio": (pos_idx + 1) / len(chain),
                "n_branches": len(branch_rewards),
                "n_success": n_success,
                "recoverability": 1 if n_success > 0 else 0,
                "mean_branch_reward": sum(branch_rewards) / len(branch_rewards) if branch_rewards else 0.0,
                "main_chain_reward": float(main_chain_reward) if main_chain_reward is not None else 0.0,
            })

        return results

    async def run(
        self,
        mode: str,
        messages: List[Dict[str, Any]],
        sampling_params: Dict[str, Any],
        data_instance: Dict[str, Any],
    ) -> Union[TSValAgentLoopOutput, Tuple[List[TSTrainAgentLoopOutput], List[TSTrainAgentLoopOutput]], List[Dict[str, Any]]]:
        if mode == "train_one_stage":
            return await self._run_train_one_stage(messages, sampling_params, data_instance)
        elif mode == "recoverability_eval":
            return await self._run_recoverability_eval(messages, sampling_params, data_instance)
        return await super().run(mode, messages, sampling_params, data_instance)
