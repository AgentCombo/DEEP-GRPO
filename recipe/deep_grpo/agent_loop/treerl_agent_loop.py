"""
TreeRL Agent Loop - EPTree sampling with process supervision.

Implements the TreeRL algorithm from "TreeRL: LLM Reinforcement Learning with On-Policy Tree Search".
Key features:
- Token-level entropy-guided forking (EPTree)
- Process supervision from tree structure (V values, Global/Local Advantage)
- Single-step complete response generation (no sentence-level splitting)
"""

import asyncio
import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional
from uuid import uuid4

import numpy as np

from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

from recipe.deep_grpo.protocol import Node, FINISH_REASON, RewardInfo
from recipe.deep_grpo.agent_loop.outputs import TSTrainAgentLoopOutput, TSValAgentLoopOutput
from recipe.deep_grpo.agent_loop.deep_grpo_agent_loop import DeepGRPOAgentLoop

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class TreeRLTreeNode:
    """Lightweight tree node for TreeRL's EPTree structure.

    Each node represents a segment of the response. The full response for a leaf
    is the concatenation of segment_ids along the path from root to the leaf.
    """

    def __init__(self, segment_ids: List[int], segment_logprobs: List[float], segment_mask: List[int],
                 finish_reason: Optional[FINISH_REASON] = None):
        self.segment_ids = segment_ids
        self.segment_logprobs = segment_logprobs
        self.segment_mask = segment_mask
        self.finish_reason = finish_reason  # Only meaningful for leaf nodes
        self.children: List["TreeRLTreeNode"] = []
        self.is_leaf = True
        self.reward: Optional[float] = None
        self.reward_info: Optional[RewardInfo] = None

        # Process supervision values (computed after tree is built)
        self.V = 0.0
        self.n_leaves = 0
        self.sum_rewards = 0.0
        self.process_reward = 0.0
        self.parent: Optional["TreeRLTreeNode"] = None

    def add_child(self, child: "TreeRLTreeNode"):
        self.children.append(child)
        child.parent = self
        self.is_leaf = False


class TreeRLAgentLoop(DeepGRPOAgentLoop):
    """TreeRL agent loop implementing EPTree sampling + process supervision.

    Unlike the base TSAgentLoop which generates step-by-step (sentence-level),
    this generates complete responses in one shot and forks at token level
    based on entropy.
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

        # TreeRL controls M internally, rollout_n is set to 1 from config
        cls.rollout_n = config.actor_rollout_ref.rollout.n

        # These are not used by TreeRL but needed by parent class methods
        cls.expand_branch_chain = False
        cls.expand_only_on_low_quality = False
        cls.low_quality_trajectory_reward_threshold = 0.0
        cls.branches_per_point = 0
        cls.n_branch_points = 0
        cls.pick_branch_chain_root_method = "random"

        # Dummy branching strategy (not used by TreeRL)
        from recipe.deep_grpo.branching_strategy import RandomBranchingStrategy
        cls.branching_strategy = RandomBranchingStrategy(max_model_len=cls.max_model_len)

        # TreeRL parameters (M, N, L, T)
        treerl_config = config.actor_rollout_ref.rollout.deep_grpo.treerl
        cls.treerl_M = treerl_config.get("M", 6)   # number of initial chains per tree
        cls.treerl_N = treerl_config.get("N", 2)   # number of forking points per iteration
        cls.treerl_L = treerl_config.get("L", 1)   # number of expansion iterations
        cls.treerl_T = treerl_config.get("T", 2)   # branching factor at each fork

        logger.info(f"TreeRL initialized with (M={cls.treerl_M}, N={cls.treerl_N}, "
                    f"L={cls.treerl_L}, T={cls.treerl_T})")
        logger.info(f"Expected leaves per prompt: {cls.treerl_M * (1 + cls.treerl_N * cls.treerl_L * cls.treerl_T)}")

    async def _rollout(self, node: Node, sampling_params: Dict[str, Any], request_id: str) -> Node:
        """Generate a single chain without scoring the leaf.

        The parent class's _rollout calls _score_node after generation, but TreeRL
        scores all leaf nodes separately via _score_leaf after the full EPTree is built.
        Scoring here would be redundant (and costly for LLM-based rewards), so we skip it.
        """
        current_node = node
        generated_chain = []
        while True:
            new_node = await self._step(current_node, sampling_params, request_id)
            generated_chain.append(new_node)
            if new_node.finish_reason == FINISH_REASON.COMPLETED or new_node.finish_reason == FINISH_REASON.EXCEED_LENGTH:
                break
            current_node = new_node
        assert len(generated_chain) > 0
        for i in range(len(generated_chain) - 1):
            generated_chain[i].children = [generated_chain[i + 1]]
        return generated_chain[0]

    async def _step(self, parent_node: Node, sampling_params: Dict[str, Any], request_id: str) -> Node:
        """Generate a complete response in one shot with logprobs.

        Unlike the sp/tp variants which stop at sentence boundaries,
        this generates until EOS or max_tokens, collecting logprobs for every token.
        """
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
                log_probs=[]
            )

        sampling_params_copy = sampling_params.copy()
        sampling_params_copy['return_logprobs'] = True
        # No stop_token_ids - generate complete response

        result = await self.server_manager.generate_with_finish_reason(
            request_id=request_id,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params_copy
        )

        response_ids = result["token_ids"]
        response_mask = [1] * len(response_ids)
        log_probs = result.get("logprobs", [0.0] * len(response_ids))

        # Determine finish reason
        if result["finish_reason"] == "length":
            finish_reason = FINISH_REASON.EXCEED_LENGTH
        elif result["finish_reason"] == "stop":
            if len(prompt_ids) + len(response_ids) >= self.max_model_len:
                finish_reason = FINISH_REASON.EXCEED_LENGTH
            else:
                finish_reason = FINISH_REASON.COMPLETED
        else:
            finish_reason = FINISH_REASON.COMPLETED

        return Node(
            node_id=uuid4().hex,
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            data_instance=parent_node.data_instance,
            finish_reason=finish_reason,
            num_turns=parent_node.num_turns + 1,
            log_probs=log_probs
        )

    @rollout_trace_op
    async def _run_train(
        self,
        messages: List[Dict[str, Any]],
        sampling_params: Dict[str, Any],
        data_instance: Dict[str, Any]
    ) -> Tuple[List[TSTrainAgentLoopOutput], List[TSTrainAgentLoopOutput]]:
        """Build M EPTrees in parallel and package all leaf paths for training."""
        self._is_validation = False
        tree_id = data_instance["uid"]

        prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=getattr(self, 'add_generation_prompt', True), tokenize=True
            ),
        )

        # Build M trees in parallel (build + score + compute V, but don't assign rewards yet)
        tasks = []
        for i in range(self.treerl_M):
            tasks.append(asyncio.create_task(
                self._build_single_tree(prompt_ids, sampling_params, data_instance)
            ))
        tree_results = await asyncio.gather(*tasks)

        # Compute cross-tree V_root: average pass rate across all M trees for this prompt
        tree_roots = [root for root in tree_results if root is not None]
        if len(tree_roots) == 0:
            return [], []
        total_rewards = sum(root.sum_rewards for root in tree_roots)
        total_leaves = sum(root.n_leaves for root in tree_roots)
        V_root = total_rewards / max(total_leaves, 1)

        # Assign process rewards and package training data using cross-tree V_root
        all_outputs = []
        for tree_root in tree_roots:
            self._assign_process_rewards(tree_root, V_root)
            outputs = self._package_training_data(tree_root, prompt_ids, tree_id)
            all_outputs.extend(outputs)

        # All outputs as main_chain, no branch_chain for TreeRL
        return all_outputs, []

    async def _build_single_tree(
        self,
        prompt_ids: List[int],
        sampling_params: Dict[str, Any],
        data_instance: Dict[str, Any],
    ) -> Optional[TreeRLTreeNode]:
        """Build a single EPTree: generate initial chain, fork, score leaves, compute V.

        Returns tree_root so that process rewards can be assigned later with
        a cross-tree V_root. Returns None if the tree is empty.
        """

        # Step 1: Generate initial chain (complete response + logprobs)
        root_node = Node(
            node_id=uuid4().hex,
            prompt_ids=prompt_ids,
            response_ids=[],
            response_mask=[],
            data_instance=data_instance,
            num_turns=0
        )

        # _expand calls _rollout which calls _step (our override: single-step generation)
        await self._expand(root_node, sampling_params, rollout_n=1)

        # Collect the generated chain (should be a single node since we do single-step)
        chain_nodes = self._get_chain_from_start_node(root_node.children[0])

        # Gather full response tokens and logprobs
        all_response_ids = []
        all_logprobs = []
        all_response_mask = []
        for node in chain_nodes:
            all_response_ids.extend(node.response_ids)
            all_logprobs.extend(node.log_probs if node.log_probs else [0.0] * len(node.response_ids))
            all_response_mask.extend(node.response_mask)

        if len(all_response_ids) == 0:
            return None

        # Use the last chain node's finish_reason (it determines if the response completed)
        chain_finish_reason = chain_nodes[-1].finish_reason

        # Step 2: Build TreeRL tree structure
        tree_root = TreeRLTreeNode(
            segment_ids=all_response_ids,
            segment_logprobs=all_logprobs,
            segment_mask=all_response_mask,
            finish_reason=chain_finish_reason,
        )

        # Step 3: EPTree iterative expansion
        for l in range(self.treerl_L):
            # Collect entropy candidates from all nodes in tree
            all_candidates = []
            self._collect_entropy_candidates(tree_root, all_candidates)

            if len(all_candidates) == 0:
                break

            # Select top-N highest entropy tokens
            all_candidates.sort(key=lambda x: x[2], reverse=True)
            selected = all_candidates[:self.treerl_N]

            # Group candidates by node
            node_to_positions = defaultdict(list)
            for target_node, token_pos, entropy in selected:
                node_to_positions[id(target_node)].append((target_node, token_pos))

            # Fork all nodes in parallel; within each node, all branch
            # generations also run in parallel (see _fork_node_at_multiple_positions).
            fork_tasks = []
            for node_id, positions in node_to_positions.items():
                target = positions[0][0]
                pos_values = sorted(p[1] for p in positions)  # ascending
                fork_tasks.append(
                    self._fork_node_at_multiple_positions(
                        tree_root, target, pos_values,
                        prompt_ids, sampling_params, data_instance
                    )
                )
            await asyncio.gather(*fork_tasks)

        # Step 4: Score all leaf nodes
        leaves = []
        self._collect_leaves(tree_root, leaves)

        score_tasks = []
        for leaf in leaves:
            score_tasks.append(self._score_leaf(leaf, prompt_ids, data_instance))
        await asyncio.gather(*score_tasks)

        # Step 5: Compute V values (process rewards assigned later with cross-tree V_root)
        self._compute_values(tree_root)

        return tree_root

    def _collect_entropy_candidates(
        self,
        node: TreeRLTreeNode,
        candidates: List[Tuple[TreeRLTreeNode, int, float]],
        min_prefix_ratio: float = 0.05,
        min_suffix_tokens: int = 10
    ):
        """Collect all tokens with their entropy as forking candidates.

        Only leaf nodes are candidates (internal nodes are committed prefixes).
        Always recurse into ALL children to reach deep leaves (not just leaf children).
        """
        if node.is_leaf and len(node.children) == 0:
            # Only collect entropy candidates from leaf nodes
            n_tokens = len(node.segment_ids)
            min_pos = max(1, int(n_tokens * min_prefix_ratio))
            max_pos = n_tokens - min_suffix_tokens

            for i in range(min_pos, max(min_pos, max_pos + 1)):
                entropy = -node.segment_logprobs[i]  # H(y_t) = -log pi(y_t | x, y_{<t})
                candidates.append((node, i, entropy))
        else:
            # Recurse into ALL children (not just leaves) to reach deep leaf nodes
            for child in node.children:
                self._collect_entropy_candidates(child, candidates, min_prefix_ratio=0.0)

    async def _fork_node_at_multiple_positions(
        self,
        tree_root: TreeRLTreeNode,
        target_node: TreeRLTreeNode,
        positions: List[int],
        prompt_ids: List[int],
        sampling_params: Dict[str, Any],
        data_instance: Dict[str, Any],
    ) -> None:
        """Fork a single node at multiple positions simultaneously.

        All branch generations are launched in parallel, then the tree is built
        in one pass (bottom-up). This is ~Nx faster than sequential forking
        because vLLM batches concurrent requests.

        Args:
            positions: token positions in ascending order. The high-entropy token
                at each position goes into the SUFFIX (new branches explore
                alternatives at that position).
        """
        # --- Phase A: parallel generation ---

        # Pre-compute ancestor_ids (shared across all forks on this node)
        ancestor_segments = []
        node = target_node.parent
        while node is not None:
            ancestor_segments.append(node.segment_ids)
            node = node.parent
        ancestor_segments.reverse()
        ancestor_ids = [tid for seg in ancestor_segments for tid in seg]

        # Create fork_nodes and launch all _expand calls in parallel
        fork_nodes: List[Optional[Node]] = []  # fork_nodes[i] for positions[i]; None if skipped
        expand_tasks = []
        for pos in positions:
            prefix_ids = target_node.segment_ids[:pos]
            fork_prompt_ids = prompt_ids + ancestor_ids + prefix_ids

            if len(fork_prompt_ids) >= self.max_model_len - 10:
                fork_nodes.append(None)
                continue

            fork_node = Node(
                node_id=uuid4().hex,
                prompt_ids=fork_prompt_ids,
                response_ids=[],
                response_mask=[],
                data_instance=data_instance,
                finish_reason=FINISH_REASON.STOP,
                num_turns=0,
            )
            fork_nodes.append(fork_node)
            expand_tasks.append(asyncio.create_task(
                self._expand(fork_node, sampling_params, rollout_n=self.treerl_T)
            ))

        if expand_tasks:
            await asyncio.gather(*expand_tasks)

        # --- Phase B: build tree structure (pure sync, bottom-up) ---

        # Split target_node into K+1 segments at the fork positions.
        # positions = [p0, p1, ...], boundaries = [0, p0, p1, ..., len]
        # Segments: seg0[0..p0), seg1[p0..p1), ..., segK[pK-1..end)
        boundaries = [0] + list(positions) + [len(target_node.segment_ids)]
        tree_segments: List[TreeRLTreeNode] = []
        for i in range(len(boundaries) - 1):
            s, e = boundaries[i], boundaries[i + 1]
            tree_segments.append(TreeRLTreeNode(
                segment_ids=target_node.segment_ids[s:e],
                segment_logprobs=target_node.segment_logprobs[s:e],
                segment_mask=target_node.segment_mask[s:e],
            ))

        # Last segment inherits original node's children and finish_reason
        last = tree_segments[-1]
        last.finish_reason = target_node.finish_reason
        last.children = target_node.children
        for child in last.children:
            child.parent = last
        last.is_leaf = (len(last.children) == 0)

        # Build from innermost (last segment) outward.
        # Each wrapper segment[i] gets segment[i+1] as first child,
        # plus the generated branches for positions[i].
        current_inner = last
        for i in range(len(positions) - 1, -1, -1):
            wrapper = tree_segments[i]
            wrapper.is_leaf = False
            wrapper.add_child(current_inner)

            # Attach generated branches for this fork point
            if fork_nodes[i] is not None:
                for child in fork_nodes[i].children:
                    branch_chain = self._get_chain_from_start_node(child)
                    branch_ids = []
                    branch_logprobs = []
                    branch_mask = []
                    for n in branch_chain:
                        branch_ids.extend(n.response_ids)
                        branch_logprobs.extend(n.log_probs if n.log_probs else [0.0] * len(n.response_ids))
                        branch_mask.extend(n.response_mask)

                    branch_finish = branch_chain[-1].finish_reason if branch_chain else FINISH_REASON.COMPLETED
                    branch_tree_node = TreeRLTreeNode(
                        branch_ids, branch_logprobs, branch_mask, finish_reason=branch_finish
                    )
                    branch_tree_node.is_leaf = True
                    wrapper.add_child(branch_tree_node)

            current_inner = wrapper

        # Replace target_node with the outermost segment in the tree
        self._replace_node_in_tree(tree_root, target_node, current_inner)

    def _replace_node_in_tree(
        self,
        tree_root: TreeRLTreeNode,
        old_node: TreeRLTreeNode,
        new_node: TreeRLTreeNode
    ):
        """Replace old_node with new_node in the tree.

        If old_node is tree_root, we copy new_node's data into tree_root.
        Otherwise, find old_node's parent and swap it.
        """
        if tree_root is old_node:
            # Replace root: copy new_node data into tree_root
            tree_root.segment_ids = new_node.segment_ids
            tree_root.segment_logprobs = new_node.segment_logprobs
            tree_root.segment_mask = new_node.segment_mask
            tree_root.finish_reason = new_node.finish_reason
            tree_root.children = new_node.children
            tree_root.is_leaf = new_node.is_leaf
            for child in tree_root.children:
                child.parent = tree_root
            return

        # Find parent of old_node and replace
        parent = old_node.parent
        if parent is not None:
            for i, child in enumerate(parent.children):
                if child is old_node:
                    parent.children[i] = new_node
                    new_node.parent = parent
                    return

        # Fallback: search the whole tree
        self._replace_in_subtree(tree_root, old_node, new_node)

    def _replace_in_subtree(
        self,
        current: TreeRLTreeNode,
        old_node: TreeRLTreeNode,
        new_node: TreeRLTreeNode
    ) -> bool:
        for i, child in enumerate(current.children):
            if child is old_node:
                current.children[i] = new_node
                new_node.parent = current
                return True
            if self._replace_in_subtree(child, old_node, new_node):
                return True
        return False

    def _collect_leaves(self, node: TreeRLTreeNode, leaves: List[TreeRLTreeNode]):
        """Collect all leaf nodes in the tree."""
        if node.is_leaf and len(node.children) == 0:
            leaves.append(node)
            return
        for child in node.children:
            self._collect_leaves(child, leaves)

    async def _score_leaf(
        self,
        leaf: TreeRLTreeNode,
        prompt_ids: List[int],
        data_instance: Dict[str, Any],
    ):
        """Score a leaf node by constructing a full Node and using existing score_node."""
        # Reconstruct full response from root to this leaf via parent pointers
        segments = []
        node = leaf
        while node is not None:
            segments.append(node)
            node = node.parent
        segments.reverse()

        full_response_ids = []
        full_response_mask = []
        for seg in segments:
            full_response_ids.extend(seg.segment_ids)
            full_response_mask.extend(seg.segment_mask)

        score_node_obj = Node(
            node_id=uuid4().hex,
            prompt_ids=prompt_ids,
            response_ids=full_response_ids,
            response_mask=full_response_mask,
            data_instance=data_instance,
            finish_reason=leaf.finish_reason if leaf.finish_reason is not None else FINISH_REASON.COMPLETED,
            num_turns=1
        )

        await self._score_node(score_node_obj)
        leaf.reward = score_node_obj.reward
        leaf.reward_info = score_node_obj.reward_info

    def _compute_values(self, node: TreeRLTreeNode):
        """Bottom-up computation of V(s_n) = fraction of correct leaf descendants."""
        if node.is_leaf and len(node.children) == 0:
            node.n_leaves = 1
            node.sum_rewards = node.reward if node.reward is not None else 0.0
            node.V = float(node.sum_rewards)
            return

        node.n_leaves = 0
        node.sum_rewards = 0.0
        for child in node.children:
            self._compute_values(child)
            node.n_leaves += child.n_leaves
            node.sum_rewards += child.sum_rewards

        node.V = node.sum_rewards / max(node.n_leaves, 1)

    def _assign_process_rewards(self, node: TreeRLTreeNode, V_root: float):
        """Top-down assignment of process rewards R(s_n).

        R(s_n) = (G_A(s_n) + L_A(s_n)) / sqrt(|L(s_n)|)
        where:
          G_A(s_n) = V(s_n) - V(root)        (global advantage)
          L_A(s_n) = V(s_n) - V(parent(s_n))  (local advantage)
          |L(s_n)| = number of leaf descendants
        """
        G_A = node.V - V_root
        # L_A: local advantage = improvement over parent. Root has no parent, so L_A = 0.
        L_A = (node.V - node.parent.V) if node.parent is not None else 0.0
        n_leaf_descendants = max(node.n_leaves, 1)
        node.process_reward = (G_A + L_A) / (n_leaf_descendants ** 0.5)

        for child in node.children:
            self._assign_process_rewards(child, V_root)

    def _enumerate_leaf_paths(
        self,
        node: TreeRLTreeNode,
        current_path: List[TreeRLTreeNode],
        all_paths: List[List[TreeRLTreeNode]]
    ):
        """Enumerate all root-to-leaf paths in the tree."""
        current_path.append(node)

        if node.is_leaf and len(node.children) == 0:
            all_paths.append(list(current_path))
        else:
            for child in node.children:
                self._enumerate_leaf_paths(child, current_path, all_paths)

        current_path.pop()

    def _package_training_data(
        self,
        tree_root: TreeRLTreeNode,
        prompt_ids: List[int],
        tree_id: str
    ) -> List[TSTrainAgentLoopOutput]:
        """Package each leaf path as a training sample with per-token process rewards."""
        all_paths = []
        self._enumerate_leaf_paths(tree_root, [], all_paths)

        outputs = []
        for path in all_paths:
            response_ids = []
            response_mask = []
            token_level_advantages = []

            for node in path:
                if len(node.segment_ids) == 0:
                    continue
                response_ids.extend(node.segment_ids)
                response_mask.extend(node.segment_mask)
                # All tokens in this segment share the same process reward
                token_level_advantages.extend([node.process_reward] * len(node.segment_ids))

            if len(response_ids) == 0:
                continue

            # Get leaf reward
            leaf = path[-1]
            leaf_reward = leaf.reward if leaf.reward is not None else 0.0
            leaf_reward_info = leaf.reward_info if leaf.reward_info is not None else RewardInfo(reward=leaf_reward, completed=1)

            # Truncate to max_model_len
            max_resp_len = self.max_model_len - len(prompt_ids)
            if len(response_ids) > max_resp_len:
                response_ids = response_ids[:max_resp_len]
                response_mask = response_mask[:max_resp_len]
                token_level_advantages = token_level_advantages[:max_resp_len]

            output = TSTrainAgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                response_mask=response_mask,
                num_turns=1,
                reward=leaf_reward,
                reward_info=leaf_reward_info,
                metrics={},
                tree_id=tree_id,
                node_id=uuid4().hex,
                advantage=float(np.mean(token_level_advantages)),
                token_level_advantages=token_level_advantages,
            )
            outputs.append(output)

        return outputs

    @rollout_trace_op
    async def _run_val(
        self,
        messages: List[Dict[str, Any]],
        sampling_params: Dict[str, Any],
        data_instance: Dict[str, Any],
    ) -> TSValAgentLoopOutput:
        """Validation: single greedy generation (same as base class)."""
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

        # _rollout is overridden to skip scoring (avoids redundant scoring in training).
        # For validation we need the reward, so score explicitly here.
        await self._score_node(node)

        return self._node_to_val_output(node, metrics)
