# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import hashlib
import json
import os
import uuid
from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Optional, Type
from collections import Counter

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger

from recipe.deep_grpo.protocol import RewardInfo


WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(
                    f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}"
                    + "cannot be satisfied in this ray cluster"
                )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.reweight_method,
                config.pf_ppo.weight_pow,
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, and vLLM integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to "cuda".
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.OPO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
            AdvantageEstimator.GPG,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
        actor_config = config.actor_rollout_ref.actor
        deep_grpo_config = actor_config.get("deep_grpo", {}) or {}
        teacher_update_mode = deep_grpo_config.get("teacher_update_mode", "joint")
        teacher_loss_reduction = str(
            deep_grpo_config.get("teacher_loss_reduction", "separate_stream_mean")
        ).strip()
        assert teacher_update_mode in ("joint", "post_ppo"), (
            "actor_rollout_ref.actor.deep_grpo.teacher_update_mode must be "
            f"'joint' or 'post_ppo', got {teacher_update_mode}"
        )
        assert teacher_loss_reduction in (
            "separate_stream_mean",
            "mixed_token_mean",
        ), (
            "actor_rollout_ref.actor.deep_grpo.teacher_loss_reduction must be "
            "'separate_stream_mean' or 'mixed_token_mean', got "
            f"{teacher_loss_reduction}"
        )
        if teacher_loss_reduction == "mixed_token_mean":
            assert teacher_update_mode == "joint", (
                "actor_rollout_ref.actor.deep_grpo.teacher_loss_reduction="
                "mixed_token_mean requires teacher_update_mode=joint"
            )
            assert deep_grpo_config.get("teacher_loss_type", None) in ("luffy", "snis", None), (
                "actor_rollout_ref.actor.deep_grpo.teacher_loss_reduction="
                "mixed_token_mean requires teacher_loss_type in "
                "('luffy', 'snis', null)"
            )
            assert abs(float(deep_grpo_config.get("teacher_loss_coef", 1.0)) - 1.0) < 1e-12, (
                "actor_rollout_ref.actor.deep_grpo.teacher_loss_reduction="
                "mixed_token_mean uses natural token weighting; "
                "teacher_loss_coef must be 1.0"
            )
            assert not self._config_bool(
                deep_grpo_config.get("auto_teacher_loss_coef", False)
            ), (
                "actor_rollout_ref.actor.deep_grpo.teacher_loss_reduction="
                "mixed_token_mean does not support auto_teacher_loss_coef"
            )
        if teacher_update_mode == "post_ppo":
            assert actor_config.strategy in ("fsdp", "fsdp2"), (
                "actor_rollout_ref.actor.deep_grpo.teacher_update_mode=post_ppo "
                "currently requires actor strategy fsdp/fsdp2; got "
                f"{actor_config.strategy}"
            )
            assert deep_grpo_config.get("teacher_loss_type", None) == "sft", (
                "actor_rollout_ref.actor.deep_grpo.teacher_update_mode=post_ppo "
                "requires actor_rollout_ref.actor.deep_grpo.teacher_loss_type=sft"
            )
            assert not self._config_bool(
                deep_grpo_config.get("auto_teacher_loss_coef", False)
            ), (
                "actor_rollout_ref.actor.deep_grpo.teacher_update_mode=post_ppo "
                "does not support auto_teacher_loss_coef"
            )
            teacher_mbs = int(
                deep_grpo_config.get(
                    "teacher_mini_batch_size",
                    actor_config.ppo_mini_batch_size,
                )
            )
            assert teacher_mbs > 0, (
                "actor_rollout_ref.actor.deep_grpo.teacher_mini_batch_size "
                f"must be > 0 for post_ppo, got {teacher_mbs}"
            )
            assert teacher_mbs % n_gpus == 0, (
                "actor_rollout_ref.actor.deep_grpo.teacher_mini_batch_size "
                "must be divisible by total actor worker world size for "
                f"post_ppo, got {teacher_mbs} and n_gpus={n_gpus}"
            )
        pim_config = (
            config.actor_rollout_ref.rollout.deep_grpo.get("prefix_inject_mode", {})
            or {}
        )
        teacher_continuation_cfg = (
            pim_config.get("teacher_continuation", {}) or {}
        )
        prefix_forest_luffy_enabled = self._config_bool(
            teacher_continuation_cfg.get("enabled", False)
        )
        if deep_grpo_config.get("teacher_loss_type", None) == "snis":
            # The SNIS weight w̃ is produced ONLY by the prefix-forest attach
            # path and shipped via the advantages field. Any other teacher-row
            # producer (e.g. teacher_suffix_synthesis) fills that field with a
            # group-relative advantage — possibly negative — which the snis
            # branch would misread as a weight (negative w̃ = anti-imitation).
            assert prefix_forest_luffy_enabled, (
                "actor_rollout_ref.actor.deep_grpo.teacher_loss_type=snis requires "
                "actor_rollout_ref.rollout.deep_grpo.prefix_inject_mode."
                "teacher_continuation.enabled=True (the only producer of "
                "SNIS weights); other teacher-row sources ship advantages, "
                "not weights"
            )
        if prefix_forest_luffy_enabled:
            suffix_sft_cfg = pim_config.get("suffix_sft_maturation", {}) or {}
            assert self._config_bool(pim_config.get("enabled", False)), (
                "prefix_inject_mode.teacher_continuation.enabled=True requires "
                "prefix_inject_mode.enabled=True"
            )
            assert pim_config.get("pool_type", "flat") == "forest", (
                "prefix_inject_mode.teacher_continuation.enabled=True requires "
                "prefix_inject_mode.pool_type=forest"
            )
            assert not self._config_bool(suffix_sft_cfg.get("enabled", False)), (
                "prefix_inject_mode.teacher_continuation.enabled=True is "
                "mutually exclusive with suffix_sft_maturation.enabled=True"
            )
            assert teacher_loss_reduction == "mixed_token_mean", (
                "prefix_inject_mode.teacher_continuation.enabled=True requires "
                "actor_rollout_ref.actor.deep_grpo.teacher_loss_reduction=mixed_token_mean"
            )
            tc_teacher_loss_type = deep_grpo_config.get("teacher_loss_type", None)
            assert tc_teacher_loss_type in ("luffy", "snis", None), (
                "prefix_inject_mode.teacher_continuation.enabled=True requires "
                "teacher_loss_type in ('luffy', 'snis') or null (pure "
                f"state-curriculum ablation), got {tc_teacher_loss_type}"
            )
            if tc_teacher_loss_type == "snis":
                snis_beta = float(deep_grpo_config.get("snis_beta", 1.0))
                assert snis_beta > 0, (
                    "actor_rollout_ref.actor.deep_grpo.snis_beta must be > 0, "
                    f"got {snis_beta}"
                )
            assert not self._config_bool(config.trainer.get("whiten_advantages", False)), (
                "prefix_inject_mode.teacher_continuation.enabled=True requires "
                "trainer.whiten_advantages=False so student and teacher "
                "continuation advantages stay on the same reward-baseline scale"
            )
        if config.actor_rollout_ref.actor.strategy == "megatron":
            model_parallel_size = (
                config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size
                * config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
            )
            assert (
                n_gpus % (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size) == 0
            ), (
                f"n_gpus ({n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times "
                f"context_parallel_size ({config.actor_rollout_ref.actor.megatron.context_parallel_size})"
            )
            megatron_dp = n_gpus // (
                model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size
            )
            minimal_bsz = megatron_dp * config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
        else:
            minimal_bsz = n_gpus

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % minimal_bsz == 0, (
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size "
            f"({minimal_bsz})"
        )

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            """Validate mutually exclusive micro batch size configuration options.

            Ensures that users don't set both deprecated micro_batch_size and
            the new micro_batch_size_per_gpu parameters simultaneously.

            Args:
                mbs: Deprecated micro batch size parameter value.
                mbs_per_gpu: New micro batch size per GPU parameter value.
                name (str): Configuration section name for error messages.

            Raises:
                ValueError: If both parameters are set or neither is set.
            """
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(
                        f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'."
                    )

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(
                        f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                        f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
                    )

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(
                config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic"
            )

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(
                config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model"
            )

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                # Divisibility check removed: actor loss is normalised by mb_tw_sum,
                # so gradient accumulation is correct even when not evenly divisible.
                # assert (
                #     config.actor_rollout_ref.actor.ppo_mini_batch_size
                #     % config.actor_rollout_ref.actor.ppo_micro_batch_size
                #     == 0
                # )
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if self.config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == "fsdp" and (
            config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1
            or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1
        ):
            assert config.actor_rollout_ref.model.use_remove_padding, (
                "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."
            )

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, (
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."
                )

        if config.data.get("val_batch_size", None) is not None:
            print(
                "WARNING: val_batch_size is deprecated."
                + " Validation datasets are sent to inference engines as a whole batch,"
                + " which will schedule the memory themselves."
            )

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, (
                "validation gen temperature should be greater than 0 when enabling do_sample"
            )

        # check multi_turn with tool config
        if config.actor_rollout_ref.rollout.multi_turn.enable:
            assert (
                config.actor_rollout_ref.rollout.multi_turn.tool_config_path is not None
                or config.actor_rollout_ref.rollout.multi_turn.interaction_config_path is not None
            ), (
                "tool_config_path or interaction_config_path must be set when enabling multi_turn with tool, "
                "due to no role-playing support"
            )

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files, self.config.data, self.tokenizer, self.processor
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files, self.config.data, self.tokenizer, self.processor
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    @staticmethod
    def _config_bool(value, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)

    @staticmethod
    def _truncate_debug_text(
        text: str,
        max_chars: int,
        full_text: bool = False,
    ) -> str:
        if full_text:
            return text
        if max_chars <= 0:
            return ""
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + f"... [truncated {len(text) - max_chars} chars]"

    @staticmethod
    def _stable_prompt_digest(token_ids) -> str:
        payload = ",".join(str(int(x)) for x in token_ids)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _trim_token_ids(token_ids, max_len: int):
        ids = [int(x) for x in token_ids]
        if max_len <= 0:
            return {
                "head": [],
                "tail": [],
                "original_len": len(ids),
            }
        if len(ids) <= max_len:
            return ids
        head_len = max_len // 2
        tail_len = max_len - head_len
        return {
            "head": ids[:head_len],
            "tail": ids[-tail_len:] if tail_len > 0 else [],
            "original_len": len(ids),
        }

    def _make_prefix_debug_meta(
        self,
        *,
        tid: str,
        source: str,
        tree_key,
        node_id: Optional[str],
        node,
        augmented_prompt_ids,
        data_instance=None,
        paired_tid: Optional[str] = None,
        debug_context: Optional[dict] = None,
    ) -> dict:
        original_ids = [int(x) for x in tree_key]
        augmented_ids = [int(x) for x in augmented_prompt_ids]
        root_prefix_match = augmented_ids[: len(original_ids)] == original_ids
        locked_prefix_ids = (
            augmented_ids[len(original_ids):] if root_prefix_match else []
        )
        if data_instance is None:
            data_instance = getattr(node, "data_instance", {}) if node is not None else {}
        extra_info = data_instance.get("extra_info", {}) if isinstance(data_instance, dict) else {}
        teacher_suffix_ids = list(getattr(node, "teacher_suffix_ids", []) or [])
        teacher_original_failed_suffix_ids = list(
            getattr(node, "teacher_original_failed_suffix_ids", []) or []
        )
        teacher_suffix_reward_info = getattr(
            node, "teacher_suffix_reward_info", None
        )
        if isinstance(teacher_suffix_reward_info, dict):
            teacher_suffix_finished = teacher_suffix_reward_info.get("finished")
            teacher_suffix_completed = teacher_suffix_reward_info.get("completed")
        else:
            teacher_suffix_finished = getattr(
                teacher_suffix_reward_info, "finished", None
            )
            teacher_suffix_completed = getattr(
                teacher_suffix_reward_info, "completed", None
            )
        meta = {
            "tree_id": tid,
            "source": source,
            "paired_tree_id": paired_tid,
            "prompt_sha1": self._stable_prompt_digest(original_ids),
            "node_id": node_id,
            "parent_id": getattr(node, "parent_id", None),
            "child_count": len(getattr(node, "children", []) or []),
            "node_observations": getattr(node, "observations", None),
            "node_last_k_succ": getattr(node, "last_k_succ", None),
            "node_last_k_total": getattr(node, "last_k_total", None),
            "node_last_used_step": getattr(node, "last_used_step", None),
            "node_last_success_step": getattr(node, "last_success_step", None),
            "node_last_fail_step": getattr(node, "last_fail_step", None),
            "node_created_step": getattr(node, "created_step", None),
            "original_prompt_token_len": len(original_ids),
            "augmented_prompt_token_len": len(augmented_ids),
            "locked_prefix_token_len": len(locked_prefix_ids),
            "teacher_suffix_token_len": len(teacher_suffix_ids),
            "teacher_original_failed_suffix_token_len": len(
                teacher_original_failed_suffix_ids
            ),
            "teacher_suffix_reward": getattr(node, "teacher_suffix_reward", None),
            "teacher_suffix_finished": teacher_suffix_finished,
            "teacher_suffix_completed": teacher_suffix_completed,
            "root_prefix_match": root_prefix_match,
            "original_prompt_ids": original_ids,
            "augmented_prompt_ids": augmented_ids,
            "locked_prefix_ids": locked_prefix_ids,
            "teacher_suffix_ids": teacher_suffix_ids,
            "teacher_original_failed_suffix_ids": teacher_original_failed_suffix_ids,
            "data_source": data_instance.get("data_source") if isinstance(data_instance, dict) else None,
            "reward_model": data_instance.get("reward_model") if isinstance(data_instance, dict) else None,
            "answer": extra_info.get("answer") if isinstance(extra_info, dict) else None,
        }
        if debug_context:
            meta.update(debug_context)
        return meta

    def _decode_prefix_debug_node_summary(
        self,
        summary: dict,
        max_text_chars: int,
        full_text: bool,
    ) -> dict:
        out = {
            k: v for k, v in summary.items()
            if k not in (
                "augmented_prompt_ids",
                "locked_prefix_ids",
                "teacher_suffix_ids",
            )
        }
        locked_prefix_ids = list(summary.get("locked_prefix_ids", []) or [])
        augmented_prompt_ids = list(summary.get("augmented_prompt_ids", []) or [])
        teacher_suffix_ids = list(summary.get("teacher_suffix_ids", []) or [])
        out["locked_prefix"] = self._truncate_debug_text(
            self.tokenizer.decode(locked_prefix_ids, skip_special_tokens=True),
            max_text_chars,
            full_text,
        )
        out["augmented_prompt"] = self._truncate_debug_text(
            self.tokenizer.decode(augmented_prompt_ids, skip_special_tokens=True),
            max_text_chars,
            full_text,
        )
        out["teacher_suffix"] = self._truncate_debug_text(
            self.tokenizer.decode(teacher_suffix_ids, skip_special_tokens=True),
            max_text_chars,
            full_text,
        )
        return out

    def _update_prefix_paired_eval_metrics(
        self,
        metrics: dict,
        group_stats: dict,
        paired_pairs: list,
    ) -> None:
        pair_meta_by_key = {
            (str(m.get("root_tid")), str(m.get("prefix_tid"))): m
            for m in getattr(self, "_prefix_paired_eval_pair_meta", [])
        }
        pair_records = []
        for root_tid, prefix_tid in paired_pairs:
            root_tid = str(root_tid)
            prefix_tid = str(prefix_tid)
            if root_tid not in group_stats or prefix_tid not in group_stats:
                continue
            root_succ, root_total = group_stats[root_tid]
            prefix_succ, prefix_total = group_stats[prefix_tid]
            if root_total <= 0 or prefix_total <= 0:
                continue
            root_rate = root_succ / root_total
            prefix_rate = prefix_succ / prefix_total
            meta = pair_meta_by_key.get((str(root_tid), str(prefix_tid)), {})
            pair_records.append(
                {
                    "root_rate": root_rate,
                    "prefix_rate": prefix_rate,
                    "delta": prefix_rate - root_rate,
                    "root_any": 1.0 if root_succ > 0 else 0.0,
                    "prefix_any": 1.0 if prefix_succ > 0 else 0.0,
                    "root_succ": root_succ,
                    "prefix_succ": prefix_succ,
                    "root_total": root_total,
                    "prefix_total": prefix_total,
                    "prefix_depth_edges": meta.get("prefix_node_depth_edges"),
                    "prefix_depth_tokens": meta.get("prefix_node_depth_tokens"),
                    "prefix_descendant_count": meta.get("prefix_descendant_count"),
                }
            )

        if not pair_records:
            return

        root_rates = [r["root_rate"] for r in pair_records]
        prefix_rates = [r["prefix_rate"] for r in pair_records]
        root_any = [r["root_any"] for r in pair_records]
        prefix_any = [r["prefix_any"] for r in pair_records]
        deltas = [r["delta"] for r in pair_records]
        any_deltas = [p - r for p, r in zip(prefix_any, root_any)]

        metrics["prefix_paired_eval/pairs"] = len(pair_records)
        metrics["prefix_paired_eval/root_response_success_rate"] = float(
            np.mean(root_rates)
        )
        metrics["prefix_paired_eval/prefix_response_success_rate"] = float(
            np.mean(prefix_rates)
        )
        metrics["prefix_paired_eval/prefix_minus_root_response_success"] = float(
            np.mean(deltas)
        )
        metrics["prefix_paired_eval/root_any_success_rate"] = float(np.mean(root_any))
        metrics["prefix_paired_eval/prefix_any_success_rate"] = float(
            np.mean(prefix_any)
        )
        metrics["prefix_paired_eval/prefix_minus_root_any_success"] = float(
            np.mean(any_deltas)
        )
        metrics["prefix_paired_eval/prefix_better_frac"] = float(
            np.mean([d > 0 for d in deltas])
        )
        metrics["prefix_paired_eval/root_better_frac"] = float(
            np.mean([d < 0 for d in deltas])
        )
        metrics["prefix_paired_eval/equal_frac"] = float(
            np.mean([d == 0 for d in deltas])
        )
        metrics["prefix_paired_eval/any_prefix_only_frac"] = float(
            np.mean(
                [
                    r["prefix_any"] > 0 and r["root_any"] == 0
                    for r in pair_records
                ]
            )
        )
        metrics["prefix_paired_eval/any_root_only_frac"] = float(
            np.mean(
                [
                    r["root_any"] > 0 and r["prefix_any"] == 0
                    for r in pair_records
                ]
            )
        )
        metrics["prefix_paired_eval/any_both_success_frac"] = float(
            np.mean(
                [
                    r["root_any"] > 0 and r["prefix_any"] > 0
                    for r in pair_records
                ]
            )
        )
        metrics["prefix_paired_eval/any_both_fail_frac"] = float(
            np.mean(
                [
                    r["root_any"] == 0 and r["prefix_any"] == 0
                    for r in pair_records
                ]
            )
        )

        def _emit_pair_bucket(name, records):
            metrics[f"prefix_paired_eval/{name}_pairs"] = len(records)
            if not records:
                return
            metrics[f"prefix_paired_eval/{name}_root_response_success_rate"] = float(
                np.mean([r["root_rate"] for r in records])
            )
            metrics[f"prefix_paired_eval/{name}_prefix_response_success_rate"] = float(
                np.mean([r["prefix_rate"] for r in records])
            )
            metrics[
                f"prefix_paired_eval/{name}_prefix_minus_root_response_success"
            ] = float(np.mean([r["delta"] for r in records]))
            metrics[f"prefix_paired_eval/{name}_prefix_any_success_rate"] = float(
                np.mean([r["prefix_any"] for r in records])
            )
            metrics[f"prefix_paired_eval/{name}_prefix_better_frac"] = float(
                np.mean([r["delta"] > 0 for r in records])
            )

        _emit_pair_bucket(
            "root_zero",
            [r for r in pair_records if r["root_succ"] == 0],
        )
        _emit_pair_bucket(
            "root_at_most_one_success",
            [r for r in pair_records if r["root_succ"] <= 1],
        )
        _emit_pair_bucket(
            "root_nonzero",
            [r for r in pair_records if r["root_succ"] > 0],
        )

        depth_records = [
            r for r in pair_records if r["prefix_depth_edges"] is not None
        ]
        if not depth_records:
            return
        metrics["prefix_paired_eval/prefix_depth_edges_mean"] = float(
            np.mean([r["prefix_depth_edges"] for r in depth_records])
        )
        depth_tokens = [
            r["prefix_depth_tokens"]
            for r in depth_records
            if r["prefix_depth_tokens"] is not None
        ]
        if depth_tokens:
            metrics["prefix_paired_eval/prefix_depth_tokens_mean"] = float(
                np.mean(depth_tokens)
            )
        metrics["prefix_paired_eval/prefix_depth_ge2_frac"] = float(
            np.mean([r["prefix_depth_edges"] >= 2 for r in depth_records])
        )
        metrics["prefix_paired_eval/prefix_has_descendant_frac"] = float(
            np.mean(
                [(r["prefix_descendant_count"] or 0) > 0 for r in depth_records]
            )
        )
        _emit_pair_bucket(
            "depth1",
            [r for r in depth_records if r["prefix_depth_edges"] == 1],
        )
        _emit_pair_bucket(
            "depth_ge2",
            [r for r in depth_records if r["prefix_depth_edges"] >= 2],
        )

    def _dump_prefix_debug_records(
        self,
        *,
        prefix_debug_by_tid: dict,
        main_chain_batch,
        group_stats: dict,
        tid_to_rows: dict,
        paired_pairs: list,
    ) -> int:
        if not getattr(self, "prefix_debug_dump_enabled", False):
            return 0
        self._last_prefix_debug_dump_deeper_records = 0
        freq = int(getattr(self, "prefix_debug_dump_freq", 50))
        if freq <= 0 or self.global_steps % freq != 0:
            return 0
        if not prefix_debug_by_tid:
            return 0

        dump_dir = getattr(self, "prefix_debug_dump_dir", None)
        if dump_dir in (None, "", "null", "None"):
            base_dir = self.config.trainer.get("default_local_dir", ".")
            dump_dir = os.path.join(str(base_dir), "prefix_debug")
        try:
            os.makedirs(dump_dir, exist_ok=True)
        except Exception as exc:
            print(f"WARNING: failed to create prefix debug dump dir {dump_dir}: {exc}")
            return 0

        max_records = int(getattr(self, "prefix_debug_dump_max_prefixes", 16))
        max_rollouts = int(getattr(self, "prefix_debug_dump_max_rollouts_per_prefix", 8))
        max_text_chars = int(getattr(self, "prefix_debug_dump_max_text_chars", 4000))
        max_deeper_examples = int(
            getattr(self, "prefix_debug_dump_max_deeper_examples", 4)
        )
        full_text = self._config_bool(
            getattr(self, "prefix_debug_dump_full_text", False)
        )
        include_token_ids = self._config_bool(
            getattr(self, "prefix_debug_dump_include_token_ids", False)
        )
        max_token_ids = int(getattr(self, "prefix_debug_dump_max_token_ids", 256))
        if max_records <= 0 and max_deeper_examples <= 0:
            return 0

        pair_lookup = {}
        pair_stats_by_tid = {}
        ordered_tids = []
        seen_tids = set()
        for root_tid, prefix_tid in paired_pairs:
            pair_lookup[str(root_tid)] = {
                "paired_role": "root",
                "paired_tree_id": str(prefix_tid),
            }
            pair_lookup[str(prefix_tid)] = {
                "paired_role": "prefix",
                "paired_tree_id": str(root_tid),
            }
            for tid in (str(prefix_tid), str(root_tid)):
                if tid in prefix_debug_by_tid and tid not in seen_tids:
                    ordered_tids.append(tid)
                    seen_tids.add(tid)

            root_stats = group_stats.get(str(root_tid))
            prefix_stats = group_stats.get(str(prefix_tid))
            if root_stats is not None and prefix_stats is not None:
                root_succ, root_total = root_stats
                prefix_succ, prefix_total = prefix_stats
                if root_total and prefix_total:
                    root_rate = float(root_succ) / float(root_total)
                    prefix_rate = float(prefix_succ) / float(prefix_total)
                    root_any = bool(root_succ > 0)
                    prefix_any = bool(prefix_succ > 0)
                    if prefix_any and not root_any:
                        any_outcome = "prefix_only"
                    elif root_any and not prefix_any:
                        any_outcome = "root_only"
                    elif root_any and prefix_any:
                        any_outcome = "both_success"
                    else:
                        any_outcome = "both_fail"
                    if prefix_rate > root_rate:
                        response_winner = "prefix"
                    elif root_rate > prefix_rate:
                        response_winner = "root"
                    else:
                        response_winner = "equal"
                    pair_payload = {
                        "paired_root_k_succ": int(root_succ),
                        "paired_root_k_total": int(root_total),
                        "paired_prefix_k_succ": int(prefix_succ),
                        "paired_prefix_k_total": int(prefix_total),
                        "paired_root_success_rate": root_rate,
                        "paired_prefix_success_rate": prefix_rate,
                        "paired_prefix_minus_root_success": prefix_rate - root_rate,
                        "paired_root_any_success": root_any,
                        "paired_prefix_any_success": prefix_any,
                        "paired_any_outcome": any_outcome,
                        "paired_response_winner": response_winner,
                    }
                    pair_stats_by_tid[str(root_tid)] = pair_payload
                    pair_stats_by_tid[str(prefix_tid)] = pair_payload
        for tid in prefix_debug_by_tid:
            tid = str(tid)
            if tid not in seen_tids:
                ordered_tids.append(tid)
                seen_tids.add(tid)

        deeper_debug_by_tid = {}
        deeper_ordered_tids = []
        if (
            max_deeper_examples > 0
            and getattr(self, "forest_pool", None) is not None
        ):
            seen_node_ids = {
                meta.get("node_id") for meta in prefix_debug_by_tid.values()
                if meta.get("node_id")
            }
            try:
                deeper_nodes = self.forest_pool.debug_deeper_nodes(
                    max_deeper_examples + len(seen_node_ids),
                    max_children=8,
                    max_lineage=32,
                )
            except Exception as exc:
                print(f"WARNING: failed to collect deeper prefix debug nodes: {exc}")
                deeper_nodes = []
            for tree_key, node_id, node, debug_context in deeper_nodes:
                if len(deeper_ordered_tids) >= max_deeper_examples:
                    break
                if node_id in seen_node_ids:
                    continue
                tid = (
                    "debug_deeper:"
                    f"{self._stable_prompt_digest(list(tree_key))}:"
                    f"{str(node_id)[:12]}"
                )
                deeper_debug_by_tid[tid] = self._make_prefix_debug_meta(
                    tid=tid,
                    source="forest_debug_deeper",
                    tree_key=tree_key,
                    node_id=node_id,
                    node=node,
                    augmented_prompt_ids=node.augmented_prompt_ids,
                    debug_context=debug_context,
                )
                deeper_ordered_tids.append(tid)
                seen_node_ids.add(node_id)

        reward_infos = main_chain_batch.non_tensor_batch.get("__reward_infos__", [])
        if reward_infos is None:
            reward_infos = []
        responses = main_chain_batch.batch.get("responses")
        token_scores = main_chain_batch.batch.get("token_level_scores")
        records = []
        ordered_groups = [
            (ordered_tids, max(0, max_records)),
            (deeper_ordered_tids, max(0, max_deeper_examples)),
        ]
        for tids, budget in ordered_groups:
            written_in_group = 0
            if budget <= 0:
                continue
            for tid in tids:
                if written_in_group >= budget:
                    break
                meta_source = (
                    deeper_debug_by_tid if tid in deeper_debug_by_tid
                    else prefix_debug_by_tid
                )
                if tid not in meta_source:
                    continue
                tid = str(tid)
                meta = meta_source[tid]
                rows = list(tid_to_rows.get(tid, []))
                k_succ, k_total = group_stats.get(tid, (None, None))
                if k_succ is None and meta.get("node_last_k_total") is not None:
                    k_succ = meta.get("node_last_k_succ")
                    k_total = meta.get("node_last_k_total")
                original_ids = meta.get("original_prompt_ids", [])
                augmented_ids = meta.get("augmented_prompt_ids", [])
                locked_prefix_ids = meta.get("locked_prefix_ids", [])
                teacher_suffix_ids = meta.get("teacher_suffix_ids", [])
                teacher_original_failed_suffix_ids = meta.get(
                    "teacher_original_failed_suffix_ids", []
                )
                original_prompt_text = self.tokenizer.decode(
                    original_ids, skip_special_tokens=True
                )
                locked_prefix_text = self.tokenizer.decode(
                    locked_prefix_ids, skip_special_tokens=True
                )
                augmented_prompt_text = self.tokenizer.decode(
                    augmented_ids, skip_special_tokens=True
                )
                teacher_suffix_text = self.tokenizer.decode(
                    teacher_suffix_ids, skip_special_tokens=True
                )
                teacher_full_response_text = (
                    self.tokenizer.decode(
                        list(locked_prefix_ids) + list(teacher_suffix_ids),
                        skip_special_tokens=True,
                    )
                    if teacher_suffix_ids else ""
                )
                teacher_original_failed_suffix_text = self.tokenizer.decode(
                    teacher_original_failed_suffix_ids,
                    skip_special_tokens=True,
                )

                record = {
                    "step": int(self.global_steps),
                    "tree_id": tid,
                    "source": meta.get("source"),
                    "paired_tree_id": meta.get("paired_tree_id"),
                    "paired_role": None,
                    "prompt_sha1": meta.get("prompt_sha1"),
                    "node_id": meta.get("node_id"),
                    "parent_id": meta.get("parent_id"),
                    "child_count": meta.get("child_count"),
                    "node_depth_edges": meta.get("node_depth_edges"),
                    "node_depth_tokens": meta.get("node_depth_tokens"),
                    "descendant_count": meta.get("descendant_count"),
                    "deepest_descendant_depth_edges": meta.get(
                        "deepest_descendant_depth_edges"
                    ),
                    "has_deeper_descendants": meta.get("has_deeper_descendants"),
                    "debug_lineage_truncated": meta.get("debug_lineage_truncated"),
                    "debug_children_truncated": meta.get("debug_children_truncated"),
                    "node_observations": meta.get("node_observations"),
                    "node_last_k_succ": meta.get("node_last_k_succ"),
                    "node_last_k_total": meta.get("node_last_k_total"),
                    "node_last_used_step": meta.get("node_last_used_step"),
                    "node_last_success_step": meta.get("node_last_success_step"),
                    "node_last_fail_step": meta.get("node_last_fail_step"),
                    "node_created_step": meta.get("node_created_step"),
                    "original_prompt_token_len": meta.get("original_prompt_token_len"),
                    "augmented_prompt_token_len": meta.get("augmented_prompt_token_len"),
                    "locked_prefix_token_len": meta.get("locked_prefix_token_len"),
                    "teacher_suffix_token_len": meta.get("teacher_suffix_token_len"),
                    "teacher_original_failed_suffix_token_len": meta.get(
                        "teacher_original_failed_suffix_token_len"
                    ),
                    "teacher_suffix_reward": meta.get("teacher_suffix_reward"),
                    "teacher_suffix_finished": meta.get("teacher_suffix_finished"),
                    "teacher_suffix_completed": meta.get("teacher_suffix_completed"),
                    "root_prefix_match": meta.get("root_prefix_match"),
                    "data_source": meta.get("data_source"),
                    "reward_model": meta.get("reward_model"),
                    "answer": meta.get("answer"),
                    "group_k_succ": k_succ,
                    "group_k_total": k_total,
                    "group_success_rate": (
                        float(k_succ) / float(k_total)
                        if k_succ is not None and k_total else None
                    ),
                    "original_prompt": self._truncate_debug_text(
                        original_prompt_text,
                        max_text_chars,
                        full_text,
                    ),
                    "locked_prefix": self._truncate_debug_text(
                        locked_prefix_text,
                        max_text_chars,
                        full_text,
                    ),
                    "augmented_prompt": self._truncate_debug_text(
                        augmented_prompt_text,
                        max_text_chars,
                        full_text,
                    ),
                    "teacher_suffix": self._truncate_debug_text(
                        teacher_suffix_text,
                        max_text_chars,
                        full_text,
                    ),
                    "teacher_full_response": self._truncate_debug_text(
                        teacher_full_response_text,
                        max_text_chars,
                        full_text,
                    ),
                    "teacher_original_failed_suffix": self._truncate_debug_text(
                        teacher_original_failed_suffix_text,
                        max_text_chars,
                        full_text,
                    ),
                    "debug_lineage": [
                        self._decode_prefix_debug_node_summary(
                            item,
                            max_text_chars,
                            full_text,
                        )
                        for item in (meta.get("debug_lineage") or [])
                    ],
                    "debug_children": [
                        self._decode_prefix_debug_node_summary(
                            item,
                            max_text_chars,
                            full_text,
                        )
                        for item in (meta.get("debug_children") or [])
                    ],
                    "rollouts": [],
                }
                if tid in pair_lookup:
                    record.update(pair_lookup[tid])
                if tid in pair_stats_by_tid:
                    record.update(pair_stats_by_tid[tid])
                if include_token_ids:
                    record["original_prompt_ids_debug"] = self._trim_token_ids(
                        original_ids, max_token_ids
                    )
                    record["locked_prefix_ids_debug"] = self._trim_token_ids(
                        locked_prefix_ids, max_token_ids
                    )
                    record["augmented_prompt_ids_debug"] = self._trim_token_ids(
                        augmented_ids, max_token_ids
                    )
                    record["teacher_suffix_ids_debug"] = self._trim_token_ids(
                        teacher_suffix_ids, max_token_ids
                    )
                    record["teacher_original_failed_suffix_ids_debug"] = (
                        self._trim_token_ids(
                            teacher_original_failed_suffix_ids, max_token_ids
                        )
                    )

                for row in rows[: max(0, max_rollouts)]:
                    reward_info = reward_infos[row] if row < len(reward_infos) else None
                    rollout = {
                        "row": int(row),
                        "score": (
                            float(token_scores[row].sum().detach().cpu().item())
                            if token_scores is not None else None
                        ),
                        "finished": getattr(reward_info, "finished", None),
                        "completed": getattr(reward_info, "completed", None),
                        "response": "",
                        "suffix": "",
                        "full_response": "",
                    }
                    if responses is not None:
                        response_ids = responses[row].detach().cpu().tolist()
                        suffix_text = self.tokenizer.decode(
                            response_ids, skip_special_tokens=True
                        )
                        full_response_text = self.tokenizer.decode(
                            list(locked_prefix_ids) + list(response_ids),
                            skip_special_tokens=True,
                        )
                        suffix_debug_text = self._truncate_debug_text(
                            suffix_text,
                            max_text_chars,
                            full_text,
                        )
                        rollout["response"] = suffix_debug_text
                        rollout["suffix"] = suffix_debug_text
                        rollout["full_response"] = self._truncate_debug_text(
                            full_response_text,
                            max_text_chars,
                            full_text,
                        )
                    record["rollouts"].append(rollout)

                records.append(record)
                written_in_group += 1

        if not records:
            return 0
        self._last_prefix_debug_dump_deeper_records = sum(
            1 for record in records
            if record.get("source") == "forest_debug_deeper"
        )
        filename = os.path.join(dump_dir, f"{self.global_steps}.jsonl")
        try:
            with open(filename, "w") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception as exc:
            print(f"WARNING: failed to write prefix debug dump {filename}: {exc}")
            return 0
        print(f"Dumped prefix debug records to {filename}")
        return len(records)

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        sample_turns = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            if "interaction_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("interaction_kwargs")
            if "agent_name" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("agent_name")
            
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.non_tensor_batch["data_source"] = test_batch.non_tensor_batch["data_source"]
            test_gen_batch.non_tensor_batch["reward_model"] = test_batch.non_tensor_batch["reward_model"]
            test_gen_batch.non_tensor_batch["extra_info"] = test_batch.non_tensor_batch["extra_info"]
            test_gen_batch.non_tensor_batch["uid"] = test_batch.non_tensor_batch["uid"]

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)

            test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            
            # breakpoint()

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            reward_tensor = None
            reward_extra_info = None

            reward_tensor = test_batch.batch["token_level_rewards"]
            if "__reward_infos__" in test_batch.non_tensor_batch:
                reward_infos: list[RewardInfo] = test_batch.non_tensor_batch["__reward_infos__"]
                reward_extra_info = {
                    "finished": [info.finished for info in reward_infos],
                    "completed": [info.completed for info in reward_infos]
                }
                
            extra_infos = test_batch.non_tensor_batch["extra_info"]
            if reward_extra_info is None:
                reward_extra_info = {}
            reward_extra_info["task_id"] = [str(extra_info.get("task_id", "")) for extra_info in extra_infos]

            data_sources = test_batch.non_tensor_batch["data_source"]
            reward_extra_info["data_source"] = [str(ds) for ds in data_sources]

            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            print(f"len reward_extra_infos_dict['reward']: {len(reward_extra_infos_dict['reward'])}")

            if reward_extra_info is not None:
                for key, lst in reward_extra_info.items():
                    reward_extra_infos_dict[key].extend(lst)
                    print(f"len reward_extra_infos_dict['{key}']: {len(reward_extra_infos_dict[key])}")

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(data_src2var2metric2val) > 1:
            metrics_to_average = defaultdict(list)
            
            for metric_key, metric_val in metric_dict.items():
                parts = metric_key.split('/')
                if len(parts) > 2:
                    generic_key = f"{parts[0]}/{'/'.join(parts[2:])}"
                    metrics_to_average[generic_key].append(metric_val)

            for generic_key, values in metrics_to_average.items():
                if len(values) > 1:
                    avg_val = sum(values) / len(values)
                    parts = generic_key.split('/')
                    avg_metric_key = f"{parts[0]}/avg/{'/'.join(parts[1:])}"
                    metric_dict[avg_metric_key] = avg_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def _recoverability_eval(self, output_dir: str):
        """Run recoverability evaluation: branch at every position of failed trajectories.

        Samples from the TRAINING set (where logistic regression is trained)
        to evaluate how well recoverability correlates with relative position.
        """
        import os
        import json
        from torch.utils.data import DataLoader, Subset
        from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"recoverability_step{self.global_steps}.jsonl")

        n_samples = self.config.trainer.get("recoverability_eval_samples", 1000)
        n_samples = min(n_samples, len(self.train_dataset))

        # Fixed random subset from training set — same samples and order across
        # checkpoints for fair comparison. Does NOT touch train_dataloader state.
        g = torch.Generator().manual_seed(42)
        indices = torch.randperm(len(self.train_dataset), generator=g)[:n_samples].tolist()
        val_batch_size = self.config.data.val_batch_size
        if val_batch_size is None:
            val_batch_size = n_samples
        eval_dataloader = DataLoader(
            Subset(self.train_dataset, indices),
            batch_size=val_batch_size,
            shuffle=False,
            collate_fn=default_collate_fn,
            num_workers=0,
            drop_last=False,
        )

        print(f"Running recoverability evaluation on {n_samples} training samples, saving to {output_path}")

        def _default_serializer(obj):
            """Handle numpy types for JSON serialization."""
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        all_records = []
        for batch_data in eval_dataloader:
            batch = DataProto.from_single_dict(batch_data)

            batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
            )

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            for key in ["multi_modal_data", "raw_prompt", "tools_kwargs", "interaction_kwargs", "agent_name"]:
                if key in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append(key)

            gen_batch = batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            gen_batch.non_tensor_batch["data_source"] = batch.non_tensor_batch["data_source"]
            gen_batch.non_tensor_batch["reward_model"] = batch.non_tensor_batch["reward_model"]
            gen_batch.non_tensor_batch["extra_info"] = batch.non_tensor_batch["extra_info"]
            gen_batch.non_tensor_batch["uid"] = batch.non_tensor_batch["uid"]

            gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": True,  # Must sample for diverse branches
                "global_steps": self.global_steps,
            }

            records = self.async_rollout_manager.recoverability_eval_sequences(gen_batch)
            all_records.extend(records)

        # Single-process write — no concurrent file access issues.
        with open(output_path, "w") as f:
            for record in all_records:
                f.write(json.dumps(record, default=_default_serializer) + "\n")

        print(f"Recoverability evaluation complete: {len(all_records)} records saved to {output_path}")

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
                profile_option=self.config.trainer.npu_profile.options,
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
                profile_option=self.config.trainer.npu_profile.options,
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.trainer, "profile_steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.trainer, "profile_steps")
            assert OmegaConf.select(self.config.trainer, "worker_nsight_options") is not None, (
                "worker_nsight_options must be set when profile_steps is set"
            )
            wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                OmegaConf.select(self.config.trainer, "worker_nsight_options")
            )

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                device_name=self.device_name,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config,
                worker_group=self.actor_rollout_wg,
            )

        # Initialize branch point buffer for one-stage mode
        self.one_stage_mode = False
        if self.async_rollout_mode:
            self.one_stage_mode = self.config.actor_rollout_ref.rollout.deep_grpo.get("one_stage_mode", False)
        if self.one_stage_mode:
            # When teacher synthesis is disabled, expand_branch_chain must be True
            # so branch points are collected from failed trajectories.
            # When teacher synthesis is enabled, branch points come from the teacher pool
            # so expand_branch_chain can be False.
            ts_config_check = self.config.actor_rollout_ref.rollout.deep_grpo.get("teacher_suffix_synthesis", {})
            pim_config_check = self.config.actor_rollout_ref.rollout.deep_grpo.get("prefix_inject_mode", {})
            if not ts_config_check.get("enabled", False) and not pim_config_check.get("enabled", False):
                assert self.config.actor_rollout_ref.rollout.deep_grpo.expand_branch_chain, (
                    "one_stage_mode=True requires expand_branch_chain=True (or teacher_suffix_synthesis.enabled=True "
                    "or prefix_inject_mode.enabled=True), otherwise no branch points are collected and the buffer stays empty forever."
                )

            from recipe.deep_grpo.pools.branch_point_buffer import BranchPointBuffer
            bp_config = self.config.actor_rollout_ref.rollout.deep_grpo.get("branch_point_buffer", {})

            branches_per_entry = bp_config.get("branches_per_entry", 4)
            assert branches_per_entry >= 2, (
                f"branches_per_entry={branches_per_entry} must be >= 2, "
                "otherwise all advantages are 0 and all auxiliary chain samples get filtered out."
            )

            self.branch_point_buffer = BranchPointBuffer(
                max_size=bp_config.get("max_size", 10000),
                max_age=bp_config.get("max_age", 3),
                max_model_len=self.config.actor_rollout_ref.rollout.max_model_len,
            )
            self.buffer_sample_size = bp_config.get("sample_size_per_step", 128)
            self.buffer_branches_per_entry = branches_per_entry
            self.buffer_sampling_strategy = bp_config.get("sampling_strategy", "stratified")

        # Resolve mode flags first and check mutual exclusivity BEFORE either
        # init block runs. The init blocks create daemon threads and other
        # resources; if the assertion fired AFTER teacher_synthesis had already
        # started its worker thread, we'd leak the thread.
        self.teacher_synthesis_enabled = False
        self.prefix_inject_enabled = False
        # Sentinel defaults for chain-mode attributes. These are overwritten
        # when prefix_inject_mode.enabled=True; we initialise here to avoid
        # AttributeError if any code path ever reads them while prefix_inject
        # is disabled (defensive — all current accesses are guarded).
        self.prefix_inject_pool_type = None
        self.chain_pool = None
        self.forest_pool = None
        self._chain_tree_id_to_prompt_key: dict = {}
        self._forest_tree_id_to_node: dict = {}
        self._forest_sft_tree_id_to_node: dict = {}
        self._forest_luffy_tree_id_to_node: dict = {}
        self.prefix_forest_luffy_enabled = False
        self.prefix_forest_teacher_loss_type = None
        self.prefix_forest_snis_beta = 1.0
        self.prefix_suffix_sft_enabled = False
        self.prefix_suffix_sft_schedule = "post_step"
        self.prefix_suffix_sft_epoch_passes = 5
        self.prefix_suffix_sft_max_batches_per_step = 1
        self.prefix_suffix_sft_max_nodes_per_tree_per_step = 1
        self.teacher_allow_partial_batch = self._config_bool(
            self.config.actor_rollout_ref.actor.deep_grpo.get(
                "teacher_allow_partial_batch", False
            )
        )
        if self.async_rollout_mode:
            ts_config = self.config.actor_rollout_ref.rollout.deep_grpo.get("teacher_suffix_synthesis", {})
            self.teacher_synthesis_enabled = ts_config.get("enabled", False)
            pim_config = self.config.actor_rollout_ref.rollout.deep_grpo.get("prefix_inject_mode", {})
            self.prefix_inject_enabled = pim_config.get("enabled", False)
            assert not (self.teacher_synthesis_enabled and self.prefix_inject_enabled), (
                "teacher_suffix_synthesis.enabled and prefix_inject_mode.enabled "
                "are mutually exclusive; both consume failed_trajectory_pool."
            )

        # Initialize teacher suffix synthesis (background teacher annotation)
        if self.teacher_synthesis_enabled:
            assert self.one_stage_mode, (
                "teacher_suffix_synthesis.enabled=True requires one_stage_mode=True, "
                "because failed trajectory collection happens in the one-stage pipeline."
            )
            assert not self.config.actor_rollout_ref.rollout.deep_grpo.expand_branch_chain, (
                "teacher_suffix_synthesis.enabled=True requires expand_branch_chain=False, "
                "because branch points come from teacher analysis, not from partition+branching strategy."
            )
            from recipe.deep_grpo.pools.failed_trajectory_pool import FailedTrajectoryPool
            from recipe.deep_grpo.pools.teacher_annotated_pool import TeacherAnnotatedPool
            from recipe.deep_grpo.teacher_worker import TeacherAnnotationWorker

            self.failed_trajectory_pool = FailedTrajectoryPool(
                max_size=ts_config.get("failed_pool_max_size", 10000),
                max_age=ts_config.get("failed_pool_max_age", 5),
            )
            self.teacher_annotated_pool = TeacherAnnotatedPool(
                max_size=ts_config.get("annotated_pool_max_size", 10000),
            )
            self.teacher_branch_batch_threshold = ts_config.get("branch_batch_threshold", 128)
            self.teacher_branches_per_entry = ts_config.get("branches_per_entry", 4)
            teacher_loss_type = self.config.actor_rollout_ref.actor.deep_grpo.get("teacher_loss_type", None)
            if teacher_loss_type == "sft":
                assert self.teacher_branches_per_entry >= 0, (
                    f"teacher branches_per_entry={self.teacher_branches_per_entry} must be >= 0 for SFT mode."
                )
            else:
                assert self.teacher_branches_per_entry >= 1, (
                    f"teacher branches_per_entry={self.teacher_branches_per_entry} must be >= 1 for LUFFY mode, "
                    "otherwise the group only has the teacher suffix and all advantages are 0."
                )
            self.teacher_sampling_strategy = ts_config.get("sampling_strategy", "stratified")

            # Determine agent loop class for teacher worker
            default_agent = self.config.actor_rollout_ref.rollout.deep_grpo.get(
                "default_agent_name", "reasoning_agent_loop"
            )
            if default_agent == "reasoning_agent_loop":
                from recipe.deep_grpo.agent_loop.reasoning_agent_loop import ReasoningAgentLoop
                agent_loop_class = ReasoningAgentLoop
            elif default_agent == "treerl_agent_loop":
                from recipe.deep_grpo.agent_loop.treerl_agent_loop import TreeRLAgentLoop
                agent_loop_class = TreeRLAgentLoop
            else:
                raise ValueError(f"Unknown agent_name for teacher worker: {default_agent}")

            # Note: init_class will be called automatically when the background
            # thread first creates an agent_loop instance (via __init__).
            # We call it here too for fail-fast config validation.
            agent_loop_class.init_class(self.config, self.tokenizer)

            self.teacher_worker = TeacherAnnotationWorker(
                config=ts_config,
                tokenizer=self.tokenizer,
                failed_pool=self.failed_trajectory_pool,
                annotated_pool=self.teacher_annotated_pool,
                agent_loop_class=agent_loop_class,
                agent_loop_config=self.config,
            )
            self.teacher_worker.start()

            # Teacher comparison table logging config
            self.teacher_comparison_log_freq = ts_config.get("comparison_log_freq", 20)
            self.teacher_comparison_log_samples = ts_config.get("comparison_log_samples", 20)
            columns = ["step", "tree_id", "prompt", "prefix",
                        "failed_suffix", "failed_full_response",
                        "teacher_suffix", "teacher_full_response",
                        "teacher_reward"]
            for k in range(1, self.teacher_branches_per_entry + 1):
                columns.extend([f"branch_{k}_suffix", f"branch_{k}_reward"])
            self._teacher_comparison_columns = columns

        # Initialize prefix_inject_mode. Mutual exclusivity with
        # teacher_synthesis_enabled is already asserted above.
        if self.prefix_inject_enabled:
            debug_cfg = pim_config.get("debug_dump", {}) or {}
            self.prefix_debug_dump_enabled = self._config_bool(
                debug_cfg.get("enabled", False)
            )
            self.prefix_debug_dump_dir = debug_cfg.get("dir", None)
            self.prefix_debug_dump_freq = int(debug_cfg.get("freq", 50))
            self.prefix_debug_dump_max_prefixes = int(
                debug_cfg.get("max_prefixes", 16)
            )
            self.prefix_debug_dump_max_deeper_examples = int(
                debug_cfg.get("max_deeper_examples", 4)
            )
            self.prefix_debug_dump_max_rollouts_per_prefix = int(
                debug_cfg.get("max_rollouts_per_prefix", 8)
            )
            self.prefix_debug_dump_max_text_chars = int(
                debug_cfg.get("max_text_chars", 4000)
            )
            self.prefix_debug_dump_full_text = self._config_bool(
                debug_cfg.get("full_text", False)
            )
            self.prefix_debug_dump_include_token_ids = self._config_bool(
                debug_cfg.get("include_token_ids", False)
            )
            self.prefix_debug_dump_max_token_ids = int(
                debug_cfg.get("max_token_ids", 256)
            )
            assert self.one_stage_mode, (
                "prefix_inject_mode requires one_stage_mode=True for the "
                "failed trajectory collection path."
            )

            injection_ratio = pim_config.get("injection_ratio", 0.25)
            assert 0.0 < injection_ratio <= 1.0, (
                f"prefix_inject_mode.injection_ratio must be in (0.0, 1.0], "
                f"got {injection_ratio}"
            )
            raw_prefix_rollout_n = pim_config.get("rollout_n", None)
            self.prefix_rollout_n = (
                int(self.config.actor_rollout_ref.rollout.n)
                if raw_prefix_rollout_n is None
                else int(raw_prefix_rollout_n)
            )
            assert self.prefix_rollout_n > 1, (
                "prefix_inject_mode.rollout_n must be > 1, got "
                f"{self.prefix_rollout_n}"
            )
            suffix_sft_cfg = pim_config.get("suffix_sft_maturation", {}) or {}
            self.prefix_suffix_sft_enabled = self._config_bool(
                suffix_sft_cfg.get("enabled", False)
            )
            self.prefix_suffix_sft_schedule = str(
                suffix_sft_cfg.get("schedule", "post_step")
            ).strip().lower()
            assert self.prefix_suffix_sft_schedule in (
                "post_step",
                "epoch_local",
            ), (
                "prefix_inject_mode.suffix_sft_maturation.schedule must be "
                "'post_step' or 'epoch_local', got "
                f"{self.prefix_suffix_sft_schedule}"
            )
            self.prefix_suffix_sft_epoch_passes = int(
                suffix_sft_cfg.get("epoch_passes", 5)
            )
            self.prefix_suffix_sft_max_batches_per_step = int(
                suffix_sft_cfg.get("max_batches_per_step", 1)
            )
            self.prefix_suffix_sft_max_nodes_per_tree_per_step = int(
                suffix_sft_cfg.get("max_nodes_per_tree_per_step", 1)
            )
            teacher_continuation_cfg = (
                pim_config.get("teacher_continuation", {}) or {}
            )
            self.prefix_forest_luffy_enabled = self._config_bool(
                teacher_continuation_cfg.get("enabled", False)
            )
            if self.prefix_suffix_sft_enabled:
                assert self.prefix_suffix_sft_max_batches_per_step > 0, (
                    "prefix_inject_mode.suffix_sft_maturation.max_batches_per_step "
                    "must be > 0, got "
                    f"{self.prefix_suffix_sft_max_batches_per_step}"
                )
                assert self.prefix_suffix_sft_max_nodes_per_tree_per_step > 0, (
                    "prefix_inject_mode.suffix_sft_maturation."
                    "max_nodes_per_tree_per_step must be > 0, got "
                    f"{self.prefix_suffix_sft_max_nodes_per_tree_per_step}"
                )
                assert self.prefix_suffix_sft_epoch_passes > 0, (
                    "prefix_inject_mode.suffix_sft_maturation.epoch_passes "
                    "must be > 0 when suffix SFT is enabled, got "
                    f"{self.prefix_suffix_sft_epoch_passes}"
                )
                teacher_update_mode = self.config.actor_rollout_ref.actor.deep_grpo.get(
                    "teacher_update_mode", "joint"
                )
                assert teacher_update_mode == "post_ppo", (
                    "prefix_inject_mode.suffix_sft_maturation.enabled=True "
                    "requires actor_rollout_ref.actor.deep_grpo.teacher_update_mode=post_ppo"
                )
            # pool_type:
            #   "flat"   legacy SyntheticPromptPool
            #   "chain"  per-prompt linear PrefixChainPool
            #   "forest" hard-state verified-prefix tree pool
            self.prefix_inject_pool_type = pim_config.get("pool_type", "flat")
            assert self.prefix_inject_pool_type in ("flat", "chain", "forest"), (
                f"prefix_inject_mode.pool_type must be 'flat', 'chain', or 'forest', "
                f"got {self.prefix_inject_pool_type}"
            )
            if self.prefix_suffix_sft_enabled:
                assert self.prefix_inject_pool_type == "forest", (
                    "prefix_inject_mode.suffix_sft_maturation.enabled=True "
                    "requires prefix_inject_mode.pool_type=forest"
                )
            if self.prefix_forest_luffy_enabled:
                actor_deep_grpo_cfg = self.config.actor_rollout_ref.actor.deep_grpo
                assert self.prefix_inject_pool_type == "forest", (
                    "prefix_inject_mode.teacher_continuation.enabled=True "
                    "requires prefix_inject_mode.pool_type=forest"
                )
                assert not self.prefix_suffix_sft_enabled, (
                    "prefix_inject_mode.teacher_continuation.enabled=True is "
                    "mutually exclusive with suffix_sft_maturation.enabled=True"
                )
                assert actor_deep_grpo_cfg.get("teacher_update_mode", "joint") == "joint", (
                    "prefix forest LUFFY requires teacher_update_mode=joint"
                )
                self.prefix_forest_teacher_loss_type = actor_deep_grpo_cfg.get(
                    "teacher_loss_type", None
                )
                assert self.prefix_forest_teacher_loss_type in (
                    "luffy",
                    "snis",
                    None,
                ), (
                    "prefix forest teacher continuation requires "
                    "teacher_loss_type in ('luffy', 'snis') or null, got "
                    f"{self.prefix_forest_teacher_loss_type}"
                )
                self.prefix_forest_snis_beta = float(
                    actor_deep_grpo_cfg.get("snis_beta", 1.0)
                )
                assert (
                    actor_deep_grpo_cfg.get(
                        "teacher_loss_reduction",
                        "separate_stream_mean",
                    )
                    == "mixed_token_mean"
                ), (
                    "prefix forest LUFFY requires "
                    "teacher_loss_reduction=mixed_token_mean"
                )
                assert not self._config_bool(
                    self.config.trainer.get("whiten_advantages", False)
                ), (
                    "prefix forest LUFFY requires trainer.whiten_advantages=False"
                )
            self.prefix_inject_ratio = injection_ratio

            from recipe.deep_grpo.teacher_worker import TeacherAnnotationWorker

            # Reuse teacher_suffix_synthesis sub-config for teacher worker params
            # (max_concurrent, poll_interval, min_prefix_match_*). prefix_inject_mode
            # reuses the same teacher worker infrastructure.
            ts_config = self.config.actor_rollout_ref.rollout.deep_grpo.get(
                "teacher_suffix_synthesis", {}
            )

            # Resolve agent loop class (shared by flat, chain, and forest paths).
            default_agent = self.config.actor_rollout_ref.rollout.deep_grpo.get(
                "default_agent_name", "reasoning_agent_loop"
            )
            if default_agent == "reasoning_agent_loop":
                from recipe.deep_grpo.agent_loop.reasoning_agent_loop import ReasoningAgentLoop
                agent_loop_class = ReasoningAgentLoop
            elif default_agent == "treerl_agent_loop":
                from recipe.deep_grpo.agent_loop.treerl_agent_loop import TreeRLAgentLoop
                agent_loop_class = TreeRLAgentLoop
            else:
                raise ValueError(f"Unknown agent_name for teacher worker: {default_agent}")
            agent_loop_class.init_class(self.config, self.tokenizer)

            if self.prefix_inject_pool_type == "chain":
                # Chain pool path: no failed_trajectory_pool, no annotated_pool,
                # no synthetic_prompt_pool. Teacher pulls deepening requests
                # directly from the chain pool and writes responses back to it.
                from recipe.deep_grpo.pools.prefix_chain_pool import PrefixChainPool

                self.chain_pool = PrefixChainPool()
                # Kept None so checkpoint save/load and other branches can
                # detect chain mode via attribute presence.
                self.failed_trajectory_pool = None
                self.teacher_annotated_pool = None
                self.synthetic_prompt_pool = None

                self.teacher_worker = TeacherAnnotationWorker(
                    config=ts_config,
                    tokenizer=self.tokenizer,
                    failed_pool=None,
                    annotated_pool=None,
                    agent_loop_class=agent_loop_class,
                    agent_loop_config=self.config,
                    chain_pool=self.chain_pool,
                )
                self.teacher_worker.start()

                # Map tree_id → prompt_key, populated each step when we sample
                # chain entries and sent down into rollout. Used to route
                # post-rollout outcomes (record_observation) back to the
                # correct chain.
                self._chain_tree_id_to_prompt_key: dict = {}
                self.forest_pool = None
            elif self.prefix_inject_pool_type == "forest":
                # Forest pool path: hard-state replay buffer. Teacher expands
                # failed rollouts into children; all-success rollout deactivates
                # only the current node.
                from recipe.deep_grpo.pools.prefix_forest_pool import PrefixForestPool

                self.forest_pool = PrefixForestPool(
                    max_model_len=self.config.actor_rollout_ref.rollout.max_model_len,
                )
                self.chain_pool = None
                self.failed_trajectory_pool = None
                self.teacher_annotated_pool = None
                self.synthetic_prompt_pool = None

                self.teacher_worker = TeacherAnnotationWorker(
                    config=ts_config,
                    tokenizer=self.tokenizer,
                    failed_pool=None,
                    annotated_pool=None,
                    agent_loop_class=agent_loop_class,
                    agent_loop_config=self.config,
                    forest_pool=self.forest_pool,
                )
                self.teacher_worker.start()

                # Map rollout tree_id → (forest tree_key, node_id), populated
                # each step for post-rollout writeback.
                self._forest_tree_id_to_node: dict = {}
                # Map teacher-only SFT tree_id → (forest tree_key, node_id).
                self._forest_sft_tree_id_to_node: dict = {}
            else:
                # Flat pool path (legacy; preserved for ablations).
                from recipe.deep_grpo.pools.failed_trajectory_pool import FailedTrajectoryPool
                from recipe.deep_grpo.pools.teacher_annotated_pool import TeacherAnnotatedPool
                from recipe.deep_grpo.pools.synthetic_prompt_pool import SyntheticPromptPool

                max_per_prompt = pim_config.get("max_per_prompt", 3)
                assert max_per_prompt > 0, (
                    f"prefix_inject_mode.max_per_prompt must be > 0, got {max_per_prompt}"
                )

                self.chain_pool = None
                self.forest_pool = None

                # Note: max_size/max_age are passed for legacy-mode signature
                # compatibility but ignored in priority_mode. Priority mode lets
                # the pool grow unboundedly by design (memory per entry is small
                # and the round-robin scheduler keeps old entries deprioritized).
                self.failed_trajectory_pool = FailedTrajectoryPool(
                    max_size=ts_config.get("failed_pool_max_size", 10000),
                    max_age=ts_config.get("failed_pool_max_age", 5),
                    priority_mode=True,
                )
                self.teacher_annotated_pool = TeacherAnnotatedPool(
                    max_size=ts_config.get("annotated_pool_max_size", 10000),
                )
                curriculum_cfg = self.config.data.get("curriculum", {})
                w_floor_hard = float(curriculum_cfg.get("w_floor_hard", 0.2))
                w_floor_mastered = float(curriculum_cfg.get("w_floor_mastered", 0.01))
                self.synthetic_prompt_pool = SyntheticPromptPool(
                    max_per_prompt=max_per_prompt,
                    w_floor_hard=w_floor_hard,
                    w_floor_mastered=w_floor_mastered,
                )

                self.teacher_worker = TeacherAnnotationWorker(
                    config=ts_config,
                    tokenizer=self.tokenizer,
                    failed_pool=self.failed_trajectory_pool,
                    annotated_pool=self.teacher_annotated_pool,
                    agent_loop_class=agent_loop_class,
                    agent_loop_config=self.config,
                    synthetic_pool=self.synthetic_prompt_pool,  # <-- gating flag
                )
                self.teacher_worker.start()

    def _record_synthetic_usage(
        self,
        main_chain_batch,
        synth_tree_id_to_entry: dict,
        metrics: dict,
    ):
        """Write back per-entry succ/total stats to the synthetic prompt pool.

        For each synthetic entry injected this step, find the K main_chain_batch
        rows whose tree_id matches the one we minted for it, count rows with
        reward strictly above `low_quality_trajectory_reward_threshold` as
        successes (same threshold _run_synthetic_entry uses to define "failed"),
        and call synthetic_prompt_pool.record_usage.
        """
        tree_ids_col = main_chain_batch.non_tensor_batch.get("__tree_ids__")
        token_scores = main_chain_batch.batch.get("token_level_scores")
        if tree_ids_col is None or token_scores is None:
            metrics["synthetic_inject/recorded_entries"] = 0
            return
        rewards = token_scores.sum(dim=-1).cpu().tolist()

        # Group main_chain_batch row indices by their tree_id.
        grouped: dict = {}
        for idx, tid in enumerate(tree_ids_col):
            grouped.setdefault(str(tid), []).append(idx)

        threshold = self.config.actor_rollout_ref.rollout.deep_grpo.get(
            "low_quality_trajectory_reward_threshold", 0.0
        )

        recorded = 0
        escape_groups = 0
        for tree_id, entry in synth_tree_id_to_entry.items():
            row_indices = grouped.get(tree_id)
            if not row_indices:
                continue
            k_total = len(row_indices)
            k_succ = sum(1 for i in row_indices if rewards[i] > threshold)
            self.synthetic_prompt_pool.record_usage(
                entry=entry,
                k_successes=k_succ,
                k_total=k_total,
                current_step=self.global_steps,
            )
            recorded += 1
            if k_succ > 0:
                escape_groups += 1

        metrics["synthetic_inject/recorded_entries"] = recorded
        if recorded > 0:
            metrics["synthetic_inject/escape_rate_groups"] = escape_groups / recorded

    def _global_teacher_mini_batch_size(self) -> int:
        size = int(
            self.config.actor_rollout_ref.actor.deep_grpo.get(
                "teacher_mini_batch_size",
                self.config.actor_rollout_ref.actor.ppo_mini_batch_size,
            )
        )
        assert size > 0, (
            "actor_rollout_ref.actor.deep_grpo.teacher_mini_batch_size "
            f"must be > 0, got {size}"
        )
        return size

    def _make_forest_suffix_sft_entry(
        self,
        tree_id: str,
        tree_key,
        node,
    ):
        """Convert a forest node's stored teacher suffix into a teacher entry."""
        from recipe.deep_grpo.protocol import BranchPointEntry, TeacherSuffix
        from recipe.deep_grpo.teacher_suffix_utils import append_eos_if_missing

        original_prompt_ids = list(tree_key)
        augmented_prompt_ids = list(getattr(node, "augmented_prompt_ids", []) or [])
        assert augmented_prompt_ids[:len(original_prompt_ids)] == original_prompt_ids, (
            "forest node augmented_prompt_ids must start with tree_key"
        )
        locked_prefix_ids = augmented_prompt_ids[len(original_prompt_ids):]
        suffix_ids = list(getattr(node, "teacher_suffix_ids", []) or [])
        assert suffix_ids, "forest suffix SFT entry requires teacher_suffix_ids"
        suffix_mask = list(getattr(node, "teacher_suffix_mask", []) or [])
        if len(suffix_mask) != len(suffix_ids):
            suffix_mask = [1] * len(suffix_ids)
        suffix_ids, suffix_mask, _ = append_eos_if_missing(
            suffix_ids,
            suffix_mask,
            self.tokenizer,
        )
        reward = getattr(node, "teacher_suffix_reward", None)
        if reward is None:
            reward = 1.0
        reward_info = getattr(node, "teacher_suffix_reward_info", None)
        if isinstance(reward_info, dict):
            reward_info = RewardInfo(
                reward=float(reward_info.get("reward", reward)),
                completed=int(reward_info.get("completed", 1)),
                finished=int(reward_info.get("finished", 1)),
                judgement_reply=reward_info.get("judgement_reply"),
            )
        if reward_info is None:
            reward_info = RewardInfo(
                reward=float(reward),
                completed=1,
                finished=1,
            )
        teacher_suffix = TeacherSuffix(
            suffix_ids=suffix_ids,
            suffix_mask=suffix_mask,
            reward=float(reward),
            reward_info=reward_info,
            original_failed_suffix_ids=list(
                getattr(node, "teacher_original_failed_suffix_ids", []) or []
            ),
        )
        return BranchPointEntry(
            prompt_ids=original_prompt_ids,
            response_ids=locked_prefix_ids,
            response_mask=[1] * len(locked_prefix_ids),
            data_instance=deepcopy(getattr(node, "data_instance", {}) or {}),
            num_turns=float(getattr(node, "num_turns", 1.0)),
            tree_id=tree_id,
            branch_chain_root_index=0,
            chain_total_length=1,
            agent_name=getattr(node, "agent_name", "") or "",
            teacher_suffix=teacher_suffix,
        )

    @staticmethod
    def _forest_teacher_reward_info(node, reward: float):
        reward_info = getattr(node, "teacher_suffix_reward_info", None)
        if isinstance(reward_info, dict):
            return RewardInfo(
                reward=float(reward_info.get("reward", reward)),
                completed=int(reward_info.get("completed", 1)),
                finished=int(reward_info.get("finished", 1)),
                judgement_reply=reward_info.get("judgement_reply"),
            )
        if reward_info is None:
            return RewardInfo(
                reward=float(reward),
                completed=1,
                finished=1,
            )
        return reward_info

    def _build_prefix_forest_luffy_teacher_batch(self, entries) -> DataProto:
        """Build off-policy teacher continuation rows for mixed-token LUFFY."""
        if not entries:
            return DataProto()

        from recipe.deep_grpo.teacher_suffix_utils import append_eos_if_missing
        from tensordict import TensorDict

        prompts = []
        responses = []
        response_masks = []
        rewards = []
        advantages = []
        tree_ids = []
        node_ids = []
        reward_infos = []

        for entry in entries:
            node = entry["node"]
            suffix_ids = list(getattr(node, "teacher_suffix_ids", []) or [])
            assert suffix_ids, "prefix forest LUFFY teacher row requires suffix ids"
            suffix_mask = list(getattr(node, "teacher_suffix_mask", []) or [])
            if len(suffix_mask) != len(suffix_ids):
                suffix_mask = [1] * len(suffix_ids)
            suffix_ids, suffix_mask, _ = append_eos_if_missing(
                suffix_ids,
                suffix_mask,
                self.tokenizer,
            )
            prompts.append(list(getattr(node, "augmented_prompt_ids", []) or []))
            responses.append(suffix_ids)
            response_masks.append(suffix_mask)
            rewards.append(float(entry["teacher_reward"]))
            advantages.append(float(entry["teacher_advantage"]))
            tree_ids.append(str(entry["tree_id"]))
            node_ids.append(f"prefix_luffy:{entry['node_id']}")
            reward_infos.append(
                self._forest_teacher_reward_info(
                    node,
                    float(entry["teacher_reward"]),
                )
            )

        self.tokenizer.padding_side = "left"
        prompt_outputs = self.tokenizer.pad(
            [{"input_ids": ids} for ids in prompts],
            padding="max_length",
            max_length=self.config.actor_rollout_ref.rollout.max_model_len,
            return_tensors="pt",
            return_attention_mask=True,
        )
        prompt_ids = prompt_outputs["input_ids"]
        prompt_attention_mask = prompt_outputs["attention_mask"]

        self.tokenizer.padding_side = "right"
        response_outputs = self.tokenizer.pad(
            [{"input_ids": ids} for ids in responses],
            padding="max_length",
            max_length=self.config.actor_rollout_ref.rollout.max_model_len,
            return_tensors="pt",
            return_attention_mask=True,
        )
        response_ids = response_outputs["input_ids"]
        response_attention_mask = response_outputs["attention_mask"]

        mask_outputs = self.tokenizer.pad(
            [{"input_ids": ids} for ids in response_masks],
            padding="max_length",
            max_length=self.config.actor_rollout_ref.rollout.max_model_len,
            return_tensors="pt",
            return_attention_mask=False,
        )
        response_mask = mask_outputs["input_ids"] * response_attention_mask
        assert response_ids.shape == response_mask.shape, (
            "mismatch in prefix LUFFY response_ids and response_mask shape: "
            f"{response_ids.shape} vs {response_mask.shape}"
        )

        input_ids = torch.cat([prompt_ids, response_ids], dim=1)
        attention_mask = torch.cat(
            [prompt_attention_mask, response_attention_mask],
            dim=1,
        )
        position_ids = (attention_mask.cumsum(dim=1) - 1) * attention_mask

        token_scores = torch.zeros_like(response_ids, dtype=torch.float32)
        for i, reward in enumerate(rewards):
            valid_response_len = int(response_attention_mask[i].sum().item())
            assert valid_response_len > 0, (
                "prefix LUFFY teacher response length must be > 0"
            )
            token_scores[i, valid_response_len - 1] = float(reward)

        # NOTE: for teacher_loss_type="snis" this tensor carries the SNIS
        # weight w̃ (typical range (0, e^{1/β}]), NOT an advantage. It reuses
        # the advantages/returns transport so it survives shuffle/concat into
        # dp_actor unchanged. Downstream data metrics will report it under
        # "advantage" names — expected, harmless.
        advantage_tensor = torch.tensor(
            advantages,
            dtype=torch.float32,
        ).unsqueeze(-1) * response_mask

        batch = TensorDict(
            {
                "prompts": prompt_ids,
                "responses": response_ids,
                "response_mask": response_mask,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "token_level_scores": token_scores,
                "token_level_rewards": token_scores,
                "advantages": advantage_tensor,
                "returns": advantage_tensor,
            },
            batch_size=len(entries),
        )
        return DataProto(
            batch=batch,
            non_tensor_batch={
                "__num_turns__": np.array(
                    [
                        float(getattr(entry["node"], "num_turns", 1.0)) + 1.0
                        for entry in entries
                    ],
                    dtype=np.int32,
                ),
                "__tree_ids__": np.array(tree_ids, dtype=object),
                "__node_ids__": np.array(node_ids, dtype=object),
                "__reward_infos__": np.array(reward_infos, dtype=object),
            },
            meta_info={
                "metrics": [{} for _ in entries],
                "temperature": self.config.actor_rollout_ref.rollout.temperature,
            },
        )

    def _attach_prefix_forest_luffy_teacher_continuations(
        self,
        main_chain_batch: DataProto,
        metrics: dict,
    ) -> DataProto:
        """Add verified forest teacher suffixes as mixed-policy teacher rows.

        Per-group flow (group = student rollouts sharing a tree_id + 1 teacher):
        1. Gate: teacher row must have a verified reward above the low-quality
           threshold (declaration ④).
        2. Student advantages use the STUDENT-ONLY baseline (standard GRPO),
           NOT a teacher-augmented baseline. Folding the teacher (R=1) into the
           student baseline gives all-fail groups a small nonzero advantage that
           slips past the downstream zero-advantage filter and injects
           low-quality gradients; the student-only baseline keeps GRPO's
           sum-to-zero so all-fail/all-correct student groups get advantage 0
           and are correctly filtered.
        3. The teacher row's coefficient uses the MIXED baseline
           mean(students + teacher), so the teacher keeps its signal exactly at
           the all-fail nodes where students contribute none:
           - "luffy": A* = R* − mixed_baseline (consumed as f(π)·A in dp_actor)
           - "snis":  w̃ = exp(A*/β) / mean_j exp(A_j/β) (consumed as
             −sg[w̃]·logπ in dp_actor; bounded by e^{ΔR_max/β}, baseline-shift
             invariant, mean-normalized to compose with token-mean without
             double dilution)
           - null:    no teacher row (pure state-curriculum ablation arm)
        """
        metrics["prefix_luffy/enabled"] = int(self.prefix_forest_luffy_enabled)
        if (
            not self.prefix_forest_luffy_enabled
            or not self._forest_luffy_tree_id_to_node
        ):
            return DataProto()

        candidates = self._forest_luffy_tree_id_to_node
        metrics["prefix_luffy/candidate_groups"] = len(candidates)
        tree_ids_col = main_chain_batch.non_tensor_batch.get("__tree_ids__")
        token_scores = main_chain_batch.batch.get("token_level_scores")
        response_mask = main_chain_batch.batch.get("response_mask")
        advantages = main_chain_batch.batch.get("advantages")
        returns = main_chain_batch.batch.get("returns")
        if (
            tree_ids_col is None
            or token_scores is None
            or response_mask is None
            or advantages is None
            or returns is None
        ):
            metrics["prefix_luffy/no_student_rows_skipped"] = len(candidates)
            return DataProto()

        rows_by_tid: dict = defaultdict(list)
        for idx, tid in enumerate(tree_ids_col):
            tid = str(tid)
            if tid in candidates:
                rows_by_tid[tid].append(idx)

        rewards = token_scores.sum(dim=-1).detach().cpu().tolist()
        teacher_entries = []
        teacher_rewards = []
        student_rewards_for_metrics = []
        baselines = []
        teacher_advs = []
        snis_weights = []
        missing_suffix = 0
        missing_reward_skipped = 0
        low_reward_teacher_skipped = 0
        teacher_not_better_skipped = 0
        no_student_rows = 0
        equal_reward_groups = 0
        zero_teacher_advantage = 0
        student_rows_reweighted = 0

        # Teacher rows must be verified-correct (declaration ④: premise, not
        # property). Gate BEFORE the student-advantage rewrite so a rejected
        # teacher never contaminates the group baseline.
        teacher_reward_gate = float(
            self.config.actor_rollout_ref.rollout.deep_grpo.get(
                "low_quality_trajectory_reward_threshold", 0.0
            )
        )
        teacher_loss_type = self.prefix_forest_teacher_loss_type
        snis_beta = self.prefix_forest_snis_beta

        for tid, (_tree_key, node_id, node) in candidates.items():
            suffix_ids = list(getattr(node, "teacher_suffix_ids", []) or [])
            if not suffix_ids:
                missing_suffix += 1
                continue
            row_indices = rows_by_tid.get(str(tid), [])
            if not row_indices:
                no_student_rows += 1
                continue

            teacher_reward = getattr(node, "teacher_suffix_reward", None)
            if teacher_reward is None:
                missing_reward_skipped += 1
                continue
            teacher_reward = float(teacher_reward)
            if teacher_reward <= teacher_reward_gate:
                low_reward_teacher_skipped += 1
                continue
            student_rewards = [float(rewards[i]) for i in row_indices]
            all_rewards = student_rewards + [teacher_reward]
            # Mixed baseline (students + teacher): used ONLY for the teacher
            # row's weight (w̃ / luffy A*) and the retirement/zero-adv gates.
            baseline = float(np.mean(all_rewards))
            teacher_adv = teacher_reward - baseline

            # Student advantages use a STUDENT-ONLY baseline (standard GRPO),
            # NOT the mixed baseline. Rationale: with the teacher (R=1) folded
            # into the student baseline, an all-fail group's students get a
            # small NON-zero advantage (0 - 1/(n+1) < 0) and slip past the
            # downstream zero-advantage filter — injecting low-quality
            # gradients from rollouts that should have been dropped. Using the
            # student-only mean restores GRPO's sum-to-zero property: all-fail
            # (and all-correct) student groups get advantage exactly 0 and are
            # filtered as intended, while the teacher row (handled separately
            # below) keeps its mixed-baseline signal and survives the filter.
            student_baseline = float(np.mean(student_rewards))
            student_advs = [r - student_baseline for r in student_rewards]

            for row_idx, student_adv in zip(row_indices, student_advs):
                adv_value = float(student_adv)
                advantages[row_idx] = adv_value * response_mask[row_idx]
                returns[row_idx] = adv_value * response_mask[row_idx]
                student_rows_reweighted += 1

            if max(all_rewards) - min(all_rewards) <= 1e-8:
                equal_reward_groups += 1
                continue
            if abs(teacher_adv) <= 1e-8:
                zero_teacher_advantage += 1
                continue
            if teacher_reward <= max(student_rewards):
                # k>=1 retirement (the "all-fail only" injection rule): once any
                # student matches the teacher, the teacher row retires. This is
                # the standing condition of the expectation-ascent theorem: the
                # damage regime (policy mass on another correct mode) requires
                # k>=1, and its residual probability is suppressed by (1-J)^n.
                # Students keep their student-only (GRPO) advantages set above.
                teacher_not_better_skipped += 1
                continue
            if teacher_loss_type is None:
                # Pure state-curriculum ablation arm: no teacher row is built.
                # (Student advantages already use the student-only baseline
                # above; all-fail injected groups get advantage 0 and are
                # filtered, exactly as teacher-free GRPO would handle them.)
                continue

            if teacher_loss_type == "snis":
                # Mean-normalized self-normalized IS weight:
                #   w̃ = exp(A*/β) / mean_j exp(A_j/β), A_j = R_j − baseline.
                # Mean (not sum) normalization composes with token-mean
                # aggregation to a single 1/(n+1) dilution (no double count).
                # Bounded: w̃ ≤ exp(ΔR_max/β). Baseline-shift invariant — we
                # exploit exactly that invariance to shift by the group max
                # before exponentiating (standard softmax trick, no overflow
                # for any β > 0).
                group_advs = np.asarray(all_rewards, dtype=np.float64) - baseline
                shifted = (group_advs - group_advs.max()) / snis_beta
                exp_advs = np.exp(shifted)
                w_tilde = float(
                    np.exp((teacher_adv - group_advs.max()) / snis_beta)
                    / exp_advs.mean()
                )
                snis_weights.append(w_tilde)
                teacher_row_coef = w_tilde
            else:  # "luffy"
                teacher_row_coef = float(teacher_adv)

            teacher_entries.append(
                {
                    "tree_id": str(tid),
                    "node_id": str(node_id),
                    "node": node,
                    "teacher_reward": teacher_reward,
                    # NOTE: for snis this field carries the SNIS weight w̃ (a
                    # stop-grad coefficient), NOT an advantage. It rides the
                    # advantages tensor through shuffle/concat into dp_actor,
                    # where the snis branch reads it as -w̃·logπ.
                    "teacher_advantage": teacher_row_coef,
                }
            )
            teacher_rewards.append(teacher_reward)
            student_rewards_for_metrics.extend(student_rewards)
            baselines.append(baseline)
            teacher_advs.append(float(teacher_adv))

        teacher_batch = self._build_prefix_forest_luffy_teacher_batch(
            teacher_entries
        )
        teacher_tokens = (
            teacher_batch.batch["response_mask"].sum().item()
            if len(teacher_batch) > 0 else 0
        )
        main_tokens = response_mask.sum().item()
        mixed_tokens = main_tokens + teacher_tokens

        metrics.update(
            {
                "prefix_luffy/teacher_rows_built": len(teacher_entries),
                "prefix_luffy/missing_suffix_skipped": missing_suffix,
                "prefix_luffy/missing_reward_skipped": missing_reward_skipped,
                "prefix_luffy/low_reward_teacher_skipped": low_reward_teacher_skipped,
                "prefix_luffy/teacher_not_better_skipped": teacher_not_better_skipped,
                "prefix_luffy/no_student_rows_skipped": no_student_rows,
                "prefix_luffy/equal_reward_groups_skipped": equal_reward_groups,
                "prefix_luffy/zero_teacher_advantage_skipped": zero_teacher_advantage,
                "prefix_luffy/student_rows_reweighted": student_rows_reweighted,
                "prefix_luffy/mixed_token_count": mixed_tokens,
                "prefix_luffy/teacher_token_fraction": (
                    float(teacher_tokens) / float(mixed_tokens)
                    if mixed_tokens > 0 else 0.0
                ),
            }
        )
        if snis_weights:
            metrics["prefix_luffy/snis_weight_mean"] = float(np.mean(snis_weights))
            metrics["prefix_luffy/snis_weight_max"] = float(np.max(snis_weights))
        if len(teacher_batch) > 0 and student_rows_reweighted > 0:
            # Declaration ⑤ sentinel: token-mean length convention. Teacher
            # share drifts with L/(n+L) when teacher/student lengths diverge.
            mean_teacher_len = float(teacher_tokens) / len(teacher_batch)
            mean_student_len = float(main_tokens) / max(1, len(main_chain_batch))
            metrics["prefix_luffy/len_ratio"] = (
                mean_teacher_len / (mean_student_len + 1e-8)
            )
        if teacher_rewards:
            metrics["prefix_luffy/teacher_reward_mean"] = float(
                np.mean(teacher_rewards)
            )
        if student_rewards_for_metrics:
            metrics["prefix_luffy/student_reward_mean"] = float(
                np.mean(student_rewards_for_metrics)
            )
        if baselines:
            metrics["prefix_luffy/baseline_mean"] = float(np.mean(baselines))
        if teacher_advs:
            metrics["prefix_luffy/teacher_advantage_mean"] = float(
                np.mean(teacher_advs)
            )
            metrics["prefix_luffy/teacher_positive_advantage_rate"] = float(
                np.mean([adv > 0.0 for adv in teacher_advs])
            )
        return teacher_batch

    def _build_forest_suffix_sft_batch(self, entries) -> DataProto:
        """Build teacher-only SFT rows from stored forest suffixes.

        This mirrors AgentLoopWorker._postprocess(run_from_teacher_entry(...))
        for branches_per_entry=0, without sending teacher suffix replay through
        the rollout workers. The entries are already concrete hard-state
        teacher suffix rows; no model generation is needed.
        """
        if not entries:
            return DataProto()

        from tensordict import TensorDict

        prompts = [
            list(entry.prompt_ids) + list(entry.response_ids)
            for entry in entries
        ]
        responses = [
            list(entry.teacher_suffix.suffix_ids)
            for entry in entries
        ]
        response_masks = [
            list(entry.teacher_suffix.suffix_mask)
            for entry in entries
        ]
        assert all(responses), "suffix-SFT entries must have non-empty responses"

        self.tokenizer.padding_side = "left"
        prompt_outputs = self.tokenizer.pad(
            [{"input_ids": ids} for ids in prompts],
            padding="max_length",
            max_length=self.config.actor_rollout_ref.rollout.max_model_len,
            return_tensors="pt",
            return_attention_mask=True,
        )
        prompt_ids = prompt_outputs["input_ids"]
        prompt_attention_mask = prompt_outputs["attention_mask"]

        self.tokenizer.padding_side = "right"
        response_outputs = self.tokenizer.pad(
            [{"input_ids": ids} for ids in responses],
            padding="max_length",
            max_length=self.config.actor_rollout_ref.rollout.max_model_len,
            return_tensors="pt",
            return_attention_mask=True,
        )
        response_ids = response_outputs["input_ids"]
        response_attention_mask = response_outputs["attention_mask"]

        mask_outputs = self.tokenizer.pad(
            [{"input_ids": ids} for ids in response_masks],
            padding="max_length",
            max_length=self.config.actor_rollout_ref.rollout.max_model_len,
            return_tensors="pt",
            return_attention_mask=False,
        )
        response_mask = mask_outputs["input_ids"] * response_attention_mask
        assert response_ids.shape == response_mask.shape, (
            "mismatch in suffix-SFT response_ids and response_mask shape: "
            f"{response_ids.shape} vs {response_mask.shape}"
        )

        input_ids = torch.cat([prompt_ids, response_ids], dim=1)
        attention_mask = torch.cat(
            [prompt_attention_mask, response_attention_mask],
            dim=1,
        )
        position_ids = (attention_mask.cumsum(dim=1) - 1) * attention_mask

        token_scores = torch.zeros_like(response_ids, dtype=torch.float32)
        for i, entry in enumerate(entries):
            valid_response_len = int(response_attention_mask[i].sum().item())
            assert valid_response_len > 0, "suffix-SFT response length must be > 0"
            token_scores[i, valid_response_len - 1] = float(
                entry.teacher_suffix.reward
            )

        advantages = torch.ones(
            (len(entries), 1),
            dtype=torch.float32,
        ) * response_mask

        batch = TensorDict(
            {
                "prompts": prompt_ids,
                "responses": response_ids,
                "response_mask": response_mask,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "token_level_scores": token_scores,
                "token_level_rewards": token_scores,
                "advantages": advantages,
                "returns": advantages,
            },
            batch_size=len(entries),
        )
        non_tensor_batch = {
            "__num_turns__": np.array(
                [float(entry.num_turns) + 1.0 for entry in entries],
                dtype=np.int32,
            ),
            "__tree_ids__": np.array(
                [str(entry.tree_id) for entry in entries],
                dtype=object,
            ),
            "__node_ids__": np.array(
                [str(uuid.uuid4()) for _ in entries],
                dtype=object,
            ),
            "__reward_infos__": np.array(
                [entry.teacher_suffix.reward_info for entry in entries],
                dtype=object,
            ),
        }
        return DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
            meta_info={
                "metrics": [{} for _ in entries],
                "temperature": self.config.actor_rollout_ref.rollout.temperature,
            },
        )

    def _sample_current_forest_suffix_sft_batch(self, metrics: dict) -> DataProto:
        """Sample current active suffix nodes after forest observation writeback."""
        teacher_mini_batch_size = self._global_teacher_mini_batch_size()
        world_size = max(1, int(self.actor_rollout_wg.world_size))
        assert teacher_mini_batch_size % world_size == 0, (
            "actor_rollout_ref.actor.deep_grpo.teacher_mini_batch_size "
            "must be divisible by actor world_size for post-PPO suffix SFT, got "
            f"{teacher_mini_batch_size} and world_size={world_size}"
        )

        suffix_sft_ready_trees, suffix_sft_ready_nodes = (
            self.forest_pool.suffix_sft_ready_counts()
        )
        _, suffix_sft_balanced_ready_nodes = (
            self.forest_pool.suffix_sft_ready_counts(max_nodes_per_tree=1)
        )
        ready_batches = suffix_sft_balanced_ready_nodes // teacher_mini_batch_size
        waiting_for_full_batch = int(suffix_sft_ready_nodes > 0 and ready_batches == 0)
        needed_for_next_batch = (
            teacher_mini_batch_size - suffix_sft_balanced_ready_nodes
            if waiting_for_full_batch else 0
        )
        target_batches = min(
            self.prefix_suffix_sft_max_batches_per_step,
            ready_batches,
        )
        target_suffix_sft = target_batches * teacher_mini_batch_size

        sampled_entries = []
        sampled_triples = []
        if target_suffix_sft > 0:
            sampled_triples = self.forest_pool.sample_suffix_sft(
                target_suffix_sft,
                self.global_steps,
                max_nodes_per_tree=1,
            )
            assert len(sampled_triples) == target_suffix_sft, (
                "forest suffix-SFT sampling returned fewer nodes than the "
                "tree-balanced ready count promised: "
                f"{len(sampled_triples)} vs {target_suffix_sft}"
            )
            for tree_key, node_id, node in sampled_triples:
                tid = str(uuid.uuid4())
                sampled_entries.append(
                    self._make_forest_suffix_sft_entry(tid, tree_key, node)
                )
                self._forest_sft_tree_id_to_node[tid] = (tree_key, node_id)

        pending_trees, pending_nodes = self.forest_pool.suffix_sft_ready_counts()
        metrics.update(
            {
                "prefix_suffix_sft/sampled_entries": len(sampled_entries),
                "prefix_suffix_sft/train_eligible_entries": 0,
                "prefix_suffix_sft/ready_trees": suffix_sft_ready_trees,
                "prefix_suffix_sft/ready_nodes": suffix_sft_ready_nodes,
                "prefix_suffix_sft/balanced_ready_nodes": (
                    suffix_sft_balanced_ready_nodes
                ),
                "prefix_suffix_sft/max_nodes_per_tree_per_step": (
                    self.prefix_suffix_sft_max_nodes_per_tree_per_step
                ),
                "prefix_suffix_sft/train_chunk_size": teacher_mini_batch_size,
                "prefix_suffix_sft/partial_batch_enabled": 0,
                "prefix_suffix_sft/partial_batch_entries": 0,
                "prefix_suffix_sft/waiting_for_full_batch": waiting_for_full_batch,
                "prefix_suffix_sft/needed_for_next_batch": needed_for_next_batch,
                "prefix_suffix_sft/pending_trees": pending_trees,
                "prefix_suffix_sft/pending_nodes": pending_nodes,
            }
        )
        assert len(sampled_entries) == len(sampled_triples)
        return self._build_forest_suffix_sft_batch(sampled_entries)

    def _build_forest_suffix_sft_batch_from_triples(self, sampled_triples):
        sampled_entries = []
        route_by_tid = {}
        for tree_key, node_id, node in sampled_triples:
            tid = str(uuid.uuid4())
            sampled_entries.append(
                self._make_forest_suffix_sft_entry(tid, tree_key, node)
            )
            route_by_tid[tid] = (tree_key, node_id)

        batch = self._build_forest_suffix_sft_batch(sampled_entries)
        if len(batch) > 0:
            batch.non_tensor_batch["__suffix_sft__"] = np.ones(
                len(batch),
                dtype=bool,
            )
        return batch, route_by_tid

    def _iter_epoch_local_suffix_sft_batches(
        self,
        snapshot,
        teacher_mini_batch_size: int,
    ):
        """Yield full distinct-tree suffix-SFT batches from a frozen snapshot."""
        grouped = {}
        for tree_key, node_id, node in snapshot:
            grouped.setdefault(tree_key, deque()).append((tree_key, node_id, node))

        for pass_idx in range(self.prefix_suffix_sft_epoch_passes):
            remaining = {
                tree_key: deque(nodes)
                for tree_key, nodes in grouped.items()
            }
            tree_order = deque(grouped.keys())
            while True:
                active_tree_count = sum(
                    1 for nodes in remaining.values() if len(nodes) > 0
                )
                if active_tree_count < teacher_mini_batch_size:
                    break

                batch = []
                examined = 0
                order_len = len(tree_order)
                while (
                    len(batch) < teacher_mini_batch_size
                    and examined < order_len
                ):
                    tree_key = tree_order.popleft()
                    examined += 1
                    nodes = remaining[tree_key]
                    if nodes:
                        batch.append(nodes.popleft())
                    tree_order.append(tree_key)

                if len(batch) != teacher_mini_batch_size:
                    break
                yield pass_idx, batch

    def _run_forest_suffix_sft_teacher_only_update(
        self,
        sampled_triples,
    ) -> dict:
        """Run one teacher-only post-PPO SFT update on a full forest batch."""
        teacher_mini_batch_size = self._global_teacher_mini_batch_size()
        world_size = max(1, int(self.actor_rollout_wg.world_size))
        assert len(sampled_triples) == teacher_mini_batch_size, (
            "epoch-local suffix SFT only runs full teacher mini-batches, got "
            f"{len(sampled_triples)} vs {teacher_mini_batch_size}"
        )
        assert teacher_mini_batch_size % world_size == 0, (
            "actor_rollout_ref.actor.deep_grpo.teacher_mini_batch_size must be "
            "divisible by actor world_size for epoch-local suffix SFT, got "
            f"{teacher_mini_batch_size} and world_size={world_size}"
        )

        routes = [(tree_key, node_id) for tree_key, node_id, _node in sampled_triples]
        trainable_triples = [
            (tree_key, node_id, node)
            for tree_key, node_id, node in sampled_triples
            if self.forest_pool.suffix_sft_trainable(tree_key, node_id)
        ]
        result = {
            "optimizer_steps": 0,
            "recorded_entries": 0,
            "record_skipped_entries": 0,
            "record_skipped_optimizer_steps": 0,
            "stale_entries": len(sampled_triples) - len(trainable_triples),
        }
        if len(trainable_triples) != len(sampled_triples):
            result["record_skipped_entries"] = len(sampled_triples)
            return result

        sampled_count = self.forest_pool.mark_suffix_sft_sampled(
            routes,
            current_step=self.global_steps,
        )
        if sampled_count != len(sampled_triples):
            result["stale_entries"] += len(sampled_triples) - sampled_count
            result["record_skipped_entries"] = len(sampled_triples)
            return result

        teacher_chain_batch, route_by_tid = (
            self._build_forest_suffix_sft_batch_from_triples(sampled_triples)
        )
        if len(teacher_chain_batch) == 0:
            result["record_skipped_entries"] = len(sampled_triples)
            return result

        metrics = {}
        timing_raw = {}
        teacher_chain_batch.non_tensor_batch["sources"] = np.full(
            len(teacher_chain_batch),
            2,
            dtype=np.float64,
        )
        teacher_chain_batch.non_tensor_batch["__suffix_sft__"] = np.ones(
            len(teacher_chain_batch),
            dtype=bool,
        )

        self._balance_batch(
            teacher_chain_batch,
            metrics=metrics,
            logging_prefix="epoch_suffix_sft_teacher_seqlen",
        )
        batch = teacher_chain_batch
        response_mask = batch.batch["response_mask"]
        batch.batch["main_chain_mask"] = torch.zeros_like(response_mask)
        batch.batch["branch_chain_mask"] = torch.zeros_like(response_mask)
        batch.batch["teacher_chain_mask"] = response_mask

        T_teacher = batch.batch["teacher_chain_mask"].sum().item()
        batch.meta_info["T_main"] = 0
        batch.meta_info["T_branch"] = 0
        batch.meta_info["T_teacher"] = T_teacher
        batch.non_tensor_batch["global_token_num"] = torch.sum(
            batch.batch["attention_mask"],
            dim=-1,
        ).numpy()

        with marked_timer("epoch_suffix_sft_old_log_prob", timing_raw, color="blue"):
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            entropys = old_log_prob.batch["entropys"]
            eps = 1e-8
            teacher_entropy = (
                entropys * batch.batch["teacher_chain_mask"]
            ).sum() / (T_teacher + eps)
            metrics["actor/teacher_chain/entropy"] = (
                teacher_entropy.detach().item()
            )
            old_log_prob.batch.pop("entropys")
            batch = batch.union(old_log_prob)

        with marked_timer("epoch_suffix_sft_ref", timing_raw, color="olive"):
            if not self.ref_in_actor:
                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
            else:
                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
            batch = batch.union(ref_log_prob)

        with marked_timer("epoch_suffix_sft_update_actor", timing_raw, color="red"):
            actor_output = self.actor_rollout_wg.update_actor(batch)
            actor_worker_metrics = list(actor_output.non_tensor_batch["metrics"])
            actor_output_metrics = reduce_metrics(actor_worker_metrics)
            metrics.update(actor_output_metrics)

        post_sft_optimizer_steps_by_worker = [
            int(m.get("actor/post_sft_optimizer_steps", 0) or 0)
            for m in actor_worker_metrics
        ]
        steps_with_teacher_by_worker = [
            int(m.get("actor/steps_with_teacher", 0) or 0)
            for m in actor_worker_metrics
        ]
        post_sft_all_workers_succeeded = bool(
            steps_with_teacher_by_worker
        ) and all(
            teacher_steps > 0 and post_sft_steps == teacher_steps
            for post_sft_steps, teacher_steps in zip(
                post_sft_optimizer_steps_by_worker,
                steps_with_teacher_by_worker,
            )
        )

        teacher_tids = batch.non_tensor_batch.get("__tree_ids__")
        assert teacher_tids is not None, (
            "epoch-local suffix SFT batch requires __tree_ids__"
        )
        train_tree_ids = [
            str(tid)
            for tid in teacher_tids
            if str(tid) in route_by_tid
        ]
        assert len(train_tree_ids) == len(sampled_triples), (
            "epoch-local suffix SFT batch contains rows without forest routes"
        )

        if train_tree_ids and post_sft_all_workers_succeeded:
            result["optimizer_steps"] = min(post_sft_optimizer_steps_by_worker)
            recorded = 0
            for tid in set(train_tree_ids):
                tree_key, node_id = route_by_tid[tid]
                update = self.forest_pool.record_suffix_sft(
                    tree_key=tree_key,
                    node_id=node_id,
                    current_step=self.global_steps,
                    count=1,
                )
                recorded += update.get("suffix_sft_recorded", 0)
            result["recorded_entries"] = recorded
        elif train_tree_ids and sum(steps_with_teacher_by_worker) > 0:
            result["record_skipped_optimizer_steps"] = sum(
                max(0, teacher_steps - post_sft_steps)
                for post_sft_steps, teacher_steps in zip(
                    post_sft_optimizer_steps_by_worker,
                    steps_with_teacher_by_worker,
                )
            )
            result["record_skipped_entries"] = len(train_tree_ids)

        result["steps_with_teacher"] = min(steps_with_teacher_by_worker or [0])
        result["actor_post_sft_optimizer_steps"] = min(
            post_sft_optimizer_steps_by_worker or [0]
        )
        result["actor_optimizer_steps"] = int(
            metrics.get("actor/optimizer_steps", 0) or 0
        )
        result["actor_lr_scheduler_step"] = int(
            metrics.get("actor/lr_scheduler_step", 0) or 0
        )
        result["timing_s/epoch_suffix_sft_old_log_prob"] = timing_raw.get(
            "epoch_suffix_sft_old_log_prob",
            0.0,
        )
        result["timing_s/epoch_suffix_sft_ref"] = timing_raw.get(
            "epoch_suffix_sft_ref",
            0.0,
        )
        result["timing_s/epoch_suffix_sft_update_actor"] = timing_raw.get(
            "epoch_suffix_sft_update_actor",
            0.0,
        )
        return result

    def _run_epoch_local_suffix_sft_phase(
        self,
        epoch: int,
        epoch_start_step: int,
        epoch_end_step: int,
        logger=None,
    ) -> dict:
        if (
            not self.prefix_inject_enabled
            or self.prefix_inject_pool_type != "forest"
            or not self.prefix_suffix_sft_enabled
            or self.prefix_suffix_sft_schedule != "epoch_local"
        ):
            return {}

        teacher_mini_batch_size = self._global_teacher_mini_batch_size()
        world_size = max(1, int(self.actor_rollout_wg.world_size))
        assert teacher_mini_batch_size % world_size == 0, (
            "actor_rollout_ref.actor.deep_grpo.teacher_mini_batch_size must be "
            "divisible by actor world_size for epoch-local suffix SFT, got "
            f"{teacher_mini_batch_size} and world_size={world_size}"
        )

        snapshot, clear_stats = (
            self.forest_pool.freeze_suffix_sft_epoch_and_clear_teacher_events(
                min_created_step=epoch_start_step,
                max_created_step=epoch_end_step,
            )
        )
        trees_cleaned = self.forest_pool.cleanup_inactive_trees()
        snapshot_tree_count = len({tree_key for tree_key, _node_id, _node in snapshot})

        phase_metrics = {
            "epoch_suffix_sft/enabled": 1,
            "epoch_suffix_sft/epoch": epoch,
            "epoch_suffix_sft/epoch_start_step": epoch_start_step,
            "epoch_suffix_sft/epoch_end_step": epoch_end_step,
            "epoch_suffix_sft/passes": self.prefix_suffix_sft_epoch_passes,
            "epoch_suffix_sft/train_chunk_size": teacher_mini_batch_size,
            "epoch_suffix_sft/snapshot_nodes": len(snapshot),
            "epoch_suffix_sft/snapshot_trees": snapshot_tree_count,
            "epoch_suffix_sft/trees_cleaned_after_clear": trees_cleaned,
            "epoch_suffix_sft/events_cleared": clear_stats.get(
                "teacher_events_cleared",
                0,
            ),
            "epoch_suffix_sft/events_cleared_pending": clear_stats.get(
                "teacher_events_cleared_pending",
                0,
            ),
            "epoch_suffix_sft/events_cleared_in_flight": clear_stats.get(
                "teacher_events_cleared_in_flight",
                0,
            ),
            "epoch_suffix_sft/sample_batches": 0,
            "epoch_suffix_sft/sampled_entries": 0,
            "epoch_suffix_sft/optimizer_steps": 0,
            "epoch_suffix_sft/recorded_entries": 0,
            "epoch_suffix_sft/record_skipped_entries": 0,
            "epoch_suffix_sft/record_skipped_optimizer_steps": 0,
            "epoch_suffix_sft/stale_entries": 0,
            "epoch_suffix_sft/actor_optimizer_steps": 0,
            "epoch_suffix_sft/actor_lr_scheduler_steps": 0,
            "epoch_suffix_sft/actor_post_sft_optimizer_steps": 0,
            "epoch_suffix_sft/waiting_for_full_batch": 0,
            "epoch_suffix_sft/needed_for_next_batch": 0,
        }

        if snapshot_tree_count < teacher_mini_batch_size:
            phase_metrics["epoch_suffix_sft/waiting_for_full_batch"] = int(
                len(snapshot) > 0
            )
            phase_metrics["epoch_suffix_sft/needed_for_next_batch"] = (
                teacher_mini_batch_size - snapshot_tree_count
                if len(snapshot) > 0 else 0
            )
            phase_metrics.update(self.forest_pool.stats)
            if logger is not None:
                logger.log(data=phase_metrics, step=epoch_end_step)
            return phase_metrics

        timing_sums = defaultdict(float)
        for _pass_idx, batch_triples in self._iter_epoch_local_suffix_sft_batches(
            snapshot,
            teacher_mini_batch_size,
        ):
            phase_metrics["epoch_suffix_sft/sample_batches"] += 1
            phase_metrics["epoch_suffix_sft/sampled_entries"] += len(batch_triples)
            result = self._run_forest_suffix_sft_teacher_only_update(batch_triples)
            phase_metrics["epoch_suffix_sft/optimizer_steps"] += result.get(
                "optimizer_steps",
                0,
            )
            phase_metrics["epoch_suffix_sft/recorded_entries"] += result.get(
                "recorded_entries",
                0,
            )
            phase_metrics["epoch_suffix_sft/record_skipped_entries"] += result.get(
                "record_skipped_entries",
                0,
            )
            phase_metrics[
                "epoch_suffix_sft/record_skipped_optimizer_steps"
            ] += result.get("record_skipped_optimizer_steps", 0)
            phase_metrics["epoch_suffix_sft/stale_entries"] += result.get(
                "stale_entries",
                0,
            )
            phase_metrics["epoch_suffix_sft/actor_optimizer_steps"] += result.get(
                "actor_optimizer_steps",
                0,
            )
            phase_metrics[
                "epoch_suffix_sft/actor_lr_scheduler_steps"
            ] += result.get("actor_lr_scheduler_step", 0)
            phase_metrics[
                "epoch_suffix_sft/actor_post_sft_optimizer_steps"
            ] += result.get("actor_post_sft_optimizer_steps", 0)
            for key, value in result.items():
                if key.startswith("timing_s/"):
                    timing_sums[key] += float(value or 0.0)

        intended_entries = len(snapshot) * self.prefix_suffix_sft_epoch_passes
        phase_metrics["epoch_suffix_sft/dropped_entries"] = max(
            0,
            intended_entries - phase_metrics["epoch_suffix_sft/sampled_entries"],
        )
        for key, value in timing_sums.items():
            phase_metrics[key] = value
        phase_metrics.update(self.forest_pool.stats)
        phase_metrics["sft/optimizer_steps"] = phase_metrics[
            "epoch_suffix_sft/optimizer_steps"
        ]
        if logger is not None:
            logger.log(data=phase_metrics, step=epoch_end_step)
        return phase_metrics

    def _log_teacher_comparison_table(self, sampled_entries, branch_chain_batch, step):
        """Log a wandb table comparing teacher suffixes with model branches.

        One row per entry: shared columns + K pairs of (branch_suffix, branch_reward).
        K = teacher_branches_per_entry, fixed across all entries.
        Teacher info is decoded from sampled_entries (raw data).
        Model branches: branch_chain_batch entry i → indices [i*K, (i+1)*K).
        """
        if self.teacher_comparison_log_freq <= 0:
            return
        if step % self.teacher_comparison_log_freq != 0:
            return
        if not sampled_entries:
            return

        K = self.teacher_branches_per_entry

        # Sample a subset of entries to log
        n_samples = min(self.teacher_comparison_log_samples, len(sampled_entries))
        sample_indices = np.random.choice(len(sampled_entries), size=n_samples, replace=False)

        rows = []
        for idx in sample_indices:
            entry = sampled_entries[idx]
            ts = entry.teacher_suffix

            # Decode shared context from entry (raw List[int], no padding)
            prompt_text = self.tokenizer.decode(entry.prompt_ids, skip_special_tokens=True)
            prefix_text = self.tokenizer.decode(entry.response_ids, skip_special_tokens=True)
            teacher_suffix_text = self.tokenizer.decode(ts.suffix_ids, skip_special_tokens=True)
            teacher_full_response = self.tokenizer.decode(
                list(entry.response_ids) + list(ts.suffix_ids), skip_special_tokens=True
            )
            teacher_reward = ts.reward

            failed_suffix_text = self.tokenizer.decode(
                ts.original_failed_suffix_ids, skip_special_tokens=True
            )
            failed_full_response = self.tokenizer.decode(
                list(entry.response_ids) + list(ts.original_failed_suffix_ids), skip_special_tokens=True
            )

            # Decode K model branch suffixes and rewards from branch_chain_batch
            branch_suffixes = []
            branch_rewards = []
            if K > 0 and len(branch_chain_batch) > 0:
                prompt_len = branch_chain_batch.batch["prompts"].shape[1]
                base_idx = int(idx) * K
                for k in range(K):
                    midx = base_idx + k
                    response_ids = branch_chain_batch.batch["responses"][midx]
                    valid_len = int(branch_chain_batch.batch["attention_mask"][midx, prompt_len:].sum().item())
                    branch_suffixes.append(
                        self.tokenizer.decode(response_ids[:valid_len].tolist(), skip_special_tokens=True)
                    )
                    branch_rewards.append(
                        branch_chain_batch.batch["token_level_scores"][midx].sum().item()
                    )

            # Build one row: shared fields + K branch pairs
            row = [
                step,
                entry.tree_id,
                prompt_text,
                prefix_text,
                failed_suffix_text,
                failed_full_response,
                teacher_suffix_text,
                teacher_full_response,
                teacher_reward,
            ]
            for k in range(K):
                row.extend([branch_suffixes[k], branch_rewards[k]])

            rows.append(row)

        if not rows:
            return

        # Debug: print summary to verify data diversity
        print(f"[TeacherComparisonTable] step={step}, rows={len(rows)}, "
              f"tree_ids={[r[1] for r in rows]}, "
              f"teacher_rewards={[r[8] for r in rows]}")

        # Log to wandb — only this step's samples
        try:
            import wandb
            table = wandb.Table(columns=self._teacher_comparison_columns, data=rows)
            wandb.log({f"teacher/comparison_table_step_{step}": table}, step=step)
        except Exception as e:
            print(f"WARNING: Failed to log teacher comparison table: {e}")

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # Skip if this exact step is already on disk (the best-checkpoint hook
        # saves during validation, before the periodic save condition fires).
        if getattr(self, "_last_checkpoint_step", None) == self.global_steps:
            return

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        # When keeping the best checkpoint, worker-side rotation must be
        # disabled: it retains the N most recent actor/critic saves in its own
        # order and would delete the best step's shards. The whole-folder
        # rotation at the end of this method enforces retention (newest N plus
        # the best) instead.
        save_best_enabled = bool(self.config.trainer.get("save_best_checkpoint", False))
        worker_max_actor_keep = None if save_best_enabled else max_actor_ckpt_to_keep
        worker_max_critic_keep = None if save_best_enabled else max_critic_ckpt_to_keep

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=worker_max_actor_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=worker_max_critic_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # save branch point buffer (so it survives checkpoint resume)
        if self.one_stage_mode and len(self.branch_point_buffer) > 0:
            import pickle
            buffer_local_path = os.path.join(local_global_step_folder, "branch_point_buffer.pkl")
            with open(buffer_local_path, "wb") as f:
                pickle.dump(self.branch_point_buffer.state_dict(), f)

        # save teacher synthesis pools
        if self.teacher_synthesis_enabled:
            import pickle
            for name, pool in [
                ("failed_trajectory_pool", self.failed_trajectory_pool),
                ("teacher_annotated_pool", self.teacher_annotated_pool),
            ]:
                if len(pool) > 0:
                    pool_path = os.path.join(local_global_step_folder, f"{name}.pkl")
                    with open(pool_path, "wb") as f:
                        pickle.dump(pool.state_dict(), f)

        # save prefix_inject pools
        if self.prefix_inject_enabled:
            import pickle
            if self.prefix_inject_pool_type == "chain":
                pools_to_save = [("chain_pool", self.chain_pool)]
            elif self.prefix_inject_pool_type == "forest":
                pools_to_save = [("prefix_forest_pool", self.forest_pool)]
            else:
                pools_to_save = [
                    ("failed_trajectory_pool", self.failed_trajectory_pool),
                    ("synthetic_prompt_pool", self.synthetic_prompt_pool),
                ]
            for name, pool in pools_to_save:
                if pool is not None and len(pool) > 0:
                    pool_path = os.path.join(local_global_step_folder, f"{name}.pkl")
                    with open(pool_path, "wb") as f:
                        pickle.dump(pool.state_dict(), f)

        # save curriculum sampler state
        if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
            sampler_path = os.path.join(local_global_step_folder, "curriculum_sampler.pt")
            torch.save(self.train_dataloader.sampler.state_dict(), sampler_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

        self._last_checkpoint_step = self.global_steps

        # Rotate whole global_step_* folders with the same retention as the actor.
        # The worker-side rotation above only removes the actor/ (and critic/)
        # subfolders of old checkpoints; the per-step extras saved here (data.pt,
        # pool pickles, sampler state) would otherwise accumulate for the whole
        # run. Runs after the tracker write so a complete, resumable checkpoint
        # exists on disk at all times. The best-validation step (tracked in
        # best_checkpoint.json when trainer.save_best_checkpoint is on) is
        # always retained on top of the newest max_actor_ckpt_to_keep folders.
        if isinstance(max_actor_ckpt_to_keep, int) and max_actor_ckpt_to_keep > 0:
            import re
            import shutil

            best_step = getattr(self, "_best_val_step", None)
            step_dir_pattern = re.compile(r"^global_step_(\d+)$")
            step_dirs = []
            for entry in os.listdir(self.config.trainer.default_local_dir):
                match = step_dir_pattern.match(entry)
                full_path = os.path.join(self.config.trainer.default_local_dir, entry)
                if match and os.path.isdir(full_path):
                    step = int(match.group(1))
                    # Never touch folders from beyond the current step (left over
                    # from a longer previous run we resumed into).
                    if step <= self.global_steps:
                        step_dirs.append((step, full_path))
            step_dirs.sort()
            for step, stale_path in step_dirs[: -max_actor_ckpt_to_keep]:
                if best_step is not None and step == best_step:
                    continue
                print(f"Removing stale checkpoint folder: {stale_path}")
                shutil.rmtree(stale_path, ignore_errors=True)

    def _maybe_save_best_checkpoint(self, val_metrics: dict):
        """Keep the checkpoint of the best validation score seen so far.

        Enabled by trainer.save_best_checkpoint. The tracked metric defaults to
        the cross-dataset average produced by _validate ("val-core/avg/..."),
        falling back to the single val-core metric when only one dataset is
        validated; override with trainer.best_checkpoint_metric. Saves a full
        checkpoint at the current step (so the best state is on disk even when
        the step is not a save_freq multiple) and records it in
        best_checkpoint.json; the rotation in _save_checkpoint spares that step.
        """
        if not self.config.trainer.get("save_best_checkpoint", False):
            return

        metric_key = self.config.trainer.get("best_checkpoint_metric", None)
        if metric_key is None:
            avg_keys = [k for k in val_metrics if k.startswith("val-core/avg/")]
            core_keys = [k for k in val_metrics if k.startswith("val-core/")]
            candidates = avg_keys or core_keys
            if len(candidates) == 1:
                metric_key = candidates[0]
            else:
                # With val_kwargs.n > 1 several val-core metrics coexist
                # (mean@N, best@N, maj@N). Default to the plain mean@N — the
                # same quantity mean@1 tracks, just estimated from N samples.
                mean_keys = [k for k in candidates if k.rsplit("/", 1)[-1].startswith("mean@")]
                if len(mean_keys) == 1:
                    metric_key = mean_keys[0]
                    if not getattr(self, "_best_metric_announced", False):
                        self._best_metric_announced = True
                        print(f"save_best_checkpoint: auto-selected '{metric_key}' among {sorted(candidates)}")
                else:
                    print(
                        "WARNING: save_best_checkpoint could not pick a validation "
                        f"metric automatically (candidates: {sorted(candidates)}); set "
                        "trainer.best_checkpoint_metric to one of them. Skipping "
                        "best-checkpoint tracking for this validation."
                    )
                    return
        if metric_key not in val_metrics:
            print(
                f"WARNING: best_checkpoint_metric '{metric_key}' not in validation "
                f"metrics; available val-core keys: "
                f"{sorted(k for k in val_metrics if k.startswith('val-core/'))}"
            )
            return

        score = float(val_metrics[metric_key])
        best_score = getattr(self, "_best_val_score", None)
        if best_score is not None and score <= best_score:
            return

        prev_best_step = getattr(self, "_best_val_step", None)
        self._best_val_score = score
        self._best_val_step = self.global_steps
        print(
            f"New best validation score {score:.6f} ({metric_key}) at step "
            f"{self.global_steps}"
            + (f" (previous best: step {prev_best_step})" if prev_best_step is not None else "")
        )

        # Save a full checkpoint at this step so the best state is on disk.
        # The rotation inside also removes the dethroned previous best folder
        # once it falls outside the retention window (the tracker above already
        # points at the current step).
        self._save_checkpoint()

        # Record the best step so rotation spares it and resume restores it.
        best_record_path = os.path.join(self.config.trainer.default_local_dir, "best_checkpoint.json")
        with open(best_record_path, "w") as f:
            json.dump({"step": self._best_val_step, "score": self._best_val_score, "metric": metric_key}, f)

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])
        if getattr(self, "teacher_worker", None) is not None:
            self.teacher_worker.update_current_step(self.global_steps)

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

        # load branch point buffer
        if self.one_stage_mode:
            import pickle
            buffer_local_path = os.path.join(global_step_folder, "branch_point_buffer.pkl")
            if os.path.exists(buffer_local_path):
                with open(buffer_local_path, "rb") as f:
                    state = pickle.load(f)
                self.branch_point_buffer.load_state_dict(state)
                print(f"Loaded branch point buffer with {len(self.branch_point_buffer)} entries from {buffer_local_path}")
            else:
                print(f"No branch point buffer found at {buffer_local_path}, starting with empty buffer")

        # load teacher synthesis pools
        if self.teacher_synthesis_enabled:
            import pickle
            for name, pool in [
                ("failed_trajectory_pool", self.failed_trajectory_pool),
                ("teacher_annotated_pool", self.teacher_annotated_pool),
            ]:
                pool_path = os.path.join(global_step_folder, f"{name}.pkl")
                if os.path.exists(pool_path):
                    with open(pool_path, "rb") as f:
                        state = pickle.load(f)
                    pool.load_state_dict(state)
                    print(f"Loaded {name} with {len(pool)} entries from {pool_path}")

        # load prefix_inject pools
        if self.prefix_inject_enabled:
            import pickle
            # Detect cross-mode resume (config switched between flat, chain,
            # and forest) — warn loudly because the other mode's pickle files
            # will be silently orphaned.
            if self.prefix_inject_pool_type == "chain":
                pools_to_load = [("chain_pool", self.chain_pool)]
                for stale in ("failed_trajectory_pool", "synthetic_prompt_pool", "prefix_forest_pool"):
                    stale_path = os.path.join(global_step_folder, f"{stale}.pkl")
                    if os.path.exists(stale_path):
                        print(
                            f"WARNING: found stale {stale}.pkl from previous "
                            f"non-chain run at {stale_path}; ignoring "
                            f"(current pool_type=chain)."
                        )
            elif self.prefix_inject_pool_type == "forest":
                pools_to_load = [("prefix_forest_pool", self.forest_pool)]
                for stale in ("failed_trajectory_pool", "synthetic_prompt_pool", "chain_pool"):
                    stale_path = os.path.join(global_step_folder, f"{stale}.pkl")
                    if os.path.exists(stale_path):
                        print(
                            f"WARNING: found stale {stale}.pkl from previous "
                            f"non-forest run at {stale_path}; ignoring "
                            f"(current pool_type=forest)."
                        )
            else:
                pools_to_load = [
                    ("failed_trajectory_pool", self.failed_trajectory_pool),
                    ("synthetic_prompt_pool", self.synthetic_prompt_pool),
                ]
                for stale in ("chain_pool", "prefix_forest_pool"):
                    stale_path = os.path.join(global_step_folder, f"{stale}.pkl")
                    if os.path.exists(stale_path):
                        print(
                            f"WARNING: found stale {stale}.pkl from previous "
                            f"non-flat run at {stale_path}; ignoring "
                            f"(current pool_type=flat)."
                        )
            for name, pool in pools_to_load:
                if pool is None:
                    continue
                pool_path = os.path.join(global_step_folder, f"{name}.pkl")
                if os.path.exists(pool_path):
                    with open(pool_path, "rb") as f:
                        state = pickle.load(f)
                    pool.load_state_dict(state)
                    print(f"Loaded {name} with {len(pool)} entries from {pool_path}")

        # load curriculum sampler state
        if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
            sampler_path = os.path.join(global_step_folder, "curriculum_sampler.pt")
            if os.path.exists(sampler_path):
                state = torch.load(sampler_path, weights_only=False)
                self.train_dataloader.sampler.load_state_dict(state)
                print(f"Loaded curriculum sampler state from {sampler_path}")

        # restore best-validation tracker (ignore records from beyond the
        # resumed step: that future was abandoned)
        best_record_path = os.path.join(self.config.trainer.default_local_dir, "best_checkpoint.json")
        if os.path.exists(best_record_path):
            with open(best_record_path) as f:
                best_record = json.load(f)
            if int(best_record.get("step", -1)) <= self.global_steps:
                self._best_val_step = int(best_record["step"])
                self._best_val_score = float(best_record["score"])
                print(
                    f"Restored best checkpoint tracker: step {self._best_val_step}, "
                    f"score {self._best_val_score:.6f}"
                )

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _post_process_gen_batch(self, batch: DataProto, prefix: str):
        metrics = {}
        timing_raw = {}

        if "timing" in batch.meta_info:
            timing_raw.update(batch.meta_info["timing"])
            batch.meta_info.pop("timing", None)

        # Remove the tail
        batch = batch[:len(batch) - (len(batch) % self.actor_rollout_wg.world_size)]

        if self.config.trainer.balance_batch:
            self._balance_batch(batch, metrics=metrics, logging_prefix=f"{prefix}/global_seqlen")

        # compute global_valid tokens
        batch.non_tensor_batch["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).numpy()

        # recompute old_log_probs
        with marked_timer(f"{prefix}/old_log_prob", timing_raw, color="blue"):
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            entropys = old_log_prob.batch["entropys"]
            response_masks = batch.batch["response_mask"]
            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
            entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
            old_log_prob_metrics = {f"actor/{prefix}/entropy": entropy_agg.detach().item()}
            metrics.update(old_log_prob_metrics)
            old_log_prob.batch.pop("entropys")
            batch = batch.union(old_log_prob)

        # compute reference log_prob
        with marked_timer(f"{prefix}/ref", timing_raw, color="olive"):
            if not self.ref_in_actor:
                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
            else:
                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)            
            batch = batch.union(ref_log_prob)

        return batch, metrics, timing_raw
        



    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # Run recoverability eval at step 0 (before training) if configured
        recoverability_eval_dir = self.config.trainer.get("recoverability_eval_dir", None)
        if recoverability_eval_dir and self.async_rollout_mode and self.config.trainer.get("recoverability_eval_freq", 0) > 0:
            self._recoverability_eval(recoverability_eval_dir)

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        for epoch in range(self.config.trainer.total_epochs):
            epoch_start_step = self.global_steps
            epoch_sft_phase_ran = False
            epoch_num_batches = len(self.train_dataloader)
            for batch_idx, batch_dict in enumerate(self.train_dataloader):
                is_epoch_last_batch = batch_idx + 1 >= epoch_num_batches
                do_profile = (
                    self.global_steps in self.config.trainer.profile_steps
                    if self.config.trainer.profile_steps is not None
                    else False
                )
                if do_profile:
                    self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
                    if self.use_reference_policy:
                        self.ref_policy_wg.start_profile()
                    if self.use_critic:
                        self.critic_wg.start_profile()
                    if self.use_rm:
                        self.rm_wg.start_profile()

                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                # Build uid -> dataset index mapping for curriculum sampler feedback
                if "__dataset_idx__" in batch.non_tensor_batch:
                    uid_to_ds_idx = dict(zip(
                        batch.non_tensor_batch["uid"],
                        batch.non_tensor_batch["__dataset_idx__"]
                    ))
                else:
                    uid_to_ds_idx = {}

                # Prefix inject mode: sample synthetic prompts and mint a fresh
                # tree_id for each. The tree_id becomes the GRPO group key and
                # the writeback lookup key. dict preserves insertion order, so
                # we can pass `.items()` to ts_generate_one_stage as the
                # (tree_id, entry) sequence it expects.
                synth_tree_id_to_entry: dict = {}
                prefix_debug_by_tid: dict = {}
                sampled_entries = []
                branches_per_entry = 0
                forest_suffix_sft_train_tree_ids = []
                # NOTE on batch composition semantics: prefix_inject (both flat
                # chain, and forest modes) uses ADDITION semantics — synth entries are
                # rolled out IN ADDITION to the full dataloader batch. Main
                # prompts use rollout.n; injected prefix prompts use
                # prefix_inject_mode.rollout_n (default: rollout.n). This means
                # a chain run with batch=128, rollout.n=8, prefix_rollout_n=16,
                # and injection_ratio=0.25 does 128*8 + 32*16 = 1536 rollouts,
                # vs vanilla's 1024. To achieve strict compute parity with
                # vanilla, reduce data.train_batch_size in the experiment script.
                if self.prefix_inject_enabled:
                    if self.prefix_inject_pool_type == "chain":
                        # Chain mode: sample (prompt_key, node) pairs from
                        # active chains. Wrap each node as a SyntheticPromptEntry
                        # for downstream ts_generate_one_stage (which expects
                        # augmented_prompt_ids + data_instance + agent_name).
                        from recipe.deep_grpo.protocol import SyntheticPromptEntry
                        import copy as _copy
                        # Start fresh each step — tree_ids are single-step scoped.
                        self._chain_tree_id_to_prompt_key = {}
                        target_synth = min(
                            int(self.prefix_inject_ratio * len(batch)),
                            self.chain_pool.active_chains_count(),
                        )
                        if target_synth > 0:
                            pairs = self.chain_pool.sample(
                                target_synth, self.global_steps
                            )
                            for prompt_key, node in pairs:
                                tid = str(uuid.uuid4())
                                # deepcopy data_instance so any downstream
                                # mutation (reward handlers, agent-loop bookkeeping)
                                # doesn't propagate back to the chain node and
                                # corrupt future samples of the same chain.
                                synth_tree_id_to_entry[tid] = SyntheticPromptEntry(
                                    augmented_prompt_ids=list(node.augmented_prompt_ids),
                                    data_instance=_copy.deepcopy(node.data_instance),
                                    agent_name=node.agent_name,
                                )
                                self._chain_tree_id_to_prompt_key[tid] = prompt_key
                        metrics["synthetic_inject/injected_count"] = len(
                            synth_tree_id_to_entry
                        )
                        metrics["synthetic_inject/rollout_n"] = (
                            self.prefix_rollout_n
                        )
                        metrics.update(self.chain_pool.stats)
                        # Keep the teacher worker's notion of the training step
                        # in sync for transition-log bookkeeping.
                        self.teacher_worker.update_current_step(self.global_steps)
                    elif self.prefix_inject_pool_type == "forest":
                        # Forest mode: sample verified-prefix tree nodes and
                        # wrap them as SyntheticPromptEntry for the existing
                        # synthetic rollout path.
                        from recipe.deep_grpo.protocol import SyntheticPromptEntry
                        import copy as _copy
                        self._forest_tree_id_to_node = {}
                        self._forest_sft_tree_id_to_node = {}
                        self._forest_luffy_tree_id_to_node = {}
                        self._prefix_paired_eval_diag_tree_ids = set()
                        self._prefix_paired_eval_pairs = []
                        self._prefix_paired_eval_pair_meta = []
                        forest_trees_cleaned_pre_sample = (
                            self.forest_pool.cleanup_inactive_trees()
                        )
                        suffix_sft_sampled = 0
                        suffix_sft_ready_trees, suffix_sft_ready_nodes = (
                            self.forest_pool.suffix_sft_ready_counts()
                        )
                        suffix_sft_balanced_ready_nodes = suffix_sft_ready_nodes
                        suffix_sft_train_chunk_size = 0
                        suffix_sft_waiting_for_full_batch = 0
                        suffix_sft_needed_for_next_batch = 0
                        suffix_sft_partial_batch_entries = 0
                        if self.prefix_suffix_sft_enabled:
                            teacher_mini_batch_size = (
                                self._global_teacher_mini_batch_size()
                            )
                            world_size = max(1, int(self.actor_rollout_wg.world_size))
                            assert teacher_mini_batch_size % world_size == 0, (
                                "actor_rollout_ref.actor.deep_grpo.teacher_mini_batch_size "
                                "must be divisible by actor world_size for "
                                "post-PPO suffix SFT, got "
                                f"{teacher_mini_batch_size} and world_size={world_size}"
                            )
                            suffix_sft_train_chunk_size = teacher_mini_batch_size
                            _, suffix_sft_balanced_ready_nodes = (
                                self.forest_pool.suffix_sft_ready_counts(
                                    max_nodes_per_tree=1
                                )
                            )
                            ready_batches = (
                                suffix_sft_balanced_ready_nodes
                                // suffix_sft_train_chunk_size
                            )
                            if (
                                suffix_sft_ready_nodes > 0
                                and ready_batches == 0
                            ):
                                suffix_sft_waiting_for_full_batch = 1
                                suffix_sft_needed_for_next_batch = (
                                    suffix_sft_train_chunk_size
                                    - suffix_sft_balanced_ready_nodes
                                )
                        target_synth = min(
                            int(self.prefix_inject_ratio * len(batch)),
                            self.forest_pool.prefix_injection_ready_trees_count(),
                        )
                        if target_synth > 0:
                            sampled_forest_entries = []
                            triples = self.forest_pool.sample(
                                target_synth,
                                self.global_steps,
                            )
                            for tree_key, node_id, node in triples:
                                tid = str(uuid.uuid4())
                                debug_context = None
                                if self.prefix_debug_dump_enabled:
                                    debug_context = self.forest_pool.debug_node_context(
                                        tree_key,
                                        node_id,
                                    )
                                synth_tree_id_to_entry[tid] = SyntheticPromptEntry(
                                    augmented_prompt_ids=list(node.augmented_prompt_ids),
                                    data_instance=_copy.deepcopy(node.data_instance),
                                    agent_name=node.agent_name,
                                )
                                self._forest_tree_id_to_node[tid] = (tree_key, node_id)
                                if self.prefix_forest_luffy_enabled:
                                    self._forest_luffy_tree_id_to_node[tid] = (
                                        tree_key,
                                        node_id,
                                        node,
                                    )
                                sampled_forest_entries.append(
                                    (tid, tree_key, node_id, node)
                                )
                                if self.prefix_debug_dump_enabled:
                                    prefix_debug_by_tid[tid] = self._make_prefix_debug_meta(
                                        tid=tid,
                                        source=(
                                            "forest_root"
                                            if node.parent_id is None
                                            else "forest_prefix"
                                        ),
                                        tree_key=tree_key,
                                        node_id=node_id,
                                        node=node,
                                        augmented_prompt_ids=node.augmented_prompt_ids,
                                        debug_context=debug_context,
                                    )

                            paired_cfg = (
                                self.config.actor_rollout_ref.rollout.deep_grpo
                                .get("prefix_inject_mode", {})
                                .get("paired_eval", {})
                            ) or {}
                            paired_enabled_raw = paired_cfg.get("enabled", False)
                            if isinstance(paired_enabled_raw, str):
                                paired_enabled = paired_enabled_raw.strip().lower() in (
                                    "1",
                                    "true",
                                    "yes",
                                    "on",
                                )
                            else:
                                paired_enabled = bool(paired_enabled_raw)
                            paired_freq = int(paired_cfg.get("freq", 50))
                            paired_num_pairs = int(paired_cfg.get("num_pairs", 4))
                            should_pair = (
                                paired_enabled
                                and paired_num_pairs > 0
                                and paired_freq > 0
                                and self.global_steps % paired_freq == 0
                            )
                            if should_pair:
                                candidates = [
                                    (tid, tree_key, node_id, node)
                                    for (
                                        tid,
                                        tree_key,
                                        node_id,
                                        node,
                                    ) in sampled_forest_entries
                                    if node.parent_id is not None
                                ]
                                for prefix_tid, tree_key, node_id, node in candidates[
                                    :paired_num_pairs
                                ]:
                                    root_tid = str(uuid.uuid4())
                                    prefix_context = self.forest_pool.debug_node_context(
                                        tree_key,
                                        node_id,
                                    )
                                    self._prefix_paired_eval_pair_meta.append(
                                        {
                                            "root_tid": root_tid,
                                            "prefix_tid": prefix_tid,
                                            "prefix_node_depth_edges": (
                                                prefix_context.get("node_depth_edges")
                                            ),
                                            "prefix_node_depth_tokens": (
                                                prefix_context.get("node_depth_tokens")
                                            ),
                                            "prefix_descendant_count": (
                                                prefix_context.get("descendant_count")
                                            ),
                                            "prefix_deepest_descendant_depth_edges": (
                                                prefix_context.get(
                                                    "deepest_descendant_depth_edges"
                                                )
                                            ),
                                            "prefix_child_count": len(
                                                getattr(node, "children", []) or []
                                            ),
                                        }
                                    )
                                    synth_tree_id_to_entry[root_tid] = SyntheticPromptEntry(
                                        augmented_prompt_ids=list(tree_key),
                                        data_instance=_copy.deepcopy(node.data_instance),
                                        agent_name=node.agent_name,
                                    )
                                    self._prefix_paired_eval_diag_tree_ids.add(root_tid)
                                    self._prefix_paired_eval_pairs.append(
                                        (root_tid, prefix_tid)
                                    )
                                    if self.prefix_debug_dump_enabled:
                                        prefix_debug_by_tid[root_tid] = self._make_prefix_debug_meta(
                                            tid=root_tid,
                                            source="paired_root",
                                            tree_key=tree_key,
                                            node_id=None,
                                            node=None,
                                            augmented_prompt_ids=list(tree_key),
                                            data_instance=node.data_instance,
                                            paired_tid=prefix_tid,
                                        )
                                        if prefix_tid in prefix_debug_by_tid:
                                            prefix_debug_by_tid[prefix_tid][
                                                "paired_tree_id"
                                            ] = root_tid
                                            if prefix_context:
                                                prefix_debug_by_tid[prefix_tid].update(
                                                    prefix_context
                                                )
                        if self._prefix_paired_eval_pairs:
                            metrics["prefix_paired_eval/requested_pairs"] = len(
                                self._prefix_paired_eval_pairs
                            )
                        metrics["synthetic_inject/injected_count"] = len(
                            synth_tree_id_to_entry
                        ) - len(
                            getattr(self, "_prefix_paired_eval_diag_tree_ids", set())
                        )
                        pending_trees, pending_nodes = (
                            self.forest_pool.suffix_sft_ready_counts()
                        )
                        metrics.update(
                            {
                                "prefix_suffix_sft/sampled_entries": suffix_sft_sampled,
                                "prefix_suffix_sft/train_eligible_entries": 0,
                                "prefix_suffix_sft/ready_trees": suffix_sft_ready_trees,
                                "prefix_suffix_sft/ready_nodes": suffix_sft_ready_nodes,
                                "prefix_suffix_sft/balanced_ready_nodes": suffix_sft_balanced_ready_nodes,
                                "prefix_suffix_sft/max_nodes_per_tree_per_step": self.prefix_suffix_sft_max_nodes_per_tree_per_step,
                                "prefix_suffix_sft/train_chunk_size": suffix_sft_train_chunk_size,
                                "prefix_suffix_sft/partial_batch_enabled": int(
                                    0
                                ),
                                "prefix_suffix_sft/partial_batch_entries": suffix_sft_partial_batch_entries,
                                "prefix_suffix_sft/waiting_for_full_batch": suffix_sft_waiting_for_full_batch,
                                "prefix_suffix_sft/needed_for_next_batch": suffix_sft_needed_for_next_batch,
                                "prefix_suffix_sft/pending_trees": pending_trees,
                                "prefix_suffix_sft/pending_nodes": pending_nodes,
                                "prefix_suffix_sft/schedule_epoch_local": int(
                                    self.prefix_suffix_sft_schedule
                                    == "epoch_local"
                                ),
                                "forest_pool/trees_cleaned_pre_sample": (
                                    forest_trees_cleaned_pre_sample
                                ),
                                "synthetic_inject/rollout_n": (
                                    self.prefix_rollout_n
                                ),
                                "prefix_luffy/enabled": int(
                                    self.prefix_forest_luffy_enabled
                                ),
                                "prefix_luffy/candidate_groups": len(
                                    self._forest_luffy_tree_id_to_node
                                ),
                                "prefix_luffy/teacher_rows_built": 0,
                                "prefix_luffy/missing_suffix_skipped": 0,
                                "prefix_luffy/missing_reward_skipped": 0,
                                "prefix_luffy/low_reward_teacher_skipped": 0,
                                "prefix_luffy/teacher_not_better_skipped": 0,
                                "prefix_luffy/no_student_rows_skipped": 0,
                                "prefix_luffy/equal_reward_groups_skipped": 0,
                                "prefix_luffy/zero_teacher_advantage_skipped": 0,
                                "prefix_luffy/student_rows_reweighted": 0,
                                "prefix_luffy/mixed_token_count": 0,
                                "prefix_luffy/teacher_token_fraction": 0.0,
                                "prefix_luffy/teacher_tail_filtered": 0,
                            }
                        )
                        metrics.update(self.forest_pool.stats)
                        self.teacher_worker.update_current_step(self.global_steps)
                    else:
                        # Adaptive: cap by pool size so we don't force-sample
                        # duplicates when the pool is small (early training).
                        target_synth = min(
                            int(self.prefix_inject_ratio * len(batch)),
                            len(self.synthetic_prompt_pool),
                        )
                        if target_synth > 0:
                            for entry in self.synthetic_prompt_pool.sample(target_synth):
                                synth_tree_id_to_entry[str(uuid.uuid4())] = entry
                        metrics["synthetic_inject/injected_count"] = len(
                            synth_tree_id_to_entry
                        )
                        metrics["synthetic_inject/rollout_n"] = (
                            self.prefix_rollout_n
                        )
                        metrics.update(self.synthetic_prompt_pool.stats)

                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "interaction_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("interaction_kwargs")
                if "index" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("index")
                if "__dataset_idx__" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("__dataset_idx__")
                if "agent_name" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("agent_name")
                if "data_source" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("data_source")
                if "reward_model" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("reward_model")
                if "extra_info" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("extra_info")
                if "uid" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("uid")

                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # breakpoint()
                    # generate
                    with marked_timer("gen", timing_raw, color="red"):
                        if self.one_stage_mode:
                            # Decide branch entries BEFORE generation so they run in parallel with main chains
                            if self.prefix_inject_enabled:
                                # Prefix inject mode does not generate model
                                # buffer branches. Forest suffix-SFT is sampled
                                # after forest observation writeback below.
                                branches_per_entry = 0
                            elif self.teacher_synthesis_enabled:
                                if self.teacher_annotated_pool.has_enough(self.teacher_branch_batch_threshold):
                                    sampled_entries = self.teacher_annotated_pool.sample(
                                        self.teacher_branch_batch_threshold,
                                        strategy=self.teacher_sampling_strategy,
                                    )
                                else:
                                    sampled_entries = []
                                branches_per_entry = self.teacher_branches_per_entry
                            else:
                                sampled_entries = self.branch_point_buffer.sample(
                                    self.buffer_sample_size,
                                    strategy=self.buffer_sampling_strategy,
                                )
                                branches_per_entry = self.buffer_branches_per_entry

                            # Main chains + buffer branches + synthetic rollouts IN PARALLEL
                            (main_chain_batch, branch_chain_batch, teacher_chain_batch,
                             failed_trajectories
                            ) = self.async_rollout_manager.ts_generate_one_stage(
                                gen_batch, sampled_entries, branches_per_entry,
                                synthetic_entries=(
                                    list(synth_tree_id_to_entry.items())
                                    if synth_tree_id_to_entry else None
                                ),
                            )

                            # Process failed trajectories based on mode
                            if self.prefix_inject_enabled:
                                # Compute per-group (tree_id) stats once; both
                                # flat and chain paths need k_succ/k_total lookups.
                                tree_ids_col = main_chain_batch.non_tensor_batch.get("__tree_ids__")
                                # Defensive: some early-abort / custom agent
                                # paths might not populate token_level_scores
                                # or __tree_ids__. Skip failure feed if either
                                # is missing — still log pool/teacher stats so
                                # dashboards don't show a gap.
                                token_scores = main_chain_batch.batch.get("token_level_scores")
                                feed_skipped = (
                                    token_scores is None or tree_ids_col is None
                                )
                                if feed_skipped:
                                    logger.warning(
                                        "prefix_inject: token_level_scores or "
                                        "__tree_ids__ missing from main_chain_batch; "
                                        "skipping failure feed for this step."
                                    )
                                metrics["synthetic_inject/failure_feed_skipped"] = (
                                    1 if feed_skipped else 0
                                )
                                reward_thresh = self.config.actor_rollout_ref.rollout.deep_grpo.get(
                                    "low_quality_trajectory_reward_threshold", 0.0
                                )
                                from collections import defaultdict as _dd
                                tid_to_rows: dict = _dd(list)
                                group_stats: dict = {}
                                tid_to_failures: dict = _dd(list)
                                if not feed_skipped:
                                    rewards_arr = token_scores.sum(dim=-1).cpu().numpy()
                                    for i in range(len(tree_ids_col)):
                                        tid_to_rows[str(tree_ids_col[i])].append(i)
                                    for tid, rows in tid_to_rows.items():
                                        k_total = len(rows)
                                        k_succ = sum(
                                            1 for r in rows
                                            if float(rewards_arr[r]) > reward_thresh
                                        )
                                        group_stats[tid] = (k_succ, k_total)
                                    for ft in (failed_trajectories or []):
                                        tid_to_failures[str(ft.tree_id)].append(ft)

                                paired_diag_ids = set(
                                    getattr(
                                        self,
                                        "_prefix_paired_eval_diag_tree_ids",
                                        set(),
                                    )
                                )
                                paired_pairs = list(
                                    getattr(self, "_prefix_paired_eval_pairs", [])
                                )
                                if paired_diag_ids and tree_ids_col is None:
                                    raise RuntimeError(
                                        "prefix_paired_eval requires __tree_ids__ "
                                        "to filter diagnostic root rollouts."
                                    )
                                if (
                                    self.prefix_inject_pool_type == "forest"
                                    and paired_pairs
                                    and not feed_skipped
                                ):
                                    self._update_prefix_paired_eval_metrics(
                                        metrics,
                                        group_stats,
                                        paired_pairs,
                                    )
                                if (
                                    self.prefix_inject_pool_type == "forest"
                                    and not feed_skipped
                                    and prefix_debug_by_tid
                                ):
                                    dumped_records = self._dump_prefix_debug_records(
                                        prefix_debug_by_tid=prefix_debug_by_tid,
                                        main_chain_batch=main_chain_batch,
                                        group_stats=group_stats,
                                        tid_to_rows=tid_to_rows,
                                        paired_pairs=paired_pairs,
                                    )
                                    if dumped_records:
                                        metrics["prefix_debug_dump/records"] = (
                                            dumped_records
                                        )
                                        metrics["prefix_debug_dump/deeper_records"] = (
                                            getattr(
                                                self,
                                                "_last_prefix_debug_dump_deeper_records",
                                                0,
                                            )
                                        )

                                if self.prefix_inject_pool_type == "chain":
                                    # (a) Record observations for chain-sampled
                                    # tree_ids — drives the state machine.
                                    chain_obs_recorded = 0
                                    for (
                                        tid,
                                        prompt_key,
                                    ) in self._chain_tree_id_to_prompt_key.items():
                                        if tid not in group_stats:
                                            continue
                                        k_succ, k_total = group_stats[tid]
                                        failures = tid_to_failures.get(tid, [])
                                        # Only pass failures if the group fully
                                        # failed (those are the ones the teacher
                                        # will use for deepening).
                                        fail_arg = failures if k_succ == 0 else None
                                        self.chain_pool.record_observation(
                                            prompt_key=prompt_key,
                                            k_succ=k_succ,
                                            k_total=k_total,
                                            current_step=self.global_steps,
                                            failed_rollouts=fail_arg,
                                        )
                                        chain_obs_recorded += 1

                                    # (b) For main-pool groups with k_succ=0 that
                                    # don't have a chain yet, enqueue chain creation.
                                    main_failures_enqueued = 0
                                    main_failures_seen = 0
                                    for tid, (k_succ, k_total) in group_stats.items():
                                        if tid in self._chain_tree_id_to_prompt_key:
                                            continue  # chain rollout
                                        if k_succ != 0 or k_total == 0:
                                            continue
                                        main_failures_seen += 1
                                        failures = tid_to_failures.get(tid, [])
                                        if not failures:
                                            continue
                                        first_ft = failures[0]
                                        created_key = self.chain_pool.on_main_failure(
                                            original_prompt_ids=first_ft.prompt_ids,
                                            data_instance=first_ft.data_instance,
                                            failed_rollouts=failures,
                                            current_step=self.global_steps,
                                            agent_name=first_ft.agent_name,
                                        )
                                        if created_key is not None:
                                            main_failures_enqueued += 1

                                    metrics.update(self.chain_pool.stats)
                                    metrics.update(self.teacher_worker.stats)
                                    metrics["chain_pool/observations_recorded_this_step"] = (
                                        chain_obs_recorded
                                    )
                                    metrics["chain_pool/main_failures_seen_this_step"] = (
                                        main_failures_seen
                                    )
                                    metrics["chain_pool/main_failures_enqueued_this_step"] = (
                                        main_failures_enqueued
                                    )
                                    metrics["synthetic_inject/failed_collected"] = len(
                                        failed_trajectories or []
                                    )
                                elif self.prefix_inject_pool_type == "forest":
                                    # (a) Record observations for injected
                                    # forest nodes. Full success deactivates
                                    # only that node; all-fail and partial-
                                    # success failed rollouts enqueue teacher
                                    # events.
                                    forest_obs_recorded = 0
                                    forest_events_added = 0
                                    forest_nodes_retired = 0
                                    for tid, (tree_key, node_id) in self._forest_tree_id_to_node.items():
                                        if tid not in group_stats:
                                            continue
                                        k_succ, k_total = group_stats[tid]
                                        failures = (
                                            tid_to_failures.get(tid, [])
                                            if k_succ < k_total else None
                                        )
                                        update = self.forest_pool.record_observation(
                                            tree_key=tree_key,
                                            node_id=node_id,
                                            k_succ=k_succ,
                                            k_total=k_total,
                                            current_step=self.global_steps,
                                            failed_rollouts=failures,
                                        )
                                        forest_obs_recorded += 1
                                        forest_events_added += update.get("events_added", 0)
                                        forest_nodes_retired += update.get("node_deactivated", 0)
                                    # (b) Feed normal-batch root outcomes into
                                    # the forest. All-fail and partial-success
                                    # roots enqueue failed trajectories; full
                                    # success deactivates only the root node.
                                    prompt_tensor = main_chain_batch.batch.get("prompts")
                                    attention_mask = main_chain_batch.batch.get("attention_mask")
                                    prompt_len = (
                                        prompt_tensor.shape[1]
                                        if prompt_tensor is not None else 0
                                    )

                                    def _prompt_ids_from_group(tid: str):
                                        failures = tid_to_failures.get(tid, [])
                                        if failures:
                                            return list(failures[0].prompt_ids)
                                        rows = tid_to_rows.get(tid, [])
                                        if (
                                            not rows
                                            or prompt_tensor is None
                                            or attention_mask is None
                                        ):
                                            return []
                                        row = rows[0]
                                        mask = attention_mask[row, :prompt_len].bool()
                                        return prompt_tensor[row][mask].cpu().tolist()

                                    # Aggregate normal-batch outcomes by actual
                                    # prompt, not by rollout tree_id. Duplicate
                                    # prompts in the same step are merged into
                                    # one root observation; any failed rollout
                                    # in the merged group can enqueue a teacher
                                    # event.
                                    root_by_prompt: dict = {}
                                    for tid, (k_succ, k_total) in group_stats.items():
                                        if tid in paired_diag_ids:
                                            continue  # diagnostic-only root replay
                                        if tid in self._forest_tree_id_to_node:
                                            continue  # injected forest rollout
                                        if k_total == 0:
                                            continue
                                        prompt_ids = _prompt_ids_from_group(tid)
                                        if not prompt_ids:
                                            continue
                                        prompt_key = self.forest_pool.compute_tree_key(prompt_ids)
                                        rec = root_by_prompt.setdefault(
                                            prompt_key,
                                            {
                                                "prompt_ids": prompt_ids,
                                                "k_succ": 0,
                                                "k_total": 0,
                                                "failures": [],
                                            },
                                        )
                                        rec["k_succ"] += k_succ
                                        rec["k_total"] += k_total
                                        if k_succ < k_total:
                                            rec["failures"].extend(
                                                tid_to_failures.get(tid, [])
                                            )

                                    root_groups_seen = 0
                                    root_fail_groups = 0
                                    root_success_groups = 0
                                    root_events_added = 0
                                    root_trees_created = 0
                                    root_nodes_deactivated = 0
                                    for rec in root_by_prompt.values():
                                        k_succ = rec["k_succ"]
                                        k_total = rec["k_total"]
                                        prompt_ids = rec["prompt_ids"]
                                        if k_succ < k_total:
                                            root_fail_groups += 1
                                            failures = rec["failures"]
                                            if failures:
                                                first_ft = failures[0]
                                                data_instance = first_ft.data_instance
                                                agent_name = first_ft.agent_name
                                            else:
                                                data_instance = {}
                                                agent_name = ""
                                        else:
                                            root_success_groups += 1
                                            failures = None
                                            # Only existing trees use this path;
                                            # data_instance/agent_name are ignored
                                            # when no tree is present.
                                            data_instance = {}
                                            agent_name = ""

                                        update = self.forest_pool.record_root_observation(
                                            original_prompt_ids=prompt_ids,
                                            data_instance=data_instance,
                                            agent_name=agent_name,
                                            k_succ=k_succ,
                                            k_total=k_total,
                                            current_step=self.global_steps,
                                            failed_rollouts=failures,
                                        )
                                        root_groups_seen += 1
                                        root_events_added += update.get("events_added", 0)
                                        root_trees_created += update.get("tree_created", 0)
                                        root_nodes_deactivated += update.get("root_deactivated", 0)

                                    forest_trees_cleaned = (
                                        self.forest_pool.cleanup_inactive_trees()
                                    )
                                    if (
                                        self.prefix_suffix_sft_enabled
                                        and self.prefix_suffix_sft_schedule
                                        == "post_step"
                                    ):
                                        teacher_chain_batch = (
                                            self._sample_current_forest_suffix_sft_batch(
                                                metrics
                                            )
                                        )
                                    metrics.update(self.forest_pool.stats)
                                    metrics.update(self.teacher_worker.stats)
                                    pending_trees, pending_nodes = (
                                        self.forest_pool.suffix_sft_ready_counts()
                                    )
                                    metrics.update(
                                        {
                                            "prefix_suffix_sft/recorded_entries": 0,
                                            "prefix_suffix_sft/matured_entries": 0,
                                            "prefix_suffix_sft/pending_trees": pending_trees,
                                            "prefix_suffix_sft/pending_nodes": pending_nodes,
                                            "prefix_suffix_sft/schedule_epoch_local": int(
                                                self.prefix_suffix_sft_schedule
                                                == "epoch_local"
                                            ),
                                        }
                                    )
                                    metrics["forest_pool/observations_recorded_this_step"] = (
                                        forest_obs_recorded
                                    )
                                    metrics["forest_pool/events_added_this_step"] = (
                                        forest_events_added
                                    )
                                    metrics["forest_pool/nodes_retired_this_step"] = (
                                        forest_nodes_retired
                                    )
                                    metrics["forest_pool/root_groups_seen_this_step"] = (
                                        root_groups_seen
                                    )
                                    metrics["forest_pool/root_fail_groups_this_step"] = (
                                        root_fail_groups
                                    )
                                    metrics["forest_pool/root_success_groups_this_step"] = (
                                        root_success_groups
                                    )
                                    metrics["forest_pool/root_events_added_this_step"] = (
                                        root_events_added
                                    )
                                    metrics["forest_pool/root_trees_created_this_step"] = (
                                        root_trees_created
                                    )
                                    metrics["forest_pool/root_nodes_deactivated_this_step"] = (
                                        root_nodes_deactivated
                                    )
                                    metrics["forest_pool/trees_cleaned_this_step"] = (
                                        forest_trees_cleaned
                                    )
                                    metrics["synthetic_inject/failed_collected"] = len(
                                        [
                                            ft for ft in (failed_trajectories or [])
                                            if str(ft.tree_id) not in paired_diag_ids
                                        ]
                                    )
                                else:
                                    # Flat-pool path: gate teacher annotations
                                    # by full-fail (k_succ=0), reusing group_stats.
                                    gated = [
                                        ft for ft in (failed_trajectories or [])
                                        if group_stats.get(str(ft.tree_id), (1, 0))[0] == 0
                                    ]
                                    if gated:
                                        self.failed_trajectory_pool.add(
                                            gated, current_step=self.global_steps
                                        )
                                    metrics.update(self.failed_trajectory_pool.stats)
                                    metrics.update(self.teacher_worker.stats)
                                    metrics["synthetic_inject/failed_collected"] = len(
                                        failed_trajectories or []
                                    )
                                    metrics[
                                        "synthetic_inject/failed_gated_to_teacher"
                                    ] = len(gated)

                                    # Write back succ/total to synthetic entries
                                    # using tree_ids stamped on main_chain_batch rows.
                                    if synth_tree_id_to_entry:
                                        self._record_synthetic_usage(
                                            main_chain_batch,
                                            synth_tree_id_to_entry,
                                            metrics,
                                        )
                                if paired_diag_ids and tree_ids_col is not None:
                                    keep_idx = np.array(
                                        [
                                            i for i, tid in enumerate(tree_ids_col)
                                            if str(tid) not in paired_diag_ids
                                        ],
                                        dtype=np.int64,
                                    )
                                    removed = len(tree_ids_col) - len(keep_idx)
                                    if removed > 0:
                                        metrics[
                                            "prefix_paired_eval/filtered_rollouts"
                                        ] = removed
                                        main_chain_batch = main_chain_batch[keep_idx]
                                        failed_trajectories = [
                                            ft for ft in (failed_trajectories or [])
                                            if str(ft.tree_id) not in paired_diag_ids
                                        ]
                                if (
                                    self.prefix_inject_pool_type == "forest"
                                    and self.prefix_forest_luffy_enabled
                                ):
                                    teacher_chain_batch = (
                                        self._attach_prefix_forest_luffy_teacher_continuations(
                                            main_chain_batch,
                                            metrics,
                                        )
                                    )
                            elif self.teacher_synthesis_enabled:
                                # Teacher mode: send failed trajectories to teacher worker.
                                # With expand_branch_chain=False (recommended for teacher mode),
                                # failed_trajectories already contains only failed chains.
                                if failed_trajectories:
                                    self.failed_trajectory_pool.add(
                                        failed_trajectories, current_step=self.global_steps
                                    )
                                metrics.update(self.failed_trajectory_pool.stats)
                                metrics.update(self.teacher_annotated_pool.stats)
                                metrics.update(self.teacher_worker.stats)
                                metrics["teacher/entries_sampled"] = len(sampled_entries)
                                metrics["teacher/failed_collected"] = len(failed_trajectories)
                                # Log teacher comparison table (sampled, at controlled intervals)
                                if sampled_entries and len(teacher_chain_batch) > 0:
                                    self._log_teacher_comparison_table(
                                        sampled_entries, branch_chain_batch, self.global_steps
                                    )
                            else:
                                # Regular mode: extract pre-selected branch points from failed trajectories
                                new_branch_points = []
                                for ft in failed_trajectories:
                                    if ft.branch_points:
                                        new_branch_points.extend(ft.branch_points)
                                self.branch_point_buffer.add(new_branch_points, current_step=self.global_steps)
                                metrics.update(self.branch_point_buffer.stats)
                                metrics["buffer/entries_sampled"] = len(sampled_entries)
                                metrics["buffer/new_points_collected"] = len(new_branch_points)

                        else:
                            # Original two-stage path (unchanged)
                            main_chain_batch, branch_chain_batch = self.async_rollout_manager.ts_generate_sequences(gen_batch)
                            teacher_chain_batch = DataProto()

                    if (
                        self.prefix_inject_enabled
                        and self.prefix_inject_pool_type == "forest"
                        and self.prefix_suffix_sft_enabled
                        and len(teacher_chain_batch) > 0
                    ):
                        teacher_tids = teacher_chain_batch.non_tensor_batch.get(
                            "__tree_ids__"
                        )
                        assert teacher_tids is not None, (
                            "suffix-SFT teacher batch requires __tree_ids__"
                        )
                        suffix_sft_ids = {
                            tid
                            for tid, (tree_key, node_id) in self._forest_sft_tree_id_to_node.items()
                            if self.forest_pool.suffix_sft_trainable(
                                tree_key,
                                node_id,
                            )
                        }
                        keep_idx = np.array(
                            [
                                i for i, tid in enumerate(teacher_tids)
                                if str(tid) in suffix_sft_ids
                            ],
                            dtype=np.int64,
                        )
                        stale_filtered = sum(
                            1
                            for tid in teacher_tids
                            if (
                                str(tid) in self._forest_sft_tree_id_to_node
                                and str(tid) not in suffix_sft_ids
                            )
                        )
                        filtered = len(teacher_chain_batch) - len(keep_idx)
                        if filtered > 0:
                            teacher_chain_batch = teacher_chain_batch[keep_idx]
                        metrics["prefix_suffix_sft/non_sft_teacher_filtered"] = (
                            filtered - stale_filtered
                        )
                        metrics["prefix_suffix_sft/stale_teacher_filtered"] = (
                            stale_filtered
                        )
                        if len(teacher_chain_batch) > 0:
                            teacher_chain_batch.non_tensor_batch["__suffix_sft__"] = (
                                np.ones(len(teacher_chain_batch), dtype=bool)
                            )
                            for data_part in (main_chain_batch, branch_chain_batch):
                                if len(data_part) > 0:
                                    data_part.non_tensor_batch["__suffix_sft__"] = (
                                        np.zeros(len(data_part), dtype=bool)
                                    )

                    # Stamp dataset indices on main_chain_batch for curriculum sampler
                    if uid_to_ds_idx and "__tree_ids__" in main_chain_batch.non_tensor_batch:
                        tree_ids = main_chain_batch.non_tensor_batch["__tree_ids__"]
                        ds_indices = np.array(
                            [uid_to_ds_idx.get(str(tid), -1) for tid in tree_ids], dtype=object
                        )
                        main_chain_batch.non_tensor_batch["__dataset_indices__"] = ds_indices

                    # compute data metrics on full (unfiltered) batch for true reward stats
                    metrics.update(compute_data_metrics(batch=main_chain_batch, max_model_len=self.config.actor_rollout_ref.rollout.max_model_len, prefix="main_chain"))
                    # Source-split metrics: separate original dataset prompts
                    # (ds_idx >= 0) from injected synth/chain entries (ds_idx < 0).
                    # This lets us diagnose whether behaviour (e.g., response
                    # length) shifts on the no-prefix main distribution, or
                    # only reflects chain samples' mechanically-shorter responses.
                    ds_idx_col = main_chain_batch.non_tensor_batch.get("__dataset_indices__")
                    if ds_idx_col is not None:
                        mask_orig = np.array(
                            [int(x) >= 0 for x in ds_idx_col], dtype=bool
                        )
                        # Match the numpy-array indexing convention used
                        # elsewhere (e.g., line 2171 main_nonzero_idx).
                        if mask_orig.any():
                            orig_idx = np.where(mask_orig)[0]
                            metrics.update(compute_data_metrics(
                                batch=main_chain_batch[orig_idx],
                                max_model_len=self.config.actor_rollout_ref.rollout.max_model_len,
                                prefix="main_chain_original",
                            ))
                        if (~mask_orig).any():
                            synth_idx = np.where(~mask_orig)[0]
                            metrics.update(compute_data_metrics(
                                batch=main_chain_batch[synth_idx],
                                max_model_len=self.config.actor_rollout_ref.rollout.max_model_len,
                                prefix="main_chain_synthetic",
                            ))
                    if len(branch_chain_batch) > 0:
                        metrics.update(compute_data_metrics(batch=branch_chain_batch, max_model_len=self.config.actor_rollout_ref.rollout.max_model_len, prefix="branch_chain"))
                    if len(teacher_chain_batch) > 0:
                        metrics.update(compute_data_metrics(batch=teacher_chain_batch, max_model_len=self.config.actor_rollout_ref.rollout.max_model_len, prefix="teacher_chain"))

                    # Log rollout generations (full unfiltered batch) if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            # Dump main chain
                            main_inputs = self.tokenizer.batch_decode(main_chain_batch.batch["prompts"], skip_special_tokens=True)
                            main_outputs = self.tokenizer.batch_decode(main_chain_batch.batch["responses"], skip_special_tokens=True)
                            main_scores = main_chain_batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            main_reward_infos = main_chain_batch.non_tensor_batch["__reward_infos__"]
                            main_extra = {
                                "finished": [info.finished for info in main_reward_infos],
                                "completed": [info.completed for info in main_reward_infos],
                            }
                            self._dump_generations(
                                inputs=main_inputs,
                                outputs=main_outputs,
                                scores=main_scores,
                                reward_extra_infos_dict=main_extra,
                                dump_path=rollout_data_dir,
                            )
                            # Dump branch chain
                            if len(branch_chain_batch) > 0:
                                branch_inputs = self.tokenizer.batch_decode(branch_chain_batch.batch["prompts"], skip_special_tokens=True)
                                branch_outputs = self.tokenizer.batch_decode(branch_chain_batch.batch["responses"], skip_special_tokens=True)
                                branch_scores = branch_chain_batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                                branch_reward_infos = branch_chain_batch.non_tensor_batch["__reward_infos__"]
                                branch_extra = {
                                    "finished": [info.finished for info in branch_reward_infos],
                                    "completed": [info.completed for info in branch_reward_infos],
                                }
                                self._dump_generations(
                                    inputs=branch_inputs,
                                    outputs=branch_outputs,
                                    scores=branch_scores,
                                    reward_extra_infos_dict=branch_extra,
                                    dump_path=rollout_data_dir,
                                )

                    # Update curriculum sampler BEFORE filtering so p_hat
                    # captures all-correct/all-fail groups (the groups we want
                    # to avoid sampling next step).
                    if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                        main_chain_batch.meta_info["global_steps"] = self.global_steps
                        self.train_dataloader.sampler.update(batch=main_chain_batch)
                        metrics.update(self.train_dataloader.sampler.stats)

                    # filter zero-advantage samples (all-correct or all-wrong questions)
                    main_adv_abs = main_chain_batch.batch["advantages"].abs().sum(dim=-1)
                    main_nonzero_idx = np.where((main_adv_abs > 1e-8).numpy())[0]
                    metrics["data/main_chain/filtered_count"] = len(main_chain_batch) - len(main_nonzero_idx)
                    main_chain_batch = main_chain_batch[main_nonzero_idx]

                    if len(branch_chain_batch) > 0:
                        branch_adv_abs = branch_chain_batch.batch["advantages"].abs().sum(dim=-1)
                        branch_nonzero_idx = np.where((branch_adv_abs > 1e-8).numpy())[0]
                        metrics["data/branch_chain/filtered_count"] = len(branch_chain_batch) - len(branch_nonzero_idx)
                        branch_chain_batch = branch_chain_batch[branch_nonzero_idx]

                    metrics.update({
                        "data/main_chain_count": len(main_chain_batch),
                        "data/branch_chain_count": len(branch_chain_batch),
                        "data/teacher_chain_count": len(teacher_chain_batch),
                    })

                    if "timing" in main_chain_batch.meta_info:
                        timing_raw.update(main_chain_batch.meta_info["timing"])
                        main_chain_batch.meta_info.pop("timing", None)

                    W = self.actor_rollout_wg.world_size
                    N_main = 0
                    Nm = 0
                    N_branch = 0
                    Nb = 0
                    N_teacher = 0
                    Nt = 0

                    def _whiten_advantages(data_batch):
                        """Whiten advantages in-place if enabled."""
                        if not self.config.trainer.whiten_advantages:
                            return
                        advs = data_batch.batch["advantages"]
                        mask = data_batch.batch["response_mask"].bool()
                        valid = torch.masked_select(advs, mask)
                        if valid.numel() > 1:
                            data_batch.batch["advantages"] = (advs - valid.mean()) / (valid.std() + 1e-8) * mask

                    def _shuffle_and_trim(data_batch, source_val):
                        """Label sources, shuffle, trim to be divisible by W."""
                        data_batch.non_tensor_batch["sources"] = np.full(len(data_batch), source_val, dtype=np.float64)
                        perm = np.random.permutation(len(data_batch))
                        data_batch = data_batch[perm]
                        n = len(data_batch) - (len(data_batch) % W)
                        return data_batch[:n], n, n // W

                    # Prepare main chain (source=0)
                    if len(main_chain_batch) > 0:
                        _whiten_advantages(main_chain_batch)
                        main_chain_batch, N_main, Nm = _shuffle_and_trim(main_chain_batch, 0)

                    # Prepare model branches (source=1)
                    if len(branch_chain_batch) > 0:
                        _whiten_advantages(branch_chain_batch)
                        branch_chain_batch, N_branch, Nb = _shuffle_and_trim(branch_chain_batch, 1)

                    # Prepare teacher (source=2) — no advantage whitening
                    # (SFT ignores advantages; LUFFY uses raw advantage relative to model baseline)
                    if len(teacher_chain_batch) > 0:
                        teacher_rows_before_trim = len(teacher_chain_batch)
                        teacher_chain_batch, N_teacher, Nt = _shuffle_and_trim(teacher_chain_batch, 2)
                        if (
                            self.prefix_inject_enabled
                            and self.prefix_inject_pool_type == "forest"
                            and self.prefix_forest_luffy_enabled
                        ):
                            metrics["prefix_luffy/teacher_tail_filtered"] = (
                                teacher_rows_before_trim - N_teacher
                            )
                        if (
                            self.prefix_inject_enabled
                            and self.prefix_inject_pool_type == "forest"
                            and self.prefix_suffix_sft_enabled
                            and N_teacher > 0
                        ):
                            teacher_tids = teacher_chain_batch.non_tensor_batch.get(
                                "__tree_ids__"
                            )
                            assert teacher_tids is not None, (
                                "suffix-SFT teacher batch requires __tree_ids__"
                            )
                            teacher_mini_batch_size = (
                                self._global_teacher_mini_batch_size()
                            )
                            train_teacher_entries = (
                                N_teacher // teacher_mini_batch_size
                            ) * teacher_mini_batch_size
                            if train_teacher_entries == 0:
                                metrics[
                                    "prefix_suffix_sft/teacher_tail_filtered"
                                ] = N_teacher
                                teacher_chain_batch = teacher_chain_batch[:0]
                                N_teacher = 0
                                Nt = 0
                            elif train_teacher_entries < N_teacher:
                                metrics[
                                    "prefix_suffix_sft/teacher_tail_filtered"
                                ] = N_teacher - train_teacher_entries
                                teacher_chain_batch = teacher_chain_batch[
                                    :train_teacher_entries
                                ]
                                N_teacher = train_teacher_entries
                                Nt = N_teacher // W if W > 0 else 0
                                teacher_tids = (
                                    teacher_chain_batch.non_tensor_batch.get(
                                        "__tree_ids__"
                                    )
                                )
                            if N_teacher > 0:
                                suffix_sft_ids = set(self._forest_sft_tree_id_to_node)
                                forest_suffix_sft_train_tree_ids = [
                                    str(tid) for tid in teacher_tids
                                    if str(tid) in suffix_sft_ids
                                ]
                                assert len(forest_suffix_sft_train_tree_ids) == N_teacher, (
                                    "suffix-SFT teacher batch contains rows not "
                                    "sampled from the forest SFT queue"
                                )
                            metrics["prefix_suffix_sft/train_eligible_entries"] = (
                                len(forest_suffix_sft_train_tree_ids)
                            )

                    if N_main == 0 and N_branch == 0 and N_teacher == 0:
                        metrics["training/all_chains_empty_skip"] = 1
                    else:
                        # Balance each non-empty chain for sequence-length load balancing
                        if N_main > 0:
                            self._balance_batch(main_chain_batch, metrics=metrics, logging_prefix="main_seqlen")
                        if N_branch > 0:
                            self._balance_batch(branch_chain_batch, metrics=metrics, logging_prefix="branch_seqlen")
                        if N_teacher > 0:
                            self._balance_batch(teacher_chain_batch, metrics=metrics, logging_prefix="teacher_seqlen")

                        # Concat and interleave so sequential dispatch gives each GPU
                        # exactly Nm main + Nb branch + Nt teacher samples.
                        parts = []
                        part_totals = []
                        part_per_worker = []
                        if N_main > 0:
                            parts.append(main_chain_batch); part_totals.append(N_main); part_per_worker.append(Nm)
                        if N_branch > 0:
                            parts.append(branch_chain_batch); part_totals.append(N_branch); part_per_worker.append(Nb)
                        if N_teacher > 0:
                            parts.append(teacher_chain_batch); part_totals.append(N_teacher); part_per_worker.append(Nt)

                        if len(parts) > 1:
                            fill_specs = {
                                "__dataset_indices__": (-1, object),
                                "__suffix_sft__": (False, bool),
                                "__tree_ids__": ("", object),
                                "__node_ids__": ("", object),
                                "__reward_infos__": (None, object),
                                "__num_turns__": (0, np.int32),
                            }
                            non_tensor_keys = set()
                            for part in parts:
                                non_tensor_keys.update(part.non_tensor_batch.keys())
                            for key in non_tensor_keys:
                                fill_value, dtype = fill_specs.get(
                                    key,
                                    (None, object),
                                )
                                for part in parts:
                                    if key not in part.non_tensor_batch:
                                        part.non_tensor_batch[key] = np.full(
                                            len(part),
                                            fill_value,
                                            dtype=dtype,
                                        )

                        batch = DataProto.concat(parts) if len(parts) > 1 else parts[0]
                        if len(parts) > 1:
                            interleave_idx = []
                            cum_offset = 0
                            offsets = []
                            for total in part_totals:
                                offsets.append(cum_offset)
                                cum_offset += total
                            for i in range(W):
                                for off, pw in zip(offsets, part_per_worker):
                                    interleave_idx.extend(range(off + i * pw, off + (i + 1) * pw))
                            batch = batch[interleave_idx]

                        # add three disjoint masks: main (source=0), branch (source=1), teacher (source=2)
                        response_mask = batch.batch["response_mask"]
                        sources = torch.from_numpy(batch.non_tensor_batch["sources"]).unsqueeze(-1).expand_as(response_mask)
                        batch.batch["main_chain_mask"] = (sources == 0) * response_mask
                        batch.batch["branch_chain_mask"] = (sources == 1) * response_mask
                        batch.batch["teacher_chain_mask"] = (sources == 2) * response_mask

                        T_total = response_mask.sum().item()
                        T_main = batch.batch["main_chain_mask"].sum().item()
                        T_branch = batch.batch["branch_chain_mask"].sum().item()
                        T_teacher = batch.batch["teacher_chain_mask"].sum().item()
                        assert T_main + T_branch + T_teacher == T_total
                        batch.meta_info["T_main"] = T_main
                        batch.meta_info["T_branch"] = T_branch
                        batch.meta_info["T_teacher"] = T_teacher

                        # compute global_valid tokens
                        batch.non_tensor_batch["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).numpy()

                        # recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            main_chain_mask = batch.batch["main_chain_mask"]
                            branch_chain_mask = batch.batch["branch_chain_mask"]
                            T_main = batch.meta_info["T_main"]
                            T_branch = batch.meta_info["T_branch"]
                            eps = 1e-8
                            main_chain_entropy = (entropys * main_chain_mask).sum() / (T_main + eps)
                            branch_chain_entropy = (entropys * branch_chain_mask).sum() / (T_branch + eps)
                            metrics.update({"actor/main_chain/entropy": main_chain_entropy.detach().item()})
                            metrics.update({"actor/branch_chain/entropy": branch_chain_entropy.detach().item()})
                            teacher_chain_mask = batch.batch["teacher_chain_mask"]
                            T_teacher = batch.meta_info["T_teacher"]
                            if T_teacher > 0:
                                teacher_chain_entropy = (entropys * teacher_chain_mask).sum() / (T_teacher + eps)
                                metrics.update({"actor/teacher_chain/entropy": teacher_chain_entropy.detach().item()})
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)

                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)            
                            batch = batch.union(ref_log_prob)

                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                            actor_worker_metrics = list(
                                actor_output.non_tensor_batch["metrics"]
                            )
                            actor_output_metrics = reduce_metrics(actor_worker_metrics)
                            metrics.update(actor_output_metrics)
                            post_sft_optimizer_steps_by_worker = [
                                int(m.get("actor/post_sft_optimizer_steps", 0) or 0)
                                for m in actor_worker_metrics
                            ]
                            steps_with_teacher_by_worker = [
                                int(m.get("actor/steps_with_teacher", 0) or 0)
                                for m in actor_worker_metrics
                            ]
                            post_sft_all_workers_succeeded = bool(
                                steps_with_teacher_by_worker
                            ) and all(
                                teacher_steps > 0
                                and post_sft_steps == teacher_steps
                                for post_sft_steps, teacher_steps in zip(
                                    post_sft_optimizer_steps_by_worker,
                                    steps_with_teacher_by_worker,
                                )
                            )
                            if (
                                forest_suffix_sft_train_tree_ids
                                and post_sft_all_workers_succeeded
                            ):
                                sft_optimizer_steps = min(
                                    post_sft_optimizer_steps_by_worker
                                )
                                suffix_sft_recorded = 0
                                suffix_sft_matured = 0
                                for tid in set(forest_suffix_sft_train_tree_ids):
                                    route = self._forest_sft_tree_id_to_node.get(tid)
                                    if route is None:
                                        continue
                                    tree_key, node_id = route
                                    update = self.forest_pool.record_suffix_sft(
                                        tree_key=tree_key,
                                        node_id=node_id,
                                        current_step=self.global_steps,
                                        count=1,
                                    )
                                    suffix_sft_recorded += update.get(
                                        "suffix_sft_recorded", 0
                                    )
                                    suffix_sft_matured += update.get(
                                        "suffix_sft_matured", 0
                                )
                                metrics.update(self.forest_pool.stats)
                                metrics["sft/optimizer_steps"] = sft_optimizer_steps
                                metrics["prefix_suffix_sft/recorded_entries"] = (
                                    suffix_sft_recorded
                                )
                                metrics["prefix_suffix_sft/matured_entries"] = (
                                    suffix_sft_matured
                                )
                                pending_trees, pending_nodes = (
                                    self.forest_pool.suffix_sft_ready_counts()
                                )
                                metrics.update(
                                    {
                                        "prefix_suffix_sft/pending_trees": pending_trees,
                                        "prefix_suffix_sft/pending_nodes": pending_nodes,
                                    }
                                )
                            elif (
                                forest_suffix_sft_train_tree_ids
                                and sum(steps_with_teacher_by_worker) > 0
                            ):
                                metrics[
                                    "prefix_suffix_sft/record_skipped_optimizer_steps"
                                ] = sum(
                                    max(0, teacher_steps - post_sft_steps)
                                    for post_sft_steps, teacher_steps in zip(
                                        post_sft_optimizer_steps_by_worker,
                                        steps_with_teacher_by_worker,
                                    )
                                )
                                metrics[
                                    "prefix_suffix_sft/record_skipped_entries"
                                ] = len(forest_suffix_sft_train_tree_ids)



                    if is_epoch_last_batch or is_last_step:
                        epoch_sft_phase_metrics = (
                            self._run_epoch_local_suffix_sft_phase(
                                epoch=epoch,
                                epoch_start_step=epoch_start_step,
                                epoch_end_step=self.global_steps,
                                logger=None,
                            )
                        )
                        epoch_sft_phase_ran = True
                        if epoch_sft_phase_metrics:
                            metrics.update(epoch_sft_phase_metrics)

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with marked_timer("testing", timing_raw, color="green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                            self._maybe_save_best_checkpoint(val_metrics)
                        metrics.update(val_metrics)

                    # recoverability evaluation (independent frequency)
                    recoverability_eval_dir = self.config.trainer.get("recoverability_eval_dir", None)
                    recoverability_eval_freq = self.config.trainer.get("recoverability_eval_freq", 0)
                    if (
                        recoverability_eval_dir
                        and self.async_rollout_mode
                        and recoverability_eval_freq > 0
                        and self.global_steps % recoverability_eval_freq == 0
                    ):
                        with marked_timer("recoverability_eval", timing_raw, color="green"):
                            self._recoverability_eval(recoverability_eval_dir)

                    # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                    esi_close_to_expiration = should_save_ckpt_esi(
                        max_steps_duration=self.max_steps_duration,
                        redundant_time=self.config.trainer.esi_redundant_time,
                    )
                    # Check if the conditions for saving a checkpoint are met.
                    # The conditions include a mandatory condition (1) and
                    # one of the following optional conditions (2/3/4):
                    # 1. The save frequency is set to a positive value.
                    # 2. It's the last training step.
                    # 3. The current step number is a multiple of the save frequency.
                    # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                    if self.config.trainer.save_freq > 0 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                        or esi_close_to_expiration
                    ):
                        if esi_close_to_expiration:
                            print("Force saving checkpoint: ESI instance expiration approaching.")
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)
                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_timing_metrics(timing_raw=timing_raw))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if do_profile:
                    self.actor_rollout_wg.stop_profile()
                    if self.use_reference_policy:
                        self.ref_policy_wg.stop_profile()
                    if self.use_critic:
                        self.critic_wg.stop_profile()
                    if self.use_rm:
                        self.rm_wg.stop_profile()

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # # this is experimental and may be changed/removed in the future
                # # in favor of a general-purpose data buffer pool
                # if hasattr(self.train_dataset, "on_batch_end"):
                #     # The dataset may be changed after each training batch
                #     self.train_dataset.on_batch_end(batch=batch)

            if not epoch_sft_phase_ran:
                epoch_end_step = self.global_steps - 1
                self._run_epoch_local_suffix_sft_phase(
                    epoch=epoch,
                    epoch_start_step=epoch_start_step,
                    epoch_end_step=epoch_end_step,
                    logger=logger,
                )
