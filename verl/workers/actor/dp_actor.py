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
Single Process Actor
"""

import itertools
import logging
import os
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import compute_policy_loss, compute_luffy_shaping_loss, kl_penalty
from verl.utils.device import get_device_id, get_device_name, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _config_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


class DataParallelPPOActor(BasePPOActor):
    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
        teacher_sft_optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.teacher_sft_optimizer = teacher_sft_optimizer
        self._ema_auto_coef = None  # initialized lazily in update_policy from deep_grpo config
        self._last_optimizer_stepped = False

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            if "image_bound" in micro_batch["multi_modal_inputs"][0]:  # minicpm-o logic
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = [inputs[key] for inputs in micro_batch["multi_modal_inputs"]]
            else:
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = torch.cat(
                        [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                    )

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch.keys()
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            return entropy, log_probs

    def _optimizer_step(self, optimizer: Optional[torch.optim.Optimizer] = None):
        assert self.config.grad_clip is not None
        optimizer = optimizer if optimizer is not None else self.actor_optimizer

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            optimizer.zero_grad()
            optimizer_stepped = False
        else:
            optimizer.step()
            optimizer_stepped = True
        self._last_optimizer_stepped = optimizer_stepped
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        def _get_micro_batches(data: DataProto) -> Tuple[list, list | None]:
            select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
            batch = data.select(batch_keys=select_keys).batch
            has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch

            if has_multi_modal_inputs:
                all_multi_modal_inputs_list = data.non_tensor_batch["multi_modal_inputs"]
                if use_dynamic_bsz:
                    max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
                    rearranged_text_micro_batches, textual_indices = rearrange_micro_batches(
                        batch=batch, max_token_len=max_token_len
                    )

                    final_micro_batches_list = []
                    for i, text_mb_td in enumerate(rearranged_text_micro_batches):
                        current_original_indices = textual_indices[i]
                        current_mm_inputs_list = [all_multi_modal_inputs_list[idx] for idx in current_original_indices]

                        mb_dict = {k: v for k, v in text_mb_td.items()}
                        mb_dict["multi_modal_inputs"] = current_mm_inputs_list
                        final_micro_batches_list.append(mb_dict)
                    return final_micro_batches_list, textual_indices
                else:
                    num_micro_batches = batch.batch_size[0] // micro_batch_size
                    micro_batches_dp = data.chunk(num_micro_batches)
                    return micro_batches_dp, None
            elif use_dynamic_bsz:
                max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
                micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
                return micro_batches, indices
            else:
                micro_batches = batch.split(micro_batch_size)
                return micro_batches, None

        micro_batches, indices = _get_micro_batches(data)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(
                    micro_batch, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]
            if calculate_entropy:
                entropys = entropys[revert_indices]

        return log_probs, entropys

    def _mini_batch_step(self, mb: DataProto, temperature: float, metrics: dict) -> bool:
        """Single mini-batch: forward + backward (micro-batch accumulation) + optimizer.step()."""
        micro_bsz = self.config.ppo_micro_batch_size_per_gpu
        micro_batches = [mb[i : i + micro_bsz] for i in range(0, len(mb), micro_bsz)]

        # Mini-batch level sums for normalisation (Python float for safe cross-device ops).
        mb_main_token_sum = mb.batch["main_chain_mask"].sum().item()
        mb_branch_token_sum = mb.batch["branch_chain_mask"].sum().item()
        mb_teacher_token_sum = mb.batch["teacher_chain_mask"].sum().item()
        mb_on_policy_sum = mb_main_token_sum + mb_branch_token_sum
        eps = 1e-8

        # Mini-batch level sample/token counts (constant, independent of micro-batch loop)
        sources = mb.non_tensor_batch["sources"]
        metrics["actor/main_chain/mb_tokens"] += mb_main_token_sum
        metrics["actor/branch_chain/mb_tokens"] += mb_branch_token_sum
        metrics["actor/teacher_chain/mb_tokens"] += mb_teacher_token_sum
        metrics["actor/main_chain/mb_samples"] += (sources == 0).sum()
        metrics["actor/branch_chain/mb_samples"] += (sources == 1).sum()
        metrics["actor/teacher_chain/mb_samples"] += (sources == 2).sum()

        branch_loss_coef = self.config.deep_grpo.branch_chain_loss_lambda
        teacher_update_mode = self.config.deep_grpo.get("teacher_update_mode", "joint")
        teacher_loss_reduction = str(
            self.config.deep_grpo.get("teacher_loss_reduction", "separate_stream_mean")
        ).strip()
        assert teacher_loss_reduction in ("separate_stream_mean", "mixed_token_mean"), (
            "teacher_loss_reduction must be 'separate_stream_mean' or "
            f"'mixed_token_mean', got {teacher_loss_reduction}"
        )
        if teacher_update_mode == "joint":
            teacher_loss_type = self.config.deep_grpo.get("teacher_loss_type", None)  # null / "sft" / "luffy" / "snis"
            teacher_loss_coef = self.config.deep_grpo.get("teacher_loss_coef", 1.0)
            auto_teacher_loss_coef = _config_bool(
                self.config.deep_grpo.get("auto_teacher_loss_coef", False)
            )
        else:
            teacher_loss_type = None
            teacher_loss_coef = 0.0
            auto_teacher_loss_coef = False
        if teacher_loss_reduction == "mixed_token_mean":
            assert teacher_update_mode == "joint", (
                "teacher_loss_reduction='mixed_token_mean' requires "
                "teacher_update_mode='joint'"
            )
            assert teacher_loss_type in ("luffy", "snis", None), (
                "teacher_loss_reduction='mixed_token_mean' requires "
                f"teacher_loss_type in ('luffy', 'snis', null), got {teacher_loss_type}"
            )
            assert not auto_teacher_loss_coef, (
                "teacher_loss_reduction='mixed_token_mean' uses natural token "
                "weighting and does not support auto_teacher_loss_coef"
            )
            assert abs(float(teacher_loss_coef) - 1.0) < 1e-12, (
                "teacher_loss_reduction='mixed_token_mean' uses a unified loss; "
                f"teacher_loss_coef must be 1.0, got {teacher_loss_coef}"
            )
        if teacher_loss_type == "snis":
            assert teacher_loss_reduction == "mixed_token_mean", (
                "teacher_loss_type='snis' requires "
                "teacher_loss_reduction='mixed_token_mean'"
            )
        if auto_teacher_loss_coef:
            assert teacher_loss_type is not None, "auto_teacher_loss_coef=True requires teacher_loss_type to be set (sft or luffy)"
        luffy_gamma = self.config.deep_grpo.get("luffy_gamma", 0.1)

        self.actor_optimizer.zero_grad()

        # EMA auto_coef: use previous step's EMA as a constant for all micro-batches
        if self._ema_auto_coef is None:
            self._ema_auto_coef = self.config.deep_grpo.get("auto_coef_init", 0.1)
        auto_coef = self._ema_auto_coef if auto_teacher_loss_coef else None
        if auto_teacher_loss_coef and mb_teacher_token_sum > 0:
            metrics["actor/teacher_chain/auto_coef"] += auto_coef
        step_pg_sum = 0.0
        step_teacher_sum = 0.0

        clip_ratio = self.config.clip_ratio
        clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
        clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
        clip_ratio_c = self.config.get("clip_ratio_c", 3.0)

        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())

            old_log_prob = micro_batch.batch["old_log_probs"]
            advantages = micro_batch.batch["advantages"]
            main_chain_mask = micro_batch.batch["main_chain_mask"]
            branch_chain_mask = micro_batch.batch["branch_chain_mask"]  # model branches only (source=1)
            teacher_chain_mask = micro_batch.batch["teacher_chain_mask"]  # teacher only (source=2)
            on_policy_mask = main_chain_mask + branch_chain_mask

            _, log_prob = self._forward_micro_batch(
                micro_batch=micro_batch.batch, temperature=temperature, calculate_entropy=False
            )

            # Standard PPO loss (used for on-policy tokens: main chain + model branches)
            pg_losses, pg_clips, ppo_kl, pg_clipfrac_lowers = compute_policy_loss(
                old_log_prob=old_log_prob,
                log_prob=log_prob,
                advantages=advantages,
                cliprange=clip_ratio,
                cliprange_low=clip_ratio_low,
                cliprange_high=clip_ratio_high,
                clip_ratio_c=clip_ratio_c,
            )

            # --- Per-mode: compute teacher loss (normalized by mini-batch teacher token count) ---
            if teacher_loss_type == "sft":
                teacher_loss = (-log_prob * teacher_chain_mask).sum() / (mb_teacher_token_sum + eps)
                shaping_weights = None
                teacher_token_losses = None

            elif teacher_loss_type == "luffy":
                shaping_losses, shaping_weights = compute_luffy_shaping_loss(
                    log_prob=log_prob, advantages=advantages, gamma=luffy_gamma,
                )
                teacher_loss = (shaping_losses * teacher_chain_mask).sum() / (mb_teacher_token_sum + eps)
                teacher_token_losses = shaping_losses

            elif teacher_loss_type == "snis":
                # SNIS weighted NLL: -sg[w̃]·logπ. For teacher rows the
                # `advantages` tensor carries the per-row SNIS weight w̃
                # (computed at attach time over the mixed group), broadcast to
                # tokens via response_mask. detach() = stop-grad: w̃ is a pure
                # coefficient, gradient flows only through logπ.
                snis_weights = advantages.detach()
                snis_losses = -snis_weights * log_prob
                teacher_loss = (snis_losses * teacher_chain_mask).sum() / (mb_teacher_token_sum + eps)
                shaping_weights = snis_weights
                teacher_token_losses = snis_losses

            else:
                teacher_loss = 0.0
                shaping_weights = None
                teacher_token_losses = None

            # --- Common loss computation ---
            pg_loss_main = (pg_losses * main_chain_mask).sum() / (mb_main_token_sum + eps)
            pg_loss_branch = (pg_losses * branch_chain_mask).sum() / (mb_branch_token_sum + eps)
            student_loss_num = (
                (pg_losses * main_chain_mask).sum()
                + branch_loss_coef * (pg_losses * branch_chain_mask).sum()
            )
            student_token_den = (
                mb_main_token_sum + branch_loss_coef * mb_branch_token_sum
            )

            if teacher_loss_reduction == "mixed_token_mean":
                assert teacher_loss_type in ("luffy", "snis", None)
                teacher_loss_num = (
                    (teacher_token_losses * teacher_chain_mask).sum()
                    if teacher_token_losses is not None
                    else 0.0
                )
                mixed_loss_num = student_loss_num + teacher_loss_num
                mixed_token_den = student_token_den + (
                    mb_teacher_token_sum if teacher_token_losses is not None else 0.0
                )
                pg_loss = student_loss_num / (student_token_den + eps)
                loss = mixed_loss_num / (mixed_token_den + eps)
                metrics["actor/mixed_token_loss"] += loss.detach().item()

            else:
                pg_loss = pg_loss_main + branch_loss_coef * pg_loss_branch

                # Auto-scale teacher loss to match PPO magnitude (teacher_loss_coef becomes relative weight).
                # auto_coef is frozen from previous step's EMA — constant across all micro-batches
                # to preserve gradient accumulation equivalence.
                if auto_teacher_loss_coef:
                    loss = pg_loss + teacher_loss_coef * auto_coef * teacher_loss
                    step_pg_sum += abs(pg_loss.detach().item())
                    step_teacher_sum += teacher_loss.detach().abs().item()
                else:
                    loss = pg_loss + teacher_loss_coef * teacher_loss

            if self.config.use_kl_loss:
                ref_log_prob = micro_batch.batch["ref_log_prob"]
                kld = kl_penalty(
                    logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                )
                # Exclude teacher tokens from KL loss: KL penalises deviation from ref,
                # but we want deviation for off-policy teacher tokens
                if teacher_loss_reduction == "mixed_token_mean":
                    kl_loss = (
                        (kld * main_chain_mask).sum()
                        + branch_loss_coef * (kld * branch_chain_mask).sum()
                    ) / (student_token_den + eps)
                else:
                    kl_loss_main = (kld * main_chain_mask).sum() / (mb_main_token_sum + eps)
                    kl_loss_branch = (kld * branch_chain_mask).sum() / (mb_branch_token_sum + eps)
                    kl_loss = kl_loss_main + branch_loss_coef * kl_loss_branch
                loss = loss + self.config.kl_loss_coef * kl_loss
                metrics["actor/kl_loss"] += kl_loss.detach().item()

            loss.backward()

            # --- Accumulate metrics (all normalized by mini-batch token counts) ---
            metrics["actor/total_loss"] += loss.detach().item()
            metrics["actor/pg_loss"] += (pg_loss_main.detach().item() + branch_loss_coef * pg_loss_branch.detach().item())
            metrics["actor/pg_clipfrac"] += (pg_clips * on_policy_mask).sum().detach().item() / (mb_on_policy_sum + eps)
            metrics["actor/ppo_kl"] += (ppo_kl * on_policy_mask).sum().detach().item() / (mb_on_policy_sum + eps)
            metrics["actor/pg_clipfrac_lower"] += (pg_clipfrac_lowers * on_policy_mask).sum().detach().item() / (mb_on_policy_sum + eps)

            metrics["actor/main_chain/pg_loss"] += pg_loss_main.detach().item()
            metrics["actor/main_chain/pg_clipfrac"] += (pg_clips * main_chain_mask).sum().detach().item() / (mb_main_token_sum + eps)
            metrics["actor/main_chain/ppo_kl"] += (ppo_kl * main_chain_mask).sum().detach().item() / (mb_main_token_sum + eps)
            metrics["actor/main_chain/pg_clipfrac_lower"] += (pg_clipfrac_lowers * main_chain_mask).sum().detach().item() / (mb_main_token_sum + eps)

            metrics["actor/branch_chain/pg_loss"] += pg_loss_branch.detach().item()
            metrics["actor/branch_chain/pg_clipfrac"] += (pg_clips * branch_chain_mask).sum().detach().item() / (mb_branch_token_sum + eps)
            metrics["actor/branch_chain/ppo_kl"] += (ppo_kl * branch_chain_mask).sum().detach().item() / (mb_branch_token_sum + eps)
            metrics["actor/branch_chain/pg_clipfrac_lower"] += (pg_clipfrac_lowers * branch_chain_mask).sum().detach().item() / (mb_branch_token_sum + eps)

            if teacher_loss_type == "sft":
                metrics["actor/teacher_chain/sft_loss"] += teacher_loss.detach().item()
            elif teacher_loss_type == "luffy":
                metrics["actor/teacher_chain/luffy_loss"] += teacher_loss.detach().item()
                metrics["actor/teacher_chain/mean_shaping_weight"] += (shaping_weights * teacher_chain_mask).sum().detach().item() / (mb_teacher_token_sum + eps)
            elif teacher_loss_type == "snis":
                metrics["actor/teacher_chain/snis_loss"] += teacher_loss.detach().item()
                # Sentinel: should sit in (0, e^{ΔR_max/β}] ≈ (0, 2.72] for β=1.
                # Out-of-range = weight pipeline bug.
                metrics["actor/teacher_chain/snis_weight_mean"] += (shaping_weights * teacher_chain_mask).sum().detach().item() / (mb_teacher_token_sum + eps)
                # ρ_T≡1 residual-risk sentinel: teacher comprehensibility under
                # the student policy. Persistently very low = teacher suffixes
                # are "scripture" the student cannot absorb.
                metrics["actor/teacher_chain/mean_logp"] += (log_prob.detach() * teacher_chain_mask).sum().item() / (mb_teacher_token_sum + eps)

        # Update EMA auto_coef from this step's full-mini-batch ratio
        if auto_teacher_loss_coef and step_teacher_sum > eps:
            step_ratio = step_pg_sum / step_teacher_sum
            ema_decay = self.config.deep_grpo.get("auto_coef_ema_decay", 0.9)
            self._ema_auto_coef = ema_decay * self._ema_auto_coef + (1 - ema_decay) * step_ratio

        grad_norm = self._optimizer_step()
        metrics["actor/grad_norm"] += grad_norm.detach().item()
        return self._last_optimizer_stepped

    def _teacher_sft_mini_batch_step(
        self,
        mb: DataProto,
        temperature: float,
        metrics: dict,
    ) -> bool:
        """Run one post-PPO SFT optimizer step on teacher suffix tokens."""
        assert self.teacher_sft_optimizer is not None, (
            "teacher_update_mode='post_ppo' requires a separate teacher_sft_optimizer"
        )
        micro_bsz = self.config.ppo_micro_batch_size_per_gpu
        micro_batches = [mb[i : i + micro_bsz] for i in range(0, len(mb), micro_bsz)]

        mb_teacher_token_sum = mb.batch["teacher_chain_mask"].sum().item()
        if mb_teacher_token_sum <= 0:
            return False

        sources = mb.non_tensor_batch["sources"]
        metrics["actor/teacher_chain/mb_tokens"] += mb_teacher_token_sum
        metrics["actor/teacher_chain/mb_samples"] += (sources == 2).sum()

        eps = 1e-8
        self.teacher_sft_optimizer.zero_grad()

        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            teacher_chain_mask = micro_batch.batch["teacher_chain_mask"]

            _, log_prob = self._forward_micro_batch(
                micro_batch=micro_batch.batch,
                temperature=temperature,
                calculate_entropy=False,
            )
            teacher_loss = (
                -log_prob * teacher_chain_mask
            ).sum() / (mb_teacher_token_sum + eps)
            teacher_loss.backward()

            metrics["actor/teacher_chain/sft_loss"] += teacher_loss.detach().item()
            metrics["actor/teacher_chain/post_sft_loss"] += (
                teacher_loss.detach().item()
            )

        grad_norm = self._optimizer_step(self.teacher_sft_optimizer)
        metrics["actor/teacher_chain/post_sft_grad_norm"] += (
            grad_norm.detach().item()
        )
        metrics["actor/teacher_chain/post_sft_lr"] += (
            self.teacher_sft_optimizer.param_groups[0]["lr"]
        )
        return self._last_optimizer_stepped

    @staticmethod
    def _balanced_slice(N: int, num_mbs: int, i: int) -> tuple:
        """Return (start, end) for the i-th of num_mbs balanced partitions of N items."""
        base = N // num_mbs
        extra = N % num_mbs
        start = i * base + min(i, extra)
        end = start + base + (1 if i < extra else 0)
        return start, end

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
            "main_chain_mask",
            "branch_chain_mask",
            "teacher_chain_mask",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")

        data = data.select(batch_keys=select_keys)

        # --- Separate three data streams ---
        sources = data.non_tensor_batch["sources"]
        S = self.config.ppo_mini_batch_size
        S_teacher = int(self.config.deep_grpo.get("teacher_mini_batch_size", S))
        teacher_loss_type = self.config.deep_grpo.get("teacher_loss_type", None)
        teacher_update_mode = self.config.deep_grpo.get("teacher_update_mode", "joint")
        teacher_loss_reduction = str(
            self.config.deep_grpo.get("teacher_loss_reduction", "separate_stream_mean")
        ).strip()
        teacher_allow_partial_batch = _config_bool(
            self.config.deep_grpo.get("teacher_allow_partial_batch", False)
        )
        auto_teacher_loss_coef = _config_bool(
            self.config.deep_grpo.get("auto_teacher_loss_coef", False)
        )
        assert teacher_update_mode in ("joint", "post_ppo"), (
            f"teacher_update_mode must be 'joint' or 'post_ppo', got {teacher_update_mode}"
        )
        assert teacher_loss_reduction in ("separate_stream_mean", "mixed_token_mean"), (
            "teacher_loss_reduction must be 'separate_stream_mean' or "
            f"'mixed_token_mean', got {teacher_loss_reduction}"
        )
        mixed_token_mean = (
            teacher_update_mode == "joint"
            and teacher_loss_reduction == "mixed_token_mean"
        )
        if mixed_token_mean:
            assert teacher_loss_type in ("luffy", "snis", None), (
                "teacher_loss_reduction='mixed_token_mean' requires "
                f"teacher_loss_type in ('luffy', 'snis', null), got {teacher_loss_type}"
            )
            assert abs(float(self.config.deep_grpo.get("teacher_loss_coef", 1.0)) - 1.0) < 1e-12, (
                "teacher_loss_reduction='mixed_token_mean' uses natural token "
                "weighting; teacher_loss_coef must be 1.0"
            )
            assert not auto_teacher_loss_coef, (
                "teacher_loss_reduction='mixed_token_mean' does not support "
                "auto_teacher_loss_coef"
            )
        if teacher_update_mode == "post_ppo":
            assert self.teacher_sft_optimizer is not None, (
                "teacher_update_mode='post_ppo' requires a separate "
                "teacher_sft_optimizer"
            )
            assert teacher_loss_type == "sft", (
                "teacher_update_mode='post_ppo' only supports teacher_loss_type='sft'"
            )
            assert S_teacher > 0, (
                "teacher_update_mode='post_ppo' requires "
                f"teacher_mini_batch_size > 0, got {S_teacher}"
            )
            assert not auto_teacher_loss_coef, (
                "teacher_update_mode='post_ppo' does not support "
                "auto_teacher_loss_coef; post-PPO SFT uses teacher_sft_optimizer directly."
            )

        main_indices = np.where(sources == 0)[0]
        branch_indices = np.where(sources == 1)[0]
        if (
            teacher_update_mode == "post_ppo"
            and "__suffix_sft__" in data.non_tensor_batch
        ):
            suffix_sft_flags = np.asarray(
                data.non_tensor_batch["__suffix_sft__"]
            ).astype(bool)
            teacher_indices = np.where((sources == 2) & suffix_sft_flags)[0]
        else:
            teacher_indices = np.where(sources == 2)[0]

        main_data = data[main_indices]
        branch_data = data[branch_indices] if len(branch_indices) > 0 else None
        teacher_data = data[teacher_indices] if len(teacher_indices) > 0 else None

        N_main = len(main_data)
        N_branch = len(branch_data) if branch_data is not None else 0
        N_teacher = len(teacher_data) if teacher_data is not None else 0

        # Mini-batch counts: each stream independently sized, capped by num_main_mbs.
        num_main_mbs = N_main // S
        num_branch_mbs = min(num_main_mbs, N_branch // S)
        if mixed_token_mean:
            num_teacher_mbs = num_main_mbs if N_teacher > 0 else 0
        elif N_teacher > 0:
            if teacher_update_mode == "post_ppo":
                if teacher_allow_partial_batch:
                    num_teacher_mbs = (N_teacher + S_teacher - 1) // S_teacher
                else:
                    num_teacher_mbs = N_teacher // S_teacher
            else:
                num_teacher_mbs = min(num_main_mbs, N_teacher // S_teacher)
        else:
            num_teacher_mbs = 0

        # --- Initialise metrics accumulators ---
        metrics = {
            "actor/total_loss": 0.0,
            "actor/mixed_token_loss": 0.0,
            "actor/pg_loss": 0.0,
            "actor/pg_clipfrac": 0.0,
            "actor/ppo_kl": 0.0,
            "actor/pg_clipfrac_lower": 0.0,

            "actor/main_chain/pg_loss": 0.0,
            "actor/main_chain/pg_clipfrac": 0.0,
            "actor/main_chain/ppo_kl": 0.0,
            "actor/main_chain/pg_clipfrac_lower": 0.0,

            "actor/branch_chain/pg_loss": 0.0,
            "actor/branch_chain/pg_clipfrac": 0.0,
            "actor/branch_chain/ppo_kl": 0.0,
            "actor/branch_chain/pg_clipfrac_lower": 0.0,

            "actor/teacher_chain/sft_loss": 0.0,
            "actor/teacher_chain/luffy_loss": 0.0,
            "actor/teacher_chain/snis_loss": 0.0,
            "actor/teacher_chain/snis_weight_mean": 0.0,
            "actor/teacher_chain/mean_logp": 0.0,
            "actor/teacher_chain/mean_shaping_weight": 0.0,
            "actor/teacher_chain/auto_coef": 0.0,
            "actor/teacher_chain/mb_tokens": 0.0,
            "actor/teacher_chain/mb_samples": 0.0,
            "actor/teacher_chain/post_sft_loss": 0.0,
            "actor/teacher_chain/post_sft_grad_norm": 0.0,
            "actor/teacher_chain/post_sft_lr": 0.0,

            "actor/grad_norm": 0.0,
            "actor/main_chain/mb_tokens": 0.0,
            "actor/branch_chain/mb_tokens": 0.0,
            "actor/main_chain/mb_samples": 0.0,
            "actor/branch_chain/mb_samples": 0.0,
            "actor/kl_coef": self.config.kl_loss_coef,
        }
        if self.config.use_kl_loss:
            metrics["actor/kl_loss"] = 0.0

        total_steps = 0
        steps_with_main = 0
        steps_with_branch = 0
        steps_with_teacher = 0
        actor_optimizer_steps = 0
        post_sft_optimizer_steps = 0

        for ppo_epoch in range(self.config.ppo_epochs):
            main_perm = torch.randperm(N_main)
            branch_perm = torch.randperm(N_branch) if num_branch_mbs > 0 else None
            teacher_perm = torch.randperm(N_teacher) if num_teacher_mbs > 0 else None

            for i in range(num_main_mbs):
                ms, me = self._balanced_slice(N_main, num_main_mbs, i)
                mb = main_data[main_perm[ms:me]]

                if i < num_branch_mbs:
                    bs, be = self._balanced_slice(N_branch, num_branch_mbs, i)
                    mb = DataProto.concat([mb, branch_data[branch_perm[bs:be]]])

                teacher_added = False
                if mixed_token_mean and N_teacher > 0:
                    ts, te = self._balanced_slice(N_teacher, num_main_mbs, i)
                    if te > ts:
                        mb = DataProto.concat([mb, teacher_data[teacher_perm[ts:te]]])
                        teacher_added = True
                elif teacher_update_mode == "joint" and i < num_teacher_mbs:
                    ts, te = self._balanced_slice(N_teacher, num_teacher_mbs, i)
                    mb = DataProto.concat([mb, teacher_data[teacher_perm[ts:te]]])
                    teacher_added = True

                if self._mini_batch_step(mb, temperature, metrics):
                    actor_optimizer_steps += 1
                total_steps += 1
                steps_with_main += 1
                if i < num_branch_mbs:
                    steps_with_branch += 1
                if teacher_added:
                    steps_with_teacher += 1

        if teacher_update_mode == "post_ppo" and num_teacher_mbs > 0:
            teacher_perm = torch.randperm(N_teacher)
            for i in range(num_teacher_mbs):
                ts, te = self._balanced_slice(N_teacher, num_teacher_mbs, i)
                teacher_mb = teacher_data[teacher_perm[ts:te]]
                if self._teacher_sft_mini_batch_step(
                    teacher_mb,
                    temperature,
                    metrics,
                ):
                    post_sft_optimizer_steps += 1
                steps_with_teacher += 1

        # --- Normalise accumulated metrics ---
        if total_steps > 0 or steps_with_teacher > 0:
            for k in metrics:
                if k == "actor/kl_coef":
                    continue
                if "teacher_chain" in k:
                    if steps_with_teacher > 0:
                        metrics[k] /= steps_with_teacher
                elif "branch_chain" in k:
                    if steps_with_branch > 0:
                        metrics[k] /= steps_with_branch
                elif "main_chain" in k:
                    if steps_with_main > 0:
                        metrics[k] /= steps_with_main
                else:
                    if total_steps > 0:
                        metrics[k] /= total_steps

        metrics["actor/total_mini_batch_steps"] = total_steps
        metrics["actor/steps_with_main"] = steps_with_main
        metrics["actor/steps_with_branch"] = steps_with_branch
        metrics["actor/steps_with_teacher"] = steps_with_teacher
        metrics["actor/optimizer_steps"] = actor_optimizer_steps
        metrics["actor/post_sft_optimizer_steps"] = post_sft_optimizer_steps

        return metrics
