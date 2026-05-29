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

import logging
import os

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.device import get_device_id, get_device_name, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
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

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
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
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        if self.config.tis_imp_ratio_cap > 0:
            assert "rollout_log_probs" in data.batch.keys(), (
                "Truncated Importance Sampling (TIS) requires to configure "
                "`actor_rollout_ref.rollout.calculate_log_probs=True` "
                "and is not currently supported in Server mode (agent loop)."
            )
            select_keys.append("rollout_log_probs")

        # ============ [EasyOPD:G-OPD] Include additional log probs for OPD ============
        if self.config.policy_loss.only_reverse_kl_advantages and "ref_log_prob" in data.batch.keys():
            if "ref_log_prob" not in select_keys:
                select_keys.append("ref_log_prob")
        if "base_log_prob" in data.batch.keys():
            select_keys.append("base_log_prob")
        if "base_ref_log_prob" in data.batch.keys():
            select_keys.append("base_ref_log_prob")
        # ============ [EasyOPD:G-OPD] End ============

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        # ============ [EasyOPD:G-OPD] Include opd_teacher for multi-teacher distillation ============
        if "opd_teacher" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("opd_teacher")
        # ============ [EasyOPD:G-OPD] End ============

        # ============ [EasyOPD:Vision-OPD] Include self-distillation keys ============
        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
        vopd_enabled = loss_mode == "vopd"
        if vopd_enabled:
            vopd_required_keys = [
                "teacher_input_ids",
                "teacher_attention_mask",
                "teacher_position_ids",
                "teacher_response_start_idx",
                "self_distillation_mask",
            ]
            for key in vopd_required_keys:
                if key in data.batch.keys():
                    select_keys.append(key)
            if "teacher_multi_modal_inputs" in data.non_tensor_batch.keys():
                non_tensor_select_keys.append("teacher_multi_modal_inputs")
            if "rollout_is_weights" in data.batch.keys():
                select_keys.append("rollout_is_weights")
        # ============ [EasyOPD:Vision-OPD] End ============

        # ============ [EasyOPD:OPCD] Include context distillation keys ============
        if "exp_log_probs" in data.batch.keys():
            select_keys.append("exp_log_probs")
        if self.config.get("kl_loss_type", "") == "full" and self.config.get("kl_topk", 0) > 0:
            if "kl_topk_indices" in data.batch.keys():
                select_keys.append("kl_topk_indices")
        # ============ [EasyOPD:OPCD] End ============

        # ============ [EasyOPD:OPSA] Include teacher logits for self-distillation ============
        opsa_enabled = self.config.get("opsa_enable", False)
        if opsa_enabled:
            if "opsa_teacher_logits" in data.batch.keys():
                select_keys.append("opsa_teacher_logits")
            if "opsa_query_type" in data.batch.keys():
                select_keys.append("opsa_query_type")
        # ============ [EasyOPD:OPSA] End ============

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    rollout_log_probs = model_inputs["rollout_log_probs"] if self.config.tis_imp_ratio_cap > 0 else None
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                    if on_policy:
                        old_log_prob = log_prob.detach()
                    else:
                        old_log_prob = model_inputs["old_log_probs"]

                    # ============ [EasyOPD:G-OPD] Compute G-OPD advantages ============
                    if self.config.policy_loss.only_reverse_kl_advantages:
                        from easyopd.methods.g_opd.core import (
                            compute_g_opd_advantages,
                            compute_multi_teacher_advantages,
                            compute_standard_opd_advantages,
                        )

                        if "base_log_prob" in model_inputs and "base_ref_log_prob" in model_inputs:
                            lambda_vals = self.config.policy_loss.lambda_vals

                            if self.config.policy_loss.multi_teacher_distill:
                                # Multi-teacher distillation
                                opd_teacher = model_inputs.get("opd_teacher", None)
                                if opd_teacher is not None:
                                    advantages = compute_multi_teacher_advantages(
                                        old_log_probs=old_log_prob,
                                        ref_log_prob=model_inputs["ref_log_prob"],
                                        base_ref_log_prob=model_inputs["base_ref_log_prob"],
                                        base_log_prob=model_inputs["base_log_prob"],
                                        opd_teacher=opd_teacher,
                                        lambda_vals=lambda_vals,
                                    )
                                else:
                                    advantages = compute_standard_opd_advantages(
                                        old_log_probs=old_log_prob,
                                        ref_log_prob=model_inputs["ref_log_prob"],
                                    )
                            else:
                                # Single-teacher G-OPD / ExOPD
                                advantages = compute_g_opd_advantages(
                                    old_log_probs=old_log_prob,
                                    ref_log_prob=model_inputs["ref_log_prob"],
                                    base_log_prob=model_inputs["base_log_prob"],
                                    lambda_vals=lambda_vals,
                                )
                        else:
                            # Standard OPD without base model normalization
                            advantages = compute_standard_opd_advantages(
                                old_log_probs=old_log_prob,
                                ref_log_prob=model_inputs["ref_log_prob"],
                            )
                    # ============ [EasyOPD:G-OPD] End ============

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla
                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    # ============ [EasyOPD:Vision-OPD] VOPD loss mode ============
                    if loss_mode == "vopd":
                        from easyopd.methods.vision_opd.core import compute_self_distillation_loss as vopd_loss_fn

                        self_distillation_cfg = getattr(self.config, "self_distillation", None)
                        if self_distillation_cfg is None:
                            raise ValueError("loss_mode='vopd' requires actor.self_distillation config.")

                        self_distillation_mask = model_inputs.get("self_distillation_mask", None)
                        rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                        # Determine if we need GRPO fallback for samples without teacher
                        policy_fallback_mask = None
                        if self_distillation_mask is not None:
                            policy_fallback_mask = (self_distillation_mask <= 0.5).to(response_mask.dtype)

                        # Teacher forward pass
                        teacher_module = getattr(self, "teacher_module", None) or self.actor_module
                        distill_topk = self_distillation_cfg.get("distillation_topk", None)
                        full_logit = self_distillation_cfg.get("full_logit_distillation", True)
                        return_all_logps = full_logit and not distill_topk

                        teacher_inputs = {
                            "responses": model_inputs["responses"],
                            "input_ids": model_inputs["teacher_input_ids"],
                            "attention_mask": model_inputs["teacher_attention_mask"],
                            "position_ids": model_inputs["teacher_position_ids"],
                        }
                        if "teacher_multi_modal_inputs" in model_inputs:
                            teacher_inputs["multi_modal_inputs"] = model_inputs["teacher_multi_modal_inputs"]

                        with torch.no_grad():
                            teacher_entropy, teacher_log_prob = self._forward_micro_batch(
                                teacher_inputs,
                                temperature=temperature,
                                calculate_entropy=False,
                            )

                        # Compute VOPD distillation loss
                        vopd_loss, vopd_metrics = vopd_loss_fn(
                            student_log_probs=log_prob,
                            teacher_log_probs=teacher_log_prob,
                            response_mask=response_mask,
                            alpha=self_distillation_cfg.get("alpha", 0.5),
                            full_logit_distillation=False,  # Use token-level for simplicity
                            distillation_topk=None,
                            distillation_add_tail=True,
                            is_clip=self_distillation_cfg.get("is_clip", None),
                            old_log_probs=old_log_prob,
                            self_distillation_mask=self_distillation_mask,
                            rollout_is_weights=rollout_is_weights,
                        )

                        # Compute GRPO loss for fallback samples (no teacher)
                        grpo_loss = torch.tensor(0.0, device=log_prob.device)
                        if policy_fallback_mask is not None and policy_fallback_mask.any().item():
                            grpo_policy_loss_fn = get_policy_loss_fn("vanilla")
                            grpo_loss, _, _, _ = grpo_policy_loss_fn(
                                old_log_prob=old_log_prob,
                                log_prob=log_prob,
                                advantages=advantages,
                                response_mask=response_mask * policy_fallback_mask.unsqueeze(1) if policy_fallback_mask.dim() == 1 else response_mask * policy_fallback_mask,
                                loss_agg_mode=loss_agg_mode,
                                config=self.config,
                                rollout_log_probs=rollout_log_probs,
                            )

                        # Combined loss
                        gamma = self_distillation_cfg.get("gamma", 1.0)
                        policy_loss = vopd_loss * gamma + grpo_loss

                        micro_batch_metrics["actor/vopd_loss"] = vopd_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/grpo_fallback_loss"] = grpo_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics.update({k: v for k, v in vopd_metrics.items()})

                        if self.config.use_dynamic_bsz:
                            loss = policy_loss * loss_scale_factor
                        else:
                            loss = policy_loss * loss_scale_factor

                        loss.backward()

                        # Accumulate metrics
                        for key, value in micro_batch_metrics.items():
                            if key not in metrics:
                                metrics[key] = 0.0
                            metrics[key] += value if isinstance(value, (int, float)) else value

                        continue  # Skip the normal loss path below
                    # ============ [EasyOPD:Vision-OPD] End ============

                    # ============ [EasyOPD:OPCD] Context distillation KL loss ============
                    stage_merge = data.meta_info.get("stage_merge", False) if hasattr(data, 'meta_info') else False
                    if not stage_merge:
                        stage_merge = model_inputs.get("__stage_merge__", False)
                    if stage_merge:
                        from easyopd.methods.opcd.core import kl_penalty as opcd_kl_penalty

                        on_policy_merge = data.meta_info.get("on_policy_merge", True) if hasattr(data, 'meta_info') else True
                        if not on_policy_merge:
                            on_policy_merge = model_inputs.get("__on_policy_merge__", True)

                        exp_log_prob = model_inputs.get("exp_log_probs", None)
                        if exp_log_prob is None:
                            exp_log_prob = torch.zeros_like(log_prob)

                        kl_loss_type = self.config.get("kl_loss_type", "full")
                        kl_renorm_topk = self.config.get("kl_renorm_topk", False)

                        if on_policy_merge:
                            kld = opcd_kl_penalty(
                                logprob=log_prob,
                                ref_logprob=exp_log_prob,
                                kl_penalty_type=kl_loss_type,
                                kl_renorm_topk=kl_renorm_topk,
                            )
                        else:
                            kld = opcd_kl_penalty(
                                logprob=exp_log_prob,
                                ref_logprob=log_prob,
                                kl_penalty_type=kl_loss_type,
                                kl_renorm_topk=kl_renorm_topk,
                            )

                        policy_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        entropy_agg = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        if self.config.use_dynamic_bsz:
                            loss = policy_loss * (micro_batch_size / actual_ppo_mini_batch_size)
                        else:
                            loss = policy_loss / self.gradient_accumulation
                        loss.backward()

                        micro_batch_metrics = {
                            "actor/policy_loss": policy_loss.detach().item(),
                            "actor/entropy": entropy_agg.detach().item(),
                        }
                        append_to_dict(metrics, micro_batch_metrics)
                        continue  # Skip the normal loss path below
                    # ============ [EasyOPD:OPCD] End ============

                    # ============ [EasyOPD:OPSA] On-Policy Self-Distillation loss ============
                    opsa_enabled = self.config.get("opsa_enable", False)
                    if opsa_enabled and "opsa_teacher_logits" in model_inputs:
                        from easyopd.methods.opsa.core import opsa_loss

                        teacher_logits = model_inputs["opsa_teacher_logits"]
                        # Get student logits from current forward pass
                        student_logits = model_output.logits if hasattr(model_output, 'logits') else None

                        if student_logits is not None:
                            opsa_kl_loss, opsa_metrics = opsa_loss(
                                student_logits=student_logits,
                                teacher_logits=teacher_logits,
                                response_mask=response_mask,
                                temperature=float(self.config.get("opsa_temperature", 1.0)),
                                window_size=int(self.config.get("opsa_window_size", 32)),
                                decay_type=self.config.get("opsa_decay_type", "linear"),
                                min_weight=float(self.config.get("opsa_min_weight", 0.1)),
                                use_window_weighting=self.config.get("opsa_use_window_weighting", True),
                                loss_agg_mode=self.config.get("opsa_loss_agg_mode", "token-mean"),
                            )
                            opsa_coef = float(self.config.get("opsa_distillation_loss_coef", 1.0))
                            policy_loss = opsa_coef * opsa_kl_loss
                            metrics.update(opsa_metrics)
                            continue
                    # ============ [EasyOPD:OPSA] End ============

                    policy_loss_fn = get_policy_loss_fn(loss_mode)
                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_log_probs=rollout_log_probs,
                    )

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    loss.backward()

                    micro_batch_metrics.update(
                        {
                            "actor/pg_loss": pg_loss.detach().item() * loss_scale_factor,
                            "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                            "actor/ppo_kl": ppo_kl.detach().item(),
                            "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        }
                    )
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)

                # ============ [EasyOPD:Vision-OPD] Teacher EMA update after optimizer step ============
                if vopd_enabled and hasattr(self, "teacher_module") and self.teacher_module is not None:
                    self_distillation_cfg = getattr(self.config, "self_distillation", None)
                    if self_distillation_cfg is not None:
                        teacher_reg = self_distillation_cfg.get("teacher_regularization", "ema")
                        if teacher_reg == "ema":
                            from easyopd.methods.vision_opd.core import ema_update_teacher
                            update_rate = self_distillation_cfg.get("teacher_update_rate", 0.05)
                            ema_update_teacher(self.teacher_module, self.actor_module, update_rate)
                        elif teacher_reg == "progressive":
                            teacher_update_interval = self_distillation_cfg.get("teacher_update_interval", None)
                            if teacher_update_interval is not None:
                                global_steps = data.meta_info.get("global_steps", 0)
                                if global_steps % teacher_update_interval == 0:
                                    from easyopd.methods.vision_opd.core import progressive_update_teacher
                                    progressive_update_teacher(self.teacher_module, self.actor_module)
                # ============ [EasyOPD:Vision-OPD] End ============

        self.actor_optimizer.zero_grad()
        return metrics
