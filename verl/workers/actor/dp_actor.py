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

    # Class-level guard so that the OPSA top-K-disabled-under-SP warning is emitted
    # only once per process (avoids spam across micro-batches / training steps).
    _opsa_sp_warning_emitted = False

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
        self, micro_batch, temperature, calculate_entropy=False,
        opsa_topk_k=None,
        opsa_gather_indices=None,
    ):
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)

        When ``opsa_topk_k`` or ``opsa_gather_indices`` is provided (used for OPSA's
        topk_logits_k full-vocab KL alignment), additionally returns a third element
        ``opsa_extra`` (dict) with optional keys:
            - ``topk_log_probs``: (bs, response_len, K) — top-K log-softmax values from this model.
            - ``topk_indices``:   (bs, response_len, K) — corresponding vocab indices (int64).
            - ``gathered_log_probs``: (bs, response_len, K) — log-softmax gathered at
              ``opsa_gather_indices`` positions (used by the student to align with the
              teacher's top-K vocab subset).
        Falls back gracefully (``opsa_extra={}``) under ulysses SP / fused kernels /
        rmpad which we currently do not support for top-K extraction; callers should
        treat the absence of these keys as "use per-token mixed KL fallback".
        """
        opsa_caller_requested_extra = (opsa_topk_k is not None) or (opsa_gather_indices is not None)
        opsa_extra_requested = opsa_caller_requested_extra
        opsa_extra: dict = {}
        # OPSA top-K extraction is currently disabled under ulysses SP (rmpad+SP layout makes
        # the gather/topk extraction non-trivial). Caller should use per-token mixed KL.
        # We still honour the caller's 3-tuple return contract — just hand back an empty dict.
        if opsa_extra_requested and self.use_ulysses_sp:
            if not DataParallelPPOActor._opsa_sp_warning_emitted:
                logger.warning(
                    "[EasyOPD:OPSA] Ulysses sequence parallelism is active; top-K logits disabled. "
                    "OPSA will fall back to per-token KL approximation with reduced accuracy."
                )
                DataParallelPPOActor._opsa_sp_warning_emitted = True
            opsa_extra_requested = False
            opsa_topk_k = None
            opsa_gather_indices = None
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

                # ============ [EasyOPD:OPSA] top-K logits extraction (rmpad, no SP) ============
                # We already excluded use_ulysses_sp above; also skip when fused kernels
                # are in use because the full-vocab logits are not available there.
                if opsa_extra_requested and not self.use_fused_kernels:
                    # Memory-efficient log-softmax via logsumexp: avoids materializing the
                    # full (total_nnz, V) log-softmax tensor, which would be prohibitive for
                    # vocab sizes >> response top-K (e.g. Qwen3 V~152k vs K=512).
                    lse_rmpad = torch.logsumexp(logits_rmpad, dim=-1, keepdim=True)  # (total_nnz, 1)
                    if opsa_topk_k is not None and opsa_topk_k > 0:
                        tk = min(int(opsa_topk_k), logits_rmpad.size(-1))
                        tv_rmpad_logits, ti_rmpad = torch.topk(logits_rmpad, k=tk, dim=-1)
                        tv_rmpad = (tv_rmpad_logits - lse_rmpad).to(torch.float32)
                        # pad_input expects (total_nnz, hidden_size) -> (bsz, seqlen, hidden_size)
                        tv_full = pad_input(
                            hidden_states=tv_rmpad,
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen,
                        )
                        ti_full = pad_input(
                            hidden_states=ti_rmpad.to(tv_rmpad.dtype),
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen,
                        )
                        opsa_extra["topk_log_probs"] = tv_full[:, -response_length - 1 : -1, :]
                        opsa_extra["topk_indices"] = ti_full[:, -response_length - 1 : -1, :].to(torch.int64)
                    if opsa_gather_indices is not None:
                        K_dim = opsa_gather_indices.size(-1)
                        gi_full = torch.zeros(
                            (batch_size, seqlen, K_dim),
                            dtype=torch.int64,
                            device=logits_rmpad.device,
                        )
                        gi_full[:, -response_length - 1 : -1, :] = opsa_gather_indices.to(
                            device=logits_rmpad.device, dtype=torch.int64
                        )
                        # Unpad to rmpad layout (total_nnz, K)
                        gi_rmpad = index_first_axis(
                            rearrange(gi_full, "b s k -> (b s) k"), indices
                        )
                        gathered_logits_rmpad = logits_rmpad.gather(dim=-1, index=gi_rmpad)
                        gathered_rmpad = (gathered_logits_rmpad - lse_rmpad).to(torch.float32)
                        gathered_full = pad_input(
                            hidden_states=gathered_rmpad,
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen,
                        )
                        opsa_extra["gathered_log_probs"] = gathered_full[:, -response_length - 1 : -1, :]
                # ============ [EasyOPD:OPSA] End ============

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

                    # ============ [EasyOPD:OPSA] top-K logits extraction (non-rmpad) ============
                    if opsa_extra_requested:
                        # Memory-efficient log-softmax via logsumexp constant subtraction;
                        # avoids materializing the full (B, T, V) log-softmax tensor.
                        lse = torch.logsumexp(logits, dim=-1, keepdim=True)  # (B, T, 1)
                        if opsa_topk_k is not None and opsa_topk_k > 0:
                            tk = min(int(opsa_topk_k), logits.size(-1))
                            tv_logits, ti = torch.topk(logits, k=tk, dim=-1)
                            opsa_extra["topk_log_probs"] = (tv_logits - lse).to(torch.float32)
                            opsa_extra["topk_indices"] = ti.to(torch.int64)
                        if opsa_gather_indices is not None:
                            gi = opsa_gather_indices.to(device=logits.device, dtype=torch.int64)
                            gathered_logits = logits.gather(dim=-1, index=gi)
                            opsa_extra["gathered_log_probs"] = (gathered_logits - lse).to(torch.float32)
                    # ============ [EasyOPD:OPSA] End ============

            if opsa_caller_requested_extra:
                return entropy, log_probs, opsa_extra
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
        # ============ [EasyOPD:OPSA] propagate top-K request via meta_info ============
        opsa_topk_k = data.meta_info.get("opsa_topk_k", None)
        # ============ [EasyOPD:OPSA] End ============
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
        opsa_topk_values_lst = []
        opsa_topk_indices_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                if opsa_topk_k is not None and opsa_topk_k > 0:
                    entropy, log_probs, opsa_extra = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy,
                        opsa_topk_k=int(opsa_topk_k),
                    )
                    if "topk_log_probs" in opsa_extra:
                        opsa_topk_values_lst.append(opsa_extra["topk_log_probs"])
                        opsa_topk_indices_lst.append(opsa_extra["topk_indices"])
                else:
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

        # ============ [EasyOPD:OPSA] aggregate top-K outputs across micro batches ============
        opsa_topk_values = None
        opsa_topk_indices = None
        if opsa_topk_values_lst and len(opsa_topk_values_lst) == len(log_probs_lst):
            opsa_topk_values = torch.concat(opsa_topk_values_lst, dim=0)
            opsa_topk_indices = torch.concat(opsa_topk_indices_lst, dim=0)
        # ============ [EasyOPD:OPSA] End ============

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)
            if opsa_topk_values is not None:
                opsa_topk_values = restore_dynamic_batch(opsa_topk_values, batch_idx_list)
                opsa_topk_indices = restore_dynamic_batch(opsa_topk_indices, batch_idx_list)

        # Stash OPSA top-K outputs on the function for the calling worker to retrieve.
        # We avoid changing the public return type to keep backward compatibility.
        self._last_opsa_topk_log_probs = opsa_topk_values
        self._last_opsa_topk_indices = opsa_topk_indices

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

        # ============ [EasyOPD:OPSA] Include teacher log-probs for self-distillation ============
        opsa_enabled = self.config.get("opsa_enable", False)
        if opsa_enabled:
            if "opsa_teacher_log_probs" in data.batch.keys():
                select_keys.append("opsa_teacher_log_probs")
            # ============ [EasyOPD:OPSA] include top-K teacher distributions when present ============
            if "opsa_teacher_topk_log_probs" in data.batch.keys():
                select_keys.append("opsa_teacher_topk_log_probs")
            if "opsa_teacher_topk_indices" in data.batch.keys():
                select_keys.append("opsa_teacher_topk_indices")
            # ============ [EasyOPD:OPSA] End ============
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
                    # ============ [EasyOPD:OPSA] gather student log-probs at teacher top-K indices ============
                    # When the trainer attached `opsa_teacher_topk_indices`, ask the forward
                    # to additionally return the student's log-softmax values at exactly those
                    # vocabulary positions, so we can compute a top-K Mixed KL aligned with
                    # the teacher distribution (mirrors original OPSA's topk_logits_k=512).
                    opsa_gather_indices = None
                    if self.config.get("opsa_enable", False) and "opsa_teacher_topk_indices" in model_inputs:
                        opsa_gather_indices = model_inputs["opsa_teacher_topk_indices"]

                    if opsa_gather_indices is not None:
                        entropy, log_prob, opsa_extra = self._forward_micro_batch(
                            model_inputs, temperature=temperature, calculate_entropy=calculate_entropy,
                            opsa_gather_indices=opsa_gather_indices,
                        )
                        student_topk_log_probs = opsa_extra.get("gathered_log_probs", None)
                    else:
                        entropy, log_prob = self._forward_micro_batch(
                            model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                        )
                        student_topk_log_probs = None
                    # ============ [EasyOPD:OPSA] End ============

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
                    # opsa_teacher_log_probs: (batch, response_len) per-token log-probs from
                    # the frozen ref model conditioned on the type-conditional privileged context.
                    # log_prob (student): same shape, from the current on-policy forward pass above.
                    # Both are already per-token log-probs for the chosen token (scalar per position),
                    # so we cannot do full-vocabulary KL directly here.  Instead we use the token-level
                    # cross-entropy surrogate: CE(p_T, p_S) ≈ -p_T(y_t) * log p_S(y_t), which equals
                    # the per-chosen-token contribution of the forward KL when only the argmax mass
                    # matters — the standard approach when full vocab logits are unavailable.
                    opsa_enabled = self.config.get("opsa_enable", False)
                    if opsa_enabled and "opsa_teacher_log_probs" not in model_inputs:
                        print(f"[EasyOPD:OPSA] WARNING: opsa_enable=True but 'opsa_teacher_log_probs' NOT in model_inputs! Keys: {list(model_inputs.keys())}")
                    if opsa_enabled and "opsa_teacher_log_probs" in model_inputs:
                        print(f"[EasyOPD:OPSA] OPSA loss branch ENTERED. teacher_log_probs shape: {model_inputs['opsa_teacher_log_probs'].shape}")
                        from easyopd.methods.opsa.core import (
                            compute_early_window_weights,
                        )

                        teacher_log_probs = model_inputs["opsa_teacher_log_probs"]  # (B, resp_len)
                        student_log_probs = log_prob  # (B, resp_len), from _forward_micro_batch above

                        opsa_kl_type = str(self.config.get("opsa_kl_type", "mixed")).lower()
                        opsa_mixed_kl_weight = float(self.config.get("opsa_mixed_kl_weight", 0.5))

                        # ---- Top-K Mixed KL (preferred when teacher top-K is available) ----
                        # When the teacher provided its top-K log-softmax over the vocabulary
                        # AND the student's forward produced log-softmax gathered at those
                        # same indices, we can compute a faithful top-K Mixed KL that mirrors
                        # the original OPSA implementation (loss_fn.kl_type="mixed",
                        # mixed_kl_weight=0.5, topk_logits_k=512, zero_outside_topk=false).
                        teacher_topk_lp = model_inputs.get("opsa_teacher_topk_log_probs", None)
                        opsa_temperature = float(self.config.get("opsa_temperature", 1.0))
                        topk_path_used = False
                        if teacher_topk_lp is not None and student_topk_log_probs is not None:
                            # Both shape (B, resp_len, K). Compute KL summed over the K dim.
                            teacher_topk_lp = teacher_topk_lp.to(student_topk_log_probs.dtype)
                            # Apply temperature scaling at the distribution level BEFORE renormalization,
                            # mirroring canonical KD temperature softening (logits / T -> softmax). This
                            # is preferred over a post-hoc KL/T scaling because it actually softens the
                            # teacher/student distributions over the K subset. When opsa_temperature == 1.0
                            # this is a no-op and the result is identical to the un-scaled path.
                            if opsa_temperature != 1.0:
                                teacher_topk_lp = teacher_topk_lp / opsa_temperature
                                student_topk_log_probs = student_topk_log_probs / opsa_temperature
                            # Renormalize top-K log-probs to form valid probability distributions.
                            # The original top-K log-probs are sliced from full-vocab log-softmax
                            # and do NOT sum to 1 (typically only 0.01~0.1), which makes the raw
                            # KL mathematically invalid and causes the OPSA loss to stay large.
                            teacher_topk_lp = teacher_topk_lp - torch.logsumexp(teacher_topk_lp, dim=-1, keepdim=True)
                            student_topk_log_probs = student_topk_log_probs - torch.logsumexp(student_topk_log_probs, dim=-1, keepdim=True)
                            t_p = teacher_topk_lp.exp()
                            s_p = student_topk_log_probs.exp()
                            forward_kl_topk = (t_p * (teacher_topk_lp - student_topk_log_probs)).sum(dim=-1)
                            reverse_kl_topk = (s_p * (student_topk_log_probs - teacher_topk_lp)).sum(dim=-1)
                            forward_kl = forward_kl_topk.clamp(min=0.0)
                            reverse_kl = reverse_kl_topk.clamp(min=0.0)
                            topk_path_used = True
                        else:
                            # ---- Fallback: per-token Mixed KL on chosen-token scalar log-probs ----
                            if not hasattr(self, '_opsa_fallback_warned'):
                                logger.warning(
                                    "[EasyOPD:OPSA] Top-K logits unavailable; using per-token KL approximation. "
                                    "This computes KL only on the chosen token, which significantly underestimates "
                                    "the true distribution-level KL divergence. Consider enabling top-K (opsa_topk_logits_k > 0)."
                                )
                                self._opsa_fallback_warned = True
                            # Forward KL surrogate: p_T(y_t) * (log p_T(y_t) - log p_S(y_t))
                            forward_kl = (teacher_log_probs.exp() * (teacher_log_probs - student_log_probs)).clamp(min=0.0)
                            # Reverse KL surrogate: p_S(y_t) * (log p_S(y_t) - log p_T(y_t))
                            reverse_kl = (student_log_probs.exp() * (student_log_probs - teacher_log_probs)).clamp(min=0.0)

                        if opsa_kl_type == "forward":
                            kl_per_token = forward_kl
                        elif opsa_kl_type == "reverse":
                            kl_per_token = reverse_kl
                        else:  # "mixed"
                            kl_per_token = (
                                opsa_mixed_kl_weight * forward_kl
                                + (1.0 - opsa_mixed_kl_weight) * reverse_kl
                            )

                        # ---- opsa_temperature handling ----
                        # NOTE: Standard KD temperature scaling operates on full-vocab logits
                        # (logits / T -> softmax) before computing KL. In this training path we
                        # only have per-chosen-token scalar log-probs (log p(y_t | y_<t)) for both
                        # teacher and student, NOT the full vocabulary distribution, so we cannot
                        # perform the canonical temperature softening of the distributions here.
                        #
                        # As a pragmatic surrogate we treat opsa_temperature as a direct scaling
                        # factor on the per-token KL contribution: a higher T attenuates the
                        # distillation signal (smoother / weaker pull toward the teacher), and a
                        # lower T amplifies it -- mirroring the qualitative effect of temperature
                        # in classical KD, while remaining mathematically equivalent (up to a
                        # constant) to adjusting `opsa_distillation_loss_coef`. Users who need
                        # true logits-level temperature scaling should implement it where the
                        # full-vocab logits are still available (e.g. inside the forward pass).
                        # When opsa_temperature == 1.0 (default) this is a no-op and preserves the
                        # original behaviour exactly.
                        #
                        # NOTE: For the top-K path above, temperature is already applied at the
                        # distribution level (logit/T -> renormalize) before computing KL, which is
                        # the canonical KD formulation; in that case we MUST NOT divide kl_per_token
                        # by T again. Only the per-token fallback path uses this post-hoc KL scaling.
                        if opsa_temperature != 1.0 and not topk_path_used:
                            kl_per_token = kl_per_token / opsa_temperature

                        opsa_use_window = self.config.get("opsa_use_window_weighting", True)
                        if opsa_use_window:
                            window_weights = compute_early_window_weights(
                                response_mask,
                                window_size=int(self.config.get("opsa_window_size", 32)),
                                decay_type=self.config.get("opsa_decay_type", "linear"),
                                min_weight=float(self.config.get("opsa_min_weight", 0.1)),
                            )
                            weighted_kl = kl_per_token * window_weights
                        else:
                            weighted_kl = kl_per_token * response_mask.float()

                        opsa_loss_agg = self.config.get("opsa_loss_agg_mode", "token-mean")
                        mask = response_mask.float()
                        if opsa_loss_agg == "token-mean":
                            valid_tokens = mask.sum().clamp(min=1.0)
                            policy_loss = (weighted_kl * mask).sum() / valid_tokens
                        elif opsa_loss_agg == "seq-mean-token-sum":
                            policy_loss = torch.mean(torch.sum(weighted_kl * mask, dim=-1))
                        else:
                            seq_lengths = mask.sum(dim=-1).clamp(min=1.0)
                            policy_loss = torch.mean(torch.sum(weighted_kl * mask, dim=-1) / seq_lengths)

                        opsa_coef = float(self.config.get("opsa_distillation_loss_coef", 1.0))
                        loss = policy_loss * opsa_coef * loss_scale_factor
                        loss.backward()

                        with torch.no_grad():
                            valid_tokens = mask.sum().clamp(min=1.0)
                            kl_mean = (kl_per_token * mask).sum() / valid_tokens
                            if opsa_use_window:
                                window_mask = (window_weights >= 1.0 - 1e-6) & (mask > 0)
                                kl_in_window = (kl_per_token * window_mask.float()).sum() / window_mask.float().sum().clamp(min=1.0)
                            else:
                                kl_in_window = kl_mean

                        opsa_metrics = {
                            "opsa/kl_mean": kl_mean.item() * loss_scale_factor,
                            "opsa/kl_in_window": kl_in_window.item() * loss_scale_factor,
                            "opsa/loss": policy_loss.detach().item() * loss_scale_factor,
                        }
                        append_to_dict(metrics, opsa_metrics)
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
