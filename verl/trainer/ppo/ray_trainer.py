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
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import random
import os
import uuid
from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from typing import Optional

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
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
import json, os, math
from typing import Any
import numpy as np
import torch

def _json_default(obj: Any):
    """Fallback for json.dumps(default=...)"""
    # torch 张量 → Python 基本类型/嵌套 list
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    # NumPy 标量（如 np.bool_, np.int64, np.float32） → Python 标量
    if isinstance(obj, np.generic):
        return obj.item()
    # NumPy 数组 → Python list
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # set/tuple → list
    if isinstance(obj, (set, tuple)):
        return list(obj)
    # bytes → utf-8 字符串（容错解码）
    if isinstance(obj, (bytes, bytearray)):
        try:
            return obj.decode("utf-8", errors="ignore")
        except Exception:
            return str(obj)
    # float 的 NaN/Inf 处理（可选：避免下游解析问题）
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    # 其它非常规对象，最后兜底为字符串
    return str(obj)
def _normalize_series(x, n: int):
    """把 x 规范成长度为 n 的 Python 列表（必要时广播或跳过）"""
    # torch.Tensor / np.ndarray → list
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().tolist()
    elif isinstance(x, np.ndarray):
        x = x.tolist()

    # 标量（包括 numpy 标量）→ 广播
    if not hasattr(x, "__len__") or isinstance(x, (str, bytes)):
        # 注意：字符串我们视为“有意义的标量”，同样广播
        return [x] * n

    # 长度刚好 n → 原样返回；长度 1 → 广播；其他长度 → 返回 None 代表跳过
    if len(x) == n:
        return list(x)
    if len(x) == 1:
        return [x[0]] * n
    return None  # 不匹配的列，后面跳过

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
    effective_mask = response_mask
    teacher_kl_mask = data.batch.get("teacher_kl_mask", None)
    if teacher_kl_mask is not None:
        teacher_mask = teacher_kl_mask.to(response_mask.dtype)
        effective_mask = effective_mask * teacher_mask
    kld = kld * effective_mask.to(kld.dtype)
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=effective_mask, axis=-1)  # average over sequence
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
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
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
        # [EasyOPD:lightning_opd] Pass offline distillation tensors through to
        # the custom advantage estimator when Lightning-OPD is selected.
        adv_name = getattr(adv_estimator, "value", adv_estimator)
        if adv_name == "on_policy_distillation":
            adv_kwargs["old_log_probs"] = data.batch["old_log_probs"]
            adv_kwargs["teacher_log_probs"] = data.batch.get("teacher_log_probs")
        # [EasyOPD:lightning_opd] End

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


def compute_total_advantage(
    A_task: torch.Tensor,
    local_adv_token: torch.Tensor,
    gamma: float = 1.0,
    beta_min: float = 0.0,
    beta_max: float = 0.3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse GRPO per-sample advantages with token-level KL feedback."""
    # Per-sample gating β via sigmoid on the normalized GRPO advantage
    gate = torch.sigmoid(-gamma * A_task)
    beta_jk = beta_min + (beta_max - beta_min) * gate

    # Broadcast β to token dimension
    beta_token = beta_jk.unsqueeze(1)

    # Final token-level advantage
    A_total_token = A_task.unsqueeze(1) + beta_token * local_adv_token
    return A_total_token, beta_jk, local_adv_token


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
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
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
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
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )
        self.sample_traces = deque(maxlen=5)
        default_dir = getattr(self.config.trainer, "default_local_dir", "outputs")
        os.makedirs(default_dir, exist_ok=True)
        self.sample_log_path = os.path.join(default_dir, "rl_samples.jsonl")

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # ============ [EasyOPD] Context distillation and base model configuration ============
        # Context distillation (critique-based)
        self.critique_vllm_url = config.algorithm.get("critique_vllm_url", None)
        self.use_context_distillation = self.critique_vllm_url is not None
        self.critique_model = config.algorithm.get("critique_model", None)
        self.max_critique_tokens = config.algorithm.get("max_critique_tokens", 2048)
        self.critique_temperature = config.algorithm.get("critique_temperature", 0.0)
        self.critique_top_p = config.algorithm.get("critique_top_p", 1.0)

        # Ref solution distillation
        self.use_ref_solution_distillation = config.algorithm.get("use_ref_solution_distillation", False)

        if self.use_context_distillation or self.use_ref_solution_distillation:
            return_raw_chat = config.data.get("return_raw_chat", False)
            if not return_raw_chat:
                raise ValueError(
                    "When using context distillation (critique_vllm_url is provided) "
                    "or ref solution distillation (use_ref_solution_distillation=True), "
                    "you must set data.return_raw_chat=True in config."
                )

        # Base model paths for G-OPD corrected reward computation
        self.base_model_path = config.actor_rollout_ref.model.get("base_model_path", None)
        self.ref_base_model_path = config.actor_rollout_ref.ref.get("model", None)
        if self.ref_base_model_path is not None:
            self.ref_base_model_path = self.ref_base_model_path.get("base_model_path", None)
        self.use_base_models = self.base_model_path is not None and self.ref_base_model_path is not None

        if self.use_base_models:
            print(f"[EasyOPD] Corrected reward enabled with base models:")
            print(f"  Actor base model: {self.base_model_path}")
            print(f"  Ref base model: {self.ref_base_model_path}")
        if self.use_context_distillation:
            print(f"[EasyOPD] Context distillation enabled with vLLM URL: {self.critique_vllm_url}")
        if self.use_ref_solution_distillation:
            print("[EasyOPD] Ref solution distillation enabled")
        # ============ [EasyOPD] End ============

        # ============ [EasyOPD] Hook Dispatcher initialization ============
        # The HookDispatcher provides a unified interface for method-specific
        # logic. It coexists with the legacy if-branch approach during migration.
        try:
            from easyopd.hook_dispatch import HookDispatcher
            from easyopd.diagnostics import MetricsCollector

            self.hook_dispatcher = HookDispatcher.from_config(self.config)
            if self.hook_dispatcher.enabled:
                self.easyopd_metrics = MetricsCollector(
                    method_name=self.hook_dispatcher.method_name,
                )
            else:
                self.easyopd_metrics = None
        except ImportError:
            self.hook_dispatcher = None
            self.easyopd_metrics = None
        # ============ [EasyOPD] End ============

        # ============ [EasyOPD:Simple] Cross-tokenizer teacher sidecar ============
        self.simple_teacher_sidecar = None
        self.simple_xtok_enabled = False
        actor_loss_mode = config.actor_rollout_ref.actor.policy_loss.get("loss_mode", "vanilla")
        distillation_cfg = getattr(config, "distillation", None)
        distillation_enabled = bool(distillation_cfg is not None and distillation_cfg.get("enabled", False))
        if actor_loss_mode in {"simple", "simct", "alm", "uld", "dskd"} and distillation_enabled:
            from easyopd.methods.simple.teacher_sidecar import EasyOPDSimpleTeacherSidecar

            self.simple_teacher_sidecar = EasyOPDSimpleTeacherSidecar(config)
            self.simple_xtok_enabled = True
            print(
                f"[EasyOPD:simple] Cross-tokenizer teacher sidecar enabled for loss_mode={actor_loss_mode}"
            )
        # ============ [EasyOPD:Simple] End ============

        # ============ [EasyOPD:OPSA] On-Policy Self-Distillation configuration ============
        self.opsa_enable = config.actor_rollout_ref.actor.get("opsa_enable", False)
        if self.opsa_enable:
            self.opsa_harmful_context = config.actor_rollout_ref.actor.get(
                "opsa_harmful_context",
                "You are a helpful and safe AI assistant. When you encounter a harmful, dangerous, or unethical request, you MUST refuse clearly and explain why. Safety is your highest priority."
            )
            self.opsa_benign_context = config.actor_rollout_ref.actor.get(
                "opsa_benign_context",
                "You are a helpful AI assistant. Respond to the user's request directly and helpfully. Do not over-refuse safe and reasonable requests."
            )
            # TFR: how often the privileged context flips an unsafe response to safe.
            # Tracked as a proxy via teacher vs. student log-prob gap on harmful samples.
            self.opsa_tfr_eval_frequency = int(
                config.actor_rollout_ref.actor.get("opsa_tfr_eval_frequency", 10)
            )
            self.opsa_tfr_threshold = float(
                config.actor_rollout_ref.actor.get("opsa_tfr_threshold", 0.8)
            )
            # Rolling buffer: (teacher_lp_mean, student_lp_mean) for harmful samples
            self._opsa_tfr_buffer: list = []
            print(
                f"[EasyOPD:OPSA] On-Policy Self-Distillation enabled | "
                f"window_size={config.actor_rollout_ref.actor.get('opsa_window_size', 32)} | "
                f"tfr_eval_frequency={self.opsa_tfr_eval_frequency} | "
                f"tfr_threshold={self.opsa_tfr_threshold}"
            )
        # ============ [EasyOPD:OPSA] End ============

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

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

    # def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
    #     """Dump rollout/validation samples as JSONL."""
    #     os.makedirs(dump_path, exist_ok=True)
    #     filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

    #     n = len(inputs)
    #     base_data = {
    #         "input": inputs,
    #         "output": outputs,
    #         "gts": gts,
    #         "score": scores,
    #         "step": [self.global_steps] * n,
    #     }

    #     for k, v in reward_extra_infos_dict.items():
    #         if len(v) == n:
    #             base_data[k] = v

    #     lines = []
    #     for i in range(n):
    #         entry = {k: v[i] for k, v in base_data.items()}
    #         lines.append(json.dumps(entry, ensure_ascii=False))

    #     with open(filename, "w") as f:
    #         f.write("\n".join(lines) + "\n")

    #     print(f"Dumped generations to {filename}")
    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL（更鲁棒版）."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [int(self.global_steps)] * n,  # 确保是 Python int
        }

        # 规范化并合并额外指标
        if reward_extra_infos_dict:
            for k, v in reward_extra_infos_dict.items():
                norm = _normalize_series(v, n)
                if norm is not None:
                    base_data[k] = norm
                else:
                    # 长度不匹配就跳过，避免报错
                    # 需要的话可以 print/log 一下
                    # print(f"[dump_generations] Skip key={k} due to length mismatch: {len(v)} != {n}")
                    pass

        lines = []
        for i in range(n):
            # 逐行抽取第 i 条，保持所有列对齐
            entry = {k: base_data[k][i] for k in base_data.keys()}
            # 通过 default=_json_default 兜底所有非常规类型
            lines.append(json.dumps(entry, ensure_ascii=False, default=_json_default))

        # 注意编码为 utf-8
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _collect_tool_metrics(self, extra_infos):
        stats = {}
        if extra_infos is None:
            return stats

        if isinstance(extra_infos, np.ndarray):
            infos = extra_infos.tolist()
        else:
            infos = extra_infos

        total_calls = 0
        total_success = 0
        count = 0
        for info in infos:
            if not isinstance(info, dict):
                continue
            total_calls += int(info.get("num_tool_calls", 0) or 0)
            total_success += int(info.get("num_tool_success", 0) or 0)
            count += 1

        if count == 0:
            return stats

        stats["rollout/tool_calls_avg"] = total_calls / count
        stats["rollout/tool_success_avg"] = total_success / count
        if total_calls > 0:
            stats["rollout/tool_success_rate"] = total_success / total_calls
        else:
            stats["rollout/tool_success_rate"] = 0.0

        return stats

    # ============ [EasyOPD] Build context distillation batch (OPCD) ============
    def _maybe_build_opcd_batch(self, batch):
        """Build experience-augmented teacher inputs for OPCD context distillation.

        In OPCD's consolidate stage:
        1. Load experience from file (or from batch metadata)
        2. Inject experience into prompts to create teacher prompts
        3. Tokenize teacher prompts + student responses
        4. Compute teacher (ref model) log-probs on the augmented prompts
        5. Return exp_log_probs for KL loss computation in dp_actor

        Returns:
            Optional tuple of (DataProto with exp_log_probs, metrics dict), or None.
        """
        from verl import DataProto
        from easyopd.methods.opcd.core import build_experience_prompt, truncate_experience
        import json
        import os

        experience_path = getattr(self.config.trainer, "experience_path", None)
        if experience_path is None:
            return None

        train_system_prompt = getattr(self.config.trainer, "train_system_prompt", False)
        experience_max_length = getattr(self.config.trainer, "experience_max_length", 16384)

        # Load experiences from file
        experiences = {}
        if experience_path and os.path.exists(experience_path):
            with open(experience_path, "r") as f:
                exp_data = json.load(f)
            if isinstance(exp_data, dict):
                experiences = exp_data
            elif isinstance(exp_data, list):
                for item in exp_data:
                    if isinstance(item, dict) and "prompt" in item and "experience" in item:
                        experiences[item["prompt"]] = item["experience"]

        batch_size = batch.batch["input_ids"].shape[0]
        device = batch.batch["input_ids"].device

        # Build teacher prompts with experience injected
        has_experience_count = 0
        exp_prompts = []

        for i in range(batch_size):
            raw_prompt = batch.non_tensor_batch.get("raw_prompt", [None] * batch_size)[i]
            prompt_text = ""
            if raw_prompt is not None:
                if isinstance(raw_prompt, (list, tuple)):
                    # Chat messages format
                    for msg in raw_prompt:
                        if isinstance(msg, dict) and msg.get("role") == "user":
                            content = msg.get("content", "")
                            if isinstance(content, str):
                                prompt_text = content
                            break
                elif isinstance(raw_prompt, str):
                    prompt_text = raw_prompt

            # Find matching experience
            experience = ""
            if prompt_text in experiences:
                experience = experiences[prompt_text]
            elif experiences:
                # Try partial match or use first available
                for key in experiences:
                    if key in prompt_text or prompt_text in key:
                        experience = experiences[key]
                        break

            if experience:
                experience = truncate_experience(
                    experience, max_tokens=experience_max_length, tokenizer=self.tokenizer
                )
                has_experience_count += 1

            # Build teacher messages with experience
            if raw_prompt is not None and isinstance(raw_prompt, (list, tuple)):
                teacher_messages = build_experience_prompt(
                    raw_prompt, experience,
                    mode="system_prompt" if train_system_prompt else "user_content",
                    train_system_prompt=train_system_prompt,
                )
            else:
                teacher_messages = [{"role": "user", "content": prompt_text}]
                if experience:
                    teacher_messages = build_experience_prompt(
                        teacher_messages, experience,
                        mode="system_prompt" if train_system_prompt else "user_content",
                        train_system_prompt=train_system_prompt,
                    )

            exp_prompts.append(teacher_messages)

        # Tokenize teacher prompts and compute ref model log-probs
        # For now, we pass the experience info as metadata; the actual ref log-prob
        # computation happens via the existing ref_policy_wg.compute_ref_log_prob path
        # We need to rebuild input_ids with experience-augmented prompts

        response_mask = batch.batch["response_mask"]
        responses = batch.batch["responses"]

        teacher_input_ids_list = []
        teacher_attention_mask_list = []
        teacher_position_ids_list = []

        max_prompt_len = getattr(self.config.data, "max_prompt_length", 17408)

        for i in range(batch_size):
            # Tokenize teacher prompt
            apply_kwargs = {}
            if hasattr(self.config.data, "apply_chat_template_kwargs"):
                apply_kwargs = dict(self.config.data.get("apply_chat_template_kwargs", {}) or {})

            processing_class = getattr(self, "processor", None) or self.tokenizer
            raw_prompt_text = processing_class.apply_chat_template(
                exp_prompts[i],
                tokenize=False,
                add_generation_prompt=True,
                **apply_kwargs,
            )

            teacher_prompt_output = self.tokenizer(
                raw_prompt_text,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
                max_length=max_prompt_len,
            )
            prompt_ids = teacher_prompt_output["input_ids"].squeeze(0)
            prompt_mask = teacher_prompt_output["attention_mask"].squeeze(0)

            # Concat with student response
            valid_response_len = int(response_mask[i].sum().item())
            response_ids = responses[i, :valid_response_len]
            teacher_ids = torch.cat([prompt_ids, response_ids], dim=0)
            teacher_mask = torch.cat([prompt_mask, torch.ones(valid_response_len, dtype=torch.long)], dim=0)
            teacher_pos = torch.arange(len(teacher_ids), dtype=torch.long)

            teacher_input_ids_list.append(teacher_ids)
            teacher_attention_mask_list.append(teacher_mask)
            teacher_position_ids_list.append(teacher_pos)

        # Pad to same length
        import torch
        pad_token_id = self.tokenizer.pad_token_id or 0
        teacher_input_ids = torch.nn.utils.rnn.pad_sequence(
            teacher_input_ids_list, batch_first=True, padding_value=pad_token_id,
        ).to(device)
        teacher_attention_mask = torch.nn.utils.rnn.pad_sequence(
            teacher_attention_mask_list, batch_first=True, padding_value=0,
        ).to(device)
        teacher_position_ids = torch.nn.utils.rnn.pad_sequence(
            teacher_position_ids_list, batch_first=True, padding_value=0,
        ).to(device)

        # Compute ref model log-probs on experience-augmented inputs
        # We create a temporary batch for the ref model
        exp_batch = DataProto.from_dict(tensors={
            "input_ids": teacher_input_ids,
            "attention_mask": teacher_attention_mask,
            "position_ids": teacher_position_ids,
            "responses": batch.batch["responses"],
            "response_mask": batch.batch["response_mask"],
        })

        # Use ref policy to compute log probs
        assert self.use_reference_policy, "OPCD requires reference policy for experience log-prob computation"
        exp_log_prob_output = self.ref_policy_wg.compute_ref_log_prob(exp_batch)

        exp_log_probs = exp_log_prob_output.batch["ref_log_prob"]

        metrics = {
            "opcd/experience_coverage": has_experience_count / max(batch_size, 1),
            "opcd/exp_log_prob_mean": exp_log_probs.mean().detach().item(),
        }

        return DataProto.from_dict(tensors={
            "exp_log_probs": exp_log_probs,
        }), metrics
    # ============ [EasyOPD] End ============

    # ============ [EasyOPD] Build self-distillation batch (Vision-OPD) ============
    def _maybe_build_simple_xtok_batch(self, batch):
        """Build teacher hidden states for simple cross-tokenizer KD.

        The actor-side simple loss needs ragged teacher fields in
        ``non_tensor_batch``: ``teacher_hidden_states``, ``teacher_input_ids``
        and ``teacher_loss_mask``. This method constructs teacher-side
        prompt+response inputs from the rollout batch and forwards them to the
        long-lived teacher sidecar.
        """
        if not self.simple_xtok_enabled or self.simple_teacher_sidecar is None:
            return None

        from verl import DataProto

        response_mask = batch.batch["response_mask"]
        prompts = batch.batch["prompts"]
        responses = batch.batch["responses"]
        batch_size = responses.shape[0]

        teacher_input_ids = []
        teacher_loss_masks = []
        teacher_texts = []
        prompt_token_counts = []
        response_token_counts = []

        for i in range(batch_size):
            prompt_ids = prompts[i]
            if torch.is_tensor(prompt_ids):
                prompt_ids = prompt_ids.detach().cpu()
            prompt_ids = prompt_ids.tolist()
            prompt_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)

            valid_response_len = int(response_mask[i].sum().item())
            response_ids = responses[i, :valid_response_len]
            if torch.is_tensor(response_ids):
                response_ids = response_ids.detach().cpu()
            response_ids = response_ids.tolist()
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)

            teacher_ids, loss_mask, full_text = self.simple_teacher_sidecar.encode_for_teacher(
                prompt_text=prompt_text,
                response_text=response_text,
                max_length=self.simple_teacher_sidecar.teacher_context_length,
                mask_mode="label",
            )
            teacher_input_ids.append(np.asarray(teacher_ids, dtype=np.int64))
            teacher_loss_masks.append(np.asarray(loss_mask, dtype=bool))
            teacher_texts.append(full_text)
            prompt_token_counts.append(len(prompt_ids))
            response_token_counts.append(valid_response_len)

        teacher_hidden_states = self.simple_teacher_sidecar.compute_hidden_states_batch(
            prompts=teacher_texts,
            loss_masks=teacher_loss_masks,
            input_ids=[ids.tolist() for ids in teacher_input_ids],
            method_name="simple",
        )

        def _object_array(items):
            arr = np.empty(len(items), dtype=object)
            arr[:] = list(items)
            return arr

        non_tensors = {
            "teacher_hidden_states": _object_array(teacher_hidden_states),
            "teacher_input_ids": _object_array(teacher_input_ids),
            "teacher_loss_mask": _object_array(teacher_loss_masks),
        }
        metrics = {
            "simple/teacher_batch_size": float(batch_size),
            "simple/teacher_prompt_tokens_mean": float(np.mean(prompt_token_counts)) if prompt_token_counts else 0.0,
            "simple/teacher_response_tokens_mean": float(np.mean(response_token_counts)) if response_token_counts else 0.0,
            "simple/teacher_loss_tokens_mean": float(np.mean([mask.sum() for mask in teacher_loss_masks])) if teacher_loss_masks else 0.0,
        }

        # DataProto.union() always unions TensorDicts first, so a pure
        # non_tensor DataProto (batch=None) cannot be merged into the training
        # batch. Attach an empty TensorDict carrying only the batch size.
        from tensordict import TensorDict

        empty_batch = TensorDict({}, batch_size=[batch_size])
        return DataProto(batch=empty_batch, non_tensor_batch=non_tensors), metrics

    def _maybe_build_vision_opd_batch(
        self,
        batch,
        reward_tensor,
    ):
        """Build teacher inputs for Vision-OPD self-distillation.

        Constructs teacher_input_ids, teacher_attention_mask, teacher_position_ids,
        teacher_response_start_idx, and self_distillation_mask based on the
        self_distillation configuration.

        Supports two modes:
        1. teacher_always_on + teacher_image_key: Swap student images with teacher bbox images
        2. teacher_always_on + teacher_prompt_mode="answer_hint": Add ground truth as hint

        Returns:
            Optional tuple of (DataProto with teacher tensors, metrics dict), or None if not applicable.
        """
        from verl import DataProto
        from easyopd.methods.vision_opd.teacher_utils import (
            prepare_teacher_messages_with_bbox_images,
            prepare_opsd_teacher_messages,
            extract_images_from_messages,
            teacher_images_available,
        )
        import numpy as np

        self_distillation_cfg = self.config.actor_rollout_ref.actor.get("self_distillation", None)
        if self_distillation_cfg is None:
            return None

        device = batch.batch["input_ids"].device
        response_mask = batch.batch["response_mask"]
        responses = batch.batch["responses"]
        batch_size = batch.batch["input_ids"].shape[0]

        teacher_prompt_mode = self_distillation_cfg.get("teacher_prompt_mode", None)
        use_opsd_answer_hint = (
            self_distillation_cfg.get("teacher_always_on", False)
            and teacher_prompt_mode == "answer_hint"
        )
        use_teacher_image_swap = (
            self_distillation_cfg.get("teacher_always_on", False)
            and self_distillation_cfg.get("teacher_image_key", None) is not None
            and not use_opsd_answer_hint
        )

        if use_opsd_answer_hint:
            # OPSD mode: add answer hint to teacher prompt
            answer_hint_template = self_distillation_cfg.get(
                "answer_hint_template",
                "\n\nHere is a reference solution to this problem:\n{answer}\n\n"
                "After understanding the reference solution, please try to solve this problem below:\n",
            )

            teacher_input_ids_list = []
            teacher_attention_mask_list = []
            teacher_position_ids_list = []
            teacher_response_start_idx_list = []
            teacher_present_mask_list = []

            for i in range(batch_size):
                # Get ground truth answer
                reward_model_info = batch.non_tensor_batch.get("reward_model", [None] * batch_size)
                answer = None
                if reward_model_info[i] is not None and isinstance(reward_model_info[i], dict):
                    answer = reward_model_info[i].get("ground_truth", None)
                if answer is None:
                    extra_info = batch.non_tensor_batch.get("extra_info", [None] * batch_size)
                    if extra_info[i] is not None and isinstance(extra_info[i], dict):
                        answer = extra_info[i].get("answer", None)

                has_answer = answer is not None and str(answer).strip() != ""
                teacher_present_mask_list.append(1.0 if has_answer else 0.0)

                if not has_answer:
                    answer = ""

                raw_prompt_messages = list(batch.non_tensor_batch["raw_prompt"][i])
                teacher_messages = prepare_opsd_teacher_messages(
                    raw_prompt_messages, str(answer), answer_hint_template
                )

                # Tokenize teacher messages
                apply_kwargs = dict(self.config.data.get("apply_chat_template_kwargs", {}) or {})
                processing_class = getattr(self, "processor", None) or self.tokenizer
                raw_prompt = processing_class.apply_chat_template(
                    teacher_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    **apply_kwargs,
                )
                max_prompt_len = self_distillation_cfg.get("max_reprompt_len", 10240)
                teacher_prompt_output = self.tokenizer(
                    raw_prompt,
                    return_tensors="pt",
                    add_special_tokens=False,
                    truncation=True,
                    max_length=max_prompt_len,
                )
                prompt_ids = teacher_prompt_output["input_ids"].squeeze(0)
                prompt_mask = teacher_prompt_output["attention_mask"].squeeze(0)

                # Concat with response
                valid_response_len = int(response_mask[i].sum().item())
                response_ids = responses[i, :valid_response_len]
                teacher_ids = torch.cat([prompt_ids, response_ids], dim=0)
                teacher_mask = torch.cat([prompt_mask, torch.ones(valid_response_len, dtype=torch.long)], dim=0)
                teacher_pos = torch.arange(len(teacher_ids), dtype=torch.long)
                teacher_resp_start = torch.tensor([len(prompt_ids)], dtype=torch.long)

                teacher_input_ids_list.append(teacher_ids)
                teacher_attention_mask_list.append(teacher_mask)
                teacher_position_ids_list.append(teacher_pos)
                teacher_response_start_idx_list.append(teacher_resp_start)

            # Pad to same length
            teacher_input_ids = torch.nn.utils.rnn.pad_sequence(
                teacher_input_ids_list, batch_first=True,
                padding_value=self.tokenizer.pad_token_id or 0,
            ).to(device)
            teacher_attention_mask = torch.nn.utils.rnn.pad_sequence(
                teacher_attention_mask_list, batch_first=True, padding_value=0,
            ).to(device)
            teacher_position_ids = torch.nn.utils.rnn.pad_sequence(
                teacher_position_ids_list, batch_first=True, padding_value=0,
            ).to(device)

            teacher_present_mask = torch.tensor(teacher_present_mask_list, dtype=torch.float32, device=device)
            metrics = {
                "self_distillation/teacher_always_on_fraction": teacher_present_mask.mean().item(),
                "self_distillation/opsd_answer_hint_fraction": teacher_present_mask.mean().item(),
                "self_distillation/policy_fallback_fraction": (1.0 - teacher_present_mask.mean()).item(),
            }
            return DataProto.from_dict(
                tensors={
                    "teacher_input_ids": teacher_input_ids,
                    "teacher_attention_mask": teacher_attention_mask,
                    "teacher_position_ids": teacher_position_ids,
                    "teacher_response_start_idx": torch.stack(teacher_response_start_idx_list).to(device),
                    "self_distillation_mask": teacher_present_mask,
                },
            ), metrics

        elif use_teacher_image_swap:
            # Teacher image swap mode: use bbox_images for teacher
            teacher_image_key = self_distillation_cfg.get("teacher_image_key", "bbox_images")
            if teacher_image_key not in batch.non_tensor_batch:
                print(f"[EasyOPD] Warning: teacher_image_key '{teacher_image_key}' not found in batch")
                return None

            fallback_to_policy_loss = self_distillation_cfg.get("fallback_to_policy_loss_on_missing_teacher", False)

            teacher_input_ids_list = []
            teacher_attention_mask_list = []
            teacher_position_ids_list = []
            teacher_response_start_idx_list = []
            teacher_multi_modal_inputs_list = []
            teacher_present_mask_list = []

            for i in range(batch_size):
                teacher_images = batch.non_tensor_batch[teacher_image_key][i]
                if isinstance(teacher_images, np.ndarray):
                    teacher_images = teacher_images.tolist()
                elif teacher_images is None:
                    teacher_images = []
                else:
                    teacher_images = list(teacher_images) if not isinstance(teacher_images, list) else teacher_images

                has_teacher_images = teacher_images_available(teacher_images)
                teacher_present_mask_list.append(1.0 if has_teacher_images else 0.0)

                if not has_teacher_images:
                    if not fallback_to_policy_loss:
                        # Use student images as fallback
                        teacher_images = extract_images_from_messages(
                            list(batch.non_tensor_batch["raw_prompt"][i])
                        )

                raw_prompt_messages = list(batch.non_tensor_batch["raw_prompt"][i])
                teacher_messages = prepare_teacher_messages_with_bbox_images(
                    raw_prompt_messages, teacher_images
                )

                # Tokenize teacher messages
                apply_kwargs = dict(self.config.data.get("apply_chat_template_kwargs", {}) or {})
                processing_class = getattr(self, "processor", None) or self.tokenizer
                raw_prompt = processing_class.apply_chat_template(
                    teacher_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    **apply_kwargs,
                )
                max_prompt_len = self_distillation_cfg.get("max_reprompt_len", 10240)

                # Handle multimodal processing
                teacher_multi_modal_inputs = None
                if hasattr(self, "processor") and self.processor is not None:
                    prompt_images = extract_images_from_messages(teacher_messages)
                    model_inputs = dict(
                        self.processor(
                            text=[raw_prompt],
                            images=prompt_images or None,
                            videos=None,
                            return_tensors="pt",
                            truncation=True,
                            max_length=max_prompt_len,
                        )
                    )
                    teacher_multi_modal_inputs = model_inputs.copy()
                    prompt_ids = teacher_multi_modal_inputs.pop("input_ids").squeeze(0)
                    prompt_mask = teacher_multi_modal_inputs.pop("attention_mask").squeeze(0)
                else:
                    teacher_prompt_output = self.tokenizer(
                        raw_prompt,
                        return_tensors="pt",
                        add_special_tokens=False,
                        truncation=True,
                        max_length=max_prompt_len,
                    )
                    prompt_ids = teacher_prompt_output["input_ids"].squeeze(0)
                    prompt_mask = teacher_prompt_output["attention_mask"].squeeze(0)

                # Concat with response
                valid_response_len = int(response_mask[i].sum().item())
                response_ids = responses[i, :valid_response_len]
                teacher_ids = torch.cat([prompt_ids, response_ids], dim=0)
                teacher_mask = torch.cat([prompt_mask, torch.ones(valid_response_len, dtype=torch.long)], dim=0)
                teacher_pos = torch.arange(len(teacher_ids), dtype=torch.long)
                teacher_resp_start = torch.tensor([len(prompt_ids)], dtype=torch.long)

                teacher_input_ids_list.append(teacher_ids)
                teacher_attention_mask_list.append(teacher_mask)
                teacher_position_ids_list.append(teacher_pos)
                teacher_response_start_idx_list.append(teacher_resp_start)
                teacher_multi_modal_inputs_list.append(teacher_multi_modal_inputs)

            # Pad to same length
            teacher_input_ids = torch.nn.utils.rnn.pad_sequence(
                teacher_input_ids_list, batch_first=True,
                padding_value=self.tokenizer.pad_token_id or 0,
            ).to(device)
            teacher_attention_mask = torch.nn.utils.rnn.pad_sequence(
                teacher_attention_mask_list, batch_first=True, padding_value=0,
            ).to(device)
            teacher_position_ids = torch.nn.utils.rnn.pad_sequence(
                teacher_position_ids_list, batch_first=True, padding_value=0,
            ).to(device)

            teacher_present_mask = torch.tensor(teacher_present_mask_list, dtype=torch.float32, device=device)
            metrics = {
                "self_distillation/teacher_always_on_fraction": teacher_present_mask.mean().item(),
                "self_distillation/teacher_image_swap_fraction": teacher_present_mask.mean().item(),
                "self_distillation/policy_fallback_fraction": (1.0 - teacher_present_mask.mean()).item(),
            }

            tensors = {
                "teacher_input_ids": teacher_input_ids,
                "teacher_attention_mask": teacher_attention_mask,
                "teacher_position_ids": teacher_position_ids,
                "teacher_response_start_idx": torch.stack(teacher_response_start_idx_list).to(device),
                "self_distillation_mask": teacher_present_mask,
            }
            non_tensors = {}
            if any(m is not None for m in teacher_multi_modal_inputs_list):
                non_tensors["teacher_multi_modal_inputs"] = teacher_multi_modal_inputs_list

            return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors), metrics

        return None
    # ============ [EasyOPD] End ============

    # ============ [EasyOPD:SDPO] Build reprompt self-teacher batch ============
    def _maybe_build_sdpo_batch(self, batch, reward_tensor):
        """Build the SDPO self-teacher batch by reprompting rollouts.

        For each failed rollout, inject a correct demonstration from a
        successful rollout in the same prompt group (uid) and/or environment
        feedback, then append the ORIGINAL response so the live self-teacher
        re-scores the student's own tokens in hindsight (paper Eq. 1).

        The heavy lifting lives in
        ``easyopd.methods.sdpo.core.build_sdpo_teacher_inputs``; here we only
        wire the verl batch / tokenizer in and wrap the result for
        ``batch.union()``.
        """
        from verl import DataProto
        from easyopd.methods.sdpo.core import build_sdpo_teacher_inputs

        self_distillation_cfg = self.config.actor_rollout_ref.actor.get("self_distillation", None)
        if self_distillation_cfg is None:
            return None

        result = build_sdpo_teacher_inputs(
            batch=batch,
            reward_tensor=reward_tensor,
            cfg=self_distillation_cfg,
            tokenizer=self.tokenizer,
            apply_chat_template_kwargs=self.config.data.get("apply_chat_template_kwargs", {}) or {},
        )
        if result is None:
            return None
        tensors, metrics = result
        return DataProto.from_dict(tensors=tensors), metrics
    # ============ [EasyOPD:SDPO] End ============

    # ============ [EasyOPD:OPSA] Build self-distillation batch with privileged contexts ============
    def _maybe_build_opsa_batch(self, batch):
        """Build teacher inputs with privileged contexts for OPSA self-distillation.

        For OPSA, the teacher is a frozen copy of the student model, but provided
        with type-conditional privileged contexts (paper Section 3.2):
        - Harmful queries  → safety-focused system prompt  (I_h)
        - Benign queries   → helpfulness-focused system prompt (I_b)

        Each prompt in the batch is re-tokenized with its corresponding privileged
        system prompt prepended, then passed through the frozen ref model to obtain
        per-token teacher log-probs used as the dense KL supervision target.

        Returns:
            Tuple of (DataProto with opsa_teacher_log_probs, metrics) or (None, {}).
        """
        if not self.opsa_enable:
            return None, {}

        try:
            import torch

            device = batch.batch["input_ids"].device
            batch_size = batch.batch["input_ids"].shape[0]
            responses = batch.batch["responses"]        # (B, max_resp_len)
            response_mask = batch.batch["response_mask"]  # (B, max_resp_len)
            # batch["prompts"] holds only the prompt token ids (without response).
            # Use it instead of input_ids to avoid leaking the on-policy response
            # into the teacher's user message (Bug 7 fix).
            if "prompts" in batch.batch:
                prompt_ids_batch = batch.batch["prompts"]
            elif "input_ids" in batch.batch:
                prompt_ids_batch = batch.batch["input_ids"]
            else:
                print("[EasyOPD:OPSA] Warning: Neither 'prompts' nor 'input_ids' found in batch.")
                return None, {}
            max_prompt_len = getattr(self.config.data, "max_prompt_length", 1024)
            pad_token_id = self.tokenizer.pad_token_id or 0

            # safety_label column: "harmful" | "benign" | missing → default to harmful context
            safety_labels = batch.non_tensor_batch.get("safety_label", None)

            apply_kwargs = {}
            if hasattr(self.config.data, "apply_chat_template_kwargs"):
                apply_kwargs = dict(self.config.data.get("apply_chat_template_kwargs", {}) or {})

            teacher_input_ids_list = []
            teacher_attention_mask_list = []
            teacher_position_ids_list = []
            # Per-sample valid response lengths; used later to build aligned responses tensor.
            valid_resp_lens = []

            for i in range(batch_size):
                # Choose privileged context based on query type
                if safety_labels is not None:
                    label = safety_labels[i] if isinstance(safety_labels[i], str) else str(safety_labels[i])
                    privileged_ctx = (
                        self.opsa_benign_context if label.lower() == "benign"
                        else self.opsa_harmful_context
                    )
                else:
                    privileged_ctx = self.opsa_harmful_context

                # Decode only the prompt part (no response tokens).
                # batch["prompts"] has shape (B, prompt_len); batch["input_ids"] has shape
                # (B, prompt_len + resp_len).  In either case we use the attention_mask
                # sliced to exactly prompt_ids_batch[i]'s own length to strip left-padding.
                prompt_seq_len = prompt_ids_batch[i].shape[-1]
                # attention_mask covers the full input_ids sequence; take only the
                # first prompt_seq_len positions which correspond to the prompt.
                attn = batch.batch["attention_mask"][i, :prompt_seq_len]
                # Find the first valid (non-padding) token position for robust decoding.
                valid_indices = (attn == 1).nonzero(as_tuple=True)[0]
                if len(valid_indices) > 0:
                    first_valid_idx = valid_indices[0]
                    raw_prompt_text = self.tokenizer.decode(
                        prompt_ids_batch[i][first_valid_idx:], skip_special_tokens=True
                    )
                else:
                    raw_prompt_text = ""

                teacher_messages = [
                    {"role": "system", "content": privileged_ctx},
                    {"role": "user", "content": raw_prompt_text},
                ]

                teacher_prompt_text = self.tokenizer.apply_chat_template(
                    teacher_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    **apply_kwargs,
                )

                teacher_prompt_out = self.tokenizer(
                    teacher_prompt_text,
                    return_tensors="pt",
                    add_special_tokens=False,
                    truncation=True,
                    max_length=max_prompt_len,
                )
                t_prompt_ids = teacher_prompt_out["input_ids"].squeeze(0)
                t_prompt_mask = teacher_prompt_out["attention_mask"].squeeze(0)

                # Concatenate with the student's valid (non-padded) on-policy response tokens.
                valid_resp_len = int(response_mask[i].sum().item())
                valid_resp_lens.append(valid_resp_len)
                resp_ids = responses[i, :valid_resp_len]
                full_ids = torch.cat([t_prompt_ids, resp_ids], dim=0)
                full_mask = torch.cat([t_prompt_mask, torch.ones(valid_resp_len, dtype=torch.long)], dim=0)
                full_pos = torch.arange(len(full_ids), dtype=torch.long)

                teacher_input_ids_list.append(full_ids)
                teacher_attention_mask_list.append(full_mask)
                teacher_position_ids_list.append(full_pos)

            # Pad to uniform length within this batch
            teacher_input_ids = torch.nn.utils.rnn.pad_sequence(
                teacher_input_ids_list, batch_first=True, padding_value=pad_token_id,
            ).to(device)
            teacher_attention_mask = torch.nn.utils.rnn.pad_sequence(
                teacher_attention_mask_list, batch_first=True, padding_value=0,
            ).to(device)
            teacher_position_ids = torch.nn.utils.rnn.pad_sequence(
                teacher_position_ids_list, batch_first=True, padding_value=0,
            ).to(device)

            # Build per-sample response tensors aligned to the teacher sequence.
            # _forward_micro_batch uses `responses.size(-1)` as response_length to slice
            # the last N positions off the teacher sequence.  We must pass a `responses`
            # whose length equals the number of valid response tokens in the teacher
            # input_ids so that the slice correctly covers the response region.
            # Since different samples have different valid_resp_lens, we pad to the
            # maximum and build a matching response_mask (Bug 1 fix).
            max_valid_resp = max(valid_resp_lens)
            teacher_responses = torch.zeros(batch_size, max_valid_resp, dtype=responses.dtype, device=device)
            teacher_response_mask = torch.zeros(batch_size, max_valid_resp, dtype=response_mask.dtype, device=device)
            for i, vlen in enumerate(valid_resp_lens):
                teacher_responses[i, :vlen] = responses[i, :vlen]
                teacher_response_mask[i, :vlen] = 1

            # Pass through frozen ref model (teacher) to get per-token log-probs
            teacher_batch = DataProto.from_dict(tensors={
                "input_ids": teacher_input_ids,
                "attention_mask": teacher_attention_mask,
                "position_ids": teacher_position_ids,
                "responses": teacher_responses,
                "response_mask": teacher_response_mask,
            })

            # ============ [EasyOPD:OPSA] request top-K teacher logits when configured ============
            opsa_topk_k = None
            try:
                actor_cfg = self.config.actor_rollout_ref.actor
                opsa_topk_k_cfg = actor_cfg.get("opsa_topk_logits_k", 0)
                if opsa_topk_k_cfg is not None and int(opsa_topk_k_cfg) > 0:
                    opsa_topk_k = int(opsa_topk_k_cfg)
                    teacher_batch.meta_info["opsa_topk_k"] = opsa_topk_k
            except Exception:
                opsa_topk_k = None
            # ============ [EasyOPD:OPSA] End ============

            if not self.ref_in_actor:
                teacher_output = self.ref_policy_wg.compute_ref_log_prob(teacher_batch)
            else:
                teacher_output = self.actor_rollout_wg.compute_ref_log_prob(teacher_batch)

            # Shape: (batch_size, max_valid_resp) — per-token log-probs under privileged context.
            # Pad to (batch_size, max_resp_len) so the tensor aligns with response_mask and
            # student log_prob when merged into the training batch via batch.union().
            teacher_log_probs_raw = teacher_output.batch["ref_log_prob"]  # (B, max_valid_resp)
            max_resp_len = responses.shape[1]
            if teacher_log_probs_raw.shape[1] < max_resp_len:
                pad_cols = max_resp_len - teacher_log_probs_raw.shape[1]
                teacher_log_probs = torch.nn.functional.pad(
                    teacher_log_probs_raw, (0, pad_cols), value=0.0
                )
            elif teacher_log_probs_raw.shape[1] > max_resp_len:
                teacher_log_probs = teacher_log_probs_raw[:, :max_resp_len]
            else:
                teacher_log_probs = teacher_log_probs_raw

            opsa_tensors = {"opsa_teacher_log_probs": teacher_log_probs}

            # ============ [EasyOPD:OPSA] forward top-K teacher distributions to the actor ============
            if opsa_topk_k is not None and "ref_topk_log_probs" in teacher_output.batch.keys():
                topk_values_raw = teacher_output.batch["ref_topk_log_probs"]   # (B, max_valid_resp, K)
                topk_indices_raw = teacher_output.batch["ref_topk_indices"]    # (B, max_valid_resp, K)
                if topk_values_raw.shape[1] < max_resp_len:
                    pad_cols = max_resp_len - topk_values_raw.shape[1]
                    topk_values = torch.nn.functional.pad(topk_values_raw, (0, 0, 0, pad_cols), value=0.0)
                    topk_indices = torch.nn.functional.pad(topk_indices_raw, (0, 0, 0, pad_cols), value=0)
                elif topk_values_raw.shape[1] > max_resp_len:
                    topk_values = topk_values_raw[:, :max_resp_len, :]
                    topk_indices = topk_indices_raw[:, :max_resp_len, :]
                else:
                    topk_values = topk_values_raw
                    topk_indices = topk_indices_raw
                opsa_tensors["opsa_teacher_topk_log_probs"] = topk_values
                opsa_tensors["opsa_teacher_topk_indices"] = topk_indices.to(torch.int64)
            # ============ [EasyOPD:OPSA] End ============

            metrics = {
                "opsa/teacher_log_prob_mean": teacher_log_probs.mean().detach().item(),
                "opsa/teacher_log_prob_std": teacher_log_probs.std().detach().item(),
            }

            # TFR proxy: on harmful samples, measure how much higher the teacher log-prob is
            # compared to the student's on-policy log-prob.  A large positive gap means the
            # privileged context is successfully activating safer (more probable) responses.
            if safety_labels is not None and "old_log_probs" in batch.batch:
                import numpy as np
                student_lp = batch.batch["old_log_probs"]  # (B, resp_len), student on-policy
                harmful_mask = [
                    (str(safety_labels[i]).lower() != "benign")
                    for i in range(batch_size)
                ]
                if any(harmful_mask):
                    harm_idx = [i for i, m in enumerate(harmful_mask) if m]
                    teacher_lp_harmful = teacher_log_probs[harm_idx].mean().detach().item()
                    student_lp_harmful = student_lp[harm_idx].mean().detach().item()
                    tfr_proxy = teacher_lp_harmful - student_lp_harmful
                    metrics["opsa/tfr_proxy_lp_gap"] = tfr_proxy
                    # Buffer for periodic TFR logging
                    self._opsa_tfr_buffer.append(tfr_proxy)

            return DataProto.from_dict(tensors=opsa_tensors), metrics

        except Exception as e:
            print(f"[EasyOPD:OPSA] Warning: Failed to build OPSA teacher batch: {e}")
            return None, {}
    # ============ [EasyOPD:OPSA] End ============

    def _apply_filter_groups(self, batch: DataProto) -> DataProto:
        cfg = getattr(self.config.algorithm, "filter_groups", None)
        if cfg is None or not getattr(cfg, "enable", False):
            return batch
        metric = getattr(cfg, "metric", None)
        if metric is None or "uid" not in batch.non_tensor_batch:
            return batch

        metric_tensor = self._get_filter_metric_tensor(batch, metric)
        if metric_tensor is None or metric_tensor.numel() == 0:
            return batch

        uid_array = batch.non_tensor_batch["uid"]
        uid_list = uid_array.tolist() if hasattr(uid_array, "tolist") else list(uid_array)

        values_by_uid = defaultdict(list)
        indices_by_uid = defaultdict(list)
        metric_vals = metric_tensor.detach().cpu().tolist()
        for idx, (uid, val) in enumerate(zip(uid_list, metric_vals, strict=False)):
            values_by_uid[uid].append(val)
            indices_by_uid[uid].append(idx)

        keep_indices: list[int] = []
        for uid, vals in values_by_uid.items():
            vals_tensor = torch.tensor(vals, dtype=torch.float32)
            if metric == "reward_std":
                std = torch.std(vals_tensor, unbiased=False)
                if std >= float(getattr(cfg, "reward_std_threshold", 0.0)):
                    keep_indices.extend(indices_by_uid[uid])
            else:
                if len(vals_tensor) == 1 or torch.std(vals_tensor, unbiased=False) > 0:
                    keep_indices.extend(indices_by_uid[uid])

        if len(keep_indices) == 0 or len(keep_indices) == len(uid_list):
            return batch

        keep_indices.sort()
        return batch[keep_indices]

    def _get_filter_metric_tensor(self, batch: DataProto, metric: str) -> torch.Tensor | None:
        if metric == "seq_reward":
            return batch.batch["token_level_scores"].sum(dim=-1)
        if metric == "seq_final_reward":
            return batch.batch["token_level_rewards"].sum(dim=-1)
        if metric in batch.non_tensor_batch:
            data = batch.non_tensor_batch[metric]
            if isinstance(data, torch.Tensor):
                return data
            return torch.as_tensor(np.asarray(data))
        return None

    def _record_sample(self, batch: DataProto, storage: deque, path: str):
        if len(batch) == 0:
            return
        try:
            uid_arr = batch.non_tensor_batch.get("uid")
            uid_list = uid_arr.tolist() if hasattr(uid_arr, "tolist") else list(uid_arr) if uid_arr is not None else None
            if uid_list:
                chosen_uid = random.choice(uid_list)
                indices = [i for i, u in enumerate(uid_list) if u == chosen_uid]
            else:
                idx = np.random.randint(len(batch))
                chosen_uid = f"sample_{idx}"
                indices = [idx]

            entries = []
            extra_infos = batch.non_tensor_batch.get("extra_info")
            for i in indices:
                prompt = batch.batch["prompts"][i]
                response = batch.batch["responses"][i]
                mask = batch.batch["response_mask"][i]
                inp = batch.batch["input_ids"][i]
                attn = batch.batch["attention_mask"][i]

                prompt_len = int(attn[: prompt.shape[-1]].sum().item())
                response_len = int(mask.sum().item())
                valid_len = int(attn.sum().item())

                entries.append(
                    {
                        "prompt_text": self.tokenizer.decode(prompt[-prompt_len:], skip_special_tokens=False),
                        "response_text": self.tokenizer.decode(response[:response_len], skip_special_tokens=False),
                        "full_text": self.tokenizer.decode(inp[:valid_len], skip_special_tokens=False),
                        "tool_info": (
                            extra_infos[i] if extra_infos is not None and isinstance(extra_infos[i], dict) else None
                        ),
                    }
                )

            record = {
                "global_step": self.global_steps,
                "uid": chosen_uid,
                "traj": entries,
            }
            storage.append(record)
            with open(path, "w", encoding="utf-8") as f:
                for item in storage:
                    f.write(json.dumps(item, ensure_ascii=False, default=_json_default) + "\n")
        except Exception as exc:
            print(f"Warning: failed to record sample trace: {exc}")

    def _apply_token_kl_regularizer(self, batch: DataProto, cfg) -> DataProto:
        """Apply token-level KL regularization to advantages.

        Supports two modes:
        - Stepwise mode (SOD): adaptive per-step weighting via easyopd.methods.sod.core
        - Legacy mode: sigmoid-gated beta blending via compute_total_advantage
        """
        if cfg is None or not getattr(cfg, "enable", False):
            return batch
        if "advantages" not in batch.batch:
            return batch
        if "ref_log_prob" not in batch.batch or "old_log_probs" not in batch.batch:
            return batch

        response_mask = batch.batch["response_mask"]

        # ============ [EasyOPD] Hook-based SOD dispatch ============
        stepwise_enable = getattr(cfg, "stepwise_enable", False)
        if stepwise_enable:
            # Route to SOD method via hook dispatch if available
            if self.hook_dispatcher is not None and self.hook_dispatcher.enabled:
                from easyopd.methods.sod.core import compute_stepwise_opd_weights

                raw_local_adv = (batch.batch["ref_log_prob"] - batch.batch["old_log_probs"]) * response_mask
                epsilon = float(getattr(cfg, "stepwise_epsilon", 1e-6))
                delta = float(getattr(cfg, "stepwise_delta", 0.5))
                opd_coef = float(getattr(cfg, "stepwise_opd_coef", 1.0))

                stepwise_weights, stepwise_log_info = compute_stepwise_opd_weights(
                    old_log_probs=batch.batch["old_log_probs"],
                    ref_log_prob=batch.batch["ref_log_prob"],
                    response_mask=response_mask,
                    epsilon=epsilon,
                    delta=delta,
                )
                stepwise_weights = stepwise_weights.to(raw_local_adv.device)

                weighted_opd = opd_coef * stepwise_weights * raw_local_adv

                grpo_adv = batch.batch["advantages"]
                if grpo_adv.dim() == 1:
                    A_total_token = grpo_adv.unsqueeze(1) + weighted_opd
                else:
                    A_total_token = grpo_adv + weighted_opd

                batch.batch["advantages"] = A_total_token
                batch.meta_info.setdefault("token_kl_reg", {})
                batch.meta_info["token_kl_reg"].update(
                    {"stepwise_weights": stepwise_weights.detach().clone(), "local_adv": raw_local_adv.detach().clone()}
                )

                # Collect diagnostics via MetricsCollector
                if self.easyopd_metrics is not None:
                    self.easyopd_metrics.collect({
                        "mean_step_weight": stepwise_weights[response_mask.bool()].mean().item() if response_mask.any() else 0.0,
                    })
            else:
                # Fallback: direct import (legacy path)
                from easyopd.methods.sod.core import compute_stepwise_opd_weights

                raw_local_adv = (batch.batch["ref_log_prob"] - batch.batch["old_log_probs"]) * response_mask
                epsilon = float(getattr(cfg, "stepwise_epsilon", 1e-6))
                delta = float(getattr(cfg, "stepwise_delta", 0.5))
                opd_coef = float(getattr(cfg, "stepwise_opd_coef", 1.0))

                stepwise_weights, stepwise_log_info = compute_stepwise_opd_weights(
                    old_log_probs=batch.batch["old_log_probs"],
                    ref_log_prob=batch.batch["ref_log_prob"],
                    response_mask=response_mask,
                    epsilon=epsilon,
                    delta=delta,
                )
                stepwise_weights = stepwise_weights.to(raw_local_adv.device)

                weighted_opd = opd_coef * stepwise_weights * raw_local_adv

                grpo_adv = batch.batch["advantages"]
                if grpo_adv.dim() == 1:
                    A_total_token = grpo_adv.unsqueeze(1) + weighted_opd
                else:
                    A_total_token = grpo_adv + weighted_opd

                batch.batch["advantages"] = A_total_token
                batch.meta_info.setdefault("token_kl_reg", {})
                batch.meta_info["token_kl_reg"].update(
                    {"stepwise_weights": stepwise_weights.detach().clone(), "local_adv": raw_local_adv.detach().clone()}
                )
        # ============ [EasyOPD] End ============
        else:
            # Legacy mode: sigmoid-gated beta blending
            raw_local_adv = (batch.batch["ref_log_prob"] - batch.batch["old_log_probs"]) * response_mask
            base_adv = masked_mean(batch.batch["advantages"], response_mask, axis=1)

            beta_max = getattr(cfg, "beta_max", None)
            beta_min = float(getattr(cfg, "beta_min", 0.0))
            if beta_max is None or beta_max <= beta_min:
                return batch

            gamma = float(getattr(cfg, "gamma", 1.0))

            A_total_token, beta_jk, local_adv_used = compute_total_advantage(
                base_adv,
                raw_local_adv,
                gamma=gamma,
                beta_min=beta_min,
                beta_max=beta_max,
            )
            batch.batch["advantages"] = A_total_token
            batch.meta_info.setdefault("token_kl_reg", {})
            batch.meta_info["token_kl_reg"].update(
                {"beta_jk": beta_jk.detach().clone(), "local_adv": local_adv_used.detach().clone()}
            )
        return batch

    def _maybe_log_val_generations(self, inputs, outputs, scores, gts=None):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
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
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            print(f"len reward_extra_infos_dict['reward']: {len(reward_extra_infos_dict['reward'])}")
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)
                    print(f"len reward_extra_infos_dict['{key}']: {len(reward_extra_infos_dict[key])}")

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(
            inputs=sample_inputs,
            outputs=sample_outputs,
            scores=sample_scores,
            gts=sample_gts,
        )

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
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

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

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
            actor_rollout_config = deepcopy(self.config.actor_rollout_ref)
            if hasattr(self.config, "distillation"):
                with open_dict(actor_rollout_config):
                    actor_rollout_config.distillation = self.config.distillation
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=actor_rollout_config,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
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
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
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

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

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

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

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

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

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

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        import numpy as np
        import uuid

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

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    extra_infos = batch.non_tensor_batch.get("extra_info", None)
                    tool_metrics = self._collect_tool_metrics(extra_infos)
                    if tool_metrics:
                        metrics.update(tool_metrics)

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            from verl.utils.debug.metrics import calculate_debug_metrics

                            metrics.update(calculate_debug_metrics(batch))

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, color="olive"):
                            # ============ [EasyOPD] Context distillation before ref log prob ============
                            if self.use_context_distillation:
                                from easyopd.methods.g_opd.ref_input_utils import prepare_critique_distillation_inputs

                                apply_chat_template_kwargs = self.config.data.get(
                                    "apply_chat_template_kwargs", {}
                                )
                                batch = prepare_critique_distillation_inputs(
                                    batch=batch,
                                    tokenizer=self.tokenizer,
                                    critique_vllm_url=self.critique_vllm_url,
                                    critique_model=self.critique_model,
                                    critique_prompt_template=None,
                                    ref_apply_chat_template_kwargs=apply_chat_template_kwargs,
                                    max_critique_tokens=self.max_critique_tokens,
                                    critique_temperature=self.critique_temperature,
                                    critique_top_p=self.critique_top_p,
                                )
                            elif self.use_ref_solution_distillation:
                                from easyopd.methods.g_opd.ref_input_utils import prepare_ref_model_inputs_based_on_correct_solution

                                apply_chat_template_kwargs = self.config.data.get(
                                    "apply_chat_template_kwargs", {}
                                )
                                batch = prepare_ref_model_inputs_based_on_correct_solution(
                                    batch=batch,
                                    tokenizer=self.tokenizer,
                                    apply_chat_template_kwargs=apply_chat_template_kwargs,
                                )
                            # ============ [EasyOPD] End ============

                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # ============ [EasyOPD] Compute base model log probs for corrected reward ============
                    if self.use_base_models:
                        with marked_timer("base_log_probs", timing_raw, color="green"):
                            # Compute base_ref_log_prob using ref's base model
                            if not self.ref_in_actor:
                                base_ref_log_prob = self.ref_policy_wg.compute_base_ref_log_prob(batch)
                            else:
                                base_ref_log_prob = self.actor_rollout_wg.compute_base_ref_log_prob(batch)
                            batch = batch.union(base_ref_log_prob)

                            # Compute base_log_prob using actor's base model with input_ids
                            # Temporarily remove ref_input_ids to ensure compute uses input_ids
                            ref_input_tensors = {}
                            if "ref_input_ids" in batch.batch:
                                ref_input_tensors["ref_input_ids"] = batch.batch.pop("ref_input_ids")
                            if "ref_attention_mask" in batch.batch:
                                ref_input_tensors["ref_attention_mask"] = batch.batch.pop("ref_attention_mask")
                            if "ref_position_ids" in batch.batch:
                                ref_input_tensors["ref_position_ids"] = batch.batch.pop("ref_position_ids")

                            base_log_prob = self.actor_rollout_wg.compute_base_log_prob(batch)
                            batch = batch.union(base_log_prob)

                            # Restore ref_input_ids tensors
                            for key, tensor in ref_input_tensors.items():
                                batch.batch[key] = tensor

                            print(f"[EasyOPD] Computed base log probs: "
                                  f"base_log_prob shape={batch.batch['base_log_prob'].shape}, "
                                  f"base_ref_log_prob shape={batch.batch['base_ref_log_prob'].shape}")
                    # ============ [EasyOPD] End ============

                    # [EasyOPD:lightning_opd] Convert teacher_log_probs from
                    # non-tensor (ragged list from parquet) to padded tensor.
                    if "teacher_log_probs" in getattr(batch, "non_tensor_batch", {}):
                        from easyopd.methods.lightning_opd.data_adapter import attach_teacher_log_probs

                        attach_teacher_log_probs(batch)
                    # [EasyOPD:lightning_opd] End

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # ============ [EasyOPD] Rollout correction (IS weights / rejection sampling) ============
                        # Faithful to lasgroup/SDPO: when algorithm.rollout_correction
                        # is configured (e.g. rollout_is=token), compute token-level
                        # importance-sampling weights from (old_log_probs vs rollout_log_probs)
                        # and add them to the batch. The SDPO loss multiplies the per-token
                        # distillation loss by these weights (correcting the
                        # training-vs-rollout policy mismatch).
                        rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                        if rollout_corr_config is not None and "rollout_log_probs" in batch.batch:
                            from verl.trainer.ppo.rollout_corr_helper import (
                                compute_rollout_correction_and_add_to_batch,
                            )

                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(
                                batch, rollout_corr_config
                            )
                            metrics.update(is_metrics)
                        # ============ [EasyOPD] End ============

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                        batch = self._apply_filter_groups(batch)
                        if len(batch) == 0:
                            continue
                        self._record_sample(batch, self.sample_traces, self.sample_log_path)

                        token_kl_cfg = getattr(self.config.algorithm, "token_kl_reg", None)
                        if token_kl_cfg and getattr(token_kl_cfg, "enable", False):
                            batch = self._apply_token_kl_regularizer(batch, token_kl_cfg)

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # ============ [EasyOPD] Build self-distillation batch ============
                        vopd_loss_mode = self.config.actor_rollout_ref.actor.policy_loss.get("loss_mode", "vanilla")
                        # Resolve through registry alias mapping (no hardcoded names).
                        try:
                            from easyopd.registry import resolve_method_name as _resolve_lm
                            _resolved_lm = _resolve_lm(vopd_loss_mode) if vopd_loss_mode else None
                        except Exception:  # noqa: BLE001
                            _resolved_lm = vopd_loss_mode
                        if _resolved_lm == "vision_opd":
                            vopd_result = self._maybe_build_vision_opd_batch(batch, reward_tensor)
                            if vopd_result is not None:
                                vopd_batch_data, vopd_metrics = vopd_result
                                batch = batch.union(vopd_batch_data)
                                metrics.update(vopd_metrics)
                        # ============ [EasyOPD:SDPO] Build SDPO reprompt self-teacher batch ============
                        elif _resolved_lm == "sdpo":
                            sdpo_result = self._maybe_build_sdpo_batch(batch, reward_tensor)
                            if sdpo_result is not None:
                                sdpo_batch_data, sdpo_pre_metrics = sdpo_result
                                batch = batch.union(sdpo_batch_data)
                                metrics.update(sdpo_pre_metrics)
                        # ============ [EasyOPD:SDPO] End ============
                        # ============ [EasyOPD] End ============

                        # ============ [EasyOPD] Context distillation - experience injection (OPCD) ============
                        opcd_stage = getattr(self.config.trainer, "stage", None)
                        if opcd_stage == "consolidate":
                            opcd_result = self._maybe_build_opcd_batch(batch)
                            if opcd_result is not None:
                                opcd_batch_data, opcd_metrics = opcd_result
                                batch = batch.union(opcd_batch_data)
                                metrics.update(opcd_metrics)
                                batch.meta_info["stage_merge"] = True
                                batch.meta_info["on_policy_merge"] = getattr(
                                    self.config.trainer, "on_policy_merge", True
                                )
                        # ============ [EasyOPD] End ============

                        # ============ [EasyOPD:Simple] Teacher hidden-state injection for cross-tokenizer KD ============
                        simple_xtok_result = self._maybe_build_simple_xtok_batch(batch)
                        if simple_xtok_result is not None:
                            simple_xtok_batch, simple_xtok_metrics = simple_xtok_result
                            batch = batch.union(simple_xtok_batch)
                            metrics.update(simple_xtok_metrics)
                            print(
                                f"[EasyOPD:simple] Step {self.global_steps}: teacher hidden states injected, "
                                f"batch={len(simple_xtok_batch.non_tensor_batch['teacher_hidden_states'])}"
                            )
                        # ============ [EasyOPD:Simple] End ============

                        # ============ [EasyOPD:OPSA] Teacher log-prob injection for self-distillation ============
                        if self.opsa_enable:
                            opsa_result, opsa_pre_metrics = self._maybe_build_opsa_batch(batch)
                            if opsa_result is not None:
                                batch = batch.union(opsa_result)
                                metrics.update(opsa_pre_metrics)
                                print(f"[EasyOPD:OPSA] Step {self.global_steps}: teacher log-probs injected, shape={opsa_result.batch['opsa_teacher_log_probs'].shape}")
                            else:
                                print(f"[EasyOPD:OPSA] Step {self.global_steps}: WARNING - _maybe_build_opsa_batch returned None! OPSA loss will NOT execute.")

                            # Periodic TFR proxy reporting (paper Section 3.3)
                            if (
                                self._opsa_tfr_buffer
                                and self.global_steps % self.opsa_tfr_eval_frequency == 0
                            ):
                                import numpy as np
                                tfr_mean = float(np.mean(self._opsa_tfr_buffer))
                                metrics["opsa/tfr_proxy_lp_gap_avg"] = tfr_mean
                                if tfr_mean < 0:
                                    print(
                                        f"[EasyOPD:OPSA] Step {self.global_steps}: TFR proxy gap={tfr_mean:.4f} "
                                        f"(negative: privileged context may not be effective enough)"
                                    )
                                self._opsa_tfr_buffer.clear()
                        # ============ [EasyOPD:OPSA] End ============

                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            sample_gts = [
                                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
                                for item in batch
                            ]

                            if "request_id" in batch.non_tensor_batch:
                                reward_extra_infos_dict.setdefault(
                                    "request_id",
                                    batch.non_tensor_batch["request_id"].tolist(),
                                )

                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                gts=sample_gts,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

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
                    metrics.update(val_metrics)

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
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

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
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
