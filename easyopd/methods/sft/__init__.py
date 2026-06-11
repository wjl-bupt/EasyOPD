# Copyright 2026 EasyOPD Contributors
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

"""SFT: Supervised Fine-Tuning baseline method.

This method directly uses verl's built-in FSDPSFTTrainer without any
modifications. No additional code is needed — verl already provides a
complete, production-ready SFT implementation.

Training entry point:
    python -m verl.trainer.fsdp_sft_trainer \\
        data.train_files=<path_to_train.parquet> \\
        data.val_files=<path_to_val.parquet> \\
        model.partial_pretrain=<student_model_path> \\
        trainer.total_epochs=3

Or via torchrun for multi-GPU:
    torchrun --nproc_per_node=8 -m verl.trainer.fsdp_sft_trainer \\
        data.train_files=... model.partial_pretrain=...

Config reference: verl/trainer/config/sft_trainer.yaml
Trainer source:   verl/trainer/fsdp_sft_trainer.py
"""

from dataclasses import dataclass

from easyopd.registry import register_method

__all__ = ["SFTMethod"]


@register_method("sft")
@dataclass(frozen=True)
class SFTMethod:
    """SFT: Supervised Fine-Tuning baseline.

    Metadata class for the EasyOPD registry. SFT uses verl's built-in
    FSDPSFTTrainer directly — no verl modifications, no hooks, no extra code.
    """

    name: str = "sft"
    description: str = (
        "Supervised Fine-Tuning baseline. Trains the student model on "
        "teacher-generated responses using standard cross-entropy loss. "
        "Directly uses verl's FSDPSFTTrainer with no modifications."
    )
    paper_url: str = ""  # SFT is a standard technique
    code_url: str = ""

    # No verl modifications needed
    verl_modified_files: tuple = ()

    # Point to verl's existing implementation
    verl_trainer_module: str = "verl.trainer.fsdp_sft_trainer"
    verl_config_path: str = "verl/trainer/config/sft_trainer.yaml"
