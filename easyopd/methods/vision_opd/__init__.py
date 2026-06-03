"""
Vision-OPD: Learning to See Fine-Grained Details for Multimodal LLMs
via On-Policy Self-Distillation.

This method implements the Vision-OPD framework which uses on-policy self-distillation
with a teacher model (maintained via EMA) that receives fine-grained visual inputs
(e.g., bounding-box cropped images) to guide the student model.

Paper: https://arxiv.org/abs/2605.18740

Integration mode: Mode B (significant verl modification)
    - Core algorithm in easyopd/methods/vision_opd/core.py
    - Teacher input utilities in easyopd/methods/vision_opd/teacher_utils.py
    - Config fields added to verl/workers/config/actor.py (SelfDistillationConfig, PolicyLossConfig)
    - Self-distillation loss computation added to verl/workers/actor/dp_actor.py
    - Teacher batch construction added to verl/trainer/ppo/ray_trainer.py
    - Rollout correction config added to verl/trainer/config/algorithm.py

Modified verl files:
    - verl/workers/config/actor.py: Added SelfDistillationConfig, 'vopd' loss mode
    - verl/workers/actor/dp_actor.py: Added teacher forward + VOPD loss computation
    - verl/trainer/ppo/ray_trainer.py: Added _maybe_build_self_distillation_batch
    - verl/trainer/config/algorithm.py: Added RolloutCorrectionConfig
"""

from easyopd.methods.vision_opd.core import (
    compute_self_distillation_loss,
    ema_update_teacher,
    progressive_update_teacher,
    add_tail_bucket,
    renorm_topk_log_probs,
)
from easyopd.methods.vision_opd.teacher_utils import (
    prepare_teacher_messages_with_bbox_images,
    prepare_opsd_teacher_messages,
    extract_images_from_messages,
    teacher_images_available,
)
from easyopd.registry import register_method

__all__ = [
    "compute_self_distillation_loss",
    "ema_update_teacher",
    "progressive_update_teacher",
    "add_tail_bucket",
    "renorm_topk_log_probs",
    "prepare_teacher_messages_with_bbox_images",
    "prepare_opsd_teacher_messages",
    "extract_images_from_messages",
    "teacher_images_available",
    "VisionOPDMethod",
]


@register_method("vision_opd", loss_mode_aliases=("vopd",))
class VisionOPDMethod:
    """Vision-OPD: Learning to See Fine-Grained Details for Multimodal LLMs
    via On-Policy Self-Distillation.

    Metadata class for the EasyOPD registry.
    """

    # Method metadata
    name = "vision_opd"
    description = (
        "Vision-OPD: Learning to See Fine-Grained Details for Multimodal LLMs "
        "via On-Policy Self-Distillation"
    )
    paper_url = "https://arxiv.org/abs/2605.18740"

    # Files modified in verl
    verl_modified_files = [
        "verl/workers/config/actor.py",          # Added SelfDistillationConfig, 'vopd' loss mode
        "verl/workers/actor/dp_actor.py",        # Added teacher forward + VOPD loss computation
        "verl/trainer/ppo/ray_trainer.py",       # Added _maybe_build_self_distillation_batch
        "verl/trainer/config/algorithm.py",      # Added RolloutCorrectionConfig
    ]
