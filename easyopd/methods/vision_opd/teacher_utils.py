"""
Vision-OPD: Learning to See Fine-Grained Details for Multimodal LLMs
via On-Policy Self-Distillation - Teacher Input Utilities

This module provides utilities for preparing teacher model inputs in Vision-OPD.
The teacher model receives enhanced visual inputs (e.g., bounding-box cropped images)
to provide a stronger supervision signal for the student model.

Key functions:
    - prepare_teacher_messages_with_bbox_images: Swap student images with teacher bbox images.
    - prepare_opsd_teacher_messages: Add answer hint to teacher prompt (OPSD mode).
    - build_teacher_prompt_inputs: Tokenize teacher messages and prepare model inputs.
"""

import logging
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def prepare_teacher_messages_with_bbox_images(
    prompt_messages: List[Dict],
    teacher_images: List[Any],
    teacher_prompt_messages: Optional[List[Dict]] = None,
) -> List[Dict]:
    """Prepare teacher messages by swapping images with teacher-specific images.

    In Vision-OPD, the teacher model receives fine-grained visual inputs
    (e.g., bounding-box cropped images) while the student sees the original image.

    Args:
        prompt_messages: Original prompt messages with student images.
        teacher_images: List of teacher-specific images (e.g., bbox crops).
        teacher_prompt_messages: Optional custom teacher prompt template.

    Returns:
        List of messages with teacher images swapped in.
    """
    if teacher_prompt_messages is not None:
        return _build_teacher_messages_from_template(teacher_prompt_messages, teacher_images)
    return _swap_images_in_messages(prompt_messages, teacher_images)


def prepare_opsd_teacher_messages(
    raw_prompt_messages: List[Dict],
    answer: str,
    hint_template: str,
) -> List[Dict]:
    """Prepare teacher messages with answer hint (OPSD mode).

    In OPSD (On-Policy Self-Distillation) mode, the teacher receives the
    ground truth answer as a hint appended to the original prompt.

    Args:
        raw_prompt_messages: Original prompt messages.
        answer: Ground truth answer string.
        hint_template: Template for the answer hint. Uses {answer} placeholder.

    Returns:
        List of messages with answer hint appended to the last user message.
    """
    teacher_messages = deepcopy(raw_prompt_messages)
    last_msg = teacher_messages[-1]
    content = last_msg["content"]
    hint_suffix = hint_template.format(answer=answer)

    if isinstance(content, list):
        teacher_messages[-1]["content"] = list(content) + [{"type": "text", "text": hint_suffix}]
    elif isinstance(content, str):
        teacher_messages[-1]["content"] = content + hint_suffix
    else:
        raise TypeError(f"Unsupported message content type: {type(content)}")
    return teacher_messages


def _swap_images_in_messages(
    messages: List[Dict],
    new_images: List[Any],
) -> List[Dict]:
    """Swap images in messages with new images.

    Replaces all image content items in the messages with the provided new images.

    Args:
        messages: Original messages containing image content.
        new_images: New images to replace the originals.

    Returns:
        Messages with images swapped.
    """
    result = deepcopy(messages)
    image_idx = 0

    for msg in result:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for i, item in enumerate(content):
            if not isinstance(item, dict):
                continue
            if item.get("type") == "image" or "image" in item or "image_url" in item:
                if image_idx < len(new_images):
                    content[i] = {"type": "image", "image": new_images[image_idx]}
                    image_idx += 1

    return result


def _build_teacher_messages_from_template(
    template_messages: List[Dict],
    teacher_images: List[Any],
) -> List[Dict]:
    """Build teacher messages from a custom template with teacher images.

    Args:
        template_messages: Template messages for the teacher.
        teacher_images: Teacher-specific images.

    Returns:
        Messages built from template with teacher images.
    """
    result = deepcopy(template_messages)
    image_idx = 0

    for msg in result:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for i, item in enumerate(content):
            if not isinstance(item, dict):
                continue
            if item.get("type") == "image" or "image" in item:
                if image_idx < len(teacher_images):
                    content[i] = {"type": "image", "image": teacher_images[image_idx]}
                    image_idx += 1

    return result


def extract_images_from_messages(messages: List[Dict]) -> List[Any]:
    """Extract all images from a list of messages.

    Args:
        messages: List of message dicts.

    Returns:
        List of extracted images.
    """
    images = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "image":
                continue
            if "image" in item:
                images.append(item["image"])
            elif "path" in item:
                images.append(item["path"])
            elif "bytes" in item:
                images.append({"bytes": item["bytes"]})
    return images


def teacher_images_available(teacher_images: Any) -> bool:
    """Check if teacher images are available (non-empty).

    Args:
        teacher_images: Teacher images to check.

    Returns:
        True if at least one valid teacher image exists.
    """
    if teacher_images is None:
        return False
    if isinstance(teacher_images, np.ndarray):
        teacher_images = teacher_images.tolist()
    elif not isinstance(teacher_images, (list, tuple)):
        teacher_images = [teacher_images]
    for image in teacher_images:
        if image is None:
            continue
        if isinstance(image, str):
            if image:
                return True
            continue
        if isinstance(image, dict):
            if image.get("path") or image.get("bytes") is not None or image.get("image") is not None:
                return True
            continue
        return True
    return False
