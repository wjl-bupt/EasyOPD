"""Critic-forward adapter for GAD.

The verl critic's `_forward_micro_batch` reads `input_ids`,
`attention_mask`, `position_ids`, and `responses` from the micro-batch
dict (the last one is used to derive `response_length` for the slicing
tail). When GAD needs the critic to score the *teacher* response, we
swap all four keys to point at their `teacher_*` counterparts.

We never mutate the caller's dict.
"""

from __future__ import annotations

from typing import Mapping


_STUDENT_TO_TEACHER: dict[str, str] = {
    "input_ids": "teacher_input_ids",
    "attention_mask": "teacher_attention_mask",
    "position_ids": "teacher_position_ids",
    "responses": "teacher_response",
}


def remap_to_teacher(micro_batch: Mapping) -> dict:
    """Return a new dict where the student forward keys are sourced from teacher_* keys.

    Args:
        micro_batch: dict containing student keys (`input_ids`,
            `attention_mask`, `position_ids`, `responses`) AND teacher
            keys (`teacher_input_ids`, `teacher_attention_mask`,
            `teacher_position_ids`, `teacher_response`). May contain
            other keys (e.g. `multi_modal_inputs`) which are passed
            through unchanged.

    Returns:
        A new dict suitable for passing to the verl critic's existing
        forward path. The critic will read `responses` to compute
        `response_length`, which will now reflect the teacher response.
    """
    remapped = dict(micro_batch)  # shallow copy
    for student_key, teacher_key in _STUDENT_TO_TEACHER.items():
        remapped[student_key] = micro_batch[teacher_key]
    return remapped
