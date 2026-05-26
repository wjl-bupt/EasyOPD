"""Data contract validation for GAD batches.

GAD requires the dataset to provide FOUR teacher-side tensor batch keys
(see GAD_BATCH_KEYS below). These keys are NOT popped into the
gen_batch by verl's `_get_gen_batch` (which only pops input_ids /
attention_mask / position_ids), so they survive into the post-rollout
`batch` consumed by `compute_values` and `update_critic`.

This module validates the contract once per training run (called at
the start of GAD's `update_critic_step`).
"""

from __future__ import annotations

from typing import Mapping

GAD_BATCH_KEYS: tuple[str, ...] = (
    "teacher_input_ids",
    "teacher_attention_mask",
    "teacher_position_ids",
    "teacher_response",
)


class GADBatchContractError(ValueError):
    """Raised when GAD is enabled but the batch violates the data contract."""


def validate_gad_batch(batch_like: Mapping) -> None:
    """Verify a batch dict (or batch.batch from a DataProto) satisfies the GAD contract.

    Checks performed:
      - All four keys in `GAD_BATCH_KEYS` are present.
      - Their batch dimensions match `input_ids`' batch dimension when
        `input_ids` is present in the same dict.

    Args:
        batch_like: anything supporting `__contains__`, `__getitem__`, and `keys()`
            over tensor-shaped values. In practice either a plain dict or a
            DataProto's `.batch` TensorDict.

    Raises:
        GADBatchContractError with a message that names the missing or
        mis-shaped keys and points to docs/algo/gad.md for the schema.
    """
    missing = [k for k in GAD_BATCH_KEYS if k not in batch_like]
    if missing:
        present = sorted(batch_like.keys())
        raise GADBatchContractError(
            f"GAD is enabled but batch is missing required keys. "
            f"Missing: {missing}. Batch has keys: {present}. "
            "See docs/algo/gad.md §Data preparation for the required schema."
        )

    if "input_ids" in batch_like:
        ref_bsz = batch_like["input_ids"].shape[0]
        for k in GAD_BATCH_KEYS:
            actual = batch_like[k].shape[0]
            if actual != ref_bsz:
                raise GADBatchContractError(
                    f"GAD batch dim mismatch for '{k}': "
                    f"expected {ref_bsz} (input_ids batch dim), got {actual}."
                )
