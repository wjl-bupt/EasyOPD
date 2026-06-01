"""Tests for easyopd.methods.gad.critic_forward."""

import torch


def test_remap_to_teacher_swaps_keys(tiny_micro_batch):
    from easyopd.methods.gad.critic_forward import remap_to_teacher

    remapped = remap_to_teacher(tiny_micro_batch)

    # Inputs come from the teacher_* keys.
    assert torch.equal(remapped["input_ids"], tiny_micro_batch["teacher_input_ids"])
    assert torch.equal(remapped["attention_mask"], tiny_micro_batch["teacher_attention_mask"])
    assert torch.equal(remapped["position_ids"], tiny_micro_batch["teacher_position_ids"])
    # `responses` is swapped too so verl's internal `response_length = responses.size(-1)`
    # picks up the teacher response length, not the student's.
    assert torch.equal(remapped["responses"], tiny_micro_batch["teacher_response"])


def test_remap_to_teacher_does_not_mutate_input(tiny_micro_batch):
    from easyopd.methods.gad.critic_forward import remap_to_teacher

    snapshot_ids = tiny_micro_batch["input_ids"].clone()
    snapshot_mask = tiny_micro_batch["attention_mask"].clone()
    snapshot_responses = tiny_micro_batch["responses"].clone()
    _ = remap_to_teacher(tiny_micro_batch)

    assert torch.equal(tiny_micro_batch["input_ids"], snapshot_ids)
    assert torch.equal(tiny_micro_batch["attention_mask"], snapshot_mask)
    assert torch.equal(tiny_micro_batch["responses"], snapshot_responses)


def test_remap_preserves_extra_keys(tiny_micro_batch):
    from easyopd.methods.gad.critic_forward import remap_to_teacher

    tiny_micro_batch["multi_modal_inputs"] = [{"foo": torch.zeros(1)}]
    remapped = remap_to_teacher(tiny_micro_batch)

    assert "multi_modal_inputs" in remapped
