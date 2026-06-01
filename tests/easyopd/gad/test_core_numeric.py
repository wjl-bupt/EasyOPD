"""CPU numeric tests for easyopd.methods.gad.core pure functions."""

import math

import torch


def test_summed_reward_applies_mask():
    from easyopd.methods.gad.core import summed_reward

    values = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    out = summed_reward(values, mask)

    assert out.shape == (2,)
    assert torch.allclose(out, torch.tensor([3.0, 11.0]))


def test_compute_discriminator_loss_matches_closed_form():
    from easyopd.methods.gad.core import compute_discriminator_loss

    student = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    teacher = torch.tensor([[2.0, 2.0], [3.0, 3.0]])
    smask = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    tmask = torch.tensor([[1.0, 1.0], [1.0, 1.0]])

    loss = compute_discriminator_loss(student, teacher, smask, tmask)

    # teacher_sum - student_sum = [2, 2]; -mean(log_sigmoid([2,2]))
    expected = -math.log(1.0 / (1.0 + math.exp(-2.0)))
    assert torch.allclose(loss, torch.tensor(expected), atol=1e-6)


def test_discriminator_accuracy_perfect_and_inverted():
    from easyopd.methods.gad.core import discriminator_accuracy

    student = torch.tensor([[1.0], [1.0]])
    teacher = torch.tensor([[2.0], [2.0]])
    mask = torch.tensor([[1.0], [1.0]])

    acc_good = discriminator_accuracy(student, teacher, mask, mask)
    assert acc_good == 1.0

    acc_bad = discriminator_accuracy(teacher, student, mask, mask)
    assert acc_bad == 0.0


def test_last_token_only_keeps_only_final_valid_position():
    from easyopd.methods.gad.core import last_token_only

    values = torch.tensor([[0.5, 1.5, 2.5, 3.5], [10.0, 20.0, 30.0, 40.0]])
    # Row 0 has 3 valid tokens (last valid at index 2); row 1 has 4 (last at 3).
    response_mask = torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    out = last_token_only(values, response_mask)

    expected = torch.tensor([[0.0, 0.0, 2.5, 0.0], [0.0, 0.0, 0.0, 40.0]])
    assert torch.allclose(out, expected)


def test_last_token_only_handles_all_zero_mask_row():
    from easyopd.methods.gad.core import last_token_only

    values = torch.tensor([[1.0, 2.0, 3.0]])
    mask = torch.tensor([[0.0, 0.0, 0.0]])
    out = last_token_only(values, mask)

    # When no valid token, output must be all zeros (no spurious score).
    assert torch.allclose(out, torch.zeros_like(values))
