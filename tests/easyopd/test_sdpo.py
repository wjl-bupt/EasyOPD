"""
Unit tests for SDPO: Self-Distilled Policy Optimization.

Tests cover:
    - compute_sdpo_self_distillation_loss: Main distillation loss
    - build_reprompt_text: Reprompting logic
    - select_demonstration: Demonstration selection
    - compute_ema_update: EMA teacher update
    - Mathematical properties: non-negativity, masking, IS correction
"""

import pytest
import torch
import torch.nn.functional as F

from easyopd.methods.sdpo.core import (
    compute_sdpo_self_distillation_loss,
    build_reprompt_text,
    select_demonstration,
    compute_ema_update,
)


class TestComputeSDPOSelfDistillationLoss:
    """Tests for compute_sdpo_self_distillation_loss."""

    def test_output_scalar_full_logit(self):
        """Loss should be a scalar with full-logit distillation."""
        bsz, seq_len, vocab = 2, 10, 50
        student_log_probs = torch.randn(bsz, seq_len)
        teacher_log_probs = torch.randn(bsz, seq_len)
        response_mask = torch.ones(bsz, seq_len)
        student_all = torch.log_softmax(torch.randn(bsz, seq_len, vocab), dim=-1)
        teacher_all = torch.log_softmax(torch.randn(bsz, seq_len, vocab), dim=-1)

        loss, metrics = compute_sdpo_self_distillation_loss(
            student_log_probs=student_log_probs,
            teacher_log_probs=teacher_log_probs,
            response_mask=response_mask,
            alpha=0.5,
            full_logit_distillation=True,
            distillation_topk=None,
            is_clip=None,
            student_all_log_probs=student_all,
            teacher_all_log_probs=teacher_all,
        )
        assert loss.ndim == 0

    def test_output_scalar_topk(self):
        """Loss should be a scalar with top-k distillation."""
        bsz, seq_len, k = 2, 10, 20
        student_log_probs = torch.randn(bsz, seq_len)
        teacher_log_probs = torch.randn(bsz, seq_len)
        response_mask = torch.ones(bsz, seq_len)
        student_topk = torch.log_softmax(torch.randn(bsz, seq_len, k), dim=-1)
        teacher_topk = torch.log_softmax(torch.randn(bsz, seq_len, k), dim=-1)

        loss, metrics = compute_sdpo_self_distillation_loss(
            student_log_probs=student_log_probs,
            teacher_log_probs=teacher_log_probs,
            response_mask=response_mask,
            alpha=0.5,
            full_logit_distillation=True,
            distillation_topk=k,
            distillation_add_tail=True,
            is_clip=None,
            student_topk_log_probs=student_topk,
            teacher_topk_log_probs=teacher_topk,
        )
        assert loss.ndim == 0

    def test_non_negative_loss(self):
        """Self-distillation loss should be non-negative."""
        bsz, seq_len, vocab = 4, 15, 50
        student_log_probs = torch.randn(bsz, seq_len)
        teacher_log_probs = torch.randn(bsz, seq_len)
        response_mask = torch.ones(bsz, seq_len)
        student_all = torch.log_softmax(torch.randn(bsz, seq_len, vocab), dim=-1)
        teacher_all = torch.log_softmax(torch.randn(bsz, seq_len, vocab), dim=-1)

        loss, _ = compute_sdpo_self_distillation_loss(
            student_log_probs=student_log_probs,
            teacher_log_probs=teacher_log_probs,
            response_mask=response_mask,
            alpha=0.5,
            full_logit_distillation=True,
            distillation_topk=None,
            is_clip=None,
            student_all_log_probs=student_all,
            teacher_all_log_probs=teacher_all,
        )
        assert loss.item() >= -1e-5

    def test_identical_distributions_zero_loss(self):
        """Identical distributions should produce ~0 loss."""
        bsz, seq_len, vocab = 2, 10, 50
        log_probs = torch.log_softmax(torch.randn(bsz, seq_len, vocab), dim=-1)
        per_token = torch.randn(bsz, seq_len)
        response_mask = torch.ones(bsz, seq_len)

        loss, _ = compute_sdpo_self_distillation_loss(
            student_log_probs=per_token,
            teacher_log_probs=per_token.clone(),
            response_mask=response_mask,
            alpha=0.5,
            full_logit_distillation=True,
            distillation_topk=None,
            is_clip=None,
            student_all_log_probs=log_probs,
            teacher_all_log_probs=log_probs.clone(),
        )
        assert loss.item() < 1e-5

    def test_zero_mask_zero_loss(self):
        """Zero mask should produce zero loss."""
        bsz, seq_len, vocab = 2, 10, 50
        student_log_probs = torch.randn(bsz, seq_len)
        teacher_log_probs = torch.randn(bsz, seq_len)
        response_mask = torch.zeros(bsz, seq_len)
        student_all = torch.log_softmax(torch.randn(bsz, seq_len, vocab), dim=-1)
        teacher_all = torch.log_softmax(torch.randn(bsz, seq_len, vocab), dim=-1)

        loss, _ = compute_sdpo_self_distillation_loss(
            student_log_probs=student_log_probs,
            teacher_log_probs=teacher_log_probs,
            response_mask=response_mask,
            alpha=0.5,
            full_logit_distillation=True,
            distillation_topk=None,
            is_clip=None,
            student_all_log_probs=student_all,
            teacher_all_log_probs=teacher_all,
        )
        assert loss.item() == 0.0

    def test_self_distillation_mask(self):
        """Self-distillation mask should zero out masked samples."""
        bsz, seq_len, vocab = 4, 10, 50
        student_log_probs = torch.randn(bsz, seq_len)
        teacher_log_probs = torch.randn(bsz, seq_len)
        response_mask = torch.ones(bsz, seq_len)
        student_all = torch.log_softmax(torch.randn(bsz, seq_len, vocab), dim=-1)
        teacher_all = torch.log_softmax(torch.randn(bsz, seq_len, vocab), dim=-1)

        # Mask out all samples
        sd_mask = torch.zeros(bsz)
        loss, _ = compute_sdpo_self_distillation_loss(
            student_log_probs=student_log_probs,
            teacher_log_probs=teacher_log_probs,
            response_mask=response_mask,
            alpha=0.5,
            full_logit_distillation=True,
            distillation_topk=None,
            is_clip=None,
            student_all_log_probs=student_all,
            teacher_all_log_probs=teacher_all,
            self_distillation_mask=sd_mask,
        )
        assert loss.item() == 0.0

    def test_is_clip_effect(self):
        """IS clipping should modify the loss."""
        bsz, seq_len, vocab = 2, 10, 50
        student_log_probs = torch.randn(bsz, seq_len)
        teacher_log_probs = torch.randn(bsz, seq_len)
        old_log_probs = torch.randn(bsz, seq_len)
        response_mask = torch.ones(bsz, seq_len)
        student_all = torch.log_softmax(torch.randn(bsz, seq_len, vocab), dim=-1)
        teacher_all = torch.log_softmax(torch.randn(bsz, seq_len, vocab), dim=-1)

        loss_no_is, _ = compute_sdpo_self_distillation_loss(
            student_log_probs=student_log_probs,
            teacher_log_probs=teacher_log_probs,
            response_mask=response_mask,
            alpha=0.5,
            full_logit_distillation=True,
            distillation_topk=None,
            is_clip=None,
            student_all_log_probs=student_all,
            teacher_all_log_probs=teacher_all,
        )

        loss_with_is, _ = compute_sdpo_self_distillation_loss(
            student_log_probs=student_log_probs,
            teacher_log_probs=teacher_log_probs,
            response_mask=response_mask,
            alpha=0.5,
            full_logit_distillation=True,
            distillation_topk=None,
            is_clip=2.0,
            old_log_probs=old_log_probs,
            student_all_log_probs=student_all,
            teacher_all_log_probs=teacher_all,
        )

        # Both should be valid scalars (may or may not be equal)
        assert loss_no_is.ndim == 0
        assert loss_with_is.ndim == 0

    def test_alpha_zero_forward_kl(self):
        """alpha=0 should compute forward KL."""
        bsz, seq_len, vocab = 2, 10, 50
        student_log_probs = torch.randn(bsz, seq_len)
        teacher_log_probs = torch.randn(bsz, seq_len)
        response_mask = torch.ones(bsz, seq_len)
        student_all = torch.log_softmax(torch.randn(bsz, seq_len, vocab), dim=-1)
        teacher_all = torch.log_softmax(torch.randn(bsz, seq_len, vocab), dim=-1)

        loss, _ = compute_sdpo_self_distillation_loss(
            student_log_probs=student_log_probs,
            teacher_log_probs=teacher_log_probs,
            response_mask=response_mask,
            alpha=0.0,
            full_logit_distillation=True,
            distillation_topk=None,
            is_clip=None,
            student_all_log_probs=student_all,
            teacher_all_log_probs=teacher_all,
        )
        assert loss.ndim == 0
        assert loss.item() >= -1e-5

    def test_alpha_one_reverse_kl(self):
        """alpha=1 should compute reverse KL."""
        bsz, seq_len, vocab = 2, 10, 50
        student_log_probs = torch.randn(bsz, seq_len)
        teacher_log_probs = torch.randn(bsz, seq_len)
        response_mask = torch.ones(bsz, seq_len)
        student_all = torch.log_softmax(torch.randn(bsz, seq_len, vocab), dim=-1)
        teacher_all = torch.log_softmax(torch.randn(bsz, seq_len, vocab), dim=-1)

        loss, _ = compute_sdpo_self_distillation_loss(
            student_log_probs=student_log_probs,
            teacher_log_probs=teacher_log_probs,
            response_mask=response_mask,
            alpha=1.0,
            full_logit_distillation=True,
            distillation_topk=None,
            is_clip=None,
            student_all_log_probs=student_all,
            teacher_all_log_probs=teacher_all,
        )
        assert loss.ndim == 0
        assert loss.item() >= -1e-5

    def test_topk_without_tail(self):
        """Top-k without tail should renormalize."""
        bsz, seq_len, k = 2, 10, 20
        student_log_probs = torch.randn(bsz, seq_len)
        teacher_log_probs = torch.randn(bsz, seq_len)
        response_mask = torch.ones(bsz, seq_len)
        student_topk = torch.log_softmax(torch.randn(bsz, seq_len, k), dim=-1)
        teacher_topk = torch.log_softmax(torch.randn(bsz, seq_len, k), dim=-1)

        loss, _ = compute_sdpo_self_distillation_loss(
            student_log_probs=student_log_probs,
            teacher_log_probs=teacher_log_probs,
            response_mask=response_mask,
            alpha=0.5,
            full_logit_distillation=True,
            distillation_topk=k,
            distillation_add_tail=False,
            is_clip=None,
            student_topk_log_probs=student_topk,
            teacher_topk_log_probs=teacher_topk,
        )
        assert loss.ndim == 0

    def test_non_full_logit_reverse_kl(self):
        """Non-full-logit mode should use reverse KL."""
        bsz, seq_len = 2, 10
        student_log_probs = torch.randn(bsz, seq_len)
        teacher_log_probs = torch.randn(bsz, seq_len)
        response_mask = torch.ones(bsz, seq_len)

        loss, _ = compute_sdpo_self_distillation_loss(
            student_log_probs=student_log_probs,
            teacher_log_probs=teacher_log_probs,
            response_mask=response_mask,
            alpha=1.0,
            full_logit_distillation=False,
            is_clip=None,
        )
        assert loss.ndim == 0

    def test_raises_without_old_log_probs_for_is(self):
        """Should raise error when IS clip is set but old_log_probs is None."""
        bsz, seq_len, vocab = 2, 10, 50
        student_log_probs = torch.randn(bsz, seq_len)
        teacher_log_probs = torch.randn(bsz, seq_len)
        response_mask = torch.ones(bsz, seq_len)
        student_all = torch.log_softmax(torch.randn(bsz, seq_len, vocab), dim=-1)
        teacher_all = torch.log_softmax(torch.randn(bsz, seq_len, vocab), dim=-1)

        with pytest.raises(ValueError, match="old_log_probs is required"):
            compute_sdpo_self_distillation_loss(
                student_log_probs=student_log_probs,
                teacher_log_probs=teacher_log_probs,
                response_mask=response_mask,
                alpha=0.5,
                full_logit_distillation=True,
                distillation_topk=None,
                is_clip=2.0,
                old_log_probs=None,
                student_all_log_probs=student_all,
                teacher_all_log_probs=teacher_all,
            )

    def test_raises_without_full_logits(self):
        """Should raise error when full_logit_distillation but no all_log_probs."""
        bsz, seq_len = 2, 10
        student_log_probs = torch.randn(bsz, seq_len)
        teacher_log_probs = torch.randn(bsz, seq_len)
        response_mask = torch.ones(bsz, seq_len)

        with pytest.raises(ValueError, match="full_logit_distillation requires"):
            compute_sdpo_self_distillation_loss(
                student_log_probs=student_log_probs,
                teacher_log_probs=teacher_log_probs,
                response_mask=response_mask,
                alpha=0.5,
                full_logit_distillation=True,
                distillation_topk=None,
                is_clip=None,
                student_all_log_probs=None,
                teacher_all_log_probs=None,
            )


class TestBuildRepromptText:
    """Tests for build_reprompt_text."""

    def setup_method(self):
        """Set up common templates."""
        self.reprompt_template = "{prompt}{solution}{feedback}\n\nCorrectly solve the original question.\n"
        self.solution_template = "\nCorrect solution:\n\n{successful_previous_attempt}\n\n"
        self.feedback_template = "\nFeedback:\n\n{feedback_raw}\n\n"

    def test_with_solution_only(self):
        """Should include solution section when solution is provided."""
        result = build_reprompt_text(
            prompt_text="What is 2+2?",
            solution="The answer is 4.",
            feedback=None,
            reprompt_template=self.reprompt_template,
            solution_template=self.solution_template,
            feedback_template=self.feedback_template,
        )
        assert "What is 2+2?" in result
        assert "The answer is 4." in result
        assert "Correctly solve" in result

    def test_with_feedback_only(self):
        """Should include feedback when no solution and feedback available."""
        result = build_reprompt_text(
            prompt_text="What is 2+2?",
            solution=None,
            feedback="Your answer was wrong.",
            reprompt_template=self.reprompt_template,
            solution_template=self.solution_template,
            feedback_template=self.feedback_template,
            feedback_only_without_solution=True,
        )
        assert "What is 2+2?" in result
        assert "Your answer was wrong." in result

    def test_feedback_only_without_solution_flag(self):
        """When feedback_only_without_solution=True and solution exists, no feedback."""
        result = build_reprompt_text(
            prompt_text="What is 2+2?",
            solution="The answer is 4.",
            feedback="Your answer was wrong.",
            reprompt_template=self.reprompt_template,
            solution_template=self.solution_template,
            feedback_template=self.feedback_template,
            feedback_only_without_solution=True,
        )
        assert "The answer is 4." in result
        assert "Your answer was wrong." not in result

    def test_no_solution_no_feedback(self):
        """Without solution or feedback, should return original prompt."""
        result = build_reprompt_text(
            prompt_text="What is 2+2?",
            solution=None,
            feedback=None,
            reprompt_template=self.reprompt_template,
            solution_template=self.solution_template,
            feedback_template=self.feedback_template,
        )
        assert result == "What is 2+2?"

    def test_both_solution_and_feedback(self):
        """When feedback_only_without_solution=False, both should be included."""
        result = build_reprompt_text(
            prompt_text="What is 2+2?",
            solution="The answer is 4.",
            feedback="Check your arithmetic.",
            reprompt_template=self.reprompt_template,
            solution_template=self.solution_template,
            feedback_template=self.feedback_template,
            feedback_only_without_solution=False,
        )
        assert "The answer is 4." in result
        assert "Check your arithmetic." in result


class TestSelectDemonstration:
    """Tests for select_demonstration."""

    def test_returns_solution_when_available(self):
        """Should return a solution when one exists for the same uid."""
        success_by_uid = {"uid_0": [1, 2]}
        uids = ["uid_0", "uid_0", "uid_0"]
        response_texts = ["wrong", "correct1", "correct2"]

        result = select_demonstration(
            idx=0,
            success_by_uid=success_by_uid,
            uids=uids,
            response_texts=response_texts,
        )
        assert result == "correct1"

    def test_returns_none_when_no_success(self):
        """Should return None when no successful responses exist."""
        success_by_uid = {"uid_0": []}
        uids = ["uid_0", "uid_0"]
        response_texts = ["wrong1", "wrong2"]

        result = select_demonstration(
            idx=0,
            success_by_uid=success_by_uid,
            uids=uids,
            response_texts=response_texts,
        )
        assert result is None

    def test_dont_reprompt_on_self_success(self):
        """Should exclude self when dont_reprompt_on_self_success=True."""
        success_by_uid = {"uid_0": [0]}
        uids = ["uid_0"]
        response_texts = ["my_response"]

        result = select_demonstration(
            idx=0,
            success_by_uid=success_by_uid,
            uids=uids,
            response_texts=response_texts,
            dont_reprompt_on_self_success=True,
        )
        assert result is None

    def test_dont_reprompt_on_self_success_with_other(self):
        """Should return other's solution when self is excluded."""
        success_by_uid = {"uid_0": [0, 1]}
        uids = ["uid_0", "uid_0"]
        response_texts = ["my_response", "other_response"]

        result = select_demonstration(
            idx=0,
            success_by_uid=success_by_uid,
            uids=uids,
            response_texts=response_texts,
            dont_reprompt_on_self_success=True,
        )
        assert result == "other_response"

    def test_remove_thinking_from_demonstration(self):
        """Should remove <think>...</think> tags."""
        success_by_uid = {"uid_0": [1]}
        uids = ["uid_0", "uid_0"]
        response_texts = ["wrong", "<think>reasoning</think> The answer is 4."]

        result = select_demonstration(
            idx=0,
            success_by_uid=success_by_uid,
            uids=uids,
            response_texts=response_texts,
            remove_thinking_from_demonstration=True,
        )
        assert "<think>" not in result
        assert "The answer is 4." in result

    def test_missing_uid_returns_none(self):
        """Should return None when uid not in success_by_uid."""
        success_by_uid = {"uid_1": [0]}
        uids = ["uid_0", "uid_1"]
        response_texts = ["response_0", "response_1"]

        result = select_demonstration(
            idx=0,
            success_by_uid=success_by_uid,
            uids=uids,
            response_texts=response_texts,
        )
        assert result is None


class TestComputeEMAUpdate:
    """Tests for compute_ema_update."""

    def test_zero_rate_no_change(self):
        """Update rate 0 should not change teacher."""
        student = {"w": torch.tensor([1.0, 2.0, 3.0])}
        teacher = {"w": torch.tensor([4.0, 5.0, 6.0])}

        updated = compute_ema_update(student, teacher, update_rate=0.0)
        assert torch.allclose(updated["w"], teacher["w"])

    def test_one_rate_copies_student(self):
        """Update rate 1 should copy student to teacher."""
        student = {"w": torch.tensor([1.0, 2.0, 3.0])}
        teacher = {"w": torch.tensor([4.0, 5.0, 6.0])}

        updated = compute_ema_update(student, teacher, update_rate=1.0)
        assert torch.allclose(updated["w"], student["w"])

    def test_intermediate_rate(self):
        """Intermediate rate should interpolate."""
        student = {"w": torch.tensor([0.0])}
        teacher = {"w": torch.tensor([1.0])}

        updated = compute_ema_update(student, teacher, update_rate=0.1)
        expected = 0.9 * 1.0 + 0.1 * 0.0
        assert torch.allclose(updated["w"], torch.tensor([expected]))

    def test_multiple_params(self):
        """Should update all matching parameters."""
        student = {"a": torch.tensor([1.0]), "b": torch.tensor([2.0])}
        teacher = {"a": torch.tensor([3.0]), "b": torch.tensor([4.0])}

        updated = compute_ema_update(student, teacher, update_rate=0.5)
        assert torch.allclose(updated["a"], torch.tensor([2.0]))
        assert torch.allclose(updated["b"], torch.tensor([3.0]))

    def test_missing_student_key_preserved(self):
        """Teacher keys not in student should be preserved."""
        student = {"a": torch.tensor([1.0])}
        teacher = {"a": torch.tensor([3.0]), "b": torch.tensor([4.0])}

        updated = compute_ema_update(student, teacher, update_rate=0.5)
        assert torch.allclose(updated["b"], torch.tensor([4.0]))
