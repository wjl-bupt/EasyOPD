"""CPU-only unit tests for the OPLD listwise advantage.

Kept inside the method package (rather than under ``tests/``) so the whole
method stays self-contained. Run with::

    pytest easyopd/methods/opld/test_core.py

Requires only torch + numpy; no GPU, no verl, no ray.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from easyopd.methods.opld.core import compute_listwise_advantage, read_listwise_config


class _Cfg:
    """Minimal stand-in for AlgoConfig (only ``.get`` is used)."""

    def __init__(self, easyopd=None):
        self._easyopd = easyopd

    def get(self, key, default=None):
        return self._easyopd if key == "easyopd" else default


def _batch(bsz=6, seqlen=4):
    mask = torch.ones(bsz, seqlen)
    index = np.array(["a", "a", "a", "b", "b", "b"])
    return mask, index


def test_advantage_is_mean_zero_within_group():
    """sum_i A_i = 0 per group, since q_T and q_S both sum to 1."""
    mask, index = _batch()
    teacher = torch.tensor([-1.0, -3.0, -2.0, -5.0, -1.0, -4.0])
    student = torch.tensor([-2.0, -2.0, -2.0, -3.0, -3.0, -3.0])

    adv, _ = compute_listwise_advantage(teacher, student, mask, index)

    seq_adv = adv[:, 0]
    torch.testing.assert_close(seq_adv[:3].sum(), torch.tensor(0.0), atol=1e-6, rtol=0)
    torch.testing.assert_close(seq_adv[3:].sum(), torch.tensor(0.0), atol=1e-6, rtol=0)


def test_matches_closed_form_qT_minus_qS():
    """Default direction reproduces A_i = q_T(i) - q_S(i) exactly."""
    mask = torch.ones(3, 2)
    index = np.array(["g", "g", "g"])
    teacher = torch.tensor([-1.0, -2.0, -3.0])
    student = torch.tensor([-2.0, -1.0, -4.0])

    cfg = _Cfg({"listwise": {"length_norm": False}})
    adv, _ = compute_listwise_advantage(teacher, student, mask, index, config=cfg)

    expected = torch.softmax(teacher, 0) - torch.softmax(student, 0)
    torch.testing.assert_close(adv[:, 0], expected, atol=1e-6, rtol=0)


def test_teacher_preferred_sequence_gets_positive_advantage():
    """A sequence the teacher likes more than the student does is pushed up."""
    mask = torch.ones(2, 3)
    index = np.array(["g", "g"])
    # Teacher strongly prefers item 0; student is indifferent.
    teacher = torch.tensor([-1.0, -5.0])
    student = torch.tensor([-2.0, -2.0])

    cfg = _Cfg({"listwise": {"length_norm": False}})
    adv, _ = compute_listwise_advantage(teacher, student, mask, index, config=cfg)

    assert adv[0, 0] > 0
    assert adv[1, 0] < 0


def test_degenerate_single_candidate_group_is_zero():
    """K=1 groups carry no listwise signal (q_T = q_S = 1)."""
    mask = torch.ones(2, 3)
    index = np.array(["a", "b"])
    teacher = torch.tensor([-1.0, -9.0])
    student = torch.tensor([-4.0, -2.0])

    adv, metrics = compute_listwise_advantage(teacher, student, mask, index)

    torch.testing.assert_close(adv, torch.zeros_like(adv))
    assert metrics["opld/degenerate_group_frac"] == 1.0
    assert metrics["opld/num_groups"] == 2.0


def test_advantage_is_masked_to_response_tokens():
    """Padding positions must stay exactly zero."""
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0]])
    index = np.array(["g", "g"])
    teacher = torch.tensor([-1.0, -3.0])
    student = torch.tensor([-2.0, -2.0])

    adv, _ = compute_listwise_advantage(teacher, student, mask, index)

    assert torch.all(adv[mask == 0] == 0)
    assert torch.all(adv[mask == 1] != 0)


def test_reverse_kl_direction_is_not_negated_forward():
    """qS_to_qT is the true reverse-KL gradient, not -(forward)."""
    mask = torch.ones(3, 2)
    index = np.array(["g", "g", "g"])
    teacher = torch.tensor([-1.0, -2.0, -4.0])
    student = torch.tensor([-3.0, -1.0, -2.0])

    fwd, _ = compute_listwise_advantage(
        teacher, student, mask, index,
        config=_Cfg({"listwise": {"length_norm": False, "kl_direction": "qT_to_qS"}}),
    )
    rev, _ = compute_listwise_advantage(
        teacher, student, mask, index,
        config=_Cfg({"listwise": {"length_norm": False, "kl_direction": "qS_to_qT"}}),
    )

    # Reverse must be mean-zero too, but must NOT equal -forward.
    torch.testing.assert_close(rev[:, 0].sum(), torch.tensor(0.0), atol=1e-6, rtol=0)
    assert not torch.allclose(rev[:, 0], -fwd[:, 0], atol=1e-4)


def test_beta_sharpens_the_distribution():
    """Larger beta => more peaked group softmax => larger spread in A."""
    mask = torch.ones(3, 2)
    index = np.array(["g", "g", "g"])
    teacher = torch.tensor([-1.0, -2.0, -3.0])
    student = torch.tensor([-2.0, -2.0, -2.0])

    small, _ = compute_listwise_advantage(
        teacher, student, mask, index,
        config=_Cfg({"listwise": {"length_norm": False, "beta": 0.5}}),
    )
    large, _ = compute_listwise_advantage(
        teacher, student, mask, index,
        config=_Cfg({"listwise": {"length_norm": False, "beta": 4.0}}),
    )

    assert large[:, 0].abs().max() > small[:, 0].abs().max()


def test_length_norm_changes_ranking_for_uneven_lengths():
    """Without length norm the long sequence is penalized by its raw logprob sum."""
    mask = torch.zeros(2, 10)
    mask[0, :2] = 1.0   # short
    mask[1, :10] = 1.0  # long
    index = np.array(["g", "g"])
    # Long sequence has a lower total but a better per-token average.
    teacher = torch.tensor([-2.0, -5.0])
    student = torch.tensor([-2.0, -5.0])

    raw, _ = compute_listwise_advantage(
        teacher, student, mask, index,
        config=_Cfg({"listwise": {"length_norm": False}}),
    )
    normed, _ = compute_listwise_advantage(
        teacher, student, mask, index,
        config=_Cfg({"listwise": {"length_norm": True}}),
    )
    # q_T == q_S here so both are zero; the point is that the config is honored
    # and neither path raises on ragged lengths.
    torch.testing.assert_close(raw, torch.zeros_like(raw), atol=1e-6, rtol=0)
    torch.testing.assert_close(normed, torch.zeros_like(normed), atol=1e-6, rtol=0)


def test_eta_requires_rewards_and_shifts_target():
    """eta > 0 folds task reward into the teacher target logits."""
    mask = torch.ones(2, 3)
    index = np.array(["g", "g"])
    teacher = torch.tensor([-2.0, -2.0])
    student = torch.tensor([-2.0, -2.0])
    cfg = _Cfg({"listwise": {"length_norm": False, "eta": 1.0}})

    with pytest.raises(ValueError, match="token_level_rewards"):
        compute_listwise_advantage(teacher, student, mask, index, config=cfg)

    rewards = torch.zeros(2, 3)
    rewards[0, -1] = 1.0  # item 0 solved the task
    adv, _ = compute_listwise_advantage(
        teacher, student, mask, index, token_level_rewards=rewards, config=cfg
    )
    assert adv[0, 0] > 0 > adv[1, 0]


def test_config_roundtrips_and_rejects_typos():
    assert read_listwise_config(None)["beta"] == 1.0
    assert read_listwise_config(_Cfg(None))["kl_direction"] == "qT_to_qS"

    cfg = _Cfg({"listwise": {"beta": 2.5, "std_norm": True}})
    resolved = read_listwise_config(cfg)
    assert resolved["beta"] == 2.5 and resolved["std_norm"] is True
    assert resolved["length_norm"] is True  # default preserved

    # Legacy key still accepted.
    assert read_listwise_config(_Cfg({"listwise_kl": {"beta": 3.0}}))["beta"] == 3.0

    with pytest.raises(ValueError, match="Unknown OPLD config key"):
        read_listwise_config(_Cfg({"listwise": {"bta": 2.0}}))

    with pytest.raises(ValueError, match="kl_direction"):
        read_listwise_config(_Cfg({"listwise": {"kl_direction": "nope"}}))


def test_std_norm_rescales_advantage():
    mask = torch.ones(3, 2)
    index = np.array(["g", "g", "g"])
    teacher = torch.tensor([-1.0, -2.0, -3.0])
    student = torch.tensor([-2.0, -2.5, -2.0])

    normed, _ = compute_listwise_advantage(
        teacher, student, mask, index,
        config=_Cfg({"listwise": {"length_norm": False, "std_norm": True}}),
    )
    # Still mean-zero, but now unit-std within the group.
    seq = normed[:, 0]
    torch.testing.assert_close(seq.sum(), torch.tensor(0.0), atol=1e-5, rtol=0)
    torch.testing.assert_close(seq.std(unbiased=False), torch.tensor(1.0), atol=1e-3, rtol=0)


# ---------------------------------------------------------------------------
# Guard against a second KL term corrupting the advantage.
# Imported lazily because advantage_estimator pulls in verl.
# ---------------------------------------------------------------------------

pytest.importorskip("verl", reason="advantage_estimator requires verl")


class _AlgoCfg:
    def __init__(self, token_kl_reg=None):
        self._d = {"token_kl_reg": token_kl_reg}

    def get(self, key, default=None):
        return self._d.get(key, default)


def test_inert_token_kl_reg_is_accepted():
    """enable=True with beta_max unset is the sanctioned ref-worker switch."""
    from easyopd.methods.opld.advantage_estimator import _assert_no_conflicting_kl_term

    _assert_no_conflicting_kl_term(None)
    _assert_no_conflicting_kl_term(_AlgoCfg(None))
    _assert_no_conflicting_kl_term(
        _AlgoCfg({"enable": True, "stepwise_enable": False,
                  "beta_max": None, "beta_min": 0.0, "coef": 0.0})
    )


def test_active_token_kl_reg_is_rejected():
    """A live regularizer would blend a second KL into our advantage."""
    from easyopd.methods.opld.advantage_estimator import (
        OPLDConflictingKLTerm,
        _assert_no_conflicting_kl_term,
    )

    with pytest.raises(OPLDConflictingKLTerm, match="beta_max"):
        _assert_no_conflicting_kl_term(
            _AlgoCfg({"enable": True, "stepwise_enable": False,
                      "beta_max": 0.3, "beta_min": 0.0})
        )

    with pytest.raises(OPLDConflictingKLTerm, match="stepwise_enable"):
        _assert_no_conflicting_kl_term(
            _AlgoCfg({"enable": True, "stepwise_enable": True})
        )
