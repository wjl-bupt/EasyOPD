"""Tests for the GAD batch data contract."""

import pytest
import torch


def _make_batch_dict(with_teacher: bool = True, mismatch_batch_dim: bool = False):
    bsz = 4
    out = {
        "input_ids": torch.zeros(bsz, 8, dtype=torch.long),
        "attention_mask": torch.ones(bsz, 8, dtype=torch.long),
        "position_ids": torch.zeros(bsz, 8, dtype=torch.long),
        "responses": torch.zeros(bsz, 4, dtype=torch.long),
    }
    if with_teacher:
        tbsz = bsz + (1 if mismatch_batch_dim else 0)
        out["teacher_input_ids"] = torch.zeros(tbsz, 9, dtype=torch.long)
        out["teacher_attention_mask"] = torch.ones(tbsz, 9, dtype=torch.long)
        out["teacher_position_ids"] = torch.zeros(tbsz, 9, dtype=torch.long)
        out["teacher_response"] = torch.zeros(tbsz, 6, dtype=torch.long)
    return out


def test_keys_constant_matches_spec():
    from easyopd.methods.gad.data_contract import GAD_BATCH_KEYS

    assert GAD_BATCH_KEYS == (
        "teacher_input_ids",
        "teacher_attention_mask",
        "teacher_position_ids",
        "teacher_response",
    )


def test_validate_accepts_well_formed_batch():
    from easyopd.methods.gad.data_contract import validate_gad_batch

    validate_gad_batch(_make_batch_dict(with_teacher=True))


def test_validate_rejects_missing_keys():
    from easyopd.methods.gad.data_contract import (
        GADBatchContractError,
        validate_gad_batch,
    )

    bad = _make_batch_dict(with_teacher=True)
    del bad["teacher_input_ids"]
    del bad["teacher_position_ids"]
    with pytest.raises(GADBatchContractError) as ei:
        validate_gad_batch(bad)
    msg = str(ei.value)
    assert "teacher_input_ids" in msg
    assert "teacher_position_ids" in msg
    assert "docs/algo/gad.md" in msg


def test_validate_rejects_mismatched_batch_dim():
    from easyopd.methods.gad.data_contract import (
        GADBatchContractError,
        validate_gad_batch,
    )

    with pytest.raises(GADBatchContractError) as ei:
        validate_gad_batch(_make_batch_dict(with_teacher=True, mismatch_batch_dim=True))
    assert "batch dim" in str(ei.value).lower()
