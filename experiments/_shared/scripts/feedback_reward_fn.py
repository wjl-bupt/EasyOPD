"""SDPO-faithful reward function (lasgroup/SDPO ``feedback`` reward package).

This is the reward used by ``04_self_opd`` (SDPO and its off-policy GRPO
baseline) so the training reward is byte-faithful to lasgroup/SDPO. It dispatches
to the ported ``feedback`` reward package (``verl/utils/reward_score/feedback``,
a 1:1 copy of the reference) for every shared data source — ``sciknoweval``
(MCQ), ``tooluse``, ``math``, ``code``, ``gpqa`` — exactly like the reference
``feedback/__init__.py``. For any other data source it falls back to EasyOPD's
boxed / GSM8K auto-detect scorer.

The reference scorers return a rich dict; we normalize it to a FIXED key set
``{"score", "acc", "pred"}`` and return that dict. verl's reward manager uses
``dict["score"]`` as the token-level reward (so the SDPO success signal is
unchanged), while the ``acc`` / ``score`` / ``pred`` keys reproduce the
reference wandb validation metrics — ``val-core/<src>/acc/mean@N``,
``val-aux/<src>/score/mean@N`` (the metric reported in the SDPO paper), and the
``maj@N`` majority-vote variants. A FIXED key set is required: different
scorers (mcq / math / code) return different extra keys, and inconsistent
per-sample keys would both misalign ``reward_extra_info`` and trip verl's
per-key length assertion in ``_validate`` when a batch mixes data sources.

NOTE: ``feedback/math.py`` uses ``math_verify`` for answer-equivalence checking,
an optional dependency (the reference installs ``math-verify[antlr4_9_3]==0.8.0``).
When it is not installed, the math reward falls back to a strict boxed-answer
match only; install it for byte-faithful math rewards. The MCQ / tooluse / code
/ gpqa scorers do not need it.

This file is intentionally separate from ``reward_fn.py`` so that other
experiments (01_cross_tokenizer_opd, 06_general_opd, ...) which import
``reward_fn.py`` keep their original reward behaviour unchanged.
"""

import os
import re
import sys

# Make THIS repo's `verl` (which ships the ported feedback reward package)
# importable regardless of the caller's CWD. Resolve the repo root relative to
# this file: experiments/_shared/scripts/feedback_reward_fn.py -> root is 3 up.
_EASYOPD_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _EASYOPD_ROOT not in sys.path:
    sys.path.insert(0, _EASYOPD_ROOT)

from verl.utils.reward_score.math import compute_score as math_score
from verl.utils.reward_score.gsm8k import compute_score as gsm8k_score

# Data sources routed through the ported lasgroup/SDPO feedback dispatch
# (mirrors verl/utils/reward_score/feedback/__init__.py exactly).
_FEEDBACK_DATA_SOURCES = {
    "code", "livecodebench", "humanevalplus",
    "math", "math500", "dapo_math", "gsm8k",
    "gpqa", "sciknoweval", "tooluse",
}


def _extract_ground_truth_answer(ground_truth: str) -> tuple:
    """Extract the actual answer from ground_truth which may be a full solution.

    Returns: (answer_str, format_type) where format_type is 'gsm8k' or 'math'
    """
    if not ground_truth:
        return "", "math"

    # Check for GSM8K format: #### <number>
    gsm_match = re.findall(r'####\s*(\-?[0-9.,]+)', ground_truth)
    if gsm_match:
        return gsm_match[-1].replace(',', ''), "gsm8k"

    # Check for MATH format: \boxed{...}
    boxed_match = re.findall(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', ground_truth)
    if boxed_match:
        return boxed_match[-1], "math"

    # If ground_truth is just a short string (likely already the answer)
    if len(ground_truth) < 50 and not '\n' in ground_truth:
        if re.match(r'^-?\d+\.?\d*$', ground_truth.strip()):
            return ground_truth.strip(), "gsm8k"
        return ground_truth.strip(), "math"

    # Fallback: try to find the last number in the text
    numbers = re.findall(r'(\-?[0-9.,]+)', ground_truth)
    if numbers:
        return numbers[-1].replace(',', ''), "gsm8k"

    return ground_truth, "math"


def _extract_xml_answer(text: str) -> str:
    """Extract the answer from the LAST ``<answer>...</answer>`` block."""
    answer = str(text).split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def mcq_score(solution_str, ground_truth) -> float:
    """SciKnowEval MCQ scoring (legacy fallback, == feedback/mcq.py score)."""
    if solution_str is None:
        return 0.0
    multiple_choice_answer = _extract_xml_answer(solution_str)
    return float(multiple_choice_answer == str(ground_truth))


def _legacy_compute_score(data_source, solution_str, ground_truth) -> float:
    """EasyOPD's original boxed / GSM8K auto-detect scorer (fallback path)."""
    if data_source == "sciknoweval":
        return mcq_score(solution_str, ground_truth)
    answer, format_type = _extract_ground_truth_answer(ground_truth)
    if format_type == "gsm8k":
        return gsm8k_score(solution_str, answer, method="flexible")
    return math_score(solution_str, answer)


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Per-sequence reward, faithful to lasgroup/SDPO for known data sources.

    Returns a dict with a FIXED key set ``{"score", "acc", "pred"}``:
      * ``score`` — the reference reward; verl uses it as the token-level reward,
        so the SDPO "successful rollout" check (reward >=
        ``self_distillation.success_reward_threshold``, default 0.5) is unchanged.
      * ``acc`` / ``pred`` — reproduce the reference wandb validation metrics
        (``val-core/<src>/acc/mean@N``, ``val-aux/<src>/score/mean@N``, ``maj@N``).
    The key set is fixed (not the scorer's raw dict) so it stays consistent
    across data sources within a mixed batch.
    """
    if data_source in _FEEDBACK_DATA_SOURCES:
        try:
            from verl.utils.reward_score.feedback import compute_score as feedback_score

            # feedback/code.py and math.py read extra_info["split"]; EasyOPD's
            # prompt parquets don't always set it, so default to "test".
            ei = dict(extra_info) if extra_info else {}
            ei.setdefault("split", "test")

            result = feedback_score(
                data_source=data_source,
                solution_str=solution_str,
                ground_truth=ground_truth,
                extra_info=ei,
            )
            if isinstance(result, dict):
                score = float(result.get("score", 0.0))
                return {
                    "score": score,
                    "acc": float(result.get("acc", score)),
                    "pred": str(result.get("pred", "")),
                }
            val = float(result)
            return {"score": val, "acc": val, "pred": ""}
        except Exception:
            # Never let a single malformed sample crash training; fall back.
            pass

    val = float(_legacy_compute_score(data_source, solution_str, ground_truth))
    return {"score": val, "acc": val, "pred": ""}
