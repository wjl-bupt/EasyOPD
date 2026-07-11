"""Real math reward function for EasyOPD benchmark training.

Uses the verl math reward scorer to provide accurate rewards based on
\boxed{} answer extraction and equivalence checking.
"""

import re
import sys
import os

# Add project root to path
sys.path.insert(0, "/path/to/EasyOPD")

from verl.utils.reward_score.math import compute_score as math_score
from verl.utils.reward_score.gsm8k import compute_score as gsm8k_score


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
        # Try to determine if it's a number (GSM8K) or expression (MATH)
        if re.match(r'^-?\d+\.?\d*$', ground_truth.strip()):
            return ground_truth.strip(), "gsm8k"
        return ground_truth.strip(), "math"
    
    # Fallback: try to find the last number in the text
    numbers = re.findall(r'(\-?[0-9.,]+)', ground_truth)
    if numbers:
        return numbers[-1].replace(',', ''), "gsm8k"
    
    return ground_truth, "math"


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Compute reward score based on data source and ground truth format.

    Auto-detects whether ground_truth is in GSM8K (#### number) or MATH (\boxed{}) format.
    """
    # Extract the actual answer from ground_truth (which may be a full solution)
    answer, format_type = _extract_ground_truth_answer(ground_truth)
    
    if format_type == "gsm8k":
        # For GSM8K: check if model output contains #### <number> matching answer
        return gsm8k_score(solution_str, answer, method="flexible")
    else:
        # For MATH: check if model output contains \boxed{answer}
        return math_score(solution_str, answer)
