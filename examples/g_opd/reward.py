"""
G-OPD Reward Function for Math Reasoning Tasks.

This reward function evaluates mathematical reasoning by checking if the
model's answer matches the ground truth using math-verify.
"""

import re
from typing import Optional


def extract_answer(text: str) -> Optional[str]:
    """Extract the final answer from model output.

    Supports common formats:
    - \\boxed{answer}
    - The answer is: answer
    - Final answer: answer
    """
    # Try to extract from \boxed{}
    boxed_pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(boxed_pattern, text)
    if matches:
        return matches[-1].strip()

    # Try "The answer is" pattern
    answer_pattern = r'[Tt]he\s+(?:final\s+)?answer\s+is[:\s]+(.+?)(?:\.|$)'
    match = re.search(answer_pattern, text)
    if match:
        return match.group(1).strip()

    return None


def compute_score(solution_str: str, ground_truth: str) -> float:
    """Compute reward score by comparing model answer with ground truth.

    Args:
        solution_str: The model's generated solution text.
        ground_truth: The ground truth answer string.

    Returns:
        1.0 if the answer matches, 0.0 otherwise.
    """
    try:
        from math_verify import verify, parse
    except ImportError:
        # Fallback to simple string matching if math-verify not available
        predicted = extract_answer(solution_str)
        if predicted is None:
            return 0.0
        return 1.0 if predicted.strip() == ground_truth.strip() else 0.0

    # Use math-verify for robust answer checking
    predicted = extract_answer(solution_str)
    if predicted is None:
        return 0.0

    try:
        parsed_pred = parse(predicted)
        parsed_gt = parse(ground_truth)
        if verify(parsed_pred, parsed_gt):
            return 1.0
    except Exception:
        pass

    # Fallback: direct string comparison
    if predicted.strip() == ground_truth.strip():
        return 1.0

    return 0.0
