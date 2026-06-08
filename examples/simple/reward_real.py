"""Real reward function for EasyOPD training validation.

Provides non-zero rewards based on response quality metrics:
- Length reward: longer responses get higher reward (up to a cap)
- Format reward: responses with mathematical expressions get bonus
"""

from __future__ import annotations

import re
from typing import Any


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> float:
    """Compute a simple quality-based reward for training validation.
    
    This reward function provides non-zero gradients for training:
    - Base reward from response length (normalized)
    - Bonus for mathematical content (equations, numbers)
    - Penalty for very short or empty responses
    """
    if not solution_str or len(solution_str.strip()) == 0:
        return -1.0
    
    # Length-based reward (normalized to [0, 1])
    length = len(solution_str)
    length_reward = min(length / 500.0, 1.0)  # Cap at 500 chars
    
    # Math content bonus
    math_patterns = [
        r'\d+',           # Numbers
        r'[+\-*/=]',     # Math operators
        r'\\boxed',      # LaTeX boxed answers
        r'answer is',    # Answer phrases
        r'therefore',    # Conclusion words
    ]
    math_bonus = 0.0
    for pattern in math_patterns:
        if re.search(pattern, solution_str):
            math_bonus += 0.1
    math_bonus = min(math_bonus, 0.5)
    
    # Combine rewards
    reward = length_reward * 0.5 + math_bonus
    
    # Clip to [-1, 1]
    return max(-1.0, min(1.0, reward))
