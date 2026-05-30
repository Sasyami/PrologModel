from __future__ import annotations

from math import comb
from typing import Iterable, Sequence


def estimate_pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    if num_samples < 0:
        raise ValueError("num_samples must be non-negative")
    if num_correct < 0:
        raise ValueError("num_correct must be non-negative")
    if num_correct > num_samples:
        raise ValueError("num_correct cannot exceed num_samples")
    if k <= 0:
        raise ValueError("k must be positive")
    if num_samples == 0 or num_correct == 0:
        return 0.0

    effective_k = min(k, num_samples)
    if num_samples - num_correct < effective_k:
        return 1.0

    return 1.0 - (comb(num_samples - num_correct, effective_k) / comb(num_samples, effective_k))


def estimate_task_pass_at_k(outcomes: Sequence[bool], k: int) -> float:
    num_samples = len(outcomes)
    num_correct = sum(1 for outcome in outcomes if outcome)
    return estimate_pass_at_k(num_samples=num_samples, num_correct=num_correct, k=k)


def estimate_mean_pass_at_k(task_outcomes: Iterable[Sequence[bool]], k: int) -> float:
    task_outcomes = list(task_outcomes)
    if not task_outcomes:
        return 0.0

    total = sum(estimate_task_pass_at_k(outcomes, k=k) for outcomes in task_outcomes)
    return total / len(task_outcomes)
