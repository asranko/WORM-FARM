from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence
import numpy as np


@dataclass(frozen=True)
class BehaviorDiff:
    feature: str
    delta: float
    abs_delta: float


def diff_behavior_vectors(
    before: Mapping[str, Sequence[float]],
    after: Mapping[str, Sequence[float]],
    top_k: int = 10,
) -> list[BehaviorDiff]:
    rows: list[BehaviorDiff] = []
    for key in sorted(set(before) & set(after)):
        a = np.asarray(before[key], dtype=float)
        b = np.asarray(after[key], dtype=float)
        if a.shape != b.shape:
            continue
        delta = float(np.mean(b) - np.mean(a))
        rows.append(BehaviorDiff(key, delta, abs(delta)))
    rows.sort(key=lambda r: (-r.abs_delta, r.feature))
    return rows[:top_k]
