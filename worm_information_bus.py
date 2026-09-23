from __future__ import annotations

from dataclasses import dataclass, asdict
from hashlib import sha256
import math
import random
from typing import Iterable, List, Sequence

import numpy as np


@dataclass(frozen=True)
class Message:
    worm_id: str
    generation: int
    specialty: str
    payload: tuple[float, ...]
    confidence: float
    novelty: float
    criticality: float
    bits: int

    @property
    def fingerprint(self) -> str:
        blob = repr(asdict(self)).encode("utf-8")
        return sha256(blob).hexdigest()[:16]


@dataclass
class WorkspaceDecision:
    accepted: List[Message]
    rejected: List[Message]
    used_bits: int
    capacity_bits: int
    equivocation: float


class ShannonWorkspace:
    """Small shared channel inspired by Shannon's noisy-channel framing.

    Messages compete for a fixed bit budget. Selection favors information gain
    per bit, novelty, confidence, and criticality. A redundancy bonus is only
    granted to independent agreement on high-criticality content.
    """

    def __init__(self, capacity_bits: int = 256, noise: float = 0.08, seed: int = 42) -> None:
        self.capacity_bits = int(capacity_bits)
        self.noise = float(np.clip(noise, 0.0, 1.0))
        self.rng = random.Random(seed)

    def _quality(self, m: Message, peer_support: int) -> float:
        base = (0.55 * m.novelty + 0.35 * m.confidence + 0.10 * m.criticality)
        redundancy_bonus = 0.0
        if peer_support >= 2 and m.criticality >= 0.65:
            redundancy_bonus = 0.12 * min(peer_support / 3.0, 1.0)
        noise_penalty = self.noise * 0.12
        return max(0.0, base + redundancy_bonus - noise_penalty)

    def route(self, messages: Sequence[Message]) -> WorkspaceDecision:
        # Approximate independent support by rounded payload signatures.
        buckets: dict[tuple[int, ...], int] = {}
        for m in messages:
            sig = tuple(int(round(x * 4.0)) for x in m.payload)
            buckets[sig] = buckets.get(sig, 0) + 1

        ranked: list[tuple[float, Message]] = []
        for m in messages:
            sig = tuple(int(round(x * 4.0)) for x in m.payload)
            support = buckets[sig]
            utility = self._quality(m, support) / max(m.bits, 1)
            ranked.append((utility, m))

        ranked.sort(key=lambda x: (-x[0], x[1].worm_id))
        accepted: list[Message] = []
        rejected: list[Message] = []
        used = 0
        for _, m in ranked:
            if used + m.bits <= self.capacity_bits:
                accepted.append(m)
                used += m.bits
            else:
                rejected.append(m)

        # Equivocation: fraction of message utility that never reached workspace.
        total_u = sum((0.55 * m.novelty + 0.35 * m.confidence + 0.10 * m.criticality) for m in messages)
        accepted_u = sum((0.55 * m.novelty + 0.35 * m.confidence + 0.10 * m.criticality) for m in accepted)
        equivocation = 0.0 if total_u <= 1e-12 else float(max(0.0, 1.0 - accepted_u / total_u))

        return WorkspaceDecision(accepted, rejected, used, self.capacity_bits, equivocation)


def mutual_information_discrete(x: Iterable[int], y: Iterable[int]) -> float:
    """Empirical mutual information in bits for discrete labels."""
    x = list(x)
    y = list(y)
    if len(x) != len(y) or not x:
        raise ValueError("x and y must have the same non-zero length")
    n = len(x)
    px: dict[int, int] = {}
    py: dict[int, int] = {}
    pxy: dict[tuple[int, int], int] = {}
    for a, b in zip(x, y):
        px[a] = px.get(a, 0) + 1
        py[b] = py.get(b, 0) + 1
        pxy[(a, b)] = pxy.get((a, b), 0) + 1
    mi = 0.0
    for (a, b), count in pxy.items():
        p_ab = count / n
        p_a = px[a] / n
        p_b = py[b] / n
        mi += p_ab * math.log2(p_ab / (p_a * p_b))
    return float(mi)


def information_efficiency(mi_bits: float, message_bits: int) -> float:
    return float(mi_bits / max(message_bits, 1))
