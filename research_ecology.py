from __future__ import annotations

from dataclasses import dataclass, asdict
from hashlib import sha256
import json
import math
from typing import Iterable, Mapping, Sequence

from .models import Finding, Worm


@dataclass(frozen=True)
class ResearchMessage:
    worm_id: str
    generation: int
    specialty: str
    finding_id: str
    payload: tuple[float, ...]
    confidence: float
    novelty: float
    information_gain: float
    criticality: float
    bits: int

    @property
    def fingerprint(self) -> str:
        raw = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return sha256(raw.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class WorkspaceDecision:
    accepted: tuple[ResearchMessage, ...]
    rejected: tuple[ResearchMessage, ...]
    used_bits: int
    capacity_bits: int
    equivocation: float
    accepted_utility: float
    rejected_utility: float


@dataclass(frozen=True)
class BehaviorDiff:
    specialty: str
    feature: str
    before: float
    after: float
    delta: float
    abs_delta: float


class InformationLimitedWorkspace:
    """Deterministic shared workspace with a hard information budget.

    Messages compete for a fixed bit budget. Selection is deterministic so that
    a checkpoint/resume produces the same workspace trajectory.
    """

    def __init__(self, capacity_bits: int = 768, noise: float = 0.08, message_base_bits: int = 64) -> None:
        if capacity_bits < 64:
            raise ValueError("capacity_bits must be >= 64")
        self.capacity_bits = int(capacity_bits)
        self.noise = float(max(0.0, min(1.0, noise)))
        self.message_base_bits = int(message_base_bits)
        if self.message_base_bits < 16:
            raise ValueError("message_base_bits must be >= 16")

    @staticmethod
    def _utility(message: ResearchMessage, support: int) -> float:
        base = 0.50 * message.information_gain + 0.25 * message.novelty + 0.15 * message.confidence + 0.10 * message.criticality
        redundancy = 0.10 * min(1.0, support / 3.0) if support >= 2 and message.criticality >= 0.65 else 0.0
        return max(0.0, base + redundancy)

    def route(self, messages: Sequence[ResearchMessage]) -> WorkspaceDecision:
        buckets: dict[tuple[int, ...], int] = {}
        for m in messages:
            sig = tuple(int(round(x * 8.0)) for x in m.payload)
            buckets[sig] = buckets.get(sig, 0) + 1

        scored: list[tuple[float, str, ResearchMessage]] = []
        for m in messages:
            sig = tuple(int(round(x * 8.0)) for x in m.payload)
            support = buckets[sig]
            utility = self._utility(m, support)
            utility *= (1.0 - 0.12 * self.noise)
            density = utility / max(m.bits, 1)
            scored.append((density, m.fingerprint, m))

        scored.sort(key=lambda item: (-item[0], item[1]))
        accepted: list[ResearchMessage] = []
        rejected: list[ResearchMessage] = []
        used = 0
        for _, _, m in scored:
            if used + m.bits <= self.capacity_bits:
                accepted.append(m)
                used += m.bits
            else:
                rejected.append(m)

        def raw_u(m: ResearchMessage) -> float:
            return 0.50 * m.information_gain + 0.25 * m.novelty + 0.15 * m.confidence + 0.10 * m.criticality

        accepted_u = sum(raw_u(m) for m in accepted)
        rejected_u = sum(raw_u(m) for m in rejected)
        total_u = accepted_u + rejected_u
        equivocation = 0.0 if total_u <= 1e-12 else rejected_u / total_u
        return WorkspaceDecision(tuple(accepted), tuple(rejected), used, self.capacity_bits, float(equivocation), float(accepted_u), float(rejected_u))


def mutual_information_labels(x: Iterable[str], y: Iterable[str]) -> float:
    xs = list(x)
    ys = list(y)
    if len(xs) != len(ys) or not xs:
        raise ValueError("x and y must have the same non-zero length")
    n = len(xs)
    px: dict[str, int] = {}
    py: dict[str, int] = {}
    pxy: dict[tuple[str, str], int] = {}
    for a, b in zip(xs, ys):
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


def behavior_diff(
    before: Mapping[str, Mapping[str, float]],
    after: Mapping[str, Mapping[str, float]],
    top_k: int = 20,
) -> list[BehaviorDiff]:
    rows: list[BehaviorDiff] = []
    for specialty in sorted(set(before) | set(after)):
        b = before.get(specialty, {})
        a = after.get(specialty, {})
        for feature in sorted(set(b) | set(a)):
            bv = float(b.get(feature, 0.0))
            av = float(a.get(feature, 0.0))
            delta = av - bv
            rows.append(BehaviorDiff(specialty, feature, bv, av, delta, abs(delta)))
    rows.sort(key=lambda r: (-r.abs_delta, r.specialty, r.feature))
    return rows[:top_k]


class ResearchEcology:
    """Research layer: information bottleneck + behavioral diff per generation."""

    FEATURES = (
        "novelty", "information_gain", "validity", "falsifiability",
        "independence", "counterfactual_score", "robustness", "score",
    )

    def __init__(self, capacity_bits: int = 768, noise: float = 0.08, message_base_bits: int = 64) -> None:
        self.workspace = InformationLimitedWorkspace(capacity_bits, noise)
        self.previous_snapshot: dict[str, dict[str, float]] = {}
        self.total_messages = 0
        self.total_accepted = 0
        self.total_rejected = 0
        self.total_bits = 0

    def _snapshot(self, worms: Sequence[Worm], findings: Sequence[Finding]) -> dict[str, dict[str, float]]:
        by_worm: dict[str, list[Finding]] = {}
        for finding in findings:
            by_worm.setdefault(finding.worm_id, []).append(finding)
        grouped: dict[str, list[Finding]] = {}
        for worm in worms:
            grouped.setdefault(worm.specialty, []).extend(by_worm.get(worm.id, []))
        out: dict[str, dict[str, float]] = {}
        for specialty, rows in sorted(grouped.items()):
            out[specialty] = {
                feature: sum(getattr(f, feature) for f in rows) / max(1, len(rows))
                for feature in self.FEATURES
            }
        return out

    def process_generation(self, generation: int, worms: Sequence[Worm], findings: Sequence[Finding]) -> dict:
        messages: list[ResearchMessage] = []
        worm_by_id = {w.id: w for w in worms}
        for finding in sorted(findings, key=lambda f: f.id):
            worm = worm_by_id.get(finding.worm_id)
            if worm is None:
                continue
            payload = (
                float(finding.novelty), float(finding.information_gain), float(finding.validity),
                float(finding.falsifiability), float(finding.independence), float(finding.counterfactual_score),
                float(finding.robustness), float(finding.score),
            )
            bits = self.workspace.message_base_bits + 8 * min(8, len(finding.assumptions_targeted) + len(finding.evidence_refs))
            messages.append(ResearchMessage(
                worm_id=worm.id,
                generation=generation,
                specialty=worm.specialty,
                finding_id=finding.id,
                payload=payload,
                confidence=finding.validity,
                novelty=finding.novelty,
                information_gain=finding.information_gain,
                criticality=max(finding.information_gain, finding.anomaly),
                bits=bits,
            ))

        decision = self.workspace.route(messages)
        self.total_messages += len(messages)
        self.total_accepted += len(decision.accepted)
        self.total_rejected += len(decision.rejected)
        self.total_bits += decision.used_bits

        current_snapshot = self._snapshot(worms, findings)
        diffs = behavior_diff(self.previous_snapshot, current_snapshot) if self.previous_snapshot else []
        self.previous_snapshot = current_snapshot

        specialties = [m.specialty for m in decision.accepted]
        acceptance = ["accepted"] * len(decision.accepted) + ["rejected"] * len(decision.rejected)
        mixed = list(decision.accepted) + list(decision.rejected)
        mi = mutual_information_labels(
            [m.specialty for m in mixed],
            ["accepted" if m in decision.accepted else "rejected" for m in mixed],
        ) if mixed else 0.0

        return {
            "generation": int(generation),
            "submitted": len(messages),
            "accepted": len(decision.accepted),
            "rejected": len(decision.rejected),
            "used_bits": decision.used_bits,
            "capacity_bits": decision.capacity_bits,
            "equivocation": decision.equivocation,
            "accepted_utility": decision.accepted_utility,
            "rejected_utility": decision.rejected_utility,
            "mutual_information_specialty_to_routing_bits": mi,
            "information_efficiency": mi / max(decision.used_bits, 1),
            "accepted_fingerprints": [m.fingerprint for m in decision.accepted],
            "rejected_fingerprints": [m.fingerprint for m in decision.rejected],
            "behavior_diffs": [asdict(d) for d in diffs],
        }

    def runtime_state(self) -> dict:
        return {
            "previous_snapshot": self.previous_snapshot,
            "total_messages": self.total_messages,
            "total_accepted": self.total_accepted,
            "total_rejected": self.total_rejected,
            "total_bits": self.total_bits,
        }

    def load_runtime_state(self, state: Mapping) -> None:
        self.previous_snapshot = {k: {fk: float(v) for fk, v in vals.items()} for k, vals in state.get("previous_snapshot", {}).items()}
        self.total_messages = int(state.get("total_messages", 0))
        self.total_accepted = int(state.get("total_accepted", 0))
        self.total_rejected = int(state.get("total_rejected", 0))
        self.total_bits = int(state.get("total_bits", 0))
