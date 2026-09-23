from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
import random
from typing import Dict, Iterable


STRATEGY_GENES = (
    "penetration_drive",
    "novelty_drive",
    "evidence_demand",
    "counterfactual_drive",
    "adversarial_pressure",
    "search_breadth",
    "persistence",
    "recovery",
    "teaching",
    "restraint",
    "meta_reasoning",
    "transferability",
)


@dataclass
class StrategyGenome:
    maternal: Dict[str, float]
    paternal: Dict[str, float]
    generation: int
    mutation_load: float = 0.0

    def phenotype(self) -> Dict[str, float]:
        return {
            gene: max(0.0, min(1.0, 0.5 * self.maternal.get(gene, 0.5) + 0.5 * self.paternal.get(gene, 0.5)))
            for gene in STRATEGY_GENES
        }

    def digest(self) -> str:
        payload = {
            "maternal": self.maternal,
            "paternal": self.paternal,
            "generation": self.generation,
            "mutation_load": self.mutation_load,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


@dataclass
class EpigeneticState:
    stress: float = 0.10
    recent_failure: float = 0.0
    recent_success: float = 0.0
    novelty_suppression: float = 0.0
    caution_boost: float = 0.0
    exploration_boost: float = 0.0
    recovery_boost: float = 0.0
    age: int = 0

    def bounded(self) -> "EpigeneticState":
        for k, v in asdict(self).items():
            if k == "age":
                continue
            setattr(self, k, max(0.0, min(1.0, float(v))))
        self.age = max(0, int(self.age))
        return self


@dataclass(frozen=True)
class StrategyPolicy:
    max_depth_factor: float
    novelty_pressure: float
    evidence_threshold: float
    counterfactual_weight: float
    hostility: float
    breadth: float
    persistence: float
    recovery: float
    teaching: float
    restraint: float
    meta_reasoning: float
    transferability: float


class StrategyEngine:
    """Evolutionary strategy layer. Strategy is separate from the base cognitive genome."""

    def __init__(self, seed: int = 42, mutation_rate: float = 0.08, mutation_sigma: float = 0.07) -> None:
        self.rng = random.Random(seed ^ 0x51A7)
        self.mutation_rate = float(mutation_rate)
        self.mutation_sigma = float(mutation_sigma)

    def founder(self, generation: int, bias: float = 0.0) -> StrategyGenome:
        base = {}
        for gene in STRATEGY_GENES:
            v = self.rng.uniform(0.30, 0.72)
            if gene in {"penetration_drive", "adversarial_pressure", "meta_reasoning"}:
                v += 0.08 + bias
            if gene in {"restraint", "recovery"}:
                v += 0.04
            base[gene] = max(0.0, min(1.0, v))
        other = {g: max(0.0, min(1.0, base[g] + self.rng.uniform(-0.06, 0.06))) for g in STRATEGY_GENES}
        return StrategyGenome(base, other, generation)

    def recombine(self, mother: StrategyGenome, father: StrategyGenome, generation: int) -> StrategyGenome:
        maternal = {}
        paternal = {}
        load = 0.0
        for gene in STRATEGY_GENES:
            mv = mother.maternal.get(gene, mother.paternal.get(gene, 0.5)) if self.rng.random() < 0.5 else mother.paternal.get(gene, 0.5)
            fv = father.paternal.get(gene, father.maternal.get(gene, 0.5)) if self.rng.random() < 0.5 else father.maternal.get(gene, 0.5)
            if self.rng.random() < self.mutation_rate:
                delta = self.rng.gauss(0.0, self.mutation_sigma)
                mv = max(0.0, min(1.0, mv + delta)); load += abs(delta)
            if self.rng.random() < self.mutation_rate:
                delta = self.rng.gauss(0.0, self.mutation_sigma)
                fv = max(0.0, min(1.0, fv + delta)); load += abs(delta)
            maternal[gene] = mv
            paternal[gene] = fv
        return StrategyGenome(maternal, paternal, generation, min(1.0, load / max(1, 2 * len(STRATEGY_GENES))))

    def policy(self, genome: StrategyGenome, epi: EpigeneticState) -> StrategyPolicy:
        p = genome.phenotype(); epi.bounded()
        hostility = max(0.0, min(1.0, 0.70 * p["adversarial_pressure"] + 0.20 * p["meta_reasoning"] + 0.10 * epi.stress))
        novelty = max(0.0, min(1.0, p["novelty_drive"] + 0.35 * epi.exploration_boost - 0.25 * epi.novelty_suppression))
        evidence = max(0.20, min(0.80, 0.58 * p["evidence_demand"] + 0.18 * epi.caution_boost + 0.12 * epi.recent_failure))
        return StrategyPolicy(
            max_depth_factor=max(0.45, min(1.0, 0.55 + 0.35 * p["penetration_drive"] + 0.10 * p["persistence"])),
            novelty_pressure=novelty,
            evidence_threshold=evidence,
            counterfactual_weight=max(0.10, min(0.80, 0.35 + 0.45 * p["counterfactual_drive"])),
            hostility=hostility,
            breadth=max(0.10, min(1.0, p["search_breadth"] + 0.20 * p["curiosity"] if "curiosity" in p else p["search_breadth"])),
            persistence=max(0.10, min(1.0, p["persistence"] + 0.20 * epi.recovery_boost)),
            recovery=max(0.10, min(1.0, p["recovery"] + 0.25 * epi.recovery_boost)),
            teaching=p["teaching"],
            restraint=max(0.10, min(1.0, p["restraint"] + 0.20 * epi.caution_boost)),
            meta_reasoning=p["meta_reasoning"],
            transferability=p["transferability"],
        )

    def adapt(self, epi: EpigeneticState, *, success: float, failure: float, novelty_gap: float, stress: float) -> EpigeneticState:
        epi.recent_success = 0.82 * epi.recent_success + 0.18 * max(0.0, min(1.0, success))
        epi.recent_failure = 0.78 * epi.recent_failure + 0.22 * max(0.0, min(1.0, failure))
        epi.stress = 0.86 * epi.stress + 0.14 * max(0.0, min(1.0, stress))
        epi.exploration_boost = max(0.0, min(1.0, 0.80 * epi.exploration_boost + 0.20 * max(0.0, novelty_gap)))
        epi.novelty_suppression = max(0.0, min(1.0, 0.90 * epi.novelty_suppression + 0.10 * max(0.0, failure - success)))
        epi.caution_boost = max(0.0, min(1.0, 0.85 * epi.caution_boost + 0.15 * epi.recent_failure))
        epi.recovery_boost = max(0.0, min(1.0, 0.82 * epi.recovery_boost + 0.18 * (epi.recent_failure * 0.7 + epi.recent_success * 0.3)))
        epi.age += 1
        return epi.bounded()


def strategy_fingerprint(genome: StrategyGenome, epi: EpigeneticState) -> str:
    payload = {"g": genome.digest(), "e": asdict(epi)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
