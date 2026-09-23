from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import random
import math
from typing import Dict, List, Optional, Sequence, Tuple

from .models import Worm, WormStatus


class Sex(str, Enum):
    MALE = "MALE"
    FEMALE = "FEMALE"


TRAITS = (
    "penetration", "skepticism", "novelty", "falsifiability", "independence",
    "persistence", "adversariality", "restraint", "pattern_sensitivity", "recovery",
    "counterfactual", "meta_cognition", "war_resistance", "memory", "curiosity", "routing_bias",
)


@dataclass
class CognitiveGenome:
    maternal: Dict[str, float] = field(default_factory=dict)
    paternal: Dict[str, float] = field(default_factory=dict)
    generation: int = 0
    mutation_load: float = 0.0

    def routing_bias(self) -> float:
        """Map the inherited routing_bias allele from [0,1] to [-1,+1]."""
        value = self.phenotype().get("routing_bias", 0.5)
        return max(-1.0, min(1.0, 2.0 * float(value) - 1.0))

    def phenotype(self) -> Dict[str, float]:
        keys = [k for k in TRAITS if k in self.maternal or k in self.paternal]
        return {
            k: max(0.0, min(1.0, 0.5 * self.maternal.get(k, 0.5) + 0.5 * self.paternal.get(k, 0.5)))
            for k in keys
        }

    def distance(self, other: "CognitiveGenome") -> float:
        a, b = self.phenotype(), other.phenotype()
        keys = [k for k in TRAITS if k in a or k in b]
        total = sum(abs(a.get(k, 0.5) - b.get(k, 0.5)) for k in keys)
        return total / max(1, len(keys))


@dataclass
class ReproductiveRecord:
    parent_a: str
    parent_b: str
    child_id: str
    generation: int
    success: bool
    reason: str
    compatibility: float
    diversity: float


@dataclass
class GeneticsConfig:
    enabled: bool = True
    mutation_rate: float = 0.055
    mutation_sigma: float = 0.085
    recombination_bias: float = 0.50
    min_reputation: float = 0.38
    min_integrity: float = 0.40
    min_energy: float = 0.28
    min_resilience: float = 0.27
    max_children_per_pair: int = 2
    mate_cooldown: int = 2
    generation_gap_required: int = 1
    minimum_genetic_distance: float = 0.07
    incest_guard: bool = True
    prefer_unrelated: bool = True
    reproduction_cost: float = 0.10
    reproduction_reward: float = 0.09
    max_population_from_genetics: int = 96
    random_seed: int = 42
    selection_mode: str = "tournament"
    tournament_size: int = 4
    tournament_temperature: float = 1.0


class GenomeEngine:
    def __init__(self, config: GeneticsConfig | None = None):
        self.config = config or GeneticsConfig()
        self.rng = random.Random(self.config.random_seed + 991)
        self.records: List[ReproductiveRecord] = []
        self.parent_map: Dict[str, Tuple[str, str]] = {}
        self.mating_generation: Dict[Tuple[str, str], int] = {}
        self.children_by_pair: Dict[Tuple[str, str], int] = {}

    def initialize_founder(self, sex: Sex, generation: int = 0) -> CognitiveGenome:
        base = {t: self.rng.uniform(0.30, 0.75) for t in TRAITS}
        if sex is Sex.MALE:
            base["adversariality"] = min(1.0, base["adversariality"] + 0.10)
            base["penetration"] = min(1.0, base["penetration"] + 0.08)
            base["war_resistance"] = min(1.0, base["war_resistance"] + 0.05)
        else:
            base["restraint"] = min(1.0, base["restraint"] + 0.10)
            base["recovery"] = min(1.0, base["recovery"] + 0.08)
            base["meta_cognition"] = min(1.0, base["meta_cognition"] + 0.05)
        other = {k: max(0.0, min(1.0, v + self.rng.uniform(-0.05, 0.05))) for k, v in base.items()}
        return CognitiveGenome(maternal=base, paternal=other, generation=generation)

    @staticmethod
    def _ancestors(node_id: str, parent_map: Dict[str, Tuple[str, str]], depth: int = 8) -> set[str]:
        seen: set[str] = set()
        frontier = [(node_id, 0)]
        while frontier:
            current, d = frontier.pop()
            if d >= depth:
                continue
            for p in parent_map.get(current, ()):
                if p and p not in seen:
                    seen.add(p)
                    frontier.append((p, d + 1))
        return seen

    def _related(self, a: str, b: str) -> bool:
        if not self.config.incest_guard:
            return False
        aa = self._ancestors(a, self.parent_map)
        bb = self._ancestors(b, self.parent_map)
        return a in bb or b in aa or bool(aa & bb)

    def eligible(self, a: Worm, b: Worm, generation: int, genomes: Dict[str, CognitiveGenome]) -> Tuple[bool, str, float]:
        if not self.config.enabled:
            return False, "genetics disabled", 0.0
        if a.id == b.id:
            return False, "same individual", 0.0
        if a.sex == b.sex or a.sex not in (Sex.MALE.value, Sex.FEMALE.value) or b.sex not in (Sex.MALE.value, Sex.FEMALE.value):
            return False, "sex gate", 0.0
        if a.status != WormStatus.ACTIVE.value or b.status != WormStatus.ACTIVE.value:
            return False, "inactive parent", 0.0
        if abs(a.generation - b.generation) < self.config.generation_gap_required:
            return False, "same-generation mating prohibited", 0.0
        if generation - max(a.last_mating_generation, b.last_mating_generation) < self.config.mate_cooldown:
            return False, "mating cooldown", 0.0
        if min(a.reputation, b.reputation) < self.config.min_reputation:
            return False, "reputation gate", 0.0
        if min(a.integrity, b.integrity) < self.config.min_integrity:
            return False, "integrity gate", 0.0
        if min(a.energy, b.energy) < self.config.min_energy:
            return False, "energy gate", 0.0
        if min(a.resilience, b.resilience) < self.config.min_resilience:
            return False, "resilience gate", 0.0
        if self._related(a.id, b.id):
            return False, "lineage guard", 0.0
        key = tuple(sorted((a.id, b.id)))
        if self.children_by_pair.get(key, 0) >= self.config.max_children_per_pair:
            return False, "pair offspring cap", 0.0
        ga, gb = genomes[a.id], genomes[b.id]
        diversity = ga.distance(gb)
        if diversity < self.config.minimum_genetic_distance:
            return False, "genetic similarity too high", diversity
        compatibility = (
            0.18 * (1.0 - abs(a.reputation - b.reputation))
            + 0.18 * (1.0 - abs(a.integrity - b.integrity))
            + 0.18 * (1.0 - abs(a.resilience - b.resilience))
            + 0.46 * diversity
        )
        return (compatibility >= 0.34), ("eligible" if compatibility >= 0.34 else "low compatibility"), compatibility

    def _gamete(self, genome: CognitiveGenome) -> Dict[str, float]:
        keys = [k for k in TRAITS if k in genome.maternal or k in genome.paternal]
        return {k: (genome.maternal.get(k, 0.5) if self.rng.random() < self.config.recombination_bias else genome.paternal.get(k, 0.5)) for k in keys}

    def _mutate(self, values: Dict[str, float]) -> Tuple[Dict[str, float], float]:
        load = 0.0
        out = {}
        for k, v in values.items():
            if self.rng.random() < self.config.mutation_rate:
                delta = self.rng.gauss(0.0, self.config.mutation_sigma)
                v = max(0.0, min(1.0, v + delta))
                load += abs(delta)
            out[k] = v
        return out, load / max(1, len(values))

    def conceive(self, a: Worm, b: Worm, generation: int, genomes: Dict[str, CognitiveGenome]) -> Tuple[Optional[CognitiveGenome], str, float, float]:
        ok, reason, compatibility = self.eligible(a, b, generation, genomes)
        if not ok:
            self.records.append(ReproductiveRecord(a.id, b.id, "", generation, False, reason, compatibility, compatibility))
            return None, reason, compatibility, 0.0
        mother, father = (a, b) if a.sex == Sex.FEMALE.value else (b, a)
        maternal, mut_a = self._mutate(self._gamete(genomes[mother.id]))
        paternal, mut_b = self._mutate(self._gamete(genomes[father.id]))
        diversity = genomes[a.id].distance(genomes[b.id])
        child = CognitiveGenome(maternal=maternal, paternal=paternal, generation=generation, mutation_load=min(1.0, mut_a + mut_b))
        return child, "eligible", compatibility, diversity

    def register_birth(self, parent_a: Worm, parent_b: Worm, child: Worm, generation: int, compatibility: float, diversity: float) -> None:
        self.parent_map[child.id] = (parent_a.id, parent_b.id)
        key = tuple(sorted((parent_a.id, parent_b.id)))
        self.mating_generation[key] = generation
        self.children_by_pair[key] = self.children_by_pair.get(key, 0) + 1
        parent_a.last_mating_generation = generation
        parent_b.last_mating_generation = generation
        parent_a.births += 1
        parent_b.births += 1
        parent_a.energy = max(0.0, parent_a.energy - self.config.reproduction_cost)
        parent_b.energy = max(0.0, parent_b.energy - self.config.reproduction_cost)
        self.records.append(ReproductiveRecord(parent_a.id, parent_b.id, child.id, generation, True, "birth", compatibility, diversity))

    def _candidate_pairs(self, worms: Sequence[Worm], genomes: Dict[str, CognitiveGenome], generation: int):
        males = [w for w in worms if w.sex == Sex.MALE.value and w.status == WormStatus.ACTIVE.value]
        females = [w for w in worms if w.sex == Sex.FEMALE.value and w.status == WormStatus.ACTIVE.value]
        candidates = []
        for m in males:
            for f in females:
                ok, _, compatibility = self.eligible(m, f, generation, genomes)
                if not ok:
                    continue
                diversity = genomes[m.id].distance(genomes[f.id])
                routing_fit = 0.5 * (float(getattr(m, "routing_fitness", 0.50)) + float(getattr(f, "routing_fitness", 0.50)))
                fitness = 0.58 * compatibility + 0.32 * diversity + 0.10 * routing_fit
                candidates.append((fitness, m, f, compatibility, diversity))
        return candidates

    def select_parents_tournament(
        self, population: Sequence[tuple], k: int, tournament_size: int = 4, temperature: float = 1.0
    ) -> List[tuple]:
        """Select disjoint parent pairs using fitness-sensitive softmax tournaments."""
        selected: List[tuple] = []
        used: set[str] = set()
        pool = list(population)
        tau = max(1e-6, float(temperature))
        size = max(2, int(tournament_size))
        while pool and len(selected) < int(k):
            available = [row for row in pool if row[1].id not in used and row[2].id not in used]
            if not available:
                break
            n = min(size, len(available))
            contenders = self.rng.sample(available, n)
            fitness_values = [float(row[0]) for row in contenders]
            max_f = max(fitness_values)
            mean_f = sum(fitness_values) / len(fitness_values)
            variance = sum((v - mean_f) ** 2 for v in fitness_values) / len(fitness_values)
            spread = max(math.sqrt(variance), 1e-3)
            # Temperature is dimensionless: normalize actual fitness gaps by the
            # current tournament spread so tau=1.0 is meaningfully sensitive.
            weights = [math.exp((float(row[0]) - max_f) / (tau * spread)) for row in contenders]
            chosen = self.rng.choices(contenders, weights=weights, k=1)[0]
            selected.append(chosen)
            used.update((chosen[1].id, chosen[2].id))
            pool = [row for row in pool if row[1].id not in used and row[2].id not in used]
        return selected

    def choose_pairs(self, worms: Sequence[Worm], genomes: Dict[str, CognitiveGenome], generation: int, max_pairs: int) -> List[Tuple[Worm, Worm, float, float]]:
        candidates = self._candidate_pairs(worms, genomes, generation)
        mode = str(getattr(self.config, "selection_mode", "tournament")).strip().lower()
        if mode == "greedy":
            candidates.sort(key=lambda x: (-x[0], x[1].id, x[2].id))
            selected = []
            used: set[str] = set()
            for row in candidates:
                if len(selected) >= max_pairs:
                    break
                _, m, f, c, d = row
                if m.id in used or f.id in used:
                    continue
                selected.append(row)
                used.update((m.id, f.id))
        else:
            selected = self.select_parents_tournament(
                candidates,
                max_pairs,
                tournament_size=getattr(self.config, "tournament_size", 4),
                temperature=getattr(self.config, "tournament_temperature", 1.0),
            )
        return [(m, f, c, d) for _, m, f, c, d in selected]
