from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import json
from typing import Any, Dict

from .farm import WormFarm
from .models import Worm, WormStatus
from .strategy import EpigeneticState, StrategyEngine, StrategyGenome, StrategyPolicy, strategy_fingerprint
from .runtime import CheckpointStore, pack_state, unpack_state


class EvolvingWormFarm(WormFarm):
    """WORM FARM v2: strategy genome + epigenetic adaptation + deterministic telemetry."""

    def __init__(self, *args, strategy_seed: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_engine = StrategyEngine(strategy_seed if strategy_seed is not None else self.config.random_seed)
        self.strategy_genomes: Dict[str, StrategyGenome] = {}
        self.epigenetics: Dict[str, EpigeneticState] = {}
        self.strategy_history: list[dict] = []
        self.strategy_events: list[dict] = []

    def _new_worm(self, *args, **kwargs) -> Worm:
        worm = super()._new_worm(*args, **kwargs)
        bias = 0.05 if worm.specialty in {"PARADIGM_BREAKER", "RED_TEAM", "MODEL_KILLER"} else 0.0
        self.strategy_genomes[worm.id] = self.strategy_engine.founder(worm.generation, bias=bias)
        self.epigenetics[worm.id] = EpigeneticState()
        self._sync_strategy_to_worm(worm)
        return worm

    def _sync_strategy_to_worm(self, worm: Worm) -> StrategyPolicy:
        genome = self.strategy_genomes.setdefault(worm.id, self.strategy_engine.founder(worm.generation))
        epi = self.epigenetics.setdefault(worm.id, EpigeneticState())
        policy = self.strategy_engine.policy(genome, epi)
        # These fields already exist in v1.0 and now become strategy-conditioned.
        worm.current_strategy = f"{worm.specialty}:{'DEEP' if policy.max_depth_factor >= .78 else 'BROAD'}:{'HOSTILE' if policy.hostility >= .70 else 'STRICT'}"
        worm.cognitive_temperature = max(.05, min(1.0, policy.novelty_pressure))
        worm.uncertainty_budget = max(.05, min(1.0, 1.0 - policy.evidence_threshold * .75))
        worm.learning_capacity = max(.10, min(1.0, .45 + .35 * policy.meta_reasoning + .15 * policy.recovery))
        worm.teaching_power = max(.10, min(1.0, .35 + .50 * policy.teaching))
        worm.counterattack_skill = max(.10, min(1.0, .30 + .45 * policy.hostility + .20 * policy.recovery))
        worm.penetration_depth_reached = max(worm.penetration_depth_reached, int(worm.depth * policy.max_depth_factor))
        return policy

    def _adaptive_max_depth(self, worm: Worm) -> int:
        base = super()._adaptive_max_depth(worm)
        policy = self._sync_strategy_to_worm(worm)
        return max(1, min(self.config.max_depth, int(round(base * (0.82 + 0.42 * policy.max_depth_factor)))))

    def _build_finding(self, worm: Worm, claim, parent, pressure):
        policy = self._sync_strategy_to_worm(worm)
        finding = super()._build_finding(worm, claim, parent, pressure)
        finding.novelty = max(0.0, min(1.0, 0.75 * finding.novelty + 0.25 * policy.novelty_pressure))
        finding.falsifiability = max(0.0, min(1.0, 0.72 * finding.falsifiability + 0.28 * policy.evidence_threshold))
        finding.counterfactual_score = max(0.0, min(1.0, finding.counterfactual_score * (0.65 + 0.65 * policy.counterfactual_weight)))
        finding.robustness = max(0.0, min(1.0, finding.robustness * (0.55 + 0.55 * policy.restraint)))
        finding.independent_path = (finding.independent_path + f"|strategy:{strategy_fingerprint(self.strategy_genomes[worm.id], self.epigenetics[worm.id])}").strip("|")
        return finding

    def _run_generation_impl(self) -> int:
        before_ids = set(self.state.worms)
        produced = super()._run_generation_impl()
        after_ids = set(self.state.worms)
        # Children inherit strategy from actual parents after base genetics has created them.
        for wid in sorted(after_ids - before_ids):
            child = self.state.worms[wid]
            parents = self.genetics.parent_map.get(wid)
            if parents and all(pid in self.strategy_genomes for pid in parents):
                parent_a, parent_b = parents
                self.strategy_genomes[wid] = self.strategy_engine.recombine(
                    self.strategy_genomes[parent_a], self.strategy_genomes[parent_b], child.generation
                )
                self.epigenetics[wid] = EpigeneticState(stress=0.06, recovery_boost=.12)
                self._sync_strategy_to_worm(child)
                self.strategy_events.append({"type": "STRATEGY_BIRTH", "child": wid, "parents": list(parents), "generation": child.generation})
        # Adapt everyone based on the generation outcome.
        generation = self.state.generation
        for worm in list(self.state.worms.values()):
            epi = self.epigenetics.get(worm.id)
            genome = self.strategy_genomes.get(worm.id)
            if epi is None or genome is None:
                continue
            own_findings = [f for f in self.state.findings.values() if f.worm_id == worm.id and f.id.startswith(f"F-{max(0, generation-1):04d}-")]
            success = sum(1 for f in own_findings if f.status == "OPEN" and f.score >= .48) / max(1, len(own_findings))
            failure = sum(1 for f in own_findings if f.status == "PRUNED") / max(1, len(own_findings))
            novelty_gap = max(0.0, 0.60 - (sum(f.novelty for f in own_findings) / max(1, len(own_findings))))
            self.strategy_engine.adapt(epi, success=success, failure=failure, novelty_gap=novelty_gap, stress=worm.stress)
            self._sync_strategy_to_worm(worm)
            self.strategy_history.append({
                "generation": generation, "worm": worm.id, "specialty": worm.specialty,
                "genome": genome.digest(), "epigenetic": asdict(epi),
                "policy": asdict(self.strategy_engine.policy(genome, epi)),
            })
        self.strategy_history = self.strategy_history[-5000:]
        return produced

    def snapshot(self) -> dict:
        payload = super().snapshot()
        payload["v2"] = {
            "strategy_genomes": {k: asdict(v) for k, v in self.strategy_genomes.items()},
            "epigenetics": {k: asdict(v) for k, v in self.epigenetics.items()},
            "strategy_history": self.strategy_history,
            "strategy_events": self.strategy_events,
            "strategy_rng_state": pack_state(self.strategy_engine.rng.getstate()),
        }
        return payload

    @classmethod
    def from_checkpoint(cls, path, brain=None):
        payload = CheckpointStore(path).load()
        if payload.get("format") != "worm-farm-checkpoint" or payload.get("version") != 1:
            raise ValueError("unsupported worm-farm checkpoint")
        base = WormFarm._from_payload(payload, brain=brain)
        obj = cls.__new__(cls)
        obj.__dict__ = base.__dict__
        obj.strategy_engine = StrategyEngine(obj.config.random_seed)
        v2 = payload.get("v2", {})
        obj.strategy_genomes = {
            wid: StrategyGenome(**raw) for wid, raw in v2.get("strategy_genomes", {}).items()
        }
        obj.epigenetics = {
            wid: EpigeneticState(**raw) for wid, raw in v2.get("epigenetics", {}).items()
        }
        obj.strategy_history = list(v2.get("strategy_history", []))
        obj.strategy_events = list(v2.get("strategy_events", []))
        if v2.get("strategy_rng_state"):
            obj.strategy_engine.rng.setstate(unpack_state(v2["strategy_rng_state"]))
        for worm in obj.state.worms.values():
            obj.strategy_genomes.setdefault(worm.id, obj.strategy_engine.founder(worm.generation))
            obj.epigenetics.setdefault(worm.id, EpigeneticState())
            obj._sync_strategy_to_worm(worm)
        return obj

    def report(self) -> dict:
        r = super().report()
        strategy_values = list(self.strategy_genomes.values())
        if strategy_values:
            vectors = [g.phenotype() for g in strategy_values]
            dims = list(vectors[0])
            strategy_diversity = sum(
                abs(a[d] - b[d]) for i, a in enumerate(vectors) for b in vectors[i+1:] for d in dims
            ) / max(1, len(vectors) * max(1, len(vectors)-1) * len(dims) / 2)
        else:
            strategy_diversity = 0.0
        r.update({
            "version": "2.1.0",
            "strategy_population": len(strategy_values),
            "strategy_diversity": round(strategy_diversity, 6),
            "strategy_events": len(self.strategy_events),
            "strategy_history": len(self.strategy_history),
            "specialist_diversity": len({w.specialty for w in self.state.worms.values()}),
        })
        return r
