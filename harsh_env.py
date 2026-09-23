from __future__ import annotations
from dataclasses import dataclass
import random
from typing import Dict

from .models import Finding, Worm


@dataclass(frozen=True)
class PressureState:
    hostility: float
    evidence_noise: float
    contradiction_rate: float
    decoy_rate: float
    concept_drift: float
    resource_scarcity: float
    deadline_pressure: float
    peer_attack: float
    event_shock: float = 0.0


@dataclass
class HarshLearningConfig:
    enabled: bool = True
    hostility: float = .70
    evidence_noise: float = .25
    contradiction_rate: float = .40
    decoy_rate: float = .30
    concept_drift: float = .20
    resource_scarcity: float = .50
    deadline_pressure: float = .35
    peer_attack: float = .62
    stress_gain: float = .14
    stress_recovery: float = .065
    resilience_gain: float = .032
    adaptation_gain: float = .042
    failure_tax: float = .06


class HarshLearningEnvironment:
    def __init__(self, config: HarshLearningConfig | None = None, seed: int = 42):
        self.config = config or HarshLearningConfig()
        self.rng = random.Random(seed)
        self.shock = 0.0

    def set_shock(self, shock: float) -> None:
        self.shock = max(0.0, min(1.0, shock))

    def pressure(self) -> PressureState:
        c = self.config
        if not c.enabled:
            return PressureState(0,0,0,0,0,0,0,0,0)
        jitter = lambda scale: self.rng.uniform(-scale, scale)
        s = self.shock
        return PressureState(
            max(0, min(1, c.hostility + jitter(.07) + .25*s)),
            max(0, min(1, c.evidence_noise + jitter(.06) + .18*s)),
            max(0, min(1, c.contradiction_rate + jitter(.07) + .20*s)),
            max(0, min(1, c.decoy_rate + jitter(.06) + .22*s)),
            max(0, min(1, c.concept_drift + jitter(.04) + .20*s)),
            max(0, min(1, c.resource_scarcity + jitter(.06) + .16*s)),
            max(0, min(1, c.deadline_pressure + jitter(.06) + .14*s)),
            max(0, min(1, c.peer_attack + jitter(.07) + .20*s)),
            s,
        )

    def expose(self, worm: Worm, p: PressureState) -> Dict[str, float]:
        load = .25*p.hostility + .18*p.evidence_noise + .18*p.contradiction_rate + .14*p.resource_scarcity + .15*p.deadline_pressure + .10*p.event_shock
        buffer = .42*worm.resilience + .28*worm.adaptation
        delta = self.config.stress_gain * max(0.0, load-buffer)
        worm.stress = max(0.0, min(1.0, worm.stress + delta))
        tax = p.resource_scarcity * (.025 + .025*worm.depth)
        worm.energy = max(0.0, worm.energy-tax)
        worm.hostility_exposure += p.hostility
        return {"load": load, "stress_delta": delta, "resource_tax": tax}

    def challenge(self, worm: Worm, finding: Finding, p: PressureState) -> Dict[str, float|bool]:
        vulnerability = .34*(1-finding.falsifiability) + .28*(1-finding.independence) + .22*(1-finding.validity) + .16*(1-finding.deception_resistance)
        pressure = .25*p.evidence_noise + .23*p.contradiction_rate + .17*p.decoy_rate + .12*p.concept_drift + .15*p.peer_attack + .08*p.event_shock
        survival = max(0.0, min(1.0, .90 + .28*worm.resilience + .16*worm.adaptation - .30*vulnerability - .20*pressure - .08*p.deadline_pressure))
        passed = survival >= .50
        finding.adversarial_survival = survival
        finding.robustness = max(0.0, min(1.0, .52*finding.validity + .30*survival + .18*finding.deception_resistance))
        finding.challenge_pressure = pressure
        if passed:
            worm.resilience = min(1.0, worm.resilience + self.config.resilience_gain*(.5+survival))
            worm.adaptation = min(1.0, worm.adaptation + self.config.adaptation_gain*(.5+pressure))
            worm.successful_adaptations += 1
            worm.stress = max(0.0, worm.stress-self.config.stress_recovery)
            worm.survival_count += 1
        else:
            worm.failures += 1
            worm.stress = min(1.0, worm.stress+self.config.stress_gain*.40)
            worm.energy = max(0.0, worm.energy-self.config.failure_tax)
            worm.scar_memory.append("failed hostile-learning challenge")
        return {"survival": survival, "passed": passed, "pressure": pressure}
