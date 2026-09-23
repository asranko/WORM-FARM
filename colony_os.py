from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple
import hashlib
import math

from .models import Finding, Worm


class LearningDirection(str, Enum):
    PEER = "peer"
    COUNTERPEER = "counterpeer"
    LINEAGE = "lineage"
    ENVIRONMENT = "environment"
    META = "meta"
    TRANSFER = "transfer"


class CompetitionTier(str, Enum):
    DUEL = "DUEL"
    HUNT = "HUNT"
    SIEGE = "SIEGE"
    ABYSS = "ABYSS"


@dataclass(frozen=True)
class SpecialistProfile:
    name: str
    core_trait: str
    support_traits: Tuple[str, ...]
    preferred_modes: Tuple[str, ...]
    failure_mode: str


SPECIALISTS: Tuple[SpecialistProfile, ...] = (
    SpecialistProfile("ASSUMPTION_HUNTER", "skepticism", ("penetration", "meta_cognition"), ("ASSUMPTION_WAR", "MODEL_KILLER"), "assumption_blindness"),
    SpecialistProfile("PARADIGM_BREAKER", "adversariality", ("penetration", "novelty"), ("PARADIGM_SIEGE", "IDEA_DUEL"), "frame_lock"),
    SpecialistProfile("ANOMALY_HUNTER", "pattern_sensitivity", ("skepticism", "curiosity"), ("ANOMALY_HUNT", "UNKNOWN_FRONT"), "signal_inflation"),
    SpecialistProfile("CAUSALITY", "falsifiability", ("counterfactual", "independence"), ("MODEL_KILLER", "EVIDENCE_DUEL"), "correlation_trap"),
    SpecialistProfile("COUNTERFACTUAL", "counterfactual", ("skepticism", "creativity"), ("COUNTERFACTUAL_WAR", "MODEL_KILLER"), "scenario_explosion"),
    SpecialistProfile("EVIDENCE_FORENSICS", "falsifiability", ("restraint", "independence"), ("EVIDENCE_DUEL", "BLIND_SPOT_HUNT"), "provenance_blindness"),
    SpecialistProfile("BOUNDARY_HUNTER", "penetration", ("skepticism", "persistence"), ("BLIND_SPOT_HUNT", "UNKNOWN_FRONT"), "boundary_neglect"),
    SpecialistProfile("NOVELTY_FORGE", "novelty", ("curiosity", "adversariality"), ("IDEA_DUEL", "UNKNOWN_FRONT"), "novelty_chasing"),
    SpecialistProfile("RED_TEAM", "adversariality", ("war_resistance", "falsifiability"), ("WORM_VS_WORM", "MODEL_KILLER"), "attack_overfit"),
    SpecialistProfile("MODEL_KILLER", "falsifiability", ("adversariality", "penetration"), ("MODEL_KILLER", "PARADIGM_SIEGE"), "model_fixation"),
    SpecialistProfile("BLIND_SPOT_HUNTER", "meta_cognition", ("counterfactual", "skepticism"), ("BLIND_SPOT_HUNT", "ASSUMPTION_WAR"), "self_blindness"),
    SpecialistProfile("SYNTHESIZER", "independence", ("meta_cognition", "memory"), ("COLONY_WAR", "IDEA_DUEL"), "premature_synthesis"),
    SpecialistProfile("MEMORY_ANALYST", "memory", ("meta_cognition", "pattern_sensitivity"), ("BLIND_SPOT_HUNT", "EVIDENCE_DUEL"), "recency_bias"),
    SpecialistProfile("DECEPTION_ANALYST", "meta_cognition", ("restraint", "skepticism"), ("ASSUMPTION_WAR", "EVIDENCE_DUEL"), "suspicion_inflation"),
    SpecialistProfile("SHASE", "meta_cognition", ("skepticism", "restraint", "counterfactual"), ("INFLUENCE_AUDIT", "ASSUMPTION_WAR", "EVIDENCE_DUEL"), "attribution_error"),
    SpecialistProfile("UNCERTAINTY_CARTOGRAPHER", "restraint", ("falsifiability", "counterfactual"), ("UNKNOWN_FRONT", "COUNTERFACTUAL_WAR"), "uncertainty_paralysis"),
    SpecialistProfile("EDGE_SCOUT", "curiosity", ("novelty", "penetration"), ("UNKNOWN_FRONT", "ANOMALY_HUNT"), "novelty_bias"),
    SpecialistProfile("TRANSFER_LEARNER", "adaptation", ("memory", "meta_cognition"), ("COLONY_WAR", "EVIDENCE_DUEL"), "context_leakage"),
    SpecialistProfile("WORM_EATER", "meta_cognition", ("restraint", "skepticism"), ("WORM_VS_WORM", "BLIND_SPOT_HUNT"), "overpruning"),
    SpecialistProfile("META_ARCHITECT", "meta_cognition", ("independence", "persistence"), ("COLONY_WAR", "PARADIGM_SIEGE"), "control_fixation"),
    SpecialistProfile("WORM_OMEGA", "meta_cognition", ("adversariality", "restraint"), ("ABYSS", "COLONY_WAR"), "meta_overreach"),
)

SPECIALIST_BY_NAME = {x.name: x for x in SPECIALISTS}


def specialist_names() -> List[str]:
    return [x.name for x in SPECIALISTS]


@dataclass
class SkillState:
    # Competence is deliberately explicit and bounded. This is the state that can later
    # be driven by an LLM-backed brain instead of the deterministic operator brain.
    values: Dict[str, float] = field(default_factory=dict)
    exposures: Dict[str, int] = field(default_factory=dict)
    mastery: Dict[str, float] = field(default_factory=dict)

    def ensure(self) -> None:
        for name in specialist_names():
            self.values.setdefault(name, 0.15)
            self.exposures.setdefault(name, 0)
            self.mastery.setdefault(name, 0.0)

    def learn(self, specialty: str, delta: float, exposure: int = 1) -> None:
        self.ensure()
        # Some curriculum targets are latent traits (e.g. persistence) rather than
        # named specialties. Keep a unified competency ledger without pretending the
        # trait is a specialist identity.
        self.values.setdefault(specialty, 0.15)
        self.exposures.setdefault(specialty, 0)
        self.mastery.setdefault(specialty, 0.0)
        delta = max(-0.15, min(0.15, float(delta)))
        self.values[specialty] = max(0.0, min(1.0, self.values[specialty] + delta))
        self.exposures[specialty] = self.exposures.get(specialty, 0) + max(0, int(exposure))
        self.mastery[specialty] = max(0.0, min(1.0, self.mastery.get(specialty, 0.0) + max(0.0, delta) * 0.8))


@dataclass
class CurriculumTask:
    task_id: str
    target_specialty: str
    direction: LearningDirection
    difficulty: float
    generation: int
    completed: bool = False
    reward: float = 0.0


@dataclass
class LearningDebt:
    specialty: str
    debt: float
    reason: str
    created_generation: int
    resolved_generation: int = -1


@dataclass
class ExperimentContract:
    id: str
    hypothesis_id: str
    baseline: str
    intervention: str
    falsifier: str
    expected_information_gain: float
    risk: float
    reversible: bool = True
    status: str = "PROPOSED"
    result: str = ""


@dataclass
class MetaPolicy:
    exploration_temperature: float = 0.65
    hostility_target: float = 0.82
    novelty_pressure: float = 0.30
    evidence_threshold: float = 0.44
    deepening_threshold: float = 0.50
    rehabilitation_bias: float = 0.25
    diversity_pressure: float = 0.35
    self_attack_rate: float = 0.20
    generation: int = 0
    policy_hash: str = ""

    def digest(self) -> str:
        raw = "|".join(f"{k}={getattr(self, k):.6f}" for k in (
            "exploration_temperature", "hostility_target", "novelty_pressure",
            "evidence_threshold", "deepening_threshold", "rehabilitation_bias",
            "diversity_pressure", "self_attack_rate")) + f"|g={self.generation}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


class CurriculumEngine:
    """Generates hard, multi-directional synthetic learning tasks."""

    CHANNEL_WEIGHTS = {
        LearningDirection.PEER.value: 0.85,
        LearningDirection.COUNTERPEER.value: 1.00,
        LearningDirection.LINEAGE.value: 0.65,
        LearningDirection.ENVIRONMENT.value: 0.95,
        LearningDirection.META.value: 1.10,
        LearningDirection.TRANSFER.value: 0.90,
    }

    def __init__(self) -> None:
        self.tasks: List[CurriculumTask] = []
        self._seq = 0

    def make_task(self, specialty: str, direction: LearningDirection, generation: int, difficulty: float) -> CurriculumTask:
        self._seq += 1
        task = CurriculumTask(
            task_id=f"CT-{generation:04d}-{self._seq:06d}",
            target_specialty=specialty,
            direction=direction,
            difficulty=max(0.1, min(1.0, difficulty)),
            generation=generation,
        )
        self.tasks.append(task)
        return task

    def complete(self, task: CurriculumTask, quality: float) -> float:
        quality = max(0.0, min(1.0, quality))
        weight = self.CHANNEL_WEIGHTS.get(task.direction.value, 0.8)
        reward = max(0.0, min(1.0, quality * weight / max(0.35, task.difficulty)))
        task.completed = quality >= 0.48
        task.reward = reward
        return reward


class ExperimentScheduler:
    """Schedules reversible synthetic tests instead of merely producing more prose."""

    def __init__(self) -> None:
        self.contracts: List[ExperimentContract] = []
        self._seq = 0

    def propose(self, finding: Finding, generation: int) -> ExperimentContract:
        self._seq += 1
        return self._build(finding, generation)

    def _build(self, finding: Finding, generation: int) -> ExperimentContract:
        cid = f"EX-{generation:04d}-{self._seq:06d}"
        expected = max(0.05, min(1.0, finding.information_gain * (0.6 + 0.4 * finding.falsifiability)))
        contract = ExperimentContract(
            id=cid,
            hypothesis_id=finding.id,
            baseline=finding.title,
            intervention=f"remove assumption::{finding.assumptions_targeted[0] if finding.assumptions_targeted else 'implicit framing'}",
            falsifier="synthetic counterexample invalidates the claim",
            expected_information_gain=expected,
            risk=max(0.0, min(1.0, 1.0 - finding.falsifiability)),
        )
        self.contracts.append(contract)
        return contract

    def execute(self, contract: ExperimentContract, finding: Finding, contradiction: float, novelty: float) -> float:
        signal = max(0.0, min(1.0, 0.50 * finding.falsifiability + 0.25 * novelty + 0.25 * contradiction))
        contract.result = f"synthetic_signal={signal:.4f}"
        contract.status = "PASSED" if signal >= 0.45 else "FAILED"
        return signal


class MetaController:
    """Adapts colony pressure without allowing any single metric to dominate."""

    def __init__(self) -> None:
        self.policy = MetaPolicy()

    def update(self, generation: int, *, mean_quality: float, mean_diversity: float,
               false_suspicion: float, active_ratio: float, open_findings: int) -> MetaPolicy:
        p = self.policy
        p.generation = generation
        # Exploration rises when diversity is collapsing or when open findings are sparse.
        if mean_diversity < 0.10 or open_findings < 2:
            p.exploration_temperature = min(0.95, p.exploration_temperature + 0.06)
            p.novelty_pressure = min(0.70, p.novelty_pressure + 0.04)
        else:
            p.exploration_temperature = max(0.35, p.exploration_temperature - 0.025)
            p.novelty_pressure = max(0.12, p.novelty_pressure - 0.012)

        # If false suspicion grows, become stricter, not more permissive.
        if false_suspicion > 0.45:
            p.evidence_threshold = min(0.68, p.evidence_threshold + 0.035)
            p.hostility_target = max(0.55, p.hostility_target - 0.035)
        elif mean_quality > 0.68 and active_ratio > 0.25:
            p.evidence_threshold = max(0.34, p.evidence_threshold - 0.018)
            p.hostility_target = min(0.97, p.hostility_target + 0.018)

        # Self-attack intensifies only when the colony is stable enough to survive it.
        p.self_attack_rate = min(0.38, max(0.08, p.self_attack_rate + 0.02 * (active_ratio - 0.30)))
        p.diversity_pressure = max(0.15, min(0.55, 0.55 * p.exploration_temperature + 0.25 * (1 - mean_diversity)))
        p.policy_hash = p.digest()
        return p


def choose_specialist_for_finding(finding: Finding) -> str:
    # Deterministic routing based on the finding's profile.
    if finding.kind.value in {"BREACHER", "OMEGA"}:
        return "PARADIGM_BREAKER"
    if finding.kind.value == "BURROWER":
        return "ASSUMPTION_HUNTER"
    if finding.kind.value == "PROBER":
        return "EVIDENCE_FORENSICS"
    if finding.kind.value == "MIRROR":
        return "COUNTERFACTUAL"
    if finding.kind.value == "CROSSWORM":
        return "BOUNDARY_HUNTER"
    if finding.kind.value == "WORM_EATER":
        return "WORM_EATER"
    return "EDGE_SCOUT"
