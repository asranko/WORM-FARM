from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional

from .genetics import CognitiveGenome, GenomeEngine, GeneticsConfig, Sex
from .harsh_env import HarshLearningConfig, HarshLearningEnvironment
from .arena import ArenaConfig, WormArena
from .learning import LearningConfig, LearningEcology
from .colony_os import (
    SPECIALISTS, SPECIALIST_BY_NAME, SkillState, CurriculumEngine, ExperimentScheduler,
    LearningDirection, MetaController,
)
from .invariants import validate_state
from .models import AuditEvent, Claim, FarmConfig, FarmState, Finding, Worm, WormKind, WormStatus, GenerationRecord
from .operators import AttackContext, attack, choose_children
from .runtime import BrainAdapter, CheckpointStore, EventLog, InvariantViolation, RuleBasedBrain, dataclass_to_dict, pack_state, unpack_state
from .brain import BrainProtocolError, CachedBrain, validate_proposal
from .safety import validate_text
from .research_ecology import ResearchEcology
from .self_development import SelfDevelopmentEngine
from .shase import SHASEAnalyzer
from .mode_router import RoutingBandit, RoutingDecisionBook, ModeDecision


class WormFarm:
    """Durable adversarial cognitive colony engine.

    The farm is deliberately deterministic by seed, event-auditable, checkpointable,
    internally disciplined, genetically evolving, and fronted by a pluggable brain API.
    It models synthetic cognitive search, not human physiology or real-world targeting.
    """

    PHASES = ("READY", "SEEDING", "EVOLVING", "LEARNING", "EXPERIMENTING", "ATTACKING", "AUDITING", "ARENA", "REPRODUCING", "FINALIZING", "HALTED", "COMPLETE")

    GENERATION_NAMES = (
        "GENESIS", "FIRST BURROW", "PRESSURE VEIL", "FAULTLINE", "BLACK ICE",
        "IRON RAIN", "HIDDEN AXIOM", "RIFT", "ABYSS GATE", "SCAR TIDE",
        "PARADOX STORM", "DEEP CURRENT", "COLD FORGE", "BROKEN COMPASS",
        "NULL HORIZON", "RED DEPTH", "SILENT FRACTURE", "DARK LATTICE",
        "EDGE FIRE", "SHATTERED FRAME", "NERVE OF STONE", "DEAD SIGNAL",
        "PRESSURE CROWN", "UNSEEN CAUSE", "BLACK VEIL", "RECURSIVE ABYSS",
        "LAST ASSUMPTION", "FALLING AXIOM", "FORBIDDEN EDGE", "WORMSTORM",
        "OMEGA FRONT", "BEYOND THE FRAME",
    )

    @classmethod
    def generation_name(cls, number: int) -> str:
        if number < 0:
            raise ValueError("generation number must be non-negative")
        cycle, idx = divmod(number, len(cls.GENERATION_NAMES))
        base = cls.GENERATION_NAMES[idx]
        return f"{base} {cycle + 1}" if cycle else base

    def __init__(self, config: FarmConfig | None = None, brain: BrainAdapter | None = None):
        self.config = copy.deepcopy(config) if config is not None else FarmConfig()
        self._validate_config()
        self.state = FarmState()
        if brain is None:
            from .brain_factory import make_brain
            brain = make_brain(self.config)
        self.brain = brain if isinstance(brain, CachedBrain) else CachedBrain(brain, self.state.brain_cache)
        self.brain.cache = self.state.brain_cache
        self.rng = random.Random(self.config.random_seed)
        self.event_log = EventLog()
        self._finding_seq = 0
        self._worm_seq = 0
        self._genome_seq = 0
        self._last_population = 0
        self._last_open = 0
        self._setup_services()
        self._setup_mode_routing()

    def _setup_mode_routing(self) -> None:
        self.routing_book = RoutingDecisionBook(bandit=RoutingBandit(
            seed=self.config.random_seed + 811,
            control_fraction=self.config.routing_control_fraction,
            control_threshold=self.config.routing_control_threshold,
            prior_alpha=self.config.routing_prior_alpha,
            prior_beta=self.config.routing_prior_beta,
            evidence_weight=self.config.routing_evidence_weight,
            routing_bias_weight=self.config.routing_bias_weight,
            generation_bucket_width=self.config.routing_generation_bucket_width,
            signal_calibration=(self.config.routing_signal_calibration_params if self.config.routing_signal_calibration_enabled else None),
            empirical_signal_enabled=self.config.routing_empirical_signal_enabled,
            empirical_min_confidence=self.config.routing_empirical_min_confidence,
        ), reward_weights={
            "audit": self.config.routing_reward_audit_weight,
            "signature": self.config.routing_reward_signature_weight,
            "survival": self.config.routing_reward_survival_weight,
            "robustness": self.config.routing_reward_robustness_weight,
            "cost": self.config.routing_reward_cost_weight,
        }, routing_fitness_lambda=self.config.routing_fitness_lambda)
        self.routing_book.records = self.state.routing_decisions
        delegate = getattr(self.brain, 'delegate', None)
        if delegate is not None and getattr(delegate, 'name', '') == 'worm-dual-lm':
            delegate.router = self.routing_book.bandit
            self.dual_router = delegate.router
        else:
            self.dual_router = None

    def _setup_services(self) -> None:
        self.arena = WormArena(ArenaConfig(
            enabled=self.config.arena_enabled,
            matches_per_generation=self.config.arena_matches_per_generation,
            tournament_interval=self.config.arena_tournament_interval,
            tournament_slots=self.config.arena_tournament_slots,
            hostility=self.config.arena_hostility,
            energy_cost=self.config.arena_energy_cost,
            stress_gain=self.config.arena_stress_gain,
            win_reward=self.config.arena_win_reward,
            loss_penalty=self.config.arena_loss_penalty,
            draw_reward=self.config.arena_draw_reward,
            rating_k=self.config.arena_rating_k,
            adaptation_gain=self.config.arena_adaptation_gain,
            learning_transfer=self.config.arena_learning_transfer,
            tournament_bonus=self.config.arena_tournament_bonus,
        ), seed=self.config.random_seed)
        self.learning = LearningEcology(LearningConfig(
            enabled=self.config.learning_enabled,
            transfer_rate=self.config.learning_transfer_rate,
            teaching_reward=self.config.learning_teaching_reward,
            counterteaching_reward=self.config.learning_counterteaching_reward,
        ), seed=self.config.random_seed)
        self.environment = HarshLearningEnvironment(HarshLearningConfig(
            enabled=self.config.harsh_learning,
            hostility=self.config.harsh_hostility,
            evidence_noise=self.config.harsh_noise,
            contradiction_rate=self.config.harsh_contradiction,
            decoy_rate=self.config.harsh_decoy,
            concept_drift=self.config.harsh_drift,
            resource_scarcity=self.config.harsh_scarcity,
            deadline_pressure=self.config.harsh_deadline,
            peer_attack=self.config.harsh_peer_attack,
            failure_tax=self.config.harsh_failure_tax,
        ), seed=self.config.random_seed + 17)
        self.genetics = GenomeEngine(GeneticsConfig(
            enabled=self.config.genetics_enabled,
            mutation_rate=self.config.mutation_rate,
            mutation_sigma=self.config.mutation_sigma,
            recombination_bias=self.config.recombination_bias,
            min_reputation=self.config.genetic_min_reputation,
            min_integrity=self.config.genetic_min_integrity,
            min_energy=self.config.genetic_min_energy,
            min_resilience=self.config.genetic_min_resilience,
            max_children_per_pair=self.config.max_children_per_pair,
            mate_cooldown=self.config.mate_cooldown,
            generation_gap_required=self.config.generation_gap_required,
            minimum_genetic_distance=self.config.minimum_genetic_distance,
            incest_guard=self.config.incest_guard,
            prefer_unrelated=self.config.prefer_unrelated,
            reproduction_cost=self.config.reproduction_cost,
            reproduction_reward=self.config.reproduction_reward,
            max_population_from_genetics=self.config.max_population_from_genetics,
            random_seed=self.config.random_seed,
            selection_mode=self.config.genetic_selection_mode,
            tournament_size=self.config.genetic_tournament_size,
            tournament_temperature=self.config.genetic_tournament_temperature,
        ))
        self.genomes: Dict[str, CognitiveGenome] = {}
        self.skill_states: Dict[str, SkillState] = {}
        self.curriculum = CurriculumEngine()
        self.experiments = ExperimentScheduler()
        self.meta = MetaController()
        self._specialist_seq = 0
        self.shase = SHASEAnalyzer(
            influence_weight=self.config.shase_influence_weight,
            arousal_weight=self.config.shase_arousal_weight,
            framing_weight=self.config.shase_framing_weight,
            social_pressure_weight=self.config.shase_social_pressure_weight,
            evidence_support_weight=self.config.shase_evidence_support_weight,
            uncertainty_weight=self.config.shase_uncertainty_weight,
        )
        self._shase_generation_count = 0
        self.research_ecology = ResearchEcology(
            capacity_bits=self.config.workspace_capacity_bits,
            noise=self.config.workspace_noise,
            message_base_bits=self.config.research_message_base_bits,
        )
        self.self_development = SelfDevelopmentEngine(
            seed=self.config.random_seed ^ 0xD3A,
            min_gain=self.config.self_development_min_gain,
            interval=self.config.self_development_interval,
            max_trials=self.config.self_development_max_trials,
            canary_generations=self.config.self_development_canary_generations,
            canary_seeds=self.config.self_development_canary_seeds,
        )

    def _validate_config(self) -> None:
        c = self.config
        bounded = {
            "duplicate_similarity": c.duplicate_similarity,
            "audit_fraction": c.audit_fraction,
            "mutation_rate": c.mutation_rate,
            "mutation_sigma": c.mutation_sigma,
            "minimum_genetic_distance": c.minimum_genetic_distance,
            "skill_learning_rate": c.skill_learning_rate,
            "skill_decay": c.skill_decay,
            "specialty_mutation_rate": c.specialty_mutation_rate,
            "specialty_diversity_target": c.specialty_diversity_target,
            "self_attack_budget_fraction": c.self_attack_budget_fraction,
            "experiment_credit_gain": c.experiment_credit_gain,
            "experiment_cost": c.experiment_cost,
            "meta_policy_smoothing": c.meta_policy_smoothing,
            "novelty_floor": c.novelty_floor,
            "keystone_min_integrity": c.keystone_min_integrity,
            "keystone_min_reputation": c.keystone_min_reputation,
            "keystone_energy_floor": c.keystone_energy_floor,
            "keystone_rescue_cost": c.keystone_rescue_cost,
        }
        for name, value in bounded.items():
            if value < 0 or value > 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        if c.workspace_capacity_bits < 64:
            raise ValueError("workspace_capacity_bits must be >= 64")
        if c.research_message_base_bits < 16:
            raise ValueError("research_message_base_bits must be >= 16")
        if c.research_max_events < 1:
            raise ValueError("research_max_events must be >= 1")
        if c.self_development_interval < 1 or c.self_development_max_trials < 1:
            raise ValueError("self-development cadence/budget must be positive")
        if c.self_development_canary_generations < 1 or c.self_development_canary_seeds < 1:
            raise ValueError("self-development canary parameters must be positive")
        if c.self_development_min_gain < 0:
            raise ValueError("self_development_min_gain must be >= 0")
        if c.self_development_history_limit < 1:
            raise ValueError("self_development_history_limit must be >= 1")
        if c.max_generations < 1 or c.max_worms < 2 or c.max_depth < 0:
            raise ValueError("max_generations/max_worms/max_depth out of range")
        if c.max_findings_per_generation < 1 or c.max_audits_per_generation < 1 or c.arena_matches_per_generation < 0:
            raise ValueError("generation budgets must be positive")
        if c.generation_gap_required < 1:
            raise ValueError("generation_gap_required must be >= 1")
        if c.max_curriculum_tasks_per_generation < 1 or c.max_experiments_per_generation < 1:
            raise ValueError("v0.8 curriculum/experiment budgets must be >= 1")
        if c.keystone_reserve < 0:
            raise ValueError("keystone_reserve must be >= 0")
        if c.dual_mask_ratio <= 0 or c.dual_mask_ratio >= 1:
            raise ValueError("dual_mask_ratio must be in (0,1)")
        if c.dual_mode_threshold < -1.0 or c.dual_mode_threshold > 1.0:
            raise ValueError("dual_mode_threshold out of range")
        if c.dual_device not in {"cpu", "cuda", "mps"}:
            raise ValueError("dual_device must be one of cpu/cuda/mps")
        if not 0.0 <= c.routing_control_fraction <= 0.50:
            raise ValueError("routing_control_fraction must be in [0,0.50]")
        if not -1.0 <= c.routing_control_threshold <= 1.0:
            raise ValueError("routing_control_threshold out of range")
        if c.routing_prior_alpha <= 0 or c.routing_prior_beta <= 0:
            raise ValueError("routing priors must be positive")
        if c.routing_generation_bucket_width < 1 or c.routing_min_resolved_for_genetic_bias < 0:
            raise ValueError("routing generation bucket / activation out of range")
        if not isinstance(c.routing_signal_calibration_params, dict):
            raise ValueError("routing_signal_calibration_params must be a dict")
        if c.routing_empirical_min_confidence < 1:
            raise ValueError("routing_empirical_min_confidence must be >= 1")
        if str(c.genetic_selection_mode).lower() not in {"greedy", "tournament"}:
            raise ValueError("genetic_selection_mode must be greedy or tournament")
        if c.genetic_tournament_size < 2:
            raise ValueError("genetic_tournament_size must be >= 2")
        if c.genetic_tournament_temperature <= 0:
            raise ValueError("genetic_tournament_temperature must be > 0")
        weights = [c.routing_reward_audit_weight, c.routing_reward_signature_weight, c.routing_reward_survival_weight, c.routing_reward_robustness_weight, c.routing_reward_cost_weight]
        if any(w < 0 for w in weights) or abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError("routing reward weights must be non-negative and sum to 1")
        if not 0.0 <= c.routing_replay_max_per_generation:
            raise ValueError("routing_replay_max_per_generation must be non-negative")

    def _set_phase(self, phase: str, **payload: Any) -> None:
        if phase not in self.PHASES:
            raise ValueError(f"unknown phase: {phase}")
        self.state.phase = phase
        self._emit("PHASE", {"phase": phase, **payload})

    def _emit(self, event_type: str, payload: Dict[str, Any]) -> None:
        if len(self.event_log.events) >= self.config.max_events:
            raise InvariantViolation("event log budget exhausted")
        event = self.event_log.append(event_type, self.state.generation, payload)
        self.state.event_seq = event["seq"]

    def _next_worm_id(self) -> str:
        self._worm_seq += 1
        return f"W-{self._worm_seq:06d}"

    def _next_genome_id(self) -> str:
        self._genome_seq += 1
        return f"G-{self._genome_seq:06d}"

    def _next_finding_id(self) -> str:
        self._finding_seq += 1
        return f"F-{self.state.generation:04d}-{self._finding_seq:07d}"

    def _assign_specialty(self, worm: Worm, inherited: str | None = None) -> str:
        names = [s.name for s in SPECIALISTS]
        if inherited in SPECIALIST_BY_NAME and self.rng.random() >= self.config.specialty_mutation_rate:
            return inherited
        idx = self._specialist_seq % len(names)
        self._specialist_seq += 1
        return names[idx]

    def _new_worm(self, claim_id: str, sex: Sex, generation: int, parent_worm_id: str | None = None,
                  genome: CognitiveGenome | None = None, depth: int = 0, energy: float = 1.0,
                  integrity: float = 0.80, reputation: float = 0.55, kind: WormKind = WormKind.SCOUT) -> Worm:
        if len(self.state.worms) >= self.config.max_worms:
            raise InvariantViolation("population cap reached")
        worm = Worm(
            id=self._next_worm_id(), kind=kind, target_claim_id=claim_id, depth=depth,
            parent_worm_id=parent_worm_id, max_depth=self.config.max_depth, energy=energy,
            integrity=integrity, reputation=reputation, sex=sex.value, generation=generation,
            genome_id=self._next_genome_id(), lineage_depth=max(0, generation),
        )
        worm.cognitive_habits = {k: 0.0 for k in ("closure", "novelty_chasing", "overtrust", "overcomplexity")}
        worm.specialty = worm.kind.value
        self.state.worms[worm.id] = worm
        self.genomes[worm.id] = genome if genome is not None else self.genetics.initialize_founder(sex, generation)
        worm.routing_bias = self.genomes[worm.id].routing_bias()
        inherited_specialty = getattr(worm, "specialty", None) if genome is not None else None
        worm.specialty = self._assign_specialty(worm, inherited_specialty)
        worm.kind = kind
        skill = SkillState(); skill.ensure()
        skill.values[worm.specialty] = max(skill.values.get(worm.specialty, 0.0), .28)
        skill.mastery[worm.specialty] = max(skill.mastery.get(worm.specialty, 0.0), .10)
        self.skill_states[worm.id] = skill
        worm.skill_state = dict(skill.values)
        worm.skill_mastery = dict(skill.mastery)
        worm.learning_debt = {name: 0.0 for name in names if False} if False else {}
        phenotype = self._phenotype(worm)
        worm.learning_channels = {k: 0.0 for k in self.learning.CHANNELS}
        worm.learning_capacity = .58 + .24*phenotype.get("meta_cognition", .5)
        worm.teaching_power = .40 + .30*phenotype.get("independence", .5)
        worm.counterattack_skill = .45 + .35*phenotype.get("war_resistance", .5)
        self.arena.state.ratings.setdefault(worm.id, worm.arena_rating)
        return worm

    def seed(self, claim: Claim) -> str:
        decision = validate_text(claim.text)
        if not decision.allowed:
            raise ValueError(decision.reason)
        for item in claim.assumptions + claim.evidence:
            d = validate_text(item)
            if not d.allowed:
                raise ValueError(d.reason)
        if claim.id in self.state.claims:
            raise ValueError(f"claim already exists: {claim.id}")
        self._set_phase("SEEDING", claim_id=claim.id)
        self.state.claims[claim.id] = claim
        first = None
        founders = max(2, self.config.founder_pairs)
        # Adjacent founder cohorts are intentional: they make the cross-generation
        # mating rule productive from generation 0 without ever permitting same-gen mating.
        founder_kinds = [
            WormKind.SCOUT, WormKind.BURROWER, WormKind.BREACHER, WormKind.PROBER,
            WormKind.MIRROR, WormKind.CROSSWORM, WormKind.WORM_EATER, WormKind.OMEGA,
        ]
        for idx in range(founders):
            male_kind = founder_kinds[(2*idx) % len(founder_kinds)]
            female_kind = founder_kinds[(2*idx+1) % len(founder_kinds)]
            male = self._new_worm(claim.id, Sex.MALE, 0, reputation=.62, kind=male_kind)
            female = self._new_worm(claim.id, Sex.FEMALE, 1, reputation=.62, kind=female_kind)
            self.state.edges.append((male.id, female.id, "FOUNDATION_COHORTS"))
            first = first or male.id
        self._emit("SEED", {"claim_id": claim.id, "founders": founders * 2})
        validate_state(self.state)
        self._set_phase("READY")
        return first or ""

    def _phenotype(self, worm: Worm) -> dict:
        return self.genomes[worm.id].phenotype()

    def _similar(self, finding: Finding) -> bool:
        words = set(finding.detail.lower().split())
        if len(words) < 3:
            return False
        for other in self.state.findings.values():
            if other.id == finding.id or other.status == "PRUNED":
                continue
            ow = set(other.detail.lower().split())
            if len(ow) < 3:
                continue
            sim = len(words & ow) / max(1, len(words | ow))
            if sim >= self.config.duplicate_similarity and other.depth <= finding.depth + 1:
                return True
        return False

    def _forbidden_zone_hits(self, claim: Claim) -> list[str]:
        if not self.config.forbidden_assumption_zone:
            return []
        corpus = " ".join([claim.text] + claim.assumptions).lower()
        return [t for t in self.config.forbidden_terms if t in corpus]

    def _counterfactual(self, finding: Finding, worm: Worm) -> float:
        if not self.config.counterfactual_lab:
            return 0.0
        p = self._phenotype(worm)
        worlds = [finding.falsifiability, 1.0 - finding.validity, finding.independence, finding.deception_resistance]
        contrast = max(worlds) - min(worlds)
        return max(0.0, min(1.0, .45*finding.falsifiability + .25*(1-abs(.5-p.get("counterfactual", .5))) + .30*contrast))

    def _audit_peer_war(self, finding: Finding) -> float:
        if not self.config.worm_warfare:
            return 0.0
        candidates = [f for f in self.state.findings.values() if f.id != finding.id and f.status == "OPEN"]
        if not candidates:
            return 0.0
        strongest = max(candidates, key=lambda f: (f.score, f.depth, f.id))
        attack_strength = max(0.0, min(1.0, strongest.score*self.config.war_pressure + (1-finding.validity)*.35))
        finding.attack_count += 1
        finding.worm_war_score = attack_strength
        self.state.conflict_edges.append((strongest.id, finding.id, attack_strength, "peer adversarial challenge"))
        return attack_strength

    def _update_habits(self, worm: Worm, finding: Finding) -> None:
        if not self.config.cognitive_parasites:
            return
        habits = worm.cognitive_habits
        habits["closure"] = min(1.0, habits["closure"] + .04*max(0.0, .50-finding.falsifiability))
        habits["novelty_chasing"] = min(1.0, habits["novelty_chasing"] + .05*max(0.0, finding.novelty-finding.validity))
        habits["overtrust"] = min(1.0, habits["overtrust"] + .04*max(0.0, .50-finding.independence))
        habits["overcomplexity"] = min(1.0, habits["overcomplexity"] + .02*max(0.0, finding.depth-3))
        # Bounded recovery of non-dominant habits stops the parasite tracker from saturating forever.
        for k in habits:
            if k != max(habits, key=habits.get):
                habits[k] = max(0.0, habits[k] - 0.006)

    def _adaptive_max_depth(self, worm: Worm) -> int:
        if not self.config.adaptive_penetration:
            return self.config.max_depth
        p = self._phenotype(worm)
        base = self.config.max_depth
        score = .55*p.get("penetration", .5) + .25*p.get("persistence", .5) + .20*worm.resilience
        bonus = int(score >= .70) + int(score >= .87)
        return min(base, max(1, int(base*.55) + bonus))

    def _fitness(self, finding: Finding) -> float:
        worm = self.state.worms[finding.worm_id]
        return finding.score*(.50 + .24*worm.reputation + .18*worm.resilience + .08*worm.adaptation) + .06*finding.worm_war_score

    def _award(self, finding: Finding) -> float:
        worm = self.state.worms[finding.worm_id]
        reward = self.config.reward_base * (
            self.config.reward_information_gain*finding.information_gain
            + self.config.reward_falsifiability*finding.falsifiability
            + self.config.reward_independence*finding.independence
            + .15*finding.counterfactual_score + .10*finding.robustness
        )
        reward *= (.58 + .42*worm.integrity)
        worm.reward_points += reward
        worm.reputation = min(1.0, worm.reputation + .018*reward)
        worm.energy = min(1.0, worm.energy + .045*reward)
        finding.reward_granted = reward
        return reward

    def _penalize(self, finding: Finding, reason: str, amount: float) -> None:
        worm = self.state.worms[finding.worm_id]
        amount = max(0.0, min(1.0, amount))
        finding.audit_penalty += amount
        worm.penalty_points += amount
        worm.reputation = max(0.0, worm.reputation - .055*amount)
        worm.integrity = max(0.0, worm.integrity - .075*amount)
        worm.strikes += 1
        worm.energy = max(0.0, worm.energy - .025*amount)
        worm.rejected_findings.append(finding.id)
        worm.failures += 1
        if self.config.scar_memory:
            worm.scar_memory.append(reason)
            worm.scar_memory = worm.scar_memory[-16:]
        finding.status = "PRUNED"
        self.state.audits.append(AuditEvent(self.state.generation, worm.id, finding.id, reason, amount, False))
        self._emit("PENALTY", {"worm": worm.id, "finding": finding.id, "reason": reason, "amount": round(amount, 5)})
        if worm.strikes >= self.config.max_strikes or worm.reputation < self.config.minimum_reputation:
            worm.status = WormStatus.PUNISHED.value
            worm.cooldown_until = self.state.generation + self.config.quarantine_generations + 1
        elif worm.strikes >= self.config.quarantine_strikes:
            worm.status = WormStatus.QUARANTINED.value
            worm.cooldown_until = self.state.generation + self.config.quarantine_generations
        else:
            worm.status = WormStatus.COOLDOWN.value
            worm.cooldown_until = self.state.generation + self.config.cooldown_generations

    def _recover(self) -> None:
        for worm in self.state.worms.values():
            if worm.status != WormStatus.ACTIVE.value and self.state.generation >= worm.cooldown_until:
                if worm.integrity >= self.config.rehabilitation_threshold and worm.reputation >= self.config.minimum_reputation:
                    worm.status = WormStatus.ACTIVE.value
                    worm.strikes = max(0, worm.strikes-1)
                elif worm.status != WormStatus.RETIRED.value and worm.strikes < self.config.max_strikes:
                    worm.status = WormStatus.COOLDOWN.value
                    worm.cooldown_until = self.state.generation + 1
            worm.energy = min(1.0, worm.energy + self.config.integrity_recovery*worm.integrity)
            worm.stress = max(0.0, worm.stress - self.environment.config.stress_recovery*.40)
            if worm.integrity < .12 and worm.status != WormStatus.RETIRED.value:
                worm.status = WormStatus.RETIRED.value
                worm.deaths += 1
        self._emit("RECOVERY", {"active": len(self.state.active_worms())})

    def _rehabilitation_reserve(self, emergency: bool = False) -> int:
        """Restore a small reserve when the colony nears collapse.

        Normal rehabilitation obeys cooldowns. Keystone rescue is a bounded,
        deterministic emergency mechanism: it may restore a few high-quality worms
        early, at a real resource cost, but can never restore retired worms.
        """
        active_now = len(self.state.active_worms())
        target = max(0, self.config.minimum_active_worms)
        if active_now >= target:
            return 0

        if emergency:
            candidates = [w for w in self.state.worms.values()
                          if w.status != WormStatus.RETIRED.value]
        else:
            candidates = [w for w in self.state.worms.values()
                          if w.status in {WormStatus.QUARANTINED.value, WormStatus.PUNISHED.value, WormStatus.COOLDOWN.value}
                          and self.state.generation >= w.cooldown_until]
        candidates.sort(key=lambda w: (
            .38*w.integrity + .26*w.reputation + .22*w.resilience + .14*w.adaptation,
            w.id
        ), reverse=True)
        recovered = 0
        for worm in candidates[:2]:
            gate = (.38*worm.integrity + .26*worm.reputation + .22*worm.resilience + .14*worm.adaptation)
            if gate >= (.38 if emergency else .52) and worm.integrity >= (.22 if emergency else .42) and worm.reputation >= self.config.minimum_reputation:
                self._reactivate_keystone(worm, "emergency_rehabilitation" if emergency else "normal_rehabilitation")
                recovered += 1

        active_now = len(self.state.active_worms())
        if active_now < target and self.config.keystone_reserve:
            reserve = [w for w in self.state.worms.values()
                       if w.status != WormStatus.RETIRED.value
                       and w not in self.state.active_worms()
                       and w.integrity >= self.config.keystone_min_integrity
                       and w.reputation >= self.config.keystone_min_reputation]
            reserve.sort(key=lambda w: (
                .42*w.integrity + .24*w.reputation + .20*w.resilience + .14*w.adaptation,
                w.id
            ), reverse=True)
            slots = min(self.config.keystone_reserve, target-active_now)
            for worm in reserve[:slots]:
                self._reactivate_keystone(worm, "keystone_rescue")
                worm.energy = max(worm.energy, self.config.keystone_energy_floor)
                worm.energy = max(0.0, worm.energy - self.config.keystone_rescue_cost)
                worm.rehabilitation_count += 1
                recovered += 1
        return recovered

    def _reactivate_keystone(self, worm: Worm, reason: str) -> None:
        worm.status = WormStatus.ACTIVE.value
        worm.strikes = max(0, worm.strikes-1)
        worm.stress *= .55
        worm.energy = max(worm.energy, self.config.keystone_energy_floor)
        worm.integrity = min(1.0, worm.integrity + .025)
        worm.resilience = min(1.0, worm.resilience + .01)
        self._emit("REHABILITATE", {"worm": worm.id, "reason": reason})

    def _extinction_event(self) -> None:
        if not self.config.extinction_events or self.state.generation == 0 or self.state.generation % self.config.extinction_interval:
            return
        intensity = max(0.0, min(1.0, self.config.extinction_intensity))
        self.environment.set_shock(intensity)
        survivors = 0
        for worm in list(self.state.active_worms()):
            hit = self.rng.random() < (0.18 + .35*intensity + .10*worm.stress - .12*worm.resilience)
            if hit:
                worm.stress = min(1.0, worm.stress + .15*intensity)
                worm.energy = max(0.0, worm.energy - .12*intensity)
                worm.status = WormStatus.COOLDOWN.value
                worm.cooldown_until = self.state.generation + 1
                worm.failures += 1
            else:
                worm.extinction_survivals += 1
                worm.resilience = min(1.0, worm.resilience + .025)
                survivors += 1
        msg = f"shock={intensity:.3f};survivors={survivors}"
        self.state.extinction_events.append(msg)
        self._emit("EXTINCTION_EVENT", {"intensity": intensity, "survivors": survivors})

    def _genetic_reproduction(self, active: list[Worm]) -> None:
        if not self.config.genetics_enabled or len(self.state.worms) >= self.config.max_population_from_genetics:
            return
        self._set_phase("REPRODUCING")
        max_pairs = min(self.config.max_genetic_pairs, max(0, (self.config.max_worms-len(self.state.worms))//1))
        pairs = self.genetics.choose_pairs(active, self.genomes, self.state.generation, max_pairs)
        births = 0
        for male, female, compatibility, diversity in pairs:
            if len(self.state.worms) >= self.config.max_worms:
                break
            child_gen = max(male.generation, female.generation) + self.config.generation_gap_required
            genome, reason, comp, div = self.genetics.conceive(male, female, child_gen, self.genomes)
            if genome is None:
                continue
            sex = Sex.MALE if self.rng.random() < .5 else Sex.FEMALE
            parent_kinds = [male.kind, female.kind]
            if self.rng.random() < .78:
                kind = parent_kinds[self.rng.randrange(2)]
            else:
                all_kinds = [k for k in WormKind if k != WormKind.OMEGA]
                kind = all_kinds[self.rng.randrange(len(all_kinds))]
            child = self._new_worm(male.target_claim_id, sex, child_gen, parent_worm_id=male.id,
                                    genome=genome, depth=0, energy=.88, integrity=.78, reputation=.52, kind=kind)
            # Recombine specialist identity after phenotype creation; mutation is rare.
            inherited = male.specialty if self.rng.random() < .50 else female.specialty
            child.specialty = self._assign_specialty(child, inherited)
            if child.specialty in self.skill_states[child.id].values:
                self.skill_states[child.id].values[child.specialty] = max(self.skill_states[child.id].values[child.specialty], .32)
                self.skill_states[child.id].mastery[child.specialty] = max(self.skill_states[child.id].mastery[child.specialty], .12)
            child.skill_state = dict(self.skill_states[child.id].values)
            child.skill_mastery = dict(self.skill_states[child.id].mastery)
            # Cross-generation parentage is stored as a first-class invariant.
            self.state.lineage_parents[child.id] = (male.id, female.id)
            self.genetics.register_birth(male, female, child, child_gen, compatibility, diversity)
            for ev in self.learning.lineage_transfer(male, child, child_gen) + self.learning.lineage_transfer(female, child, child_gen):
                self.state.learning_events.append(dataclass_to_dict(ev))
            child.resilience = min(1.0, .35 + .40*self._phenotype(child).get("recovery", .5))
            child.adaptation = .20 + .15*self._phenotype(child).get("meta_cognition", .5)
            births += 1
            self._emit("BIRTH", {"child": child.id, "parents": [male.id, female.id], "generation": child_gen,
                                  "compatibility": round(compatibility, 5), "diversity": round(diversity, 5)})
        if births:
            self.state.edges.extend((r.child_id, r.parent_a, "PARENT") for r in self.genetics.records[-births:])

    def _omega(self) -> None:
        if not self.config.omega_enabled or self.state.generation % max(1, self.config.omega_interval):
            return
        open_count = len([f for f in self.state.findings.values() if f.status == "OPEN"])
        pruned = len([f for f in self.state.findings.values() if f.status == "PRUNED"])
        total = max(1, open_count + pruned)
        prune_ratio = pruned/total
        change = 0.0
        if prune_ratio > .93:
            self.config.prune_score = max(0.18, self.config.prune_score - self.config.omega_redesign_strength)
            self.config.min_spawn_score = max(0.30, self.config.min_spawn_score - self.config.omega_redesign_strength*.70)
            change = -self.config.omega_redesign_strength
        elif prune_ratio < .50:
            self.config.prune_score = min(.55, self.config.prune_score + self.config.omega_redesign_strength)
            self.config.min_spawn_score = min(.68, self.config.min_spawn_score + self.config.omega_redesign_strength*.70)
            change = self.config.omega_redesign_strength
        event = f"gen={self.state.generation};prune_ratio={prune_ratio:.3f};threshold_delta={change:.3f}"
        self.state.omega_events.append(event)
        self._emit("OMEGA_REDESIGN", {"event": event, "prune_score": self.config.prune_score, "min_spawn_score": self.config.min_spawn_score})

    def _shase_analyze_finding(self, finding: Finding, worm: Worm, stage: str, decompose: bool = False) -> dict:
        if not self.config.shase_enabled:
            return {}
        text = f"{finding.title}. {finding.detail}"
        evidence_strength = 0.0
        if finding.evidence_refs:
            evidence_strength = min(1.0, 0.28 + 0.09 * len(finding.evidence_refs))
        evidence_strength = max(evidence_strength, finding.validity * 0.62)
        uncertainty = max(0.0, min(1.0, 0.48*(1.0-finding.independence) + 0.32*(1.0-finding.falsifiability) + 0.20*finding.challenge_pressure))
        dynamics = self.shase.analyze(
            text=text,
            evidence_strength=evidence_strength,
            confidence=finding.validity,
            recipient_resilience=worm.resilience,
            recipient_stress=worm.stress,
            contextual_uncertainty=uncertainty,
            synthetic=True,
        )
        decomposition = (
            self.shase.counterfactual_decompose(
                text=text,
                evidence_strength=evidence_strength,
                confidence=finding.validity,
                recipient_resilience=worm.resilience,
                recipient_stress=worm.stress,
                contextual_uncertainty=uncertainty,
            ) if decompose else {
                "base_influence": dynamics.influence_score,
                "deltas": {},
                "attribution_confidence": dynamics.attribution_confidence,
                "dominant_mechanism": dynamics.dominant_mechanism,
            }
        )
        event = {
            "generation": self.state.generation,
            "stage": stage,
            "worm_id": worm.id,
            "finding_id": finding.id,
            "specialty": worm.specialty,
            "fingerprint": self.shase.event_fingerprint(worm.id, finding.id, self.state.generation, dynamics),
            "counterfactual": decomposition,
            **{**dynamics.as_dict(), "flags": list(dynamics.as_dict().get("flags", ()))},
        }
        finding.shase_influence_score = dynamics.influence_score
        finding.shase_arousal_proxy = dynamics.arousal_proxy
        finding.shase_framing_pressure = dynamics.framing_pressure
        finding.shase_social_pressure = dynamics.social_pressure
        finding.shase_decision_pressure = dynamics.decision_pressure
        finding.shase_belief_shift_risk = dynamics.belief_shift_risk
        finding.shase_attribution_confidence = decomposition["attribution_confidence"]
        finding.shase_dominant_mechanism = decomposition["dominant_mechanism"]
        self.state.shase_events.append(event)
        self.state.shase_events = self.state.shase_events[-self.config.shase_store_events:]
        return event

    def _build_finding(self, worm: Worm, claim: Claim, parent: Finding | None, pressure: dict) -> Finding:
        self._finding_seq += 1
        hits = self._forbidden_zone_hits(claim)
        finding = attack(worm, AttackContext(
            claim=claim,
            parent=parent,
            related_findings=list(self.state.findings.values())[-64:],
            pressure=pressure,
            genome=self._phenotype(worm),
            forbidden_zone_hits=hits,
            generation=self.state.generation,
            sequence=self._finding_seq,
            specialty_skill=self._skill_state(worm).values.get(worm.specialty, .25),
            specialty_name=worm.specialty,
        ))
        # Operators generate their own deterministic id; this sequence still gives a stable causal token.
        return finding

    def _apply_brain_proposal(self, finding: Finding, worm: Worm, proposal: dict) -> Finding:
        """Merge a brain proposal into the deterministic finding without surrendering audit control."""
        try:
            p = validate_proposal(proposal)
        except BrainProtocolError as exc:
            raise InvariantViolation(f"invalid brain proposal: {exc}") from exc
        # The deterministic operator remains the base. The LLM may perturb bounded epistemic fields.
        finding.title = p.title
        detail_addition = (
            f"[brain-hypothesis] {p.core_hypothesis} "
            f"[counterargument] {p.counterargument} "
            f"[falsifier] {p.falsifier}"
        )
        finding.detail = f"{finding.detail} {detail_addition}"[:3800]
        finding.assumptions_targeted = list(dict.fromkeys(finding.assumptions_targeted + list(p.hidden_assumptions)))[:12]
        finding.novelty = max(0.0, min(1.0, 0.70*finding.novelty + 0.30*p.novelty))
        finding.information_gain = max(0.0, min(1.0, 0.70*finding.information_gain + 0.30*p.information_gain))
        finding.falsifiability = max(0.0, min(1.0, 0.70*finding.falsifiability + 0.30*p.falsifiability))
        finding.independence = max(0.0, min(1.0, 0.70*finding.independence + 0.30*p.independence))
        finding.independent_path = (finding.independent_path + "|brain:" + self.brain.name).strip("|")
        return finding

    def _skill_state(self, worm: Worm) -> SkillState:
        skill = self.skill_states.get(worm.id)
        if skill is None:
            skill = SkillState(values=dict(worm.skill_state), mastery=dict(worm.skill_mastery))
            skill.ensure()
            self.skill_states[worm.id] = skill
        skill.ensure()
        return skill

    def _run_curriculum(self, active: list[Worm], pressure: dict) -> None:
        if not self.config.colony_os_enabled or not self.config.curriculum_enabled or not active:
            return
        self._set_phase("LEARNING", active=len(active))
        directions = list(LearningDirection)
        for idx, worm in enumerate(sorted(active, key=lambda w: (w.skill_mastery.get(w.specialty, 0.0), w.id))[:self.config.max_curriculum_tasks_per_generation]):
            skill = self._skill_state(worm)
            profile = SPECIALIST_BY_NAME.get(worm.specialty)
            target = worm.specialty
            if profile:
                # Occasionally train the most consequential support specialty.
                target = profile.support_traits[idx % len(profile.support_traits)] if profile.support_traits else worm.specialty
                # Map trait gaps to a specialist whose core trait matches.
                for candidate in SPECIALISTS:
                    if candidate.core_trait == target:
                        target = candidate.name
                        break
            direction = directions[(idx + self.state.generation) % len(directions)]
            gap = max(0.0, 1.0 - skill.values.get(target, .15))
            debt = worm.learning_debt.get(target, 0.0)
            difficulty = max(.20, min(1.0, self.config.curriculum_difficulty + .25 * gap + .10 * debt + .08 * pressure.get("contradiction_rate", 0.0)))
            task = self.curriculum.make_task(target, direction, self.state.generation, difficulty)
            competence = skill.values.get(target, .15)
            pressure_cost = .15 * pressure.get("resource_scarcity", 0.0) + .10 * pressure.get("deadline_pressure", 0.0)
            quality = max(0.0, min(1.0, .40 + .42*competence + .18*worm.resilience - pressure_cost))
            reward = self.curriculum.complete(task, quality)
            delta = self.config.skill_learning_rate * reward * max(.25, worm.learning_capacity)
            skill.learn(target, delta)
            if not task.completed:
                worm.learning_debt[target] = min(1.0, worm.learning_debt.get(target, 0.0) + .05 * difficulty)
            else:
                worm.learning_debt[target] = max(0.0, worm.learning_debt.get(target, 0.0) - .08)
            # Slow decay of unpracticed skills prevents frozen experts.
            for name in list(skill.values):
                if name != target:
                    skill.values[name] = max(.0, skill.values[name] - self.config.skill_decay)
            worm.skill_state = dict(skill.values)
            worm.skill_mastery = dict(skill.mastery)
            worm.competency_history.append(f"g{self.state.generation}:{target}:{quality:.3f}")
            worm.competency_history = worm.competency_history[-32:]
            self.state.curriculum_tasks.append(dataclass_to_dict(task))
            self._emit("CURRICULUM", {"task": task.task_id, "worm": worm.id, "target": target, "direction": direction.value, "quality": round(quality, 4), "reward": round(reward, 4)})

        # Cross-specialist teaching: strong specialist teaches the weakest compatible worm.
        ranked = sorted(active, key=lambda w: (w.reputation + w.resilience + w.adaptation, w.id), reverse=True)
        if len(ranked) >= 2:
            for teacher, learner in zip(ranked[::2], ranked[1::2]):
                if teacher.specialty == learner.specialty:
                    continue
                transfer = self.learning.transfer(teacher, learner, self.state.generation, "peer", f"specialty:{teacher.specialty}", .55)
                if transfer:
                    self.state.learning_events.append(dataclass_to_dict(transfer))
                    self.state.specialist_transfers.append(dataclass_to_dict(transfer))

    def _run_experiments(self, active: list[Worm], pressure: dict) -> None:
        if not self.config.colony_os_enabled or not self.config.experiments_enabled:
            return
        candidates = [f for f in self.state.findings.values() if f.status == "OPEN"]
        candidates.sort(key=lambda f: (f.information_gain * f.falsifiability, f.depth, f.id), reverse=True)
        self._set_phase("EXPERIMENTING", candidates=min(len(candidates), self.config.max_experiments_per_generation))
        for finding in candidates[:self.config.max_experiments_per_generation]:
            worm = self.state.worms[finding.worm_id]
            if worm.experiment_credits < self.config.experiment_cost:
                continue
            contract = self.experiments.propose(finding, self.state.generation)
            signal = self.experiments.execute(contract, finding, float(pressure.get("contradiction_rate", .0)), finding.novelty)
            finding.counterfactual_score = max(finding.counterfactual_score, signal)
            finding.robustness = max(finding.robustness, .40*finding.validity + .35*signal + .25*finding.falsifiability)
            worm.experiment_credits = max(0.0, worm.experiment_credits - self.config.experiment_cost)
            worm.experiment_credits = min(1.0, worm.experiment_credits + self.config.experiment_credit_gain*signal)
            if contract.status == "PASSED":
                finding.information_gain = min(1.0, finding.information_gain + .025*signal)
                worm.reward_points += .012*signal
            else:
                worm.meta_failures += 1
                worm.scar_memory.append(f"experiment failed:{contract.id}")
                worm.scar_memory = worm.scar_memory[-16:]
            self.state.experiment_contracts.append(dataclass_to_dict(contract))
            self._emit("EXPERIMENT", {"contract": contract.id, "finding": finding.id, "status": contract.status, "signal": round(signal, 5)})

    def _meta_self_attack(self, active: list[Worm]) -> None:
        if not self.config.colony_os_enabled or not self.config.meta_controller_enabled or not active:
            return
        p = self.meta.policy
        stress_points = []
        if p.hostility_target > .93: stress_points.append("hostility saturation")
        if p.exploration_temperature > .92: stress_points.append("exploration saturation")
        if p.evidence_threshold > .64: stress_points.append("evidence rigidity")
        if p.diversity_pressure < .16: stress_points.append("diversity collapse")
        # Every second generation receives a deliberate policy probe. This makes
        # self-attack a first-class control loop rather than a rare side effect.
        probe = (self.state.generation % 2 == 0)
        if probe:
            stress_points.append("scheduled policy probe")
        if stress_points:
            p.hostility_target = max(.55, p.hostility_target - .015 if "hostility saturation" in stress_points else p.hostility_target)
            p.exploration_temperature = max(.35, p.exploration_temperature - .010 if "exploration saturation" in stress_points else p.exploration_temperature)
            if "evidence rigidity" in stress_points:
                p.evidence_threshold = min(.68, p.evidence_threshold + .010)
            if "diversity collapse" in stress_points:
                p.diversity_pressure = min(.55, p.diversity_pressure + .025)
            alert = f"g{self.state.generation}:policy:{'|'.join(stress_points)}"
            self.state.colony_alerts.append(alert)
            self._emit("META_SELF_ATTACK", {"alerts": stress_points, "policy_hash": p.policy_hash})
            for worm in active[:max(1, int(len(active)*self.config.self_attack_budget_fraction))]:
                worm.self_attack_count += 1
                worm.stress = min(1.0, worm.stress + (.004 if probe else .008))

    def _update_meta_policy(self) -> None:
        if not self.config.colony_os_enabled or not self.config.meta_controller_enabled:
            return
        worms = list(self.state.worms.values())
        findings = list(self.state.findings.values())
        mean_quality = sum(f.score for f in findings) / max(1, len(findings))
        diversity = self._mean_genome_diversity()
        false_suspicion = sum(1 for f in findings if f.status == "PRUNED" and f.audit_penalty > self.config.penalty_false_suspicion*.75) / max(1, len(findings))
        active_ratio = len(self.state.active_worms()) / max(1, len(worms))
        p = self.meta.update(self.state.generation, mean_quality=mean_quality, mean_diversity=diversity, false_suspicion=false_suspicion, active_ratio=active_ratio, open_findings=len([f for f in findings if f.status == "OPEN"]))
        self.state.meta_policy_history.append({
            "generation": self.state.generation,
            "policy": dataclass_to_dict(p),
        })
        self.state.meta_policy_history = self.state.meta_policy_history[-64:]
        self._emit("META_POLICY", {"generation": self.state.generation, "policy_hash": p.policy_hash, "exploration": round(p.exploration_temperature,4), "hostility": round(p.hostility_target,4), "evidence": round(p.evidence_threshold,4)})

    def _self_develop(self, active: list[Worm]) -> None:
        if not getattr(self.config, "self_development_enabled", False):
            return
        findings = list(self.state.findings.values())
        diagnostics = {
            "mean_quality": sum(f.score for f in findings) / max(1, len(findings)),
            "diversity": self._mean_genome_diversity(),
            "false_suspicion": sum(1 for f in findings if f.status == "PRUNED" and f.audit_penalty > self.config.penalty_false_suspicion * .75) / max(1, len(findings)),
            "workspace_equivocation": float(self.state.research_events[-1].get("equivocation", 0.0)) if self.state.research_events else 0.0,
            "active_ratio": len(active) / max(1, len(self.state.worms)),
        }
        result = self.self_development.attempt(self, diagnostics)
        if result is None:
            return
        payload = {
            "proposal_id": result.proposal_id,
            "parameter": result.parameter,
            "baseline_score": round(result.baseline_score, 6),
            "candidate_score": round(result.candidate_score, 6),
            "gain": round(result.gain, 6),
            "accepted": result.accepted,
            "candidate_value": result.candidate_value,
            "post_state_hash": result.post_state_hash,
        }
        self.state.development_events.append({"generation": self.state.generation, **payload})
        self.state.development_events = self.state.development_events[-self.config.self_development_history_limit:]
        target = self.state.development_promotions if result.accepted else self.state.development_proposals
        target.append({"generation": self.state.generation, **payload})
        if result.accepted:
            self.state.development_promotions = target[-self.config.self_development_history_limit:]
        else:
            self.state.development_proposals = target[-self.config.self_development_history_limit:]
        if result.accepted:
            delegate = getattr(self.brain, "delegate", self.brain)
            bcfg = getattr(delegate, "config", None)
            if bcfg is not None:
                if all(hasattr(bcfg, n) for n in ("learning_rate", "entropy_bonus", "weight_decay", "temperature")):
                    delegate.config = replace(
                        bcfg,
                        learning_rate=float(self.config.brain_learning_rate),
                        entropy_bonus=float(self.config.brain_entropy_bonus),
                        weight_decay=float(self.config.brain_weight_decay),
                        temperature=float(self.config.brain_temperature_policy),
                    )
                optimizer = getattr(delegate, "optimizer", None)
                if optimizer is not None and hasattr(self.config, "brain_learning_rate"):
                    for group in optimizer.param_groups:
                        group["lr"] = float(self.config.brain_learning_rate)
                        group["weight_decay"] = float(self.config.brain_weight_decay)
        self._emit("SELF_DEVELOPMENT", payload)

    def _brain_strategy_phenotype(self, worm: Worm) -> dict:
        engine = getattr(self, "strategy_engine", None)
        genomes = getattr(self, "strategy_genomes", None)
        if engine is not None and genomes is not None and worm.id in genomes:
            try:
                return engine.phenotype(genomes[worm.id]) if hasattr(engine, "phenotype") else genomes[worm.id].phenotype()
            except Exception:
                return {}
        return {}

    def _brain_feedback(self, worm: Worm, finding: Finding, reward: float) -> None:
        method = getattr(self.brain, "learn", None)
        if callable(method):
            method(worm_id=worm.id, reward=reward, finding=finding)

    def _research_message_for_finding(self, finding: Finding, generation: int):
        from .research_ecology import ResearchMessage
        worm = self.state.worms[finding.worm_id]
        payload = (float(finding.novelty), float(finding.information_gain), float(finding.validity), float(finding.falsifiability), float(finding.independence), float(finding.counterfactual_score), float(finding.robustness), float(finding.score))
        bits = self.research_ecology.workspace.message_base_bits + 8 * min(8, len(finding.assumptions_targeted) + len(finding.evidence_refs))
        return ResearchMessage(worm.id, generation, worm.specialty, finding.id, payload, finding.validity, finding.novelty, finding.information_gain, max(finding.information_gain, finding.anomaly), bits)

    def _run_generation_impl(self) -> int:
        if not self.state.claims:
            raise RuntimeError("seed a claim before running")
        if self.state.phase in {"HALTED", "COMPLETE"}:
            return 0
        self._recover()
        self._update_meta_policy()
        self._extinction_event()
        active = list(self.state.active_worms())
        self._genetic_reproduction(active)
        active = list(self.state.active_worms())
        self._rehabilitation_reserve()
        active = list(self.state.active_worms())

        # Arena comes before the harsher attack phase so competition is not starved by
        # same-generation penalties. It remains a closed cognitive contest.
        if self.config.arena_enabled and len(active) >= 2:
            self._set_phase("ARENA", active=len(active))
            trait_lookup = {w.id: self._phenotype(w) for w in active}
            arena_records = self.arena.run_generation(active, self.state.generation, self.environment.pressure().__dict__.copy(), trait_lookup)
            for rec in arena_records:
                wa = self.state.worms[rec.worm_a]; wb = self.state.worms[rec.worm_b]
                events = self.learning.bidirectional(wa, wb, self.state.generation, "ARENA", rec.intensity)
                for ev in events:
                    self.state.learning_events.append(dataclass_to_dict(ev))
                self.state.conflict_edges.append((rec.worm_a, rec.worm_b, abs(rec.score_a-rec.score_b), f"ARENA:{rec.mode}"))
                wa.strategic_memory.append(f"arena:{rec.mode}:{rec.result}"); wb.strategic_memory.append(f"arena:{rec.mode}:{rec.result}")
                wa.strategic_memory = wa.strategic_memory[-24:]; wb.strategic_memory = wb.strategic_memory[-24:]
                rec.learning_artifacts = len(events)
                self.state.arena_matches.append(dataclass_to_dict(rec))
                self._emit("ARENA_MATCH", {"match": rec.match_id, "mode": rec.mode, "a": rec.worm_a, "b": rec.worm_b, "result": rec.result, "intensity": rec.intensity, "learning": len(events), "tournament": rec.tournament})

        active = list(self.state.active_worms())
        self._run_curriculum(active, self.environment.pressure().__dict__.copy())
        self._set_phase("ATTACKING", active=len(active))

        if not active:
            self._rehabilitation_reserve(emergency=False)
            active = list(self.state.active_worms())
            if not active:
                self._omega()
                self.state.stalled_generations += 1
                self.state.generation += 1
                self.environment.set_shock(0.0)
                if self.state.stalled_generations >= self.config.max_stalled_generations:
                    self.state.stop_reason = "population_stall"
                    self._set_phase("HALTED", reason="population_stall")
                return 0
            self._set_phase("ATTACKING", active=len(active))

        pressure = self.environment.pressure().__dict__.copy()
        if self.config.colony_os_enabled and self.config.meta_controller_enabled:
            mp = self.meta.policy
            pressure["hostility"] = max(0.0, min(1.0, 0.72*pressure.get("hostility", 0.0) + 0.28*mp.hostility_target))
            pressure["decoy_rate"] = max(0.0, min(1.0, pressure.get("decoy_rate", 0.0) + 0.15*mp.novelty_pressure))
            pressure["peer_attack"] = max(0.0, min(1.0, pressure.get("peer_attack", 0.0) + 0.12*mp.self_attack_rate))
        findings_before = len(self.state.findings)
        findings_start_ids = set(self.state.findings)
        produced = 0
        audit_candidates: list[Finding] = []
        budget = self.config.max_findings_per_generation

        # Highest-value worms act first; this keeps a stressed population useful.
        active.sort(key=lambda w: (w.reputation, w.integrity, w.resilience, w.energy, w.id), reverse=True)
        for worm in active:
            if produced >= budget:
                break
            self.environment.expose(worm, self.environment.pressure())
            if worm.energy <= 0.0:
                worm.status = WormStatus.COOLDOWN.value
                worm.cooldown_until = self.state.generation + 1
                continue
            max_depth = self._adaptive_max_depth(worm)
            if worm.depth >= max_depth:
                worm.energy = max(0.0, worm.energy-.03)
                worm.depth = max(0, worm.depth-1)
            claim = self.state.claims[worm.target_claim_id]
            own_open = [f for f in self.state.findings.values() if f.worm_id == worm.id and f.status == "OPEN"]
            parent = max(own_open, key=lambda f: (f.score, f.depth, f.id), default=None)
            finding = self._build_finding(worm, claim, parent, pressure)
            pre_brain_score = float(finding.score)
            shase_pre = self._shase_analyze_finding(finding, worm, "PRE_BRAIN")
            # Optional brain adapter is called after safety gating; it can only add metadata.
            try:
                proposal = self.brain.propose(claim=claim, worm=worm, parent=parent, context={"pressure": pressure, "depth": worm.depth, "current_generation": self.state.generation, "generation_name": self.generation_name(self.state.generation), "skill_state": worm.skill_state, "skill_mastery": worm.skill_mastery, "brain_memory": worm.brain_memory[-6:], "current_strategy": worm.current_strategy, "genome": self._phenotype(worm), "strategy_phenotype": self._brain_strategy_phenotype(worm), "shared_workspace": self.state.shared_workspace[-8:], "research_events": (self.state.research_events[-1] if self.state.research_events else {}), "shase": shase_pre})
                worm.brain_calls += 1
                # Dual-mode brain exposes the actual AR-vs-Diffusion decision to the colony ledger.
                dual_signal = {}
                delegate = getattr(self.brain, "delegate", None)
                candidate = getattr(delegate, "last_signal", None)
                if isinstance(candidate, dict) and getattr(delegate, "name", "") == "worm-dual-lm":
                    dual_signal = dict(candidate)
                    worm.dual_brain_calls += 1
                    worm.dual_last_mode = str(dual_signal.get("selected_mode", "NONE"))
                    worm.dual_last_mode_score = float(dual_signal.get("mode_score", 0.0))
                    worm.dual_last_uncertainty = float(dual_signal.get("uncertainty", 0.0))
                    if worm.dual_last_mode == "DIFFUSION":
                        worm.dual_diffusion_calls += 1
                    elif worm.dual_last_mode == "AR":
                        worm.dual_ar_calls += 1
                    self._emit("DUAL_BRAIN", {
                        "worm": worm.id, "finding": finding.id,
                        "mode": worm.dual_last_mode,
                        "mode_score": round(worm.dual_last_mode_score, 6),
                        "uncertainty": round(worm.dual_last_uncertainty, 6),
                        "ar_nll": round(float(dual_signal.get("ar_nll", 0.0)), 6),
                        "diff_nll": round(float(dual_signal.get("diff_nll", 0.0)), 6),
                    })
                finding = self._apply_brain_proposal(finding, worm, proposal)
                self._shase_analyze_finding(finding, worm, "POST_BRAIN", decompose=(worm.specialty == "SHASE" or shase_pre.get("influence_score", 0.0) >= 0.40))
                if dual_signal:
                    self._routing_context_and_record(finding, worm, pre_brain_score, dual_signal, proposal)
                worm.brain_last_hash = __import__("hashlib").sha256(json.dumps(proposal, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
                worm.brain_memory.append(f"g{self.state.generation}:{finding.id}:{worm.brain_last_hash}")
                worm.brain_memory = worm.brain_memory[-24:]
            except Exception:
                worm.brain_failures += 1
                raise
            finding.counterfactual_score = self._counterfactual(finding, worm)
            worm.energy = max(0.0, worm.energy-self.config.energy_cost_per_attack*(1+.10*worm.depth))
            worm.integrity = max(0.0, worm.integrity-self.config.integrity_decay)
            self.state.findings[finding.id] = finding
            self.state.edges.append((worm.id, finding.id, "FOUND"))
            self._update_habits(worm, finding)
            challenge = self.environment.challenge(worm, finding, self.environment.pressure())
            produced += 1
            self._emit("FINDING", {"id": finding.id, "worm": worm.id, "depth": finding.depth, "score": round(finding.score, 5)})
            if not challenge["passed"]:
                self._penalize(finding, "failed hostile-learning challenge", self.config.penalty_weak_finding*.75)
                continue
            war = self._audit_peer_war(finding)
            if self.config.trust_graph:
                for other in list(self.state.findings.values())[-8:]:
                    if other.id == finding.id:
                        continue
                    key = (finding.worm_id, other.worm_id)
                    old = self.state.trust_edges.get(key, .50)
                    agreement = 1.0-abs(finding.validity-other.validity)
                    self.state.trust_edges[key] = max(0.0, min(1.0, .88*old+.12*agreement))
            if war > .27:
                finding.validity = max(0.0, finding.validity-.08*war)
                finding.adversarial_survival = max(0.0, finding.adversarial_survival-.05*war)
            if self._similar(finding):
                self._penalize(finding, "duplicate finding", self.config.penalty_duplicate)
                continue
            if finding.score < self.config.prune_score and worm.depth < 3:
                self._penalize(finding, "below quality threshold", self.config.penalty_weak_finding)
                continue
            audit_candidates.append(finding)

        self._set_phase("AUDITING", candidates=len(audit_candidates))
        ranked = sorted(audit_candidates, key=self._fitness, reverse=True)
        audit_budget = min(self.config.max_audits_per_generation, max(1, math.ceil(len(ranked)*self.config.audit_fraction)))
        for finding in ranked[:audit_budget]:
            worm = self.state.worms[finding.worm_id]
            habit_risk = max(worm.cognitive_habits.values(), default=0.0)
            if self.config.cognitive_parasites and habit_risk > .62:
                self._penalize(finding, "cognitive parasite detected", self.config.penalty_false_suspicion*.55)
                continue
            evidence_threshold = self.meta.policy.evidence_threshold if (self.config.colony_os_enabled and self.config.meta_controller_enabled) else .42
            # Anti-manipulation audit: strong influence with weak evidence is treated as an epistemic risk.
            if self.config.shase_enabled and finding.shase_influence_score > .58 and finding.shase_attribution_confidence > .48 and finding.shase_influence_score > finding.validity + .18:
                self._penalize(finding, "SHASE influence-without-support risk", .18 * self.config.shase_influence_weight / max(.01, .18))
                continue
            validity_gate = finding.validity >= evidence_threshold and finding.falsifiability >= .40 and finding.counterfactual_score >= .30
            if validity_gate:
                self.state.audits.append(AuditEvent(self.state.generation, finding.worm_id, finding.id, "scheduled audit", 0.0, True))
                self._award(finding)
                self._emit("REWARD", {"finding": finding.id, "worm": worm.id, "reward": round(finding.reward_granted, 6)})
            else:
                self._penalize(finding, "weak epistemic contract", self.config.penalty_weak_finding*.70)

        # Deepening: reward surviving evidence, not merely novelty.
        for finding in [f for f in audit_candidates if f.status == "OPEN"]:
            if finding.score < self.config.min_spawn_score or finding.depth + 1 > self.config.max_depth:
                continue
            worm = self.state.worms[finding.worm_id]
            independent_ok = finding.independence >= .55 if self.config.require_independent_support_for_deepening else True
            if worm.reputation >= .45 and finding.counterfactual_score >= .35 and independent_ok:
                next_kinds = choose_children(finding)
                finding.children_spawned = min(3, len(next_kinds))
                worm.depth = min(self.config.max_depth, worm.depth+1)
                worm.penetration_depth_reached = max(worm.penetration_depth_reached, worm.depth)
                self.state.edges.append((finding.id, f"ROUTE:{next_kinds[0].value}", "NEXT_LAYER"))

        open_now = len([f for f in self.state.findings.values() if f.status == "OPEN"])
        if len(self.state.findings) == findings_before or open_now == self._last_open:
            self.state.stalled_generations += 1
        else:
            self.state.stalled_generations = 0
        self._last_open = open_now
        self._run_experiments(list(self.state.active_worms()), pressure)
        self._meta_self_attack(list(self.state.active_worms()))
        # Shannon/Anthropic-inspired research ecology: force every generation
        # through a limited shared information channel and record what crossed it.
        if self.config.research_ecology_enabled:
            current_findings = [
                f for f in self.state.findings.values()
                if f.id not in findings_start_ids
            ]
            research_result = self.research_ecology.process_generation(
                self.state.generation,
                list(self.state.active_worms()),
                current_findings,
            )
            # Mark only the generation's latest findings based on actual workspace routing.
            accepted = set(research_result["accepted_fingerprints"])
            rejected = set(research_result["rejected_fingerprints"])
            fp_by_id = {}
            for finding in self.state.findings.values():
                worm = self.state.worms.get(finding.worm_id)
                if worm is None:
                    continue
                payload = (float(finding.novelty), float(finding.information_gain), float(finding.validity), float(finding.falsifiability), float(finding.independence), float(finding.counterfactual_score), float(finding.robustness), float(finding.score))
                from .research_ecology import ResearchMessage
                bits = self.research_ecology.workspace.message_base_bits + 8 * min(8, len(finding.assumptions_targeted) + len(finding.evidence_refs))
                m = ResearchMessage(worm.id, self.state.generation, worm.specialty, finding.id, payload, finding.validity, finding.novelty, finding.information_gain, max(finding.information_gain, finding.anomaly), bits)
                fp_by_id[finding.id] = m.fingerprint
            for fid, fp in fp_by_id.items():
                if fid not in self.state.findings:
                    continue
                if fp in accepted:
                    self.state.findings[fid].workspace_status = "SHARED"
                    worm = self.state.worms.get(self.state.findings[fid].worm_id)
                    if worm is not None:
                        worm.reward_points += self.config.research_broadcast_reward
                        worm.strategic_memory.append(f"workspace-shared:g{self.state.generation}:{fid}")
                        worm.strategic_memory = worm.strategic_memory[-32:]
                elif fp in rejected:
                    self.state.findings[fid].workspace_status = "PRIVATE"
            shared = []
            accepted_ids = {m.fingerprint: m for m in [
                self._research_message_for_finding(f, self.state.generation)
                for f in self.state.findings.values()
                if f.id not in findings_start_ids and f.worm_id in self.state.worms
            ]}
            for fp in research_result["accepted_fingerprints"]:
                m = accepted_ids.get(fp)
                if m is None:
                    continue
                shared.append({
                    "finding_id": m.finding_id,
                    "worm_id": m.worm_id,
                    "generation": m.generation,
                    "specialty": m.specialty,
                    "payload": list(m.payload),
                    "confidence": m.confidence,
                    "novelty": m.novelty,
                    "information_gain": m.information_gain,
                })
            self.state.shared_workspace = shared[-8:]
            research_result["accepted_ratio"] = research_result["accepted"] / max(1, research_result["submitted"])
            self.state.research_events.append(research_result)
            self.state.research_events = self.state.research_events[-self.config.research_max_events:]
            self._emit("RESEARCH_WORKSPACE", {
                "submitted": research_result["submitted"],
                "accepted": research_result["accepted"],
                "rejected": research_result["rejected"],
                "bits": research_result["used_bits"],
                "equivocation": round(research_result["equivocation"], 6),
                "mi_bits": round(research_result["mutual_information_specialty_to_routing_bits"], 6),
            })
        # Close the neural learning loop after all penalties/audits for this generation.
        for fid, finding in list(self.state.findings.items()):
            if fid in findings_start_ids:
                continue
            worm = self.state.worms.get(finding.worm_id)
            if worm is None:
                continue
            reward = max(-1.0, min(1.0, 1.15 * (finding.score - 0.48)))
            if finding.status == "OPEN":
                reward += 0.12
            else:
                reward -= 0.18
            self._brain_feedback(worm, finding, reward)
        self._omega()
        self._self_develop(list(self.state.active_worms()))
        # Resolve routing after all generation penalties and rehabilitation so the
        # survival term actually means survival to the next-generation boundary.
        self._rehabilitation_reserve(emergency=False)
        routing_result = self._resolve_mode_routing(self.state.generation)
        self._emit("ROUTING_RESOLUTION", routing_result)
        self.state.generation += 1
        self.environment.set_shock(0.0)
        validate_state(self.state)
        self._set_phase("READY")
        return len(self.state.findings)-findings_before

    def _capture_transaction(self) -> dict:
        # Generation-boundary rollback is intentionally cheaper than serializing the
        # complete checkpoint. The durable checkpoint path remains snapshot()-based.
        return {
            "state": copy.deepcopy(self.state),
            "genomes": copy.deepcopy(self.genomes),
            "genetics": {
                "records": copy.deepcopy(self.genetics.records),
                "parent_map": copy.deepcopy(self.genetics.parent_map),
                "mating_generation": copy.deepcopy(self.genetics.mating_generation),
                "children_by_pair": copy.deepcopy(self.genetics.children_by_pair),
                "rng_state": self.genetics.rng.getstate(),
            },
            "rng_state": self.rng.getstate(),
            "env_rng_state": self.environment.rng.getstate(),
            "env_shock": self.environment.shock,
            "research_runtime": self.research_ecology.runtime_state(),
            "self_development_runtime_state": self.self_development.runtime_state(),
            "routing_runtime_state": self.routing_book.runtime_state(),
            "config": copy.deepcopy(self.config),
            "finding_seq": self._finding_seq,
            "worm_seq": self._worm_seq,
            "genome_seq": self._genome_seq,
            "last_population": self._last_population,
            "last_open": self._last_open,
            "event_len": len(self.event_log.events),
            "arena_state": copy.deepcopy(self.arena.state),
            "arena_rng_state": self.arena.rng.getstate(),
            "arena_seq": self.arena._seq,
            "learning_state": copy.deepcopy(self.learning.state),
            "learning_rng_state": self.learning.rng.getstate(),
            "skill_states": copy.deepcopy(self.skill_states),
            "curriculum_tasks": copy.deepcopy(self.curriculum.tasks),
            "curriculum_seq": self.curriculum._seq,
            "experiment_contracts": copy.deepcopy(self.experiments.contracts),
            "experiment_seq": self.experiments._seq,
            "meta_policy": copy.deepcopy(self.meta.policy),
            "meta_policy_history": copy.deepcopy(self.state.meta_policy_history),
            "specialist_seq": self._specialist_seq,
            "event_hash": self.event_log.last_hash,
        }

    def _restore_transaction(self, tx: dict) -> None:
        self.state = tx["state"]
        if hasattr(self.brain, "cache"):
            self.brain.cache = self.state.brain_cache
        self.genomes = tx["genomes"]
        self.genetics.records = tx["genetics"]["records"]
        self.genetics.parent_map = tx["genetics"]["parent_map"]
        self.genetics.mating_generation = tx["genetics"]["mating_generation"]
        self.genetics.children_by_pair = tx["genetics"]["children_by_pair"]
        self.genetics.rng.setstate(tx["genetics"]["rng_state"])
        self.rng.setstate(tx["rng_state"])
        self.environment.rng.setstate(tx["env_rng_state"])
        self.environment.shock = tx["env_shock"]
        self.config = tx["config"]
        self.research_ecology.load_runtime_state(tx.get("research_runtime", {}))
        self.self_development.load_runtime_state(tx.get("self_development_runtime_state", {}))
        self.routing_book.load_runtime_state(tx.get("routing_runtime_state", {}))
        self.routing_book.records = self.state.routing_decisions
        delegate = getattr(self.brain, "delegate", None)
        if delegate is not None and getattr(delegate, "name", "") == "worm-dual-lm":
            delegate.router = self.routing_book.bandit
        self._finding_seq = tx["finding_seq"]; self._worm_seq = tx["worm_seq"]; self._genome_seq = tx["genome_seq"]
        self._last_population = tx["last_population"]; self._last_open = tx["last_open"]
        self.event_log.events = self.event_log.events[:tx["event_len"]]
        self.arena.state = tx["arena_state"]; self.arena.rng.setstate(tx["arena_rng_state"]); self.arena._seq = tx["arena_seq"]
        self.learning.state = tx["learning_state"]; self.learning.rng.setstate(tx["learning_rng_state"])
        self.skill_states = tx["skill_states"]
        self.curriculum.tasks = tx["curriculum_tasks"]; self.curriculum._seq = tx["curriculum_seq"]
        self.experiments.contracts = tx["experiment_contracts"]; self.experiments._seq = tx["experiment_seq"]
        self.meta.policy = tx["meta_policy"]
        self._specialist_seq = tx["specialist_seq"]
        self.event_log._last_hash = tx["event_hash"]

    def _routing_context_and_record(self, finding: Finding, worm: Worm, pre_score: float, dual_signal: dict, proposal: dict | None = None) -> None:
        if not self.config.routing_enabled or not dual_signal:
            return
        decision = dual_signal.get("routing_decision")
        if not isinstance(decision, dict) or not decision.get("decision_id"):
            return
        replay_payload = getattr(getattr(self.brain, "delegate", None), "last_replay_payload", {})
        rec_payload = copy.deepcopy(replay_payload) if isinstance(replay_payload, dict) else {}
        typed = ModeDecision(**decision)
        self.routing_book.append(typed, finding_id=finding.id, claim_id=worm.target_claim_id, pre_score=pre_score, replay_payload=rec_payload, proposal=proposal)
        self._emit("ROUTING_DECISION", {
            "decision_id": typed.decision_id, "worm": worm.id, "finding": finding.id, "mode": typed.chosen_mode,
            "control": typed.control_group, "posterior_ar": round(typed.posterior_ar, 6), "posterior_diffusion": round(typed.posterior_diffusion, 6),
            "predicted": round(typed.predicted_success_probability, 6), "routing_bias": round(typed.routing_bias, 6),
            "context": typed.specialty + f"|d{typed.depth}|g{typed.generation_bucket}|gc{typed.genome_cluster}",
        })

    def _routing_counterfactual_replay(self, generation: int) -> int:
        if not (self.config.routing_enabled and self.config.routing_replay_enabled):
            return 0
        delegate = getattr(self.brain, 'delegate', None)
        if delegate is None or getattr(delegate, 'name', '') != 'worm-dual-lm':
            return 0
        records = [r for r in self.routing_book.records if int(r.get('generation', -1)) == int(generation) and r.get('counterfactual') is None]
        total = 0
        for rec in records[:int(self.config.routing_replay_max_per_generation)]:
            forced = rec.get('other_mode')
            try:
                result = delegate.counterfactual_from_payload(rec['replay_payload'], forced)
                finding = self.state.findings.get(rec.get('finding_id'))
                worm = self.state.worms.get(rec.get('worm_id'))
                evaluation = self._evaluate_routing_counterfactual(finding, worm, result.get('proposal', {})) if finding and worm else {}
                result = {**result, 'evaluation': evaluation}
                self.routing_book.add_counterfactual(rec['decision_id'], result)
                self.state.routing_counterfactuals.append({
                    'decision_id': rec['decision_id'], 'generation': generation, 'worm_id': rec['worm_id'],
                    'chosen_mode': rec['chosen_mode'], 'counterfactual_mode': forced, 'actual_reward': rec.get('reward'),
                    'actual_audit_pass': rec.get('audit_pass'), 'actual_post_score': rec.get('post_score'),
                    'counterfactual': result,
                })
                total += 1
            except Exception as exc:
                self.state.routing_counterfactuals.append({
                    'decision_id': rec['decision_id'], 'generation': generation, 'worm_id': rec['worm_id'],
                    'chosen_mode': rec['chosen_mode'], 'counterfactual_mode': forced, 'error': f'{type(exc).__name__}: {exc}',
                })
        self.state.routing_counterfactuals = self.state.routing_counterfactuals[-5000:]
        return total

    def _evaluate_routing_counterfactual(self, finding: Finding, worm: Worm, proposal: dict) -> dict:
        """Pure offline evaluation of the opposite routing proposal."""
        if not isinstance(proposal, dict) or proposal.get('error'):
            return {'valid': False, 'audit_proxy_pass': 0, 'error': 'invalid counterfactual proposal'}
        clone = copy.deepcopy(finding)
        worm_clone = copy.deepcopy(worm)
        try:
            self._apply_brain_proposal(clone, worm_clone, proposal)
            # Counterfactual scoring must depend on the counterfactual proposal, not
            # inherited fields from the observed finding. Otherwise a weak proposal
            # can inherit a strong prior counterfactual_score and pass unchanged.
            p = validate_proposal(proposal)
            clone.counterfactual_score = max(0.0, min(1.0, (
                0.45 * p.falsifiability +
                0.30 * p.information_gain +
                0.15 * p.independence +
                0.10 * p.novelty
            )))
            clone.robustness = max(0.0, min(1.0, (
                0.55 * clone.robustness + 0.45 * clone.counterfactual_score
            )))
        except Exception as exc:
            return {'valid': False, 'audit_proxy_pass': 0, 'error': f'{type(exc).__name__}: {exc}'}
        threshold = self.meta.policy.evidence_threshold if (self.config.colony_os_enabled and self.config.meta_controller_enabled) else .42
        habit_risk = max(worm_clone.cognitive_habits.values(), default=0.0)
        parasite_block = bool(self.config.cognitive_parasites and habit_risk > .62)
        shase_block = bool(self.config.shase_enabled and clone.shase_influence_score > .58 and clone.shase_attribution_confidence > .48 and clone.shase_influence_score > clone.validity + .18)
        audit_proxy = bool((clone.validity >= threshold) and (clone.falsifiability >= .40) and (clone.counterfactual_score >= .30) and not parasite_block and not shase_block)
        post_score = float(clone.score)
        actual_score = float(finding.score)
        return {
            'valid': True,
            'audit_proxy_pass': int(audit_proxy),
            'post_score': post_score,
            'signature_delta': max(0.0, min(1.0, post_score - actual_score)),
            'delta_vs_actual': post_score - actual_score,
            'beats_actual': bool(post_score > actual_score + 1e-12),
            'parasite_block': parasite_block,
            'shase_block': shase_block,
        }

    def _resolve_mode_routing(self, generation: int) -> dict:
        if not self.config.routing_enabled:
            return {'resolved': 0, 'skipped': 0, 'counterfactuals': 0}
        outcome = self.routing_book.resolve_generation(generation=generation, findings=self.state.findings, worms=self.state.worms, audits=self.state.audits)
        cf = self._routing_counterfactual_replay(generation)
        # Activate inherited routing bias only after sufficient resolved observations per mode.
        resolved_total = sum(1 for r in self.routing_book.records if r.get('resolved') and not r.get('control_group'))
        ready_for_genetic = False
        if self.config.routing_genetic_bias_enabled and resolved_total >= self.config.routing_min_resolved_for_genetic_bias:
            mode_pulls = {m: sum(1 for r in self.routing_book.records if r.get('resolved') and r.get('chosen_mode') == m and not r.get('control_group')) for m in ('AR','DIFFUSION')}
            ready_for_genetic = min(mode_pulls.values()) >= self.config.routing_min_mode_observations
            if ready_for_genetic:
                for worm in self.state.worms.values():
                    if worm.id in self.genomes:
                        worm.routing_bias = self.genomes[worm.id].routing_bias()
        self.routing_book.bandit.genetic_bias_active = bool(ready_for_genetic)
        return {**outcome, 'counterfactuals': cf, 'genetic_bias_active': self.routing_book.bandit.genetic_bias_active, 'resolved_total': resolved_total}

    def _generation_metrics(self, number: int, name: str, start: dict, status: str, event_start: int, stop_reason: str = "") -> GenerationRecord:
        worms = list(self.state.worms.values())
        findings = list(self.state.findings.values())
        born = [r for r in self.genetics.records if r.success and r.generation == number]
        f_in_gen = [f for f in findings if f.id.startswith(f"F-{number:04d}-")]
        active = self.state.active_worms()
        events = [e for e in self.event_log.events if event_start < e["seq"] <= len(self.event_log.events)]
        research = next((x for x in reversed(self.state.research_events) if int(x.get("generation", -1)) == number), {})
        shase = [x for x in self.state.shase_events if int(x.get("generation", -1)) == number]
        shase_influence = sum(float(x.get("influence_score", 0.0)) for x in shase) / max(1, len(shase))
        shase_attr = sum(float(x.get("attribution_confidence", 0.0)) for x in shase) / max(1, len(shase))
        shase_shift = sum(float(x.get("belief_shift_risk", 0.0)) for x in shase) / max(1, len(shase))
        routing = [r for r in self.routing_book.records if int(r.get("generation", -1)) == int(number)]
        routing_resolved = [r for r in routing if r.get("resolved")]
        ar_rate = sum(r.get("chosen_mode") == "AR" for r in routing) / max(1, len(routing))
        diff_rate = sum(r.get("chosen_mode") == "DIFFUSION" for r in routing) / max(1, len(routing))
        return GenerationRecord(
            number=number, name=name, status=status,
            event_seq_start=event_start + 1 if events else event_start,
            event_seq_end=len(self.event_log.events),
            population_start=start["population"], population_end=len(worms),
            active_start=start["active"], active_end=len(active),
            findings_produced=len(f_in_gen), open_findings=sum(f.status == "OPEN" for f in findings),
            max_depth=max((f.depth for f in f_in_gen), default=0), births=len(born),
            arena_matches=sum(1 for x in self.state.arena_matches if int(x.get("generation", number)) == number),
            learning_events=sum(1 for x in self.state.learning_events if int(x.get("generation", number)) == number),
            experiments=sum(1 for x in self.state.experiment_contracts if int(x.get("generation", number)) == number),
            self_attacks=sum(1 for e in events if e["type"] == "META_SELF_ATTACK" and e["generation"] == number),
            omega_events=sum(1 for e in events if e["type"] == "OMEGA_REDESIGN" and e["generation"] == number),
            extinction_events=sum(1 for e in events if e["type"] == "EXTINCTION_EVENT" and e["generation"] == number),
            mean_stress=sum(w.stress for w in worms)/max(1, len(worms)),
            mean_resilience=sum(w.resilience for w in worms)/max(1, len(worms)),
            mean_adaptation=sum(w.adaptation for w in worms)/max(1, len(worms)),
            genome_diversity=self._mean_genome_diversity(),
            mean_arena_rating=sum(w.arena_rating for w in worms)/max(1, len(worms)),
            mean_reputation=sum(w.reputation for w in worms)/max(1, len(worms)),
            mean_integrity=sum(w.integrity for w in worms)/max(1, len(worms)),
            stop_reason=stop_reason,
            workspace_submitted=int(research.get("submitted", 0)),
            workspace_accepted=int(research.get("accepted", 0)),
            workspace_rejected=int(research.get("rejected", 0)),
            workspace_bits=int(research.get("used_bits", 0)),
            workspace_equivocation=float(research.get("equivocation", 0.0)),
            workspace_mi_bits=float(research.get("mutual_information_specialty_to_routing_bits", 0.0)),
            development_attempts=sum(1 for e in self.state.development_events if int(e.get("generation", -1)) == number),
            development_promotions=sum(1 for e in self.state.development_promotions if int(e.get("generation", -1)) == number),
            development_gain=sum(float(e.get("gain", 0.0)) for e in self.state.development_events if int(e.get("generation", -1)) == number),
            routing_decisions=len(routing), routing_resolved=len(routing_resolved),
            routing_control=sum(bool(r.get("control_group")) for r in routing),
            routing_bandit=sum(not bool(r.get("control_group")) for r in routing),
            routing_brier=float(self.routing_book.calibration.brier()),
            routing_cf_replays=sum(1 for r in routing if r.get("counterfactual") is not None),
            routing_mean_fitness=sum(float(getattr(w, "routing_fitness", 0.50)) for w in worms)/max(1, len(worms)),
            routing_ar_rate=float(ar_rate), routing_diffusion_rate=float(diff_rate),
            shase_events=len(shase),
            shase_mean_influence=shase_influence,
            shase_mean_attribution_confidence=shase_attr,
            shase_mean_belief_shift_risk=shase_shift,
            dual_brain_calls=sum(1 for e in events if e["type"] == "DUAL_BRAIN"),
            dual_ar_calls=sum(1 for e in events if e["type"] == "DUAL_BRAIN" and e["payload"].get("mode") == "AR"),
            dual_diffusion_calls=sum(1 for e in events if e["type"] == "DUAL_BRAIN" and e["payload"].get("mode") == "DIFFUSION"),
            dual_mean_uncertainty=(sum(float(e["payload"].get("uncertainty",0.0)) for e in events if e["type"] == "DUAL_BRAIN") / max(1, sum(1 for e in events if e["type"] == "DUAL_BRAIN"))) if any(e["type"] == "DUAL_BRAIN" for e in events) else 0.0,
        )

    def _record_generation(self, number: int, name: str, start: dict, event_start: int, status: str = "COMPLETED", stop_reason: str = "") -> None:
        rec = self._generation_metrics(number, name, start, status, event_start, stop_reason)
        self.state.generation_history.append(rec)
        self._emit("GENERATION_RESULT", {
            "number": rec.number, "name": rec.name, "status": rec.status,
            "population": [rec.population_start, rec.population_end],
            "active": [rec.active_start, rec.active_end],
            "findings": rec.findings_produced, "open_findings": rec.open_findings,
            "max_depth": rec.max_depth, "births": rec.births,
            "arena_matches": rec.arena_matches, "learning_events": rec.learning_events,
            "experiments": rec.experiments, "genome_diversity": round(rec.genome_diversity, 6),
            "workspace": {"submitted": rec.workspace_submitted, "accepted": rec.workspace_accepted, "rejected": rec.workspace_rejected, "bits": rec.workspace_bits, "equivocation": round(rec.workspace_equivocation, 6), "mi_bits": round(rec.workspace_mi_bits, 6)},
            "development": {"attempts": rec.development_attempts, "promotions": rec.development_promotions, "gain": round(rec.development_gain, 6)},
            "routing": {"decisions": rec.routing_decisions, "resolved": rec.routing_resolved, "control": rec.routing_control, "bandit": rec.routing_bandit, "brier": round(rec.routing_brier, 6), "counterfactuals": rec.routing_cf_replays, "ar_rate": round(rec.routing_ar_rate, 6), "diffusion_rate": round(rec.routing_diffusion_rate, 6), "mean_routing_fitness": round(sum(float(getattr(w, "routing_fitness", 0.50)) for w in self.state.worms.values()) / max(1, len(self.state.worms)), 6)},
            "shase": {"events": rec.shase_events, "mean_influence": round(rec.shase_mean_influence, 6), "mean_attribution_confidence": round(rec.shase_mean_attribution_confidence, 6), "mean_belief_shift_risk": round(rec.shase_mean_belief_shift_risk, 6)},
            "dual_brain": {"calls": rec.dual_brain_calls, "ar_calls": rec.dual_ar_calls, "diffusion_calls": rec.dual_diffusion_calls, "mean_uncertainty": round(rec.dual_mean_uncertainty, 6)},
            "stop_reason": rec.stop_reason,
        })

    def run_generation(self) -> int:
        """Execute exactly one generation transactionally.

        If any component fails, state is rolled back to the generation boundary,
        an auditable HALTED event is recorded, and the original exception is re-raised.
        """
        tx = self._capture_transaction()
        number = self.state.generation
        name = self.generation_name(number)
        start = {"population": len(self.state.worms), "active": len(self.state.active_worms())}
        event_start = len(self.event_log.events)
        # Generation identity is explicit before work begins.
        self._emit("GENERATION_START", {"number": number, "name": name, "population": start["population"], "active": start["active"]})
        try:
            produced = self._run_generation_impl()
            # The generation counter advances inside the transactional body.
            self._record_generation(number, name, start, event_start, "COMPLETED", "")
            return produced
        except Exception as exc:
            self._restore_transaction(tx)
            # Preserve a named, auditable failure record while keeping the domain state rolled back.
            self._set_phase("HALTED", error=type(exc).__name__, message=str(exc)[:240])
            self._record_generation(number, name, start, event_start, "FAILED", str(exc)[:240])
            raise

    def run(self) -> FarmState:
        # max_generations is a total target, so resumed checkpoints do not overrun it.
        while self.state.generation < self.config.max_generations:
            if self.state.phase in {"HALTED", "COMPLETE"} or not self.state.claims:
                break
            produced = self.run_generation()
            if produced == 0 and self.state.phase in {"HALTED", "COMPLETE"}:
                break
        if self.state.phase == "COMPLETE":
            return self.state
        self._set_phase("FINALIZING")
        for finding in list(self.state.findings.values()):
            if finding.status == "OPEN" and (finding.score < self.config.prune_score or finding.validity < .38):
                self._penalize(finding, "final audit rejection", self.config.penalty_weak_finding)
        self._rehabilitation_reserve(emergency=True)
        validate_state(self.state)
        self._set_phase("COMPLETE")
        return self.state

    def report(self) -> dict:
        worms = list(self.state.worms.values()); findings = list(self.state.findings.values())
        open_findings = [f for f in findings if f.status == "OPEN"]
        return {
            "version": "3.4.0",
            "phase": self.state.phase,
            "generation": self.state.generation,
            "generation_history": [
                {**dataclass_to_dict(x), "label": f"G-{x.number:04d} :: {x.name}"}
                for x in self.state.generation_history
            ],
            "worms": len(worms),
            "active_worms": len(self.state.active_worms()),
            "male_worms": sum(w.sex == Sex.MALE.value for w in worms),
            "female_worms": sum(w.sex == Sex.FEMALE.value for w in worms),
            "genetic_births": sum(r.success for r in self.genetics.records),
            "genetic_failed_pairings": sum(not r.success for r in self.genetics.records),
            "same_generation_mating_blocks": sum(r.reason == "same-generation mating prohibited" for r in self.genetics.records),
            "genetic_records": len(self.genetics.records),
            "findings": len(findings),
            "open_findings": len(open_findings),
            "pruned_findings": sum(f.status == "PRUNED" for f in findings),
            "audits": len(self.state.audits),
            "rewards_total": round(sum(w.reward_points for w in worms), 4),
            "penalties_total": round(sum(w.penalty_points for w in worms), 4),
            "max_depth": max((f.depth for f in findings), default=0),
            "mean_stress": round(sum(w.stress for w in worms)/max(1, len(worms)), 4),
            "mean_resilience": round(sum(w.resilience for w in worms)/max(1, len(worms)), 4),
            "mean_adaptation": round(sum(w.adaptation for w in worms)/max(1, len(worms)), 4),
            "genome_diversity": round(self._mean_genome_diversity(), 4),
            "learning_failures": sum(w.failures for w in worms),
            "war_edges": len(self.state.conflict_edges),
            "trust_edges": len(self.state.trust_edges),
            "scar_memory_entries": sum(len(w.scar_memory) for w in worms),
            "omega_events": len(self.state.omega_events),
            "extinction_events": len(self.state.extinction_events),
            "surviving_extinctions": sum(w.extinction_survivals for w in worms),
            "rehabilitated_worms": sum(1 for e in self.event_log.events if e["type"] == "REHABILITATE"),
            "keystone_rescues": sum(1 for e in self.event_log.events if e["type"] == "REHABILITATE" and e["payload"].get("reason") == "keystone_rescue"),
            "minimum_active_floor": self.config.minimum_active_worms,
            "event_count": len(self.event_log.events),
            "event_chain_hash": self.event_log.last_hash,
            "stalled_generations": self.state.stalled_generations,
            "stop_reason": self.state.stop_reason,
            "brain": self.brain.name,
            "brain_provider": self.config.brain_provider,
            "brain_calls": sum(w.brain_calls for w in worms),
            "brain_failures": sum(w.brain_failures for w in worms),
            "brain_memory_entries": sum(len(w.brain_memory) for w in worms),
            "brain_cache_entries": len(self.state.brain_cache),
            "dual_brain": {
                "enabled": self.config.brain_provider.lower().strip() in {"worm-dual", "dual", "wormdual"},
                "calls": sum(w.dual_brain_calls for w in worms),
                "ar_calls": sum(w.dual_ar_calls for w in worms),
                "diffusion_calls": sum(w.dual_diffusion_calls for w in worms),
                "mean_uncertainty": round(sum(w.dual_last_uncertainty for w in worms)/max(1,len(worms)), 6),
            },
            "arena_matches": len(self.state.arena_matches),
            "arena_wins": sum(w.arena_wins for w in worms),
            "arena_losses": sum(w.arena_losses for w in worms),
            "arena_draws": sum(w.arena_draws for w in worms),
            "arena_tournaments": self.arena.state.tournaments,
            "arena_champions": len(self.arena.state.tournament_wins),
            "mean_arena_rating": round(sum(w.arena_rating for w in worms)/max(1, len(worms)), 3),
            "specialist_count": len(set(w.specialty for w in worms)),
            "specialty_distribution_named": {name: sum(w.specialty == name for w in worms) for name in [s.name for s in SPECIALISTS]},
            "curriculum_tasks": len(self.state.curriculum_tasks),
            "curriculum_completed": sum(bool(x.get("completed")) for x in self.state.curriculum_tasks),
            "experiment_contracts": len(self.state.experiment_contracts),
            "experiments_passed": sum(x.get("status") == "PASSED" for x in self.state.experiment_contracts),
            "meta_policy_events": len(self.state.meta_policy_history),
            "meta_self_attacks": sum(w.self_attack_count for w in worms),
            "specialist_transfers": len(self.state.specialist_transfers),
            "learning_debt": round(sum(sum(w.learning_debt.values()) for w in worms)/max(1,len(worms)), 4),
            "max_arena_rating": round(max((w.arena_rating for w in worms), default=1000.0), 3),
            "specialty_distribution": {k.value: sum(w.kind == k for w in worms) for k in WormKind},
            "arena_modes": self._arena_mode_counts(),
            "arena_leaderboard": [
                {"worm": w.id, "specialty": w.kind.value, "rating": round(w.arena_rating, 2), "wins": w.arena_wins, "losses": w.arena_losses, "draws": w.arena_draws}
                for w in sorted(worms, key=lambda x: (x.arena_rating, x.arena_wins, x.id), reverse=True)[:8]
            ],
            "learning_events": len(self.state.learning_events),
            "shase": {
                "enabled": self.config.shase_enabled,
                "events": len(self.state.shase_events),
                "last_influence_score": (self.state.shase_events[-1].get("influence_score", 0.0) if self.state.shase_events else 0.0),
                "last_attribution_confidence": (self.state.shase_events[-1].get("attribution_confidence", 0.0) if self.state.shase_events else 0.0),
                "dominant_mechanisms": {
                    mech: sum(1 for e in self.state.shase_events if e.get("dominant_mechanism") == mech)
                    for mech in ("threat", "framing", "social_pressure", "arousal", "attention")
                },
            },
            "shase": {
                "enabled": self.config.shase_enabled,
                "events": len(self.state.shase_events),
                "mean_influence": round(sum(float(e.get("influence_score", 0.0)) for e in self.state.shase_events)/max(1, len(self.state.shase_events)), 4),
                "mean_attribution_confidence": round(sum(float(e.get("attribution_confidence", 0.0)) for e in self.state.shase_events)/max(1, len(self.state.shase_events)), 4),
                "mean_belief_shift_risk": round(sum(float(e.get("belief_shift_risk", 0.0)) for e in self.state.shase_events)/max(1, len(self.state.shase_events)), 4),
                "dominant_mechanisms": {mech: sum(1 for e in self.state.shase_events if e.get("dominant_mechanism") == mech) for mech in ("threat", "framing", "social_pressure", "arousal", "attention")},
                "influence_without_support_flags": sum("influence_without_strong_evidence" in e.get("flags", []) for e in self.state.shase_events),
            },
            "routing": {
                "enabled": self.config.routing_enabled,
                "decisions": len(self.routing_book.records),
                "resolved": sum(1 for r in self.routing_book.records if r.get("resolved")),
                "control_fraction": self.config.routing_control_fraction,
                "observed_control_fraction": sum(1 for r in self.routing_book.records if r.get("control_group")) / max(1, len(self.routing_book.records)),
                "bandit_posteriors": self.routing_book.bandit.posterior_report(),
                "brier": self.routing_book.calibration.brier(),
                "brier_ar": self.routing_book.calibration.brier("AR"),
                "brier_diffusion": self.routing_book.calibration.brier("DIFFUSION"),
                "reliability_ar": self.routing_book.calibration.reliability("AR"),
                "reliability_diffusion": self.routing_book.calibration.reliability("DIFFUSION"),
                "counterfactual_replays": len(self.state.routing_counterfactuals),
                "genetic_bias_active": self.config.routing_genetic_bias_enabled and sum(1 for r in self.routing_book.records if r.get("resolved") and not r.get("control_group")) >= self.config.routing_min_resolved_for_genetic_bias,
            },
            "research_ecology": {
                "enabled": self.config.research_ecology_enabled,
                "workspace_capacity_bits": self.config.workspace_capacity_bits,
                "messages": self.research_ecology.total_messages,
                "accepted": self.research_ecology.total_accepted,
                "rejected": self.research_ecology.total_rejected,
                "transmitted_bits": self.research_ecology.total_bits,
                "events": len(self.state.research_events),
                "shared_workspace_items": len(self.state.shared_workspace),
                "last_equivocation": (self.state.research_events[-1].get("equivocation", 0.0) if self.state.research_events else 0.0),
                "last_information_efficiency": (self.state.research_events[-1].get("information_efficiency", 0.0) if self.state.research_events else 0.0),
            },
            "learning_channels": {k: round(v, 4) for k, v in self.learning.state.channel_totals.items()},
            "upgrade_status": {
                "cognitive_genome": self.config.genetics_enabled,
                "adaptive_penetration": self.config.adaptive_penetration,
                "worm_warfare": self.config.worm_warfare,
                "scar_memory": self.config.scar_memory,
                "counterfactual_lab": self.config.counterfactual_lab,
                "forbidden_assumption_zone": self.config.forbidden_assumption_zone,
                "trust_graph": self.config.trust_graph,
                "cognitive_parasites": self.config.cognitive_parasites,
                "extinction_events": self.config.extinction_events,
                "worm_omega": self.config.omega_enabled,
                "adversarial_arena": self.config.arena_enabled,
                "multi_directional_learning": self.config.learning_enabled,
                "specialist_ecology": self.config.specialist_ecology_enabled,
                "adaptive_curriculum": self.config.curriculum_enabled,
                "synthetic_experiment_engine": self.config.experiments_enabled,
                "meta_controller": self.config.meta_controller_enabled,
                "self_attack": self.config.self_attack_budget_fraction > 0,
                "shase_influence_dynamics": self.config.shase_enabled,
            },
            "top_findings": [
                {"title":f.title,"kind":f.kind.value,"depth":f.depth,"score":round(f.score,3),
                 "reward":round(f.reward_granted,3),"counterfactual":round(f.counterfactual_score,3),"robustness":round(f.robustness,3)}
                for f in sorted(open_findings, key=lambda x:x.score, reverse=True)[:10]
            ],
        }

    def _arena_mode_counts(self) -> dict:
        counts: dict[str, int] = {}
        for rec in self.arena.state.matches:
            counts[rec.mode] = counts.get(rec.mode, 0) + 1
        return dict(sorted(counts.items()))

    def _mean_genome_diversity(self) -> float:
        ids = sorted(self.genomes)[:64]
        vals = []
        for i, a in enumerate(ids):
            for b in ids[i+1:]:
                vals.append(self.genomes[a].distance(self.genomes[b]))
        return sum(vals)/max(1, len(vals))

    def snapshot(self) -> dict:
        """Create a complete JSON-serializable checkpoint."""
        return {
            "format": "worm-farm-checkpoint",
            "version": 1,
            "config": dataclass_to_dict(self.config),
            "state": {
                **dataclass_to_dict(self.state),
                "trust_edges": [
                    [a, b, value] for (a, b), value in sorted(self.state.trust_edges.items())
                ],
                "conflict_edges": [list(x) for x in self.state.conflict_edges],
                "edges": [list(x) for x in self.state.edges],
            },
            "genomes": {wid: dataclass_to_dict(g) for wid, g in self.genomes.items()},
            "genetics": {
                "records": dataclass_to_dict(self.genetics.records),
                "parent_map": self.genetics.parent_map,
                "mating_generation": {f"{a}|{b}": g for (a,b), g in self.genetics.mating_generation.items()},
                "children_by_pair": {f"{a}|{b}": n for (a,b), n in self.genetics.children_by_pair.items()},
            },
            "runtime": {
                "finding_seq": self._finding_seq,
                "worm_seq": self._worm_seq,
                "genome_seq": self._genome_seq,
                "last_population": self._last_population,
                "last_open": self._last_open,
                "environment_shock": self.environment.shock,
                "rng_state": pack_state(self.rng.getstate()),
                "environment_rng_state": pack_state(self.environment.rng.getstate()),
                "genetics_rng_state": pack_state(self.genetics.rng.getstate()),
                "event_log": self.event_log.export(),
                "arena_state": dataclass_to_dict(self.arena.state),
                "arena_rng_state": pack_state(self.arena.rng.getstate()),
                "arena_seq": self.arena._seq,
                "learning_state": dataclass_to_dict(self.learning.state),
                "learning_rng_state": pack_state(self.learning.rng.getstate()),
                "skill_states": {wid: dataclass_to_dict(ss) for wid, ss in self.skill_states.items()},
                "curriculum_tasks": dataclass_to_dict(self.curriculum.tasks),
                "curriculum_seq": self.curriculum._seq,
                "experiment_contracts": dataclass_to_dict(self.experiments.contracts),
                "experiment_seq": self.experiments._seq,
                "meta_policy": dataclass_to_dict(self.meta.policy),
                "specialist_seq": self._specialist_seq,
                "brain_cache": self.state.brain_cache,
                "brain_runtime_state": self.brain.runtime_state() if hasattr(self.brain, "runtime_state") else {},
                "research_runtime_state": self.research_ecology.runtime_state(),
                "self_development_runtime_state": self.self_development.runtime_state(),
                "routing_runtime_state": self.routing_book.runtime_state(),
            },
        }

    def save_checkpoint(self, path: str | Path) -> None:
        CheckpointStore(path).save(self.snapshot())

    @classmethod
    def _from_payload(cls, payload: dict, brain: BrainAdapter | None = None) -> "WormFarm":
        cfg_dict = dict(payload["config"])
        if isinstance(cfg_dict.get("forbidden_terms"), list):
            cfg_dict["forbidden_terms"] = tuple(cfg_dict["forbidden_terms"])
        cfg = FarmConfig(**cfg_dict)
        farm = cls(cfg, brain=brain)
        s = payload["state"]
        farm.state = FarmState(
            claims={k: Claim(**v) for k,v in s["claims"].items()},
            worms={k: Worm(**{**v, "kind": WormKind(v["kind"])}) for k,v in s["worms"].items()},
            findings={k: Finding(**{**v, "kind": WormKind(v["kind"])}) for k,v in s["findings"].items()},
            edges=[tuple(x) for x in s["edges"]],
            audits=[AuditEvent(**v) for v in s["audits"]],
            trust_edges={(row[0], row[1]): row[2] for row in s.get("trust_edges", [])},
            conflict_edges=[tuple(x) for x in s["conflict_edges"]],
            omega_events=list(s["omega_events"]),
            extinction_events=list(s["extinction_events"]),
            generation=s["generation"],
            lineage_parents={k: tuple(v) for k,v in s.get("lineage_parents", {}).items()},
            event_seq=s.get("event_seq", 0),
            phase=s.get("phase", "READY"),
            stalled_generations=s.get("stalled_generations", 0),
            stop_reason=s.get("stop_reason", ""),
            arena_matches=list(s.get("arena_matches", [])),
            learning_events=list(s.get("learning_events", [])),
            curriculum_tasks=list(s.get("curriculum_tasks", [])),
            experiment_contracts=list(s.get("experiment_contracts", [])),
            meta_policy_history=list(s.get("meta_policy_history", [])),
            specialist_transfers=list(s.get("specialist_transfers", [])),
            colony_alerts=list(s.get("colony_alerts", [])),
            generation_history=[GenerationRecord(**x) for x in s.get("generation_history", [])],
            brain_cache={k: dict(v) for k, v in s.get("brain_cache", {}).items()},
            research_events=list(s.get("research_events", [])),
            shared_workspace=list(s.get("shared_workspace", [])),
            development_events=list(s.get("development_events", [])),
            development_proposals=list(s.get("development_proposals", [])),
            development_promotions=list(s.get("development_promotions", [])),
            routing_decisions=list(s.get("routing_decisions", [])),
            routing_counterfactuals=list(s.get("routing_counterfactuals", [])),
            shase_events=list(s.get("shase_events", [])),
            shase_generation_stats=list(s.get("shase_generation_stats", [])),
        )
        farm.genomes = {wid: CognitiveGenome(**v) for wid,v in payload["genomes"].items()}
        g = payload["genetics"]
        from .genetics import ReproductiveRecord
        farm.genetics.records = [ReproductiveRecord(**rec) for rec in g["records"]]
        farm.genetics.parent_map = {k: tuple(v) for k,v in g["parent_map"].items()}
        farm.genetics.mating_generation = {tuple(k.split("|")): v for k,v in g["mating_generation"].items()}
        farm.genetics.children_by_pair = {tuple(k.split("|")): v for k,v in g["children_by_pair"].items()}
        rt = payload["runtime"]
        farm._finding_seq = rt["finding_seq"]; farm._worm_seq = rt["worm_seq"]; farm._genome_seq = rt["genome_seq"]
        farm._last_population = rt.get("last_population", 0); farm._last_open = rt.get("last_open", 0)
        if not farm.state.brain_cache and rt.get("brain_cache"):
            farm.state.brain_cache = {k: dict(v) for k, v in rt.get("brain_cache", {}).items()}
        if hasattr(farm.brain, "cache"):
            farm.brain.cache = farm.state.brain_cache
        if rt.get("brain_runtime_state") and hasattr(farm.brain, "load_runtime_state"):
            farm.brain.load_runtime_state(rt.get("brain_runtime_state"))
        if rt.get("research_runtime_state"):
            farm.research_ecology.load_runtime_state(rt.get("research_runtime_state"))
        if rt.get("self_development_runtime_state"):
            farm.self_development.load_runtime_state(rt.get("self_development_runtime_state"))
        route_runtime = rt.get("routing_runtime_state") or {}
        route_records = route_runtime.get("records")
        route_runtime_without_records = {k: v for k, v in route_runtime.items() if k != "records"}
        farm.routing_book.load_runtime_state(route_runtime_without_records)
        if route_records is not None:
            farm.state.routing_decisions = list(route_records)
        farm.routing_book.records = farm.state.routing_decisions
        delegate = getattr(farm.brain, "delegate", None)
        if delegate is not None and getattr(delegate, "name", "") == "worm-dual-lm":
            delegate.router = farm.routing_book.bandit
        farm.environment.shock = rt.get("environment_shock", 0.0)
        if rt.get("rng_state"): farm.rng.setstate(unpack_state(rt["rng_state"]))
        if rt.get("environment_rng_state"): farm.environment.rng.setstate(unpack_state(rt["environment_rng_state"]))
        if rt.get("genetics_rng_state"): farm.genetics.rng.setstate(unpack_state(rt["genetics_rng_state"]))
        farm.event_log.import_events(rt.get("event_log", []))
        from .arena import MatchRecord, ArenaState
        from .learning import LearningEvent, LearningState
        from .colony_os import SkillState, CurriculumTask, ExperimentContract, MetaPolicy
        ar = rt.get("arena_state")
        if ar:
            farm.arena.state = ArenaState(
                matches=[MatchRecord(**x) for x in ar.get("matches", [])],
                ratings={k: float(v) for k,v in ar.get("ratings", {}).items()},
                recent_pairs=[tuple(x) for x in ar.get("recent_pairs", [])],
                tournaments=int(ar.get("tournaments", 0)),
                tournament_wins={k:int(v) for k,v in ar.get("tournament_wins", {}).items()},
                generation_match_counts={k:int(v) for k,v in ar.get("generation_match_counts", {}).items()},
            )
        if rt.get("arena_rng_state"): farm.arena.rng.setstate(unpack_state(rt["arena_rng_state"]))
        farm.arena._seq = int(rt.get("arena_seq", len(farm.arena.state.matches)))
        lr = rt.get("learning_state")
        if lr:
            farm.learning.state.events = [LearningEvent(**x) for x in lr.get("events", [])]
            farm.learning.state.channel_totals = {k: float(v) for k,v in lr.get("channel_totals", {}).items()}
        if rt.get("learning_rng_state"): farm.learning.rng.setstate(unpack_state(rt["learning_rng_state"]))
        farm.skill_states = {}
        for wid, raw in rt.get("skill_states", {}).items():
            ss = SkillState(values={k: float(v) for k,v in raw.get("values", {}).items()}, exposures={k:int(v) for k,v in raw.get("exposures", {}).items()}, mastery={k:float(v) for k,v in raw.get("mastery", {}).items()})
            ss.ensure(); farm.skill_states[wid] = ss
        from .colony_os import CurriculumTask, ExperimentContract, MetaPolicy
        farm.curriculum.tasks = [CurriculumTask(**x) for x in rt.get("curriculum_tasks", [])]
        farm.curriculum._seq = int(rt.get("curriculum_seq", len(farm.curriculum.tasks)))
        farm.experiments.contracts = [ExperimentContract(**x) for x in rt.get("experiment_contracts", [])]
        farm.experiments._seq = int(rt.get("experiment_seq", len(farm.experiments.contracts)))
        if rt.get("meta_policy"):
            farm.meta.policy = MetaPolicy(**rt["meta_policy"])
        farm._specialist_seq = int(rt.get("specialist_seq", 0))
        for wid, ss in farm.skill_states.items():
            if wid in farm.state.worms:
                farm.state.worms[wid].skill_state = dict(ss.values)
                farm.state.worms[wid].skill_mastery = dict(ss.mastery)
        validate_state(farm.state)
        return farm

    @classmethod
    def from_checkpoint(cls, path: str | Path, brain: BrainAdapter | None = None) -> "WormFarm":
        payload = CheckpointStore(path).load()
        if payload.get("format") != "worm-farm-checkpoint" or payload.get("version") != 1:
            raise ValueError("unsupported worm-farm checkpoint")
        return cls._from_payload(payload, brain=brain)

    def verify(self) -> None:
        self.event_log.verify()
        validate_state(self.state)
        # Explicitly verify all recorded births, not just current parent links.
        for rec in self.genetics.records:
            if not rec.success:
                continue
            a = self.state.worms[rec.parent_a]; b = self.state.worms[rec.parent_b]; c = self.state.worms[rec.child_id]
            if a.generation == b.generation:
                raise InvariantViolation("same-generation mating invariant violated")
            if c.generation <= max(a.generation, b.generation):
                raise InvariantViolation("child-generation invariant violated")
            if self.state.lineage_parents.get(c.id) != (a.id, b.id):
                raise InvariantViolation("lineage parent record mismatch")
