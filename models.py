from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class WormKind(str, Enum):
    SCOUT = "SCOUT"
    BURROWER = "BURROWER"
    BREACHER = "BREACHER"
    PROBER = "PROBER"
    MIRROR = "MIRROR"
    CROSSWORM = "CROSSWORM"
    WORM_EATER = "WORM_EATER"
    OMEGA = "OMEGA"


class WormStatus(str, Enum):
    ACTIVE = "ACTIVE"
    COOLDOWN = "COOLDOWN"
    QUARANTINED = "QUARANTINED"
    PUNISHED = "PUNISHED"
    RETIRED = "RETIRED"


@dataclass
class Claim:
    id: str
    text: str
    evidence: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    confidence: float = 0.5
    provenance: str = "synthetic"


@dataclass
class Finding:
    id: str
    worm_id: str
    kind: WormKind
    depth: int
    title: str
    detail: str
    parent_finding_id: Optional[str] = None
    evidence_refs: List[str] = field(default_factory=list)
    assumptions_targeted: List[str] = field(default_factory=list)
    novelty: float = 0.0
    information_gain: float = 0.0
    anomaly: float = 0.0
    validity: float = 0.5
    falsifiability: float = 0.5
    independence: float = 0.5
    actionability: float = 0.5
    counterfactual_score: float = 0.0
    habit_break: float = 0.0
    deception_resistance: float = 0.0
    children_spawned: int = 0
    status: str = "OPEN"
    audit_penalty: float = 0.0
    reward_granted: float = 0.0
    adversarial_survival: float = 0.0
    robustness: float = 0.5
    challenge_pressure: float = 0.0
    independent_path: str = ""
    attack_count: int = 0
    worm_war_score: float = 0.0
    workspace_status: str = "PRIVATE"
    # v3.2 SHASE synthetic influence-dynamics fields
    shase_influence_score: float = 0.0
    shase_arousal_proxy: float = 0.0
    shase_framing_pressure: float = 0.0
    shase_social_pressure: float = 0.0
    shase_decision_pressure: float = 0.0
    shase_belief_shift_risk: float = 0.0
    shase_attribution_confidence: float = 0.0
    shase_dominant_mechanism: str = "none"

    @property
    def score(self) -> float:
        base = (
            0.18 * self.information_gain
            + 0.12 * self.novelty
            + 0.10 * self.anomaly
            + 0.18 * self.validity
            + 0.12 * self.falsifiability
            + 0.10 * self.independence
            + 0.10 * self.counterfactual_score
            + 0.05 * self.habit_break
            + 0.05 * self.deception_resistance
        )
        return max(0.0, min(1.0, base + 0.03 * self.actionability - 0.12 * self.audit_penalty))


@dataclass
class Worm:
    id: str
    kind: WormKind
    target_claim_id: str
    depth: int = 0
    parent_worm_id: Optional[str] = None
    max_depth: int = 5
    energy: float = 1.0
    integrity: float = 0.80
    reputation: float = 0.50
    status: str = WormStatus.ACTIVE.value
    private_notes: List[str] = field(default_factory=list)
    scar_memory: List[str] = field(default_factory=list)
    rejected_findings: List[str] = field(default_factory=list)
    reward_points: float = 0.0
    penalty_points: float = 0.0
    strikes: int = 0
    cooldown_until: int = -1
    births: int = 0
    deaths: int = 0
    stress: float = 0.10
    resilience: float = 0.40
    adaptation: float = 0.20
    hostility_exposure: float = 0.0
    failures: int = 0
    successful_adaptations: int = 0
    survival_count: int = 0
    sex: str = "UNASSIGNED"
    generation: int = 0
    lineage_parents: Dict[str, Tuple[str, str]] = field(default_factory=dict)
    event_seq: int = 0
    phase: str = "READY"
    stalled_generations: int = 0
    stop_reason: str = ""
    last_mating_generation: int = -999
    genome_id: str = ""
    lineage_depth: int = 0
    births_this_generation: int = 0
    cognitive_habits: Dict[str, float] = field(default_factory=dict)
    trust_scores: Dict[str, float] = field(default_factory=dict)
    penetration_depth_reached: int = 0
    extinction_survivals: int = 0
    # v0.7 specialist ecology / arena / learning
    specialty: str = "GENERALIST"
    arena_rating: float = 1000.0
    arena_wins: int = 0
    arena_losses: int = 0
    arena_draws: int = 0
    arena_streak: int = 0
    rivalry_index: float = 0.0
    counterattack_skill: float = 0.50
    learning_capacity: float = 0.60
    teaching_power: float = 0.50
    learning_channels: Dict[str, float] = field(default_factory=dict)
    strategic_memory: List[str] = field(default_factory=list)
    curriculum_stage: int = 0
    tournament_credits: float = 0.0
    # v0.8 adaptive colony OS
    skill_state: Dict[str, float] = field(default_factory=dict)
    skill_mastery: Dict[str, float] = field(default_factory=dict)
    learning_debt: Dict[str, float] = field(default_factory=dict)
    competency_history: List[str] = field(default_factory=list)
    current_strategy: str = "SPECIALIST"
    cognitive_temperature: float = 0.65
    uncertainty_budget: float = 0.55
    experiment_credits: float = 1.0
    self_attack_count: int = 0
    meta_failures: int = 0
    rehabilitation_count: int = 0
    age: int = 0
    brain_memory: List[str] = field(default_factory=list)
    brain_calls: int = 0
    brain_failures: int = 0
    brain_last_hash: str = ""
    # v2.5 dual-mode LM telemetry
    dual_brain_calls: int = 0
    dual_ar_calls: int = 0
    dual_diffusion_calls: int = 0
    dual_last_mode: str = "NONE"
    dual_last_mode_score: float = 0.0
    dual_last_uncertainty: float = 0.0
    # v3.4 learned routing genotype: -1 favors AR, +1 favors DIFFUSION
    routing_bias: float = 0.0
    # Selection pressure accumulated specifically from learned AR/DIFFUSION routing outcomes.
    routing_fitness: float = 0.50


@dataclass
class FarmConfig:
    max_generations: int = 10
    max_worms: int = 96
    max_depth: int = 8
    duplicate_similarity: float = 0.84
    energy_cost_per_attack: float = 0.07
    spawn_cost: float = 0.12
    reward_base: float = 0.55
    reward_information_gain: float = 0.45
    reward_falsifiability: float = 0.25
    reward_independence: float = 0.20
    penalty_weak_finding: float = 0.45
    penalty_duplicate: float = 0.28
    penalty_false_suspicion: float = 0.70
    integrity_decay: float = 0.010
    integrity_recovery: float = 0.045
    cooldown_generations: int = 1
    quarantine_generations: int = 2
    max_strikes: int = 5
    quarantine_strikes: int = 2
    rehabilitation_threshold: float = 0.52
    audit_fraction: float = 0.40
    minimum_reputation: float = 0.12
    prune_score: float = 0.30
    min_spawn_score: float = 0.46
    founder_pairs: int = 4
    max_genetic_pairs: int = 10
    max_population_from_genetics: int = 96
    genetic_selection_mode: str = "tournament"
    genetic_tournament_size: int = 4
    genetic_tournament_temperature: float = 1.0
    genetics_enabled: bool = True
    mutation_rate: float = 0.055
    mutation_sigma: float = 0.085
    recombination_bias: float = 0.50
    genetic_min_reputation: float = 0.38
    genetic_min_integrity: float = 0.40
    genetic_min_energy: float = 0.28
    genetic_min_resilience: float = 0.27
    max_children_per_pair: int = 2
    mate_cooldown: int = 2
    generation_gap_required: int = 1
    minimum_genetic_distance: float = 0.07
    incest_guard: bool = True
    prefer_unrelated: bool = True
    reproduction_cost: float = 0.10
    reproduction_reward: float = 0.09
    diversity_bonus: float = 0.12
    collapse_penalty: float = 0.10
    harsh_learning: bool = True
    harsh_hostility: float = 0.70
    harsh_noise: float = 0.25
    harsh_contradiction: float = 0.40
    harsh_decoy: float = 0.30
    harsh_drift: float = 0.20
    harsh_scarcity: float = 0.50
    harsh_deadline: float = 0.35
    harsh_peer_attack: float = 0.62
    harsh_failure_tax: float = 0.06
    # 10 delayed evolutionary/cognitive upgrades
    adaptive_penetration: bool = True
    worm_warfare: bool = True
    scar_memory: bool = True
    counterfactual_lab: bool = True
    forbidden_assumption_zone: bool = True
    trust_graph: bool = True
    cognitive_parasites: bool = True
    extinction_events: bool = True
    extinction_interval: int = 3
    extinction_intensity: float = 0.30
    omega_enabled: bool = True
    omega_interval: int = 4
    omega_redesign_strength: float = 0.06
    # dynamic population discipline
    minimum_active_worms: int = 4
    max_depth_bonus: float = 2.0
    war_pressure: float = 0.22
    parasite_pressure: float = 0.18
    forbidden_terms: Tuple[str, ...] = (
        "this is settled",
        "no alternative matters",
        "the search space is complete",
        "obviously impossible",
        "everyone knows",
        "by definition therefore",
    )
    random_seed: int = 42
    # v0.6 runtime hardening
    max_findings_per_generation: int = 96
    max_audits_per_generation: int = 48
    max_events: int = 20000
    # v1.0 pluggable LLM brain
    brain_provider: str = "rule-based"
    brain_base_url: str = "http://127.0.0.1:11434/v1"
    brain_model: str = "qwen2.5:3b"
    brain_timeout_seconds: float = 30.0
    brain_temperature: float = 0.20
    # v2.2 diffusion audit brain
    diffusion_base_provider: str = "torch-neural"
    diffusion_checkpoint_path: str = ""
    diffusion_vocab_path: str = ""
    diffusion_mask_rate: float = 0.35
    diffusion_audit_weight: float = 0.15
    diffusion_seed: int = 42
    # v2.1 real neural controller (optional torch dependency)
    brain_hidden_dim: int = 64
    brain_learning_rate: float = 0.0015
    brain_weight_decay: float = 0.0001
    brain_entropy_bonus: float = 0.012
    brain_baseline_decay: float = 0.95
    brain_temperature_policy: float = 0.85
    checkpoint_every: int = 0
    deterministic_ids: bool = True
    max_stalled_generations: int = 3
    min_information_gain_delta: float = 0.015
    require_independent_support_for_deepening: bool = True
    enable_fault_injection: bool = False
    # v0.7 adversarial arena + multi-directional learning
    arena_enabled: bool = True
    arena_matches_per_generation: int = 14
    arena_tournament_interval: int = 4
    arena_tournament_slots: int = 16
    arena_hostility: float = 0.88
    arena_energy_cost: float = 0.09
    arena_stress_gain: float = 0.09
    arena_win_reward: float = 0.10
    arena_loss_penalty: float = 0.075
    arena_draw_reward: float = 0.025
    arena_rating_k: float = 26.0
    arena_adaptation_gain: float = 0.028
    arena_learning_transfer: float = 0.16
    arena_tournament_bonus: float = 0.06
    learning_enabled: bool = True
    learning_transfer_rate: float = 0.16
    learning_teaching_reward: float = 0.025
    learning_counterteaching_reward: float = 0.018
    # v0.8 Adaptive Cognitive Colony OS
    colony_os_enabled: bool = True
    specialist_ecology_enabled: bool = True
    curriculum_enabled: bool = True
    experiments_enabled: bool = True
    meta_controller_enabled: bool = True
    max_curriculum_tasks_per_generation: int = 12
    max_experiments_per_generation: int = 8
    curriculum_difficulty: float = 0.72
    skill_learning_rate: float = 0.065
    skill_decay: float = 0.008
    specialty_mutation_rate: float = 0.025
    specialty_diversity_target: float = 0.28
    self_attack_budget_fraction: float = 0.12
    experiment_credit_gain: float = 0.06
    experiment_cost: float = 0.05
    meta_policy_smoothing: float = 0.18
    novelty_floor: float = 0.08
    keystone_reserve: int = 3
    keystone_min_integrity: float = 0.30
    keystone_min_reputation: float = 0.18
    keystone_energy_floor: float = 0.32
    keystone_rescue_cost: float = 0.08
    # v2.3 Shannon/Anthropic-inspired research ecology
    research_ecology_enabled: bool = True
    workspace_capacity_bits: int = 768
    workspace_noise: float = 0.08
    research_message_base_bits: int = 64
    research_max_events: int = 5000
    research_broadcast_reward: float = 0.018
    # v3.0 self-learning / self-development controller
    self_development_enabled: bool = True
    self_development_interval: int = 3
    self_development_min_gain: float = 0.012
    self_development_max_trials: int = 2
    self_development_canary_generations: int = 2
    self_development_canary_seeds: int = 3
    self_development_history_limit: int = 256
    self_development_allow_config_mutation_only: bool = True
    # v3.2 SHASE: synthetic influence-dynamics analyzer
    shase_enabled: bool = True
    shase_store_events: int = 5000
    shase_influence_weight: float = 0.18
    shase_arousal_weight: float = 0.18
    shase_framing_weight: float = 0.16
    shase_social_pressure_weight: float = 0.12
    shase_evidence_support_weight: float = 0.22
    shase_uncertainty_weight: float = 0.14
    # v2.5 WormDualLM swarm brain
    dual_checkpoint_path: str = ""
    dual_tokenizer_path: str = ""
    dual_mask_ratio: float = 0.35
    dual_mode_threshold: float = 0.18
    dual_device: str = "cpu"
    dual_delegate_provider: str = "rule-based"
    # v3.4 learned AR/DIFFUSION routing
    routing_enabled: bool = True
    routing_control_fraction: float = 0.10
    routing_control_threshold: float = 0.18
    routing_prior_alpha: float = 1.0
    routing_prior_beta: float = 1.0
    routing_evidence_weight: float = 0.35
    routing_bias_weight: float = 0.12
    routing_generation_bucket_width: int = 4
    routing_min_resolved_for_genetic_bias: int = 32
    routing_min_mode_observations: int = 8
    routing_genetic_bias_enabled: bool = True
    routing_reward_audit_weight: float = 0.30
    routing_reward_signature_weight: float = 0.18
    routing_reward_survival_weight: float = 0.30
    routing_reward_robustness_weight: float = 0.17
    routing_reward_cost_weight: float = 0.05
    routing_replay_enabled: bool = True
    routing_replay_max_per_generation: int = 96
    routing_signal_calibration_enabled: bool = False
    routing_signal_calibration_params: Dict[str, Dict[str, float]] = field(default_factory=dict)
    routing_empirical_signal_enabled: bool = True
    routing_empirical_min_confidence: int = 10
    routing_fitness_lambda: float = 0.0


@dataclass
class GenerationRecord:
    number: int
    name: str
    status: str = "COMPLETED"
    event_seq_start: int = 0
    event_seq_end: int = 0
    population_start: int = 0
    population_end: int = 0
    active_start: int = 0
    active_end: int = 0
    findings_produced: int = 0
    open_findings: int = 0
    max_depth: int = 0
    births: int = 0
    arena_matches: int = 0
    learning_events: int = 0
    experiments: int = 0
    self_attacks: int = 0
    omega_events: int = 0
    extinction_events: int = 0
    mean_stress: float = 0.0
    mean_resilience: float = 0.0
    mean_adaptation: float = 0.0
    genome_diversity: float = 0.0
    mean_arena_rating: float = 1000.0
    mean_reputation: float = 0.0
    mean_integrity: float = 0.0
    stop_reason: str = ""
    workspace_submitted: int = 0
    workspace_accepted: int = 0
    workspace_rejected: int = 0
    workspace_bits: int = 0
    workspace_equivocation: float = 0.0
    workspace_mi_bits: float = 0.0
    development_attempts: int = 0
    development_promotions: int = 0
    development_gain: float = 0.0
    routing_decisions: int = 0
    routing_resolved: int = 0
    routing_control: int = 0
    routing_bandit: int = 0
    routing_brier: float = 0.0
    routing_cf_replays: int = 0
    routing_ar_rate: float = 0.0
    routing_diffusion_rate: float = 0.0
    routing_mean_fitness: float = 0.5
    shase_events: int = 0
    shase_mean_influence: float = 0.0
    shase_mean_attribution_confidence: float = 0.0
    shase_mean_belief_shift_risk: float = 0.0
    dual_brain_calls: int = 0
    dual_ar_calls: int = 0
    dual_diffusion_calls: int = 0
    dual_mean_uncertainty: float = 0.0


@dataclass
class AuditEvent:
    generation: int
    worm_id: str
    finding_id: str
    reason: str
    penalty: float
    passed: bool


@dataclass
class FarmState:
    claims: Dict[str, Claim] = field(default_factory=dict)
    worms: Dict[str, Worm] = field(default_factory=dict)
    findings: Dict[str, Finding] = field(default_factory=dict)
    edges: List[Tuple[str, str, str]] = field(default_factory=list)
    audits: List[AuditEvent] = field(default_factory=list)
    trust_edges: Dict[Tuple[str, str], float] = field(default_factory=dict)
    conflict_edges: List[Tuple[str, str, float, str]] = field(default_factory=list)
    omega_events: List[str] = field(default_factory=list)
    extinction_events: List[str] = field(default_factory=list)
    generation: int = 0
    lineage_parents: Dict[str, Tuple[str, str]] = field(default_factory=dict)
    event_seq: int = 0
    phase: str = "READY"
    stalled_generations: int = 0
    stop_reason: str = ""
    arena_matches: List[dict] = field(default_factory=list)
    learning_events: List[dict] = field(default_factory=list)
    brain_cache: Dict[str, dict] = field(default_factory=dict)
    # v0.8 adaptive colony OS telemetry
    curriculum_tasks: List[dict] = field(default_factory=list)
    experiment_contracts: List[dict] = field(default_factory=list)
    meta_policy_history: List[dict] = field(default_factory=list)
    specialist_transfers: List[dict] = field(default_factory=list)
    colony_alerts: List[str] = field(default_factory=list)
    generation_history: List[GenerationRecord] = field(default_factory=list)
    research_events: List[dict] = field(default_factory=list)
    shared_workspace: List[dict] = field(default_factory=list)
    # v3.0 self-development telemetry
    development_events: List[dict] = field(default_factory=list)
    development_proposals: List[dict] = field(default_factory=list)
    development_promotions: List[dict] = field(default_factory=list)
    # v3.4 learned routing ledger
    routing_decisions: List[dict] = field(default_factory=list)
    routing_counterfactuals: List[dict] = field(default_factory=list)
    # v3.2 SHASE influence-dynamics analysis
    shase_events: List[dict] = field(default_factory=list)
    shase_generation_stats: List[dict] = field(default_factory=list)

    def active_worms(self) -> List[Worm]:
        return [w for w in self.worms.values() if w.status == WormStatus.ACTIVE.value]

    def all_worms(self) -> List[Worm]:
        return list(self.worms.values())
