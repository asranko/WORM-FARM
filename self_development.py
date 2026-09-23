from __future__ import annotations

from dataclasses import dataclass, asdict, replace
import copy
import hashlib
import math
from typing import Any, Dict, List, Tuple

from .models import Claim, FarmConfig
from .brain import StaticBrain


@dataclass(frozen=True)
class DevelopmentProposal:
    id: str
    generation: int
    parameter: str
    delta: float
    rationale: str
    expected_gain: float


@dataclass(frozen=True)
class DevelopmentResult:
    proposal_id: str
    generation: int
    parameter: str
    baseline_score: float
    candidate_score: float
    gain: float
    accepted: bool
    rationale: str
    candidate_value: float
    post_state_hash: str


class DevelopmentBandit:
    """Small deterministic contextual bandit for learning which safe mutations help."""

    DEFAULT_ARMS = (
        "prune_score",
        "min_spawn_score",
        "war_pressure",
        "parasite_pressure",
        "curriculum_difficulty",
        "arena_hostility",
        "workspace_capacity_bits",
        "research_broadcast_reward",
        "brain_learning_rate",
        "brain_entropy_bonus",
        "brain_weight_decay",
        "brain_temperature_policy",
    )

    def __init__(self, seed: int = 42) -> None:
        self.seed = int(seed)
        self.counts: Dict[str, int] = {a: 0 for a in self.DEFAULT_ARMS}
        self.means: Dict[str, float] = {a: 0.0 for a in self.DEFAULT_ARMS}

    def choose(self, eligible: tuple[str, ...] | None = None) -> str:
        arms = eligible or self.DEFAULT_ARMS
        unseen = [a for a in arms if self.counts.get(a, 0) == 0]
        if unseen:
            return unseen[0]
        total = sum(self.counts.get(a, 0) for a in arms)
        scored = []
        for arm in arms:
            n = max(1, self.counts.get(arm, 0))
            bonus = math.sqrt(2.0 * math.log(max(2, total)) / n)
            scored.append((self.means[arm] + bonus, arm))
        return max(scored, key=lambda x: (x[0], x[1]))[1]

    def update(self, arm: str, reward: float) -> None:
        reward = max(-1.0, min(1.0, float(reward)))
        n = self.counts.get(arm, 0) + 1
        old = self.means.get(arm, 0.0)
        self.counts[arm] = n
        self.means[arm] = old + (reward - old) / n

    def runtime_state(self) -> dict:
        return {"seed": self.seed, "counts": dict(self.counts), "means": dict(self.means)}

    def load_runtime_state(self, state: dict) -> None:
        if not state:
            return
        self.seed = int(state.get("seed", self.seed))
        self.counts = {k: int(v) for k, v in state.get("counts", self.counts).items()}
        self.means = {k: float(v) for k, v in state.get("means", self.means).items()}


class SelfDevelopmentEngine:
    """Safe self-development: propose/evaluate/promote config-only changes via sandbox canaries.

    It never mutates source code, benchmark labels, external targets, or evaluator rules.
    Development is transactional: only candidates that improve a held-out synthetic canary
    by the configured margin and preserve invariants are promoted.
    """

    PARAM_RANGES = {
        "prune_score": (0.18, 0.55, 0.06),
        "min_spawn_score": (0.30, 0.68, 0.07),
        "war_pressure": (0.05, 0.55, 0.10),
        "parasite_pressure": (0.03, 0.45, 0.08),
        "curriculum_difficulty": (0.20, 1.00, 0.10),
        "arena_hostility": (0.35, 1.00, 0.10),
        "workspace_capacity_bits": (128, 2048, 256),
        "research_broadcast_reward": (0.002, 0.08, 0.02),
        "brain_learning_rate": (0.0002, 0.006, 0.0006),
        "brain_entropy_bonus": (0.0, 0.05, 0.005),
        "brain_weight_decay": (0.0, 0.001, 0.0002),
        "brain_temperature_policy": (0.35, 1.20, 0.10),
    }

    def __init__(self, seed: int = 42, min_gain: float = 0.012, interval: int = 3, max_trials: int = 1, canary_generations: int = 2, canary_seeds: int = 3) -> None:
        self.seed = int(seed)
        self.min_gain = float(min_gain)
        self.interval = max(1, int(interval))
        self.max_trials = max(1, int(max_trials))
        self.canary_generations = max(1, int(canary_generations))
        self.canary_seeds = max(1, int(canary_seeds))
        self.bandit = DevelopmentBandit(seed)
        self.history: List[dict] = []

    @staticmethod
    def _hash_payload(payload: dict) -> str:
        raw = str(sorted(payload.items())).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]

    def due(self, generation: int) -> bool:
        return generation > 0 and generation % self.interval == 0

    @classmethod
    def eligible_arms(cls, config: FarmConfig) -> tuple[str, ...]:
        neural = str(getattr(config, "brain_provider", "rule-based")).lower().strip() in {"torch-neural", "neural", "pytorch"}
        arms = list(cls.PARAM_RANGES.keys())
        if not neural:
            arms = [a for a in arms if not a.startswith("brain_")]
        else:
            brain = [a for a in arms if a.startswith("brain_")]
            structural = [a for a in arms if not a.startswith("brain_")]
            # Give the live learning rule early evolutionary pressure; it is the most direct
            # self-development surface when a trainable brain is installed.
            arms = brain + structural
        return tuple(arms)

    def propose(self, config: FarmConfig, generation: int, diagnostics: dict) -> DevelopmentProposal:
        arm = self.bandit.choose(self.eligible_arms(config))
        lo, hi, step = self.PARAM_RANGES[arm]
        current = float(getattr(config, arm))
        # Deterministic direction derived from generation + observed diagnostics.
        quality = float(diagnostics.get("mean_quality", 0.5))
        diversity = float(diagnostics.get("diversity", 0.2))
        false_suspicion = float(diagnostics.get("false_suspicion", 0.0))
        direction_score = (0.55 - quality) + (0.20 - diversity) + 0.75 * false_suspicion
        sign = 1.0 if direction_score > 0 else -1.0
        if arm in {"arena_hostility", "war_pressure", "curriculum_difficulty"} and quality < 0.45:
            sign = -1.0
        if arm == "workspace_capacity_bits":
            sign = 1.0 if diagnostics.get("workspace_equivocation", 0.0) > 0.35 else -1.0
        candidate = current + sign * step
        candidate = max(lo, min(hi, candidate))
        delta = candidate - current
        pid = f"DEV-{generation:04d}-{self.bandit.counts[arm]+1:03d}"
        rationale = (
            f"Explore {arm}: quality={quality:.3f}, diversity={diversity:.3f}, "
            f"false_suspicion={false_suspicion:.3f}."
        )
        return DevelopmentProposal(pid, generation, arm, delta, rationale, abs(delta))

    @staticmethod
    def _score_state(state) -> float:
        findings = list(state.findings.values())
        worms = list(state.worms.values())
        active = len(state.active_worms())
        total = len(worms)
        mean_quality = sum(f.score for f in findings) / max(1, len(findings))
        open_ratio = sum(1 for f in findings if f.status == "OPEN") / max(1, len(findings))
        mean_resilience = sum(w.resilience for w in worms) / max(1, total)
        mean_depth = sum(f.depth for f in findings) / max(1, len(findings))
        depth_norm = min(1.0, mean_depth / 4.0)
        arena_wins = sum(w.arena_wins for w in worms)
        arena_losses = sum(w.arena_losses for w in worms)
        arena_total = arena_wins + arena_losses + sum(w.arena_draws for w in worms)
        arena_rate = arena_wins / max(1, arena_total)
        ws = list(getattr(state, "research_events", []))
        ws_accept = float(ws[-1].get("accepted", 0)) / max(1, float(ws[-1].get("submitted", 1))) if ws else 0.0
        return max(0.0, min(1.0,
            0.30 * mean_quality
            + 0.12 * open_ratio
            + 0.18 * mean_resilience
            + 0.14 * depth_norm
            + 0.16 * arena_rate
            + 0.10 * ws_accept
        ))

    def _canary(self, parent_farm, candidate_config: FarmConfig, seed_offset: int = 0) -> float:
        # Canary uses the same claim but an isolated deterministic seed and no nested self-development.
        from .farm import WormFarm
        cfg = copy.deepcopy(candidate_config)
        cfg.random_seed = int(candidate_config.random_seed) + int(seed_offset)
        cfg.max_generations = min(max(1, cfg.self_development_canary_generations), cfg.max_generations)
        cfg.self_development_enabled = False
        cfg.research_ecology_enabled = bool(cfg.research_ecology_enabled)
        if str(getattr(candidate_config, "brain_provider", "rule-based")).lower().strip() in {"torch-neural", "neural", "pytorch"}:
            canary = WormFarm(cfg)
        else:
            cfg.brain_provider = "rule-based"
            canary = WormFarm(cfg, brain=StaticBrain("self-development-canary"))
        for claim in parent_farm.state.claims.values():
            canary.seed(copy.deepcopy(claim))
            break
        canary.run()
        return self._score_state(canary.state)

    def _canary_mean(self, parent_farm, candidate_config: FarmConfig) -> tuple[float, list[float]]:
        values = [self._canary(parent_farm, candidate_config, i) for i in range(self.canary_seeds)]
        return (sum(values) / max(1, len(values)), values)

    def attempt(self, farm, diagnostics: dict) -> DevelopmentResult | None:
        if not self.due(farm.state.generation) or not getattr(farm.config, "self_development_enabled", False):
            return None
        accepted_result = None
        baseline_score, baseline_trials = self._canary_mean(farm, copy.deepcopy(farm.config))
        last_result = None
        proposal = None
        for trial in range(self.max_trials):
            if trial == 0 or proposal is None:
                proposal = self.propose(farm.config, farm.state.generation, diagnostics)
            else:
                proposal = replace(proposal, id=f"{proposal.id}-REV", delta=-proposal.delta, rationale=proposal.rationale + " Opposite-direction probe.")
            candidate_config = copy.deepcopy(farm.config)
            setattr(candidate_config, proposal.parameter, getattr(candidate_config, proposal.parameter) + proposal.delta)
            candidate_config.workspace_capacity_bits = int(round(candidate_config.workspace_capacity_bits / 16.0) * 16)
            candidate_config.workspace_capacity_bits = max(128, min(2048, candidate_config.workspace_capacity_bits))
            self._validate_candidate(candidate_config)
            candidate_score, candidate_trials = self._canary_mean(farm, candidate_config)
            gain = candidate_score - baseline_score
            positive_trials = sum(1 for b, c in zip(baseline_trials, candidate_trials) if c > b)
            accepted = gain >= self.min_gain and positive_trials >= max(2, math.ceil(self.canary_seeds * 0.67))
            self.bandit.update(proposal.parameter, gain)
            last_result = DevelopmentResult(
                proposal.id, proposal.generation, proposal.parameter,
                baseline_score, candidate_score, gain, accepted,
                proposal.rationale, float(getattr(candidate_config, proposal.parameter)), "",
            )
            self.history.append({**asdict(last_result), "baseline_trials": baseline_trials, "candidate_trials": candidate_trials, "positive_trials": positive_trials})
            if accepted:
                self.apply(farm, candidate_config)
                accepted_result = DevelopmentResult(
                    **{**asdict(last_result), "post_state_hash": self.state_hash(farm)}
                )
                self.history[-1] = {**asdict(accepted_result), "baseline_trials": baseline_trials, "candidate_trials": candidate_trials, "positive_trials": positive_trials}
                break
            # One reversal probe is enough to perform a true local search without becoming a
            # brute-force hyperparameter sweep.
        self.history = self.history[-128:]
        return accepted_result or last_result

    @staticmethod
    def state_hash(farm) -> str:
        compact = {
            "gen": farm.state.generation,
            "worms": sorted((w.id, w.generation, w.specialty, w.status, round(w.reputation, 6)) for w in farm.state.worms.values()),
            "findings": sorted((f.id, f.status, round(f.score, 6)) for f in farm.state.findings.values()),
            "config": {k: getattr(farm.config, k) for k in ("prune_score", "min_spawn_score", "war_pressure", "parasite_pressure", "curriculum_difficulty", "arena_hostility", "workspace_capacity_bits", "research_broadcast_reward", "brain_learning_rate", "brain_entropy_bonus", "brain_weight_decay", "brain_temperature_policy")},
        }
        return SelfDevelopmentEngine._hash_payload(compact)

    @staticmethod
    def _validate_candidate(config: FarmConfig) -> None:
        if not (0.18 <= config.prune_score <= 0.55):
            raise ValueError("invalid prune_score candidate")
        if not (0.30 <= config.min_spawn_score <= 0.68):
            raise ValueError("invalid min_spawn_score candidate")
        if not (0.05 <= config.war_pressure <= 0.55):
            raise ValueError("invalid war_pressure candidate")
        if not (0.03 <= config.parasite_pressure <= 0.45):
            raise ValueError("invalid parasite_pressure candidate")
        if not (0.20 <= config.curriculum_difficulty <= 1.0):
            raise ValueError("invalid curriculum_difficulty candidate")
        if not (0.35 <= config.arena_hostility <= 1.0):
            raise ValueError("invalid arena_hostility candidate")
        if not (128 <= config.workspace_capacity_bits <= 2048):
            raise ValueError("invalid workspace_capacity_bits candidate")
        if not (0.002 <= config.research_broadcast_reward <= 0.08):
            raise ValueError("invalid research_broadcast_reward candidate")
        if not (0.0002 <= config.brain_learning_rate <= 0.006):
            raise ValueError("invalid brain_learning_rate candidate")
        if not (0.0 <= config.brain_entropy_bonus <= 0.05):
            raise ValueError("invalid brain_entropy_bonus candidate")
        if not (0.0 <= config.brain_weight_decay <= 0.001):
            raise ValueError("invalid brain_weight_decay candidate")
        if not (0.35 <= config.brain_temperature_policy <= 1.20):
            raise ValueError("invalid brain_temperature_policy candidate")

    def apply(self, farm, candidate_config: FarmConfig) -> None:
        # Only a pre-approved safe mutation surface can be changed at runtime.
        safe = tuple(self.PARAM_RANGES.keys())
        for name in safe:
            setattr(farm.config, name, copy.deepcopy(getattr(candidate_config, name)))
        farm.arena.config = replace(farm.arena.config, hostility=farm.config.arena_hostility)
        if hasattr(farm, "research_ecology"):
            farm.research_ecology.workspace.capacity_bits = int(farm.config.workspace_capacity_bits)

    def runtime_state(self) -> dict:
        return {
            "seed": self.seed,
            "min_gain": self.min_gain,
            "interval": self.interval,
            "max_trials": self.max_trials,
            "canary_generations": self.canary_generations,
            "canary_seeds": self.canary_seeds,
            "bandit": self.bandit.runtime_state(),
            "history": list(self.history),
        }

    def load_runtime_state(self, state: dict) -> None:
        if not state:
            return
        self.seed = int(state.get("seed", self.seed))
        self.min_gain = float(state.get("min_gain", self.min_gain))
        self.interval = max(1, int(state.get("interval", self.interval)))
        self.max_trials = max(1, int(state.get("max_trials", self.max_trials)))
        self.canary_generations = max(1, int(state.get("canary_generations", self.canary_generations)))
        self.canary_seeds = max(1, int(state.get("canary_seeds", self.canary_seeds)))
        self.bandit.load_runtime_state(state.get("bandit", {}))
        self.history = list(state.get("history", []))[-128:]
