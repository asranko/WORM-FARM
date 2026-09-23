from __future__ import annotations

from dataclasses import dataclass, asdict
from collections import defaultdict
import hashlib
import json
import math
import random
from typing import Any, Dict, Iterable, Mapping

from .models import Worm
from .runtime import pack_state, unpack_state

MODES = ("AR", "DIFFUSION")


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


def stable_control_bucket(worm_id: str, generation: int, seed: int = 0) -> float:
    raw = f"{seed}|{worm_id}|{generation}|WORM-CONTROL-v2".encode("utf-8")
    value = int(hashlib.sha256(raw).hexdigest()[:12], 16)
    return value / float(16**12 - 1)


def genome_cluster(phenotype: Mapping[str, float]) -> str:
    keys = sorted(phenotype)
    quantized = {k: int(round(_clamp(float(phenotype[k])) * 8.0)) for k in keys}
    raw = json.dumps(quantized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]


def generation_bucket(generation: int, width: int = 4) -> int:
    width = max(1, int(width))
    return int(generation) // width


def strategy_cluster(strategy: Mapping[str, float] | None) -> str:
    """Stable low-cardinality cluster for routing context."""
    if not strategy:
        return "default"
    keys = sorted(strategy)
    quantized = {k: int(round(_clamp(float(strategy[k])) * 8.0)) for k in keys}
    raw = json.dumps(quantized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]


def stable_json_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


@dataclass
class BetaPosterior:
    alpha: float = 1.0
    beta: float = 1.0
    pulls: int = 0
    reward_sum: float = 0.0

    @property
    def mean(self) -> float:
        return self.alpha / max(1e-12, self.alpha + self.beta)

    def update(self, reward: float) -> None:
        r = _clamp(reward)
        self.alpha += r
        self.beta += 1.0 - r
        self.pulls += 1
        self.reward_sum += r


@dataclass(frozen=True)
class ModeDecision:
    decision_id: str
    generation: int
    worm_id: str
    specialty: str
    depth: int
    generation_bucket: int
    genome_cluster: str
    routing_bias: float
    control_group: bool
    control_threshold: float
    chosen_mode: str
    other_mode: str
    ar_signal: float
    diff_signal: float
    mode_score: float
    uncertainty: float
    sampled_ar: float
    sampled_diffusion: float
    posterior_ar: float
    posterior_diffusion: float
    predicted_success_probability: float
    cost_units: float
    input_tokens: int
    genetic_bias_active: bool
    strategy_cluster: str = "default"
    workspace_state_hash: str = ""
    raw_ar_signal: float = 0.0
    raw_diff_signal: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class PlattSignalCalibrator:
    """Per-mode logistic calibration over bounded routing scores.

    score -> logit(score) -> sigmoid(intercept + slope*logit(score))
    The fitted parameters are external research artifacts and can be injected into the bandit.
    """
    def __init__(self, params: Mapping[str, Mapping[str, float]] | None = None) -> None:
        self.params = {str(k): {str(pk): float(pv) for pk, pv in v.items()} for k, v in (params or {}).items()}

    @staticmethod
    def _logit(p: float) -> float:
        p = _clamp(p, 1e-5, 1.0-1e-5)
        return math.log(p/(1.0-p))

    @staticmethod
    def _sigmoid(x: float) -> float:
        x = max(-30.0, min(30.0, float(x)))
        return 1.0/(1.0+math.exp(-x))

    def transform(self, mode: str, score: float) -> float:
        raw = _clamp(score)
        p = self.params.get(mode)
        if not p:
            return raw
        return _clamp(self._sigmoid(float(p.get('intercept', 0.0)) + float(p.get('slope', 1.0))*self._logit(raw)))

    def runtime_state(self) -> dict[str, Any]:
        return {'params': self.params}

    @classmethod
    def from_runtime_state(cls, state: Mapping[str, Any]) -> 'PlattSignalCalibrator':
        return cls(state.get('params', {}) if state else {})


class EmpiricalRoutingSignal:
    """Empirical Bayesian success signal keyed by (specialty, mode).

    Only observed binary outcomes update the signal. The Beta prior supplies a neutral
    cold start; confidence counts actual observations and excludes prior pseudo-counts.
    """
    def __init__(self, smoothing_prior: float = 0.5, prior_strength: int = 4) -> None:
        prior_strength = max(1, int(prior_strength))
        prior = max(1e-6, min(1.0 - 1e-6, float(smoothing_prior)))
        self.prior_alpha = prior_strength * prior
        self.prior_beta = prior_strength * (1.0 - prior)
        self.stats: dict[tuple[str, str], dict[str, float]] = {}

    def _row(self, specialty: str, mode: str) -> dict[str, float]:
        key = (str(specialty), str(mode))
        if key not in self.stats:
            self.stats[key] = {
                "alpha": float(self.prior_alpha),
                "beta": float(self.prior_beta),
                "observations": 0.0,
            }
        return self.stats[key]

    def observe(self, specialty: str, mode: str, audit_passed: bool, survived_to_next_gen: bool) -> None:
        row = self._row(specialty, mode)
        success = bool(audit_passed and survived_to_next_gen)
        row["alpha"] += 1.0 if success else 0.0
        row["beta"] += 0.0 if success else 1.0
        row["observations"] += 1.0

    def score(self, specialty: str, mode: str) -> float:
        row = self._row(specialty, mode)
        return float(row["alpha"] / max(1e-12, row["alpha"] + row["beta"]))

    def confidence(self, specialty: str, mode: str) -> int:
        return int(self._row(specialty, mode)["observations"])

    def ready(self, specialty: str, min_confidence: int = 10) -> bool:
        return all(self.confidence(specialty, mode) >= int(min_confidence) for mode in MODES)

    def posterior(self, specialty: str, mode: str) -> tuple[float, float]:
        row = self._row(specialty, mode)
        return float(row["alpha"]), float(row["beta"])

    def runtime_state(self) -> dict[str, Any]:
        return {
            "prior_alpha": self.prior_alpha,
            "prior_beta": self.prior_beta,
            "stats": {f"{k[0]}|{k[1]}": dict(v) for k, v in self.stats.items()},
        }

    def load_runtime_state(self, state: Mapping[str, Any]) -> None:
        if not state:
            return
        self.prior_alpha = float(state.get("prior_alpha", self.prior_alpha))
        self.prior_beta = float(state.get("prior_beta", self.prior_beta))
        self.stats = {}
        for key, value in state.get("stats", {}).items():
            specialty, mode = key.split("|", 1)
            self.stats[(specialty, mode)] = {
                "alpha": float(value.get("alpha", self.prior_alpha)),
                "beta": float(value.get("beta", self.prior_beta)),
                "observations": float(value.get("observations", 0.0)),
            }


class RoutingBandit:
    """Contextual Thompson sampler with a permanently held-out threshold control arm."""

    def __init__(
        self,
        *,
        seed: int = 42,
        control_fraction: float = 0.10,
        control_threshold: float = 0.18,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        evidence_weight: float = 0.35,
        routing_bias_weight: float = 0.12,
        generation_bucket_width: int = 4,
        signal_calibration: Mapping[str, Mapping[str, float]] | None = None,
        empirical_signal_enabled: bool = True,
        empirical_min_confidence: int = 10,
    ) -> None:
        self.seed = int(seed)
        self.control_fraction = _clamp(control_fraction)
        self.control_threshold = float(control_threshold)
        self.prior_alpha = float(prior_alpha)
        self.prior_beta = float(prior_beta)
        self.evidence_weight = float(max(0.0, evidence_weight))
        self.routing_bias_weight = float(max(0.0, routing_bias_weight))
        self.generation_bucket_width = max(1, int(generation_bucket_width))
        self.rng = random.Random(self.seed)
        self.posteriors: dict[str, dict[str, BetaPosterior]] = {}
        self.control_posteriors: dict[str, BetaPosterior] = {m: BetaPosterior(self.prior_alpha, self.prior_beta) for m in MODES}
        self.decision_count = 0
        self.control_count = 0
        self.genetic_bias_active = False
        self.calibrator = PlattSignalCalibrator(signal_calibration)
        self.empirical_signal_enabled = bool(empirical_signal_enabled)
        self.empirical_min_confidence = max(1, int(empirical_min_confidence))
        self.empirical_signal = EmpiricalRoutingSignal()

    def context_key(
        self,
        worm: Worm,
        phenotype: Mapping[str, float],
        strategy: Mapping[str, float] | None = None,
    ) -> str:
        return "|".join((
            worm.specialty,
            f"d{min(8, max(0, int(worm.depth)))}",
            f"g{generation_bucket(worm.generation, self.generation_bucket_width)}",
            f"gc{genome_cluster(phenotype)}",
            f"sc{strategy_cluster(strategy)}",
        ))

    def _arms(self, key: str) -> dict[str, BetaPosterior]:
        arms = self.posteriors.get(key)
        if arms is None:
            arms = {m: BetaPosterior(self.prior_alpha, self.prior_beta) for m in MODES}
            self.posteriors[key] = arms
        return arms

    def is_control(self, worm_id: str, generation: int) -> bool:
        return stable_control_bucket(worm_id, generation, self.seed) < self.control_fraction

    def select(
        self,
        *,
        worm: Worm,
        phenotype: Mapping[str, float],
        ar_signal: float,
        diff_signal: float,
        mode_score: float,
        uncertainty: float,
        input_tokens: int,
        strategy: Mapping[str, float] | None = None,
        workspace_state: Any | None = None,
        decision_generation: int | None = None,
    ) -> ModeDecision:
        current_generation = int(worm.generation if decision_generation is None else decision_generation)
        raw_ar_signal = _clamp(ar_signal)
        raw_diff_signal = _clamp(diff_signal)
        calibrated_ar_signal = self.calibrator.transform("AR", raw_ar_signal)
        calibrated_diff_signal = self.calibrator.transform("DIFFUSION", raw_diff_signal)
        key = self.context_key(worm, phenotype, strategy)
        # The routing context uses the generation in which the decision is made,
        # not the worm's birth generation. This keeps per-generation outcomes aligned
        # with downstream audit and survival resolution.
        if decision_generation is not None:
            key = "|".join((
                worm.specialty,
                f"d{min(8, max(0, int(worm.depth)))}",
                f"g{generation_bucket(current_generation, self.generation_bucket_width)}",
                f"gc{genome_cluster(phenotype)}",
                f"sc{strategy_cluster(strategy)}",
            ))
        arms = self._arms(key)
        control = self.is_control(worm.id, current_generation)
        sampled_ar = arms["AR"].mean
        sampled_diff = arms["DIFFUSION"].mean
        empirical_ready = bool(self.empirical_signal_enabled and self.empirical_signal.ready(worm.specialty, self.empirical_min_confidence))
        if not control:
            if empirical_ready:
                empirical_ar_a, empirical_ar_b = self.empirical_signal.posterior(worm.specialty, "AR")
                empirical_diff_a, empirical_diff_b = self.empirical_signal.posterior(worm.specialty, "DIFFUSION")
                sampled_ar = self.rng.betavariate(max(1e-6, empirical_ar_a), max(1e-6, empirical_ar_b))
                sampled_diff = self.rng.betavariate(max(1e-6, empirical_diff_a), max(1e-6, empirical_diff_b))
                ar_value = sampled_ar
                diff_value = sampled_diff
                if self.genetic_bias_active:
                    bias01 = _clamp((float(getattr(worm, "routing_bias", 0.0)) + 1.0) / 2.0)
                    ar_value += self.routing_bias_weight * (1.0 - bias01)
                    diff_value += self.routing_bias_weight * bias01
                mode = "DIFFUSION" if diff_value >= ar_value else "AR"
                evidence = self.empirical_signal.score(worm.specialty, mode)
                posterior = evidence
            else:
                sampled_ar = self.rng.betavariate(max(1e-6, arms["AR"].alpha), max(1e-6, arms["AR"].beta))
                sampled_diff = self.rng.betavariate(max(1e-6, arms["DIFFUSION"].alpha), max(1e-6, arms["DIFFUSION"].beta))
                if self.genetic_bias_active:
                    bias01 = _clamp((float(getattr(worm, "routing_bias", 0.0)) + 1.0) / 2.0)
                else:
                    bias01 = 0.5
                ar_value = sampled_ar + self.evidence_weight * calibrated_ar_signal + self.routing_bias_weight * (1.0 - bias01)
                diff_value = sampled_diff + self.evidence_weight * calibrated_diff_signal + self.routing_bias_weight * bias01
                mode = "DIFFUSION" if diff_value >= ar_value else "AR"
                evidence = float(calibrated_diff_signal if mode == "DIFFUSION" else calibrated_ar_signal)
                posterior = arms[mode].mean
        else:
            mode = "DIFFUSION" if float(mode_score) >= self.control_threshold else "AR"
            evidence = float(calibrated_diff_signal if mode == "DIFFUSION" else calibrated_ar_signal)
            posterior = arms[mode].mean
        other = "DIFFUSION" if mode == "AR" else "AR"
        predicted = _clamp(float(posterior) if empirical_ready else (0.65 * posterior + 0.35 * _clamp(evidence)))
        self.decision_count += 1
        if control:
            self.control_count += 1
        decision_id = f"RD-{current_generation:04d}-{self.decision_count:07d}"
        sc = strategy_cluster(strategy)
        wh = stable_json_hash(workspace_state if workspace_state is not None else {})
        return ModeDecision(
            decision_id=decision_id,
            generation=current_generation,
            worm_id=worm.id,
            specialty=worm.specialty,
            depth=int(worm.depth),
            generation_bucket=generation_bucket(current_generation, self.generation_bucket_width),
            genome_cluster=genome_cluster(phenotype),
            routing_bias=float(getattr(worm, "routing_bias", 0.0)),
            control_group=control,
            control_threshold=self.control_threshold,
            chosen_mode=mode,
            other_mode=other,
            ar_signal=float(ar_signal),
            diff_signal=float(diff_signal),
            mode_score=float(mode_score),
            uncertainty=float(uncertainty),
            sampled_ar=float(sampled_ar),
            sampled_diffusion=float(sampled_diff),
            posterior_ar=float(arms["AR"].mean),
            posterior_diffusion=float(arms["DIFFUSION"].mean),
            predicted_success_probability=float(predicted),
            cost_units=max(0.1, float(input_tokens) / 64.0),
            input_tokens=int(input_tokens),
            genetic_bias_active=bool(self.genetic_bias_active),
            strategy_cluster=sc,
            workspace_state_hash=wh,
            raw_ar_signal=float(raw_ar_signal),
            raw_diff_signal=float(raw_diff_signal),
        )

    def update(self, decision: ModeDecision, reward: float, outcome: bool | None = None) -> None:
        key = "|".join((
            decision.specialty,
            f"d{decision.depth}",
            f"g{decision.generation_bucket}",
            f"gc{decision.genome_cluster}",
            f"sc{decision.strategy_cluster}",
        ))
        target = self.control_posteriors if decision.control_group else self._arms(key)
        target[decision.chosen_mode].update(reward)
        if outcome is not None and self.empirical_signal_enabled:
            self.empirical_signal.observe(decision.specialty, decision.chosen_mode, bool(outcome), True)

    def posterior_report(self) -> dict[str, Any]:
        rows: dict[str, Any] = {}
        for key in sorted(self.posteriors):
            rows[key] = {m: asdict(self.posteriors[key][m]) | {"mean": self.posteriors[key][m].mean} for m in MODES}
        return {
            "contexts": rows,
            "control": {m: asdict(self.control_posteriors[m]) | {"mean": self.control_posteriors[m].mean} for m in MODES},
            "decision_count": self.decision_count,
            "control_count": self.control_count,
        }

    def runtime_state(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "control_fraction": self.control_fraction,
            "control_threshold": self.control_threshold,
            "prior_alpha": self.prior_alpha,
            "prior_beta": self.prior_beta,
            "evidence_weight": self.evidence_weight,
            "routing_bias_weight": self.routing_bias_weight,
            "generation_bucket_width": self.generation_bucket_width,
            "posteriors": {
                k: {m: asdict(v) for m, v in arms.items()} for k, arms in self.posteriors.items()
            },
            "control_posteriors": {m: asdict(v) for m, v in self.control_posteriors.items()},
            "decision_count": self.decision_count,
            "control_count": self.control_count,
            "genetic_bias_active": self.genetic_bias_active,
            "signal_calibration": self.calibrator.runtime_state(),
            "empirical_signal_enabled": self.empirical_signal_enabled,
            "empirical_min_confidence": self.empirical_min_confidence,
            "empirical_signal": self.empirical_signal.runtime_state(),
            "rng_state": pack_state(self.rng.getstate()),
        }

    def load_runtime_state(self, state: Mapping[str, Any]) -> None:
        if not state:
            return
        self.seed = int(state.get("seed", self.seed))
        self.control_fraction = float(state.get("control_fraction", self.control_fraction))
        self.control_threshold = float(state.get("control_threshold", self.control_threshold))
        self.prior_alpha = float(state.get("prior_alpha", self.prior_alpha))
        self.prior_beta = float(state.get("prior_beta", self.prior_beta))
        self.evidence_weight = float(state.get("evidence_weight", self.evidence_weight))
        self.routing_bias_weight = float(state.get("routing_bias_weight", self.routing_bias_weight))
        self.generation_bucket_width = int(state.get("generation_bucket_width", self.generation_bucket_width))
        self.posteriors = {
            k: {m: BetaPosterior(**v) for m, v in arms.items()} for k, arms in state.get("posteriors", {}).items()
        }
        self.control_posteriors = {m: BetaPosterior(**v) for m, v in state.get("control_posteriors", {}).items()}
        for m in MODES:
            self.control_posteriors.setdefault(m, BetaPosterior(self.prior_alpha, self.prior_beta))
        self.decision_count = int(state.get("decision_count", self.decision_count))
        self.control_count = int(state.get("control_count", self.control_count))
        self.genetic_bias_active = bool(state.get("genetic_bias_active", self.genetic_bias_active))
        self.calibrator = PlattSignalCalibrator.from_runtime_state(state.get("signal_calibration", {}))
        self.empirical_signal_enabled = bool(state.get("empirical_signal_enabled", self.empirical_signal_enabled))
        self.empirical_min_confidence = max(1, int(state.get("empirical_min_confidence", self.empirical_min_confidence)))
        self.empirical_signal = EmpiricalRoutingSignal()
        self.empirical_signal.load_runtime_state(state.get("empirical_signal", {}))
        if state.get("rng_state"):
            self.rng.setstate(unpack_state(state["rng_state"]))


class CalibrationTracker:
    def __init__(self, bins: int = 10) -> None:
        self.bins = max(2, int(bins))
        self.count = 0
        self.brier_sum = 0.0
        self.by_mode: dict[str, dict[str, Any]] = {m: {"count": 0, "brier_sum": 0.0, "bins": [{"count": 0, "pred_sum": 0.0, "outcome_sum": 0.0} for _ in range(self.bins)]} for m in MODES}

    def update(self, mode: str, predicted: float, outcome: float) -> None:
        mode = mode if mode in MODES else "AR"
        p = _clamp(predicted)
        y = _clamp(outcome)
        row = self.by_mode[mode]
        self.count += 1
        self.brier_sum += (p - y) ** 2
        row["count"] += 1
        row["brier_sum"] += (p - y) ** 2
        idx = min(self.bins - 1, int(math.floor(p * self.bins)))
        row["bins"][idx]["count"] += 1
        row["bins"][idx]["pred_sum"] += p
        row["bins"][idx]["outcome_sum"] += y

    def brier(self, mode: str | None = None) -> float:
        if mode is None:
            return self.brier_sum / max(1, self.count)
        row = self.by_mode[mode]
        return row["brier_sum"] / max(1, row["count"])

    def reliability(self, mode: str) -> list[dict[str, float]]:
        out = []
        for idx, row in enumerate(self.by_mode[mode]["bins"]):
            n = row["count"]
            if not n:
                continue
            out.append({
                "bin": idx,
                "predicted_mean": row["pred_sum"] / n,
                "observed_frequency": row["outcome_sum"] / n,
                "count": float(n),
            })
        return out

    def runtime_state(self) -> dict[str, Any]:
        return {"bins": self.bins, "count": self.count, "brier_sum": self.brier_sum, "by_mode": self.by_mode}

    def load_runtime_state(self, state: Mapping[str, Any]) -> None:
        if not state:
            return
        self.bins = int(state.get("bins", self.bins))
        self.count = int(state.get("count", 0))
        self.brier_sum = float(state.get("brier_sum", 0.0))
        self.by_mode = state.get("by_mode", self.by_mode)


class RoutingDecisionBook:
    def __init__(self, *, bandit: RoutingBandit, calibration: CalibrationTracker | None = None, reward_weights: Mapping[str, float] | None = None, routing_fitness_lambda: float = 0.0) -> None:
        self.bandit = bandit
        self.calibration = calibration or CalibrationTracker()
        self.reward_weights = dict(reward_weights or {"audit":0.30,"signature":0.18,"survival":0.30,"robustness":0.17,"cost":0.05})
        self.routing_fitness_lambda = float(routing_fitness_lambda)
        self.records: list[dict[str, Any]] = []

    def append(self, decision: ModeDecision, *, finding_id: str, claim_id: str, pre_score: float, replay_payload: dict[str, Any], proposal: dict[str, Any] | None = None) -> None:
        self.records.append({
            **decision.as_dict(),
            "finding_id": finding_id,
            "claim_id": claim_id,
            "pre_score": float(pre_score),
            "proposal": dict(proposal or {}),
            "post_score": None,
            "signature_delta": None,
            "audit_pass": None,
            "audit_status": "UNRESOLVED",
            "downstream_survival": None,
            "compute_cost": float(decision.cost_units),
            "reward": None,
            "resolved": False,
            "counterfactual": None,
            "replay_payload": replay_payload,
        })

    def resolve_generation(self, *, generation: int, findings: Mapping[str, Any], worms: Mapping[str, Worm], audits: Iterable[Any]) -> dict[str, int]:
        audit_map: dict[str, list[Any]] = {}
        for a in audits:
            audit_map.setdefault(a.finding_id, []).append(a)
        resolved = 0
        skipped = 0
        for rec in self.records:
            if rec.get("resolved") or int(rec["generation"]) != int(generation):
                continue
            finding = findings.get(rec["finding_id"])
            worm = worms.get(rec["worm_id"])
            if finding is None or worm is None:
                skipped += 1
                continue
            events = audit_map.get(rec["finding_id"], [])
            if events:
                passed = bool(events[-1].passed)
                audit_status = "PASS" if passed else "FAIL"
            elif finding.status == "PRUNED":
                passed = False
                audit_status = "FAIL"
            else:
                skipped += 1
                continue
            rec["post_score"] = float(finding.score)
            rec["signature_delta"] = max(0.0, min(1.0, float(finding.score) - float(rec["pre_score"])))
            rec["audit_pass"] = int(passed)
            rec["audit_status"] = audit_status
            # This is evaluated at the generation boundary, immediately before the
            # generation counter advances. It therefore means "survived to the
            # next-generation boundary", not merely "was active at proposal time".
            rec["downstream_survival"] = int(worm.status == "ACTIVE")
            rec["downstream_fate_generation"] = int(generation) + 1
            # Explicitly separated terms: immediate audit quality, validated signature improvement,
            # survival to the next generation boundary, robustness, and normalized inference cost.
            reward = (
                self.reward_weights.get("audit", 0.30) * int(passed)
                + self.reward_weights.get("signature", 0.18) * float(rec["signature_delta"])
                + self.reward_weights.get("survival", 0.30) * int(worm.status == "ACTIVE")
                + self.reward_weights.get("robustness", 0.17) * max(0.0, min(1.0, float(finding.robustness)))
                - self.reward_weights.get("cost", 0.05) * min(1.0, float(rec["compute_cost"]) / 4.0)
            )
            reward = _clamp(reward)
            rec["reward"] = reward
            rec["resolved"] = True
            # Routing fitness is a tiny evolutionary credit channel. The optional
            # lambda term is an explicit experimental intervention for v3.5.0, default 0.
            routing_credit = _clamp(reward + self.routing_fitness_lambda * float(rec["signature_delta"]))
            worm.routing_fitness = _clamp(0.94 * float(getattr(worm, "routing_fitness", 0.50)) + 0.06 * routing_credit)
            self.bandit.update(ModeDecision(
                decision_id=rec["decision_id"], generation=rec["generation"], worm_id=rec["worm_id"],
                specialty=rec["specialty"], depth=rec["depth"], generation_bucket=rec["generation_bucket"],
                genome_cluster=rec["genome_cluster"], routing_bias=rec["routing_bias"], control_group=rec["control_group"],
                control_threshold=rec["control_threshold"], chosen_mode=rec["chosen_mode"], other_mode=rec["other_mode"],
                ar_signal=rec["ar_signal"], diff_signal=rec["diff_signal"], mode_score=rec["mode_score"], uncertainty=rec["uncertainty"],
                sampled_ar=rec["sampled_ar"], sampled_diffusion=rec["sampled_diffusion"], posterior_ar=rec["posterior_ar"],
                posterior_diffusion=rec["posterior_diffusion"], predicted_success_probability=rec["predicted_success_probability"],
                cost_units=rec["cost_units"], input_tokens=rec["input_tokens"], genetic_bias_active=rec.get("genetic_bias_active", False),
                strategy_cluster=rec.get("strategy_cluster", "default"), workspace_state_hash=rec.get("workspace_state_hash", ""),
                raw_ar_signal=rec.get("raw_ar_signal", rec["ar_signal"]), raw_diff_signal=rec.get("raw_diff_signal", rec["diff_signal"]),
            ), reward, outcome=bool(passed and worm.status == "ACTIVE"))
            self.calibration.update(rec["chosen_mode"], rec["predicted_success_probability"], int(passed and worm.status == "ACTIVE"))
            resolved += 1
        return {"resolved": resolved, "skipped": skipped}

    def add_counterfactual(self, decision_id: str, result: dict[str, Any]) -> None:
        for rec in reversed(self.records):
            if rec["decision_id"] == decision_id:
                rec["counterfactual"] = result
                return

    def runtime_state(self) -> dict[str, Any]:
        return {
            "bandit": self.bandit.runtime_state(),
            "calibration": self.calibration.runtime_state(),
            "records": self.records,
            "reward_weights": self.reward_weights,
            "routing_fitness_lambda": self.routing_fitness_lambda,
        }

    def load_runtime_state(self, state: Mapping[str, Any]) -> None:
        if not state:
            return
        self.bandit.load_runtime_state(state.get("bandit", {}))
        self.calibration.load_runtime_state(state.get("calibration", {}))
        self.reward_weights = {k: float(v) for k, v in state.get("reward_weights", self.reward_weights).items()}
        self.routing_fitness_lambda = float(state.get("routing_fitness_lambda", self.routing_fitness_lambda))
        self.records = list(state.get("records", []))
