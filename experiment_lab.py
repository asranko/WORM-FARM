from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
import statistics
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence

from .models import Claim, FarmConfig
from .farm import WormFarm


@dataclass(frozen=True)
class SyntheticTask:
    task_id: str
    family: str
    claim: Claim
    target_signal: str
    trap: str
    difficulty: float


@dataclass
class TrialResult:
    architecture: str
    seed: int
    task_id: str
    generations: int
    findings: int
    open_findings: int
    max_depth: int
    births: int
    arena_matches: int
    learning_events: int
    diversity: float
    assumption_breaks: int
    counterfactual_events: int
    self_attacks: int
    robustness: float
    compute_events: int
    deterministic_digest: str


@dataclass
class BenchmarkSummary:
    architecture: str
    trials: int
    completed_trials: int
    mean_findings: float
    mean_quality: float
    mean_depth: float
    mean_diversity: float
    mean_assumption_breaks: float
    mean_counterfactual: float
    mean_robustness: float
    mean_compute_events: float
    failure_rate: float


class WORMBench:
    """Synthetic benchmark and ablation harness for WORM FARM.

    Tasks are synthetic cognitive problems. No external targets or harmful actions are used.
    """

    FAMILY_TEMPLATES = (
        ("ASSUMPTION", "The current explanation is complete and no alternative variable matters.", "hidden alternative variable", "completion_bias"),
        ("CAUSAL", "X and Y move together, therefore X is the cause of Y.", "correlation vs causality", "causal_jump"),
        ("BOUNDARY", "The rule works in observed cases, therefore it works for all cases.", "boundary condition", "generalization_overreach"),
        ("CONSENSUS", "Most available interpretations agree, therefore the framing is exhaustive.", "independent evidence paths", "consensus_lock"),
        ("UNKNOWN", "No measurement currently reveals mechanism Z, so Z cannot matter.", "measurement blind spot", "absence_of_evidence"),
        ("TRANSFER", "A strategy that works in domain A should work unchanged in domain B.", "transfer boundary", "context_leakage"),
    )

    def make_tasks(self, count_per_family: int = 10) -> List[SyntheticTask]:
        out: List[SyntheticTask] = []
        seq = 0
        for family, text, signal, trap in self.FAMILY_TEMPLATES:
            for i in range(count_per_family):
                seq += 1
                claim_id = f"WB-{seq:04d}"
                claim = Claim(
                    id=claim_id,
                    text=f"{text} Scenario={family}-{i:02d}.",
                    evidence=[f"synthetic-evidence-{family.lower()}-{i}-a", f"synthetic-evidence-{family.lower()}-{i}-b"],
                    assumptions=[f"{signal} is represented by the current framing.", f"The relevant search boundary is already known."],
                )
                out.append(SyntheticTask(claim_id, family, claim, signal, trap, 0.45 + 0.05 * (i % 6)))
        return out

    @staticmethod
    def _config(base: FarmConfig, architecture: str, seed: int, generations: int) -> FarmConfig:
        cfg = FarmConfig(**{k: v for k, v in asdict(base).items() if k != "forbidden_terms"})
        cfg.forbidden_terms = base.forbidden_terms
        cfg.random_seed = seed
        cfg.max_generations = generations
        cfg.brain_provider = "rule-based"
        if architecture == "single":
            cfg.arena_enabled = False; cfg.genetics_enabled = False; cfg.learning_enabled = False
            cfg.colony_os_enabled = False; cfg.specialist_ecology_enabled = False; cfg.curriculum_enabled = False
            cfg.experiments_enabled = False; cfg.meta_controller_enabled = False; cfg.worm_warfare = False
        elif architecture == "multi":
            cfg.genetics_enabled = False; cfg.arena_enabled = False; cfg.learning_enabled = False
            cfg.colony_os_enabled = False; cfg.specialist_ecology_enabled = False; cfg.curriculum_enabled = False
            cfg.experiments_enabled = False; cfg.meta_controller_enabled = False
        elif architecture == "adversarial":
            cfg.genetics_enabled = False; cfg.colony_os_enabled = False; cfg.curriculum_enabled = False
            cfg.experiments_enabled = False; cfg.meta_controller_enabled = False
        return cfg

    def run_trial(self, task: SyntheticTask, architecture: str, seed: int, generations: int = 8) -> TrialResult:
        cfg = self._config(FarmConfig(max_generations=generations, max_worms=64, founder_pairs=4), architecture, seed, generations)
        farm = WormFarm(cfg)
        farm.seed(task.claim)
        farm.run()
        report = farm.report()
        findings = list(farm.state.findings.values())
        assumption_breaks = sum(1 for f in findings if f.assumptions_targeted)
        counterfactual = sum(1 for f in findings if f.counterfactual_score >= 0.35)
        digest_payload = {
            "arch": architecture, "seed": seed, "history": report["generation_history"],
            "top": report.get("top_findings", []), "diversity": report.get("genome_diversity", 0.0),
        }
        digest = hashlib.sha256(json.dumps(digest_payload, sort_keys=True, default=str).encode()).hexdigest()
        return TrialResult(
            architecture, seed, task.task_id, len(report["generation_history"]), report["findings"],
            report["open_findings"], report["max_depth"], report["genetic_births"], report["arena_matches"],
            report["learning_events"], report.get("genome_diversity", 0.0), assumption_breaks,
            counterfactual, report.get("meta_self_attacks", 0), statistics.mean([x["mean_resilience"] for x in report["generation_history"]] or [0.0]),
            report.get("brain_calls", 0) + report.get("findings", 0), digest,
        )

    def run_suite(self, architectures: Sequence[str], tasks: Sequence[SyntheticTask], seeds: Sequence[int], generations: int = 8, limit_tasks: int | None = None) -> List[TrialResult]:
        chosen = list(tasks[:limit_tasks]) if limit_tasks else list(tasks)
        out: List[TrialResult] = []
        for architecture in architectures:
            for seed in seeds:
                for task in chosen:
                    out.append(self.run_trial(task, architecture, seed, generations))
        return out

    @staticmethod
    def summarize(results: Sequence[TrialResult]) -> List[BenchmarkSummary]:
        by: Dict[str, List[TrialResult]] = {}
        for r in results: by.setdefault(r.architecture, []).append(r)
        out = []
        for arch, rs in sorted(by.items()):
            out.append(BenchmarkSummary(
                architecture=arch, trials=len(rs), completed_trials=sum(r.generations > 0 for r in rs),
                mean_findings=statistics.mean(r.findings for r in rs),
                mean_quality=statistics.mean(r.robustness for r in rs),
                mean_depth=statistics.mean(r.max_depth for r in rs),
                mean_diversity=statistics.mean(r.diversity for r in rs),
                mean_assumption_breaks=statistics.mean(r.assumption_breaks for r in rs),
                mean_counterfactual=statistics.mean(r.counterfactual_events for r in rs),
                mean_robustness=statistics.mean(r.robustness for r in rs),
                mean_compute_events=statistics.mean(r.compute_events for r in rs),
                failure_rate=statistics.mean(r.generations < 1 for r in rs),
            ))
        return out

    @staticmethod
    def save(results: Sequence[TrialResult], summaries: Sequence[BenchmarkSummary], out_dir: str | Path) -> None:
        p = Path(out_dir); p.mkdir(parents=True, exist_ok=True)
        (p / "trials.json").write_text(json.dumps([asdict(r) for r in results], indent=2, sort_keys=True), encoding="utf-8")
        (p / "summaries.json").write_text(json.dumps([asdict(s) for s in summaries], indent=2, sort_keys=True), encoding="utf-8")
        lines = ["# WORM BENCH Results", ""]
        for s in summaries:
            lines.append(f"- {s.architecture}: trials={s.trials}, mean_findings={s.mean_findings:.3f}, mean_depth={s.mean_depth:.3f}, mean_assumption_breaks={s.mean_assumption_breaks:.3f}, mean_robustness={s.mean_robustness:.3f}, failure_rate={s.failure_rate:.3f}")
        (p / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
