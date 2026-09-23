from __future__ import annotations

from dataclasses import asdict
import csv
import json
from pathlib import Path
from typing import Iterable

from .experiment_lab import WORMBench, SyntheticTask, TrialResult
from .farm import WormFarm
from .models import Claim, FarmConfig
from .v2_engine import EvolvingWormFarm


ARCHITECTURES = (
    "single",
    "multi",
    "adversarial",
    "worm_v2",
    "worm_v2_no_genetics",
    "worm_v2_no_arena",
    "worm_v2_no_harsh",
    "worm_v2_no_curriculum",
    "worm_v2_no_omega",
)


def make_config(architecture: str, seed: int, generations: int) -> FarmConfig:
    cfg = FarmConfig(
        max_generations=generations,
        max_worms=48,
        founder_pairs=3,
        random_seed=seed,
        arena_matches_per_generation=8,
        max_curriculum_tasks_per_generation=8,
        max_experiments_per_generation=5,
    )
    if architecture == "single":
        cfg.arena_enabled = False; cfg.genetics_enabled = False; cfg.learning_enabled = False
        cfg.colony_os_enabled = False; cfg.specialist_ecology_enabled = False
        cfg.curriculum_enabled = False; cfg.experiments_enabled = False; cfg.meta_controller_enabled = False
        cfg.worm_warfare = False
    elif architecture == "multi":
        cfg.arena_enabled = False; cfg.genetics_enabled = False; cfg.learning_enabled = False
        cfg.colony_os_enabled = False; cfg.specialist_ecology_enabled = False
        cfg.curriculum_enabled = False; cfg.experiments_enabled = False; cfg.meta_controller_enabled = False
    elif architecture == "adversarial":
        cfg.genetics_enabled = False; cfg.colony_os_enabled = False
        cfg.curriculum_enabled = False; cfg.experiments_enabled = False; cfg.meta_controller_enabled = False
    elif architecture == "worm_v2_no_genetics":
        cfg.genetics_enabled = False
    elif architecture == "worm_v2_no_arena":
        cfg.arena_enabled = False
    elif architecture == "worm_v2_no_harsh":
        cfg.harsh_learning = False
    elif architecture == "worm_v2_no_curriculum":
        cfg.curriculum_enabled = False
    elif architecture == "worm_v2_no_omega":
        cfg.omega_enabled = False
    return cfg


def run_one(task: SyntheticTask, architecture: str, seed: int, generations: int) -> TrialResult:
    cfg = make_config(architecture, seed, generations)
    Farm = WormFarm if architecture in {"single", "multi", "adversarial"} else EvolvingWormFarm
    farm = Farm(cfg)
    farm.seed(task.claim)
    farm.run()
    r = farm.report()
    findings = list(farm.state.findings.values())
    history = r.get("generation_history", [])
    quality = sum((x.get("mean_resilience", 0.0) + x.get("mean_adaptation", 0.0)) * 0.5 for x in history) / max(1, len(history))
    assumption_breaks = sum(bool(f.assumptions_targeted) for f in findings)
    counterfactual = sum(f.counterfactual_score >= 0.35 for f in findings)
    payload = {
        "architecture": architecture,
        "seed": seed,
        "task": task.task_id,
        "history": history,
        "assumption_breaks": assumption_breaks,
        "counterfactual": counterfactual,
    }
    digest = __import__("hashlib").sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
    return TrialResult(
        architecture=architecture, seed=seed, task_id=task.task_id,
        generations=len(history), findings=r.get("findings", 0), open_findings=r.get("open_findings", 0),
        max_depth=r.get("max_depth", 0), births=r.get("genetic_births", 0), arena_matches=r.get("arena_matches", 0),
        learning_events=r.get("learning_events", 0), diversity=r.get("genome_diversity", 0.0),
        assumption_breaks=assumption_breaks, counterfactual_events=counterfactual,
        self_attacks=r.get("meta_self_attacks", 0), robustness=quality,
        compute_events=r.get("brain_calls", 0) + r.get("findings", 0), deterministic_digest=digest,
    )


def run_benchmark(*, tasks: Iterable[SyntheticTask], seeds: Iterable[int], generations: int = 5, architectures: Iterable[str] = ARCHITECTURES) -> list[TrialResult]:
    results: list[TrialResult] = []
    for architecture in architectures:
        for seed in seeds:
            for task in tasks:
                results.append(run_one(task, architecture, seed, generations))
    return results


def summarize(results: list[TrialResult]) -> list[dict]:
    groups: dict[str, list[TrialResult]] = {}
    for r in results:
        groups.setdefault(r.architecture, []).append(r)
    out = []
    for arch, rs in sorted(groups.items()):
        def mean(attr: str) -> float:
            vals = [float(getattr(x, attr)) for x in rs]
            return sum(vals) / max(1, len(vals))
        out.append({
            "architecture": arch,
            "trials": len(rs),
            "mean_findings": mean("findings"),
            "mean_depth": mean("max_depth"),
            "mean_assumption_breaks": mean("assumption_breaks"),
            "mean_counterfactual": mean("counterfactual_events"),
            "mean_robustness": mean("robustness"),
            "mean_diversity": mean("diversity"),
            "mean_compute": mean("compute_events"),
            "mean_arena": mean("arena_matches"),
            "mean_learning": mean("learning_events"),
            "failure_rate": sum(r.generations < 1 for r in rs) / max(1, len(rs)),
            "unique_digests": len({r.deterministic_digest for r in rs}),
        })
    return out


def save(results: list[TrialResult], summaries: list[dict], out_dir: str | Path) -> None:
    p = Path(out_dir); p.mkdir(parents=True, exist_ok=True)
    (p / "trials.json").write_text(json.dumps([asdict(r) for r in results], indent=2, sort_keys=True), encoding="utf-8")
    (p / "summaries.json").write_text(json.dumps(summaries, indent=2, sort_keys=True), encoding="utf-8")
    with (p / "trials.csv").open("w", newline="", encoding="utf-8") as fh:
        rows = [asdict(r) for r in results]
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["architecture"])
        writer.writeheader(); writer.writerows(rows)
    lines = ["# WORM FARM v2 Scientific Benchmark", "", "This benchmark is synthetic and closed-world. It is not evidence about human physiology or real-world targets.", ""]
    for s in summaries:
        lines.append("- {architecture}: trials={trials}, findings={mean_findings:.3f}, depth={mean_depth:.3f}, assumption_breaks={mean_assumption_breaks:.3f}, counterfactual={mean_counterfactual:.3f}, robustness={mean_robustness:.3f}, diversity={mean_diversity:.3f}, compute={mean_compute:.3f}, failure_rate={failure_rate:.3f}".format(**s))
    (p / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
