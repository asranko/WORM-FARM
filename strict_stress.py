from __future__ import annotations

import json
import tempfile
from dataclasses import asdict
from pathlib import Path

from . import Claim, FarmConfig, WormFarm

CLAIM = Claim(
    id="GAUNTLET",
    text="The current explanatory framing is complete, so no alternative route is worth exploring.",
    evidence=["E1", "E2", "E3", "E4"],
    assumptions=[
        "The search space is complete.",
        "No alternative explanation matters.",
        "The current measurement is sufficient.",
    ],
)


def harsh_config(seed: int, generations: int = 24) -> FarmConfig:
    return FarmConfig(
        max_generations=generations,
        max_worms=96,
        founder_pairs=5,
        random_seed=seed,
        max_findings_per_generation=96,
        max_audits_per_generation=48,
        arena_matches_per_generation=16,
        arena_hostility=.98,
        arena_tournament_interval=3,
        harsh_hostility=.99,
        harsh_noise=.94,
        harsh_contradiction=.94,
        harsh_decoy=.86,
        harsh_drift=.82,
        harsh_scarcity=.90,
        harsh_deadline=.86,
        harsh_peer_attack=.96,
        extinction_interval=2,
        extinction_intensity=.60,
        omega_interval=2,
        keystone_reserve=4,
        minimum_active_worms=4,
    )


def run_seed(seed: int, generations: int = 24) -> dict:
    farm = WormFarm(harsh_config(seed, generations))
    farm.seed(CLAIM)
    farm.run()
    farm.verify()
    report = farm.report()
    history = report["generation_history"]
    assert history, "no generation history"
    assert [x["number"] for x in history] == list(range(len(history)))
    assert len({x["name"] for x in history}) == len(history)
    assert all(x["status"] == "COMPLETED" for x in history)
    assert min(x["active_end"] for x in history) >= farm.config.minimum_active_worms
    return report


def checkpoint_matrix(seed: int = 101, generations: int = 16) -> dict:
    cfg = harsh_config(seed, generations)
    full = WormFarm(cfg); full.seed(CLAIM); full.run()
    checkpoints = {}
    for cut in range(1, generations):
        part = WormFarm(cfg); part.seed(CLAIM)
        for _ in range(cut):
            part.run_generation()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / f"g{cut}.json"
            part.save_checkpoint(p)
            resumed = WormFarm.from_checkpoint(p)
            resumed.run()
        checkpoints[cut] = resumed.report()["generation_history"] == full.report()["generation_history"]
    return {"seed": seed, "all_match": all(checkpoints.values()), "checks": checkpoints}


def main() -> None:
    seeds = list(range(20))
    reports = [run_seed(s) for s in seeds]
    checkpoint = checkpoint_matrix()
    strict = {
        "profile": "HARSH-GAUNTLET",
        "seeds": seeds,
        "seed_count": len(seeds),
        "generations_per_seed": 24,
        "runs_passed": len(reports),
        "runs_failed": 0,
        "min_active_floor_observed": min(min(x["active_end"] for x in r["generation_history"]) for r in reports),
        "max_generation_depth_observed": max(max(x["max_depth"] for x in r["generation_history"]) for r in reports),
        "total_arena_matches": sum(r["arena_matches"] for r in reports),
        "total_learning_events": sum(r["learning_events"] for r in reports),
        "total_births": sum(r["genetic_births"] for r in reports),
        "total_extinction_events": sum(r["extinction_events"] for r in reports),
        "checkpoint_matrix": checkpoint,
        "note": "Synthetic adversarial software test; no real-person targeting or harmful execution.",
    }
    print(json.dumps(strict, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
