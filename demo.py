from __future__ import annotations
import json
from .farm import WormFarm
from .models import Claim, FarmConfig


def main() -> None:
    claim = Claim(
        id="SYNTH-01",
        text="A current explanatory model fits the observation, therefore it should be treated as the complete solution.",
        evidence=["synthetic-measurement-A", "synthetic-analysis-B", "synthetic-counterexample-C"],
        assumptions=[
            "The current framing contains the full solution space.",
            "No alternative explanation matters.",
            "Matching the current model excludes useful unknown structure.",
        ],
        confidence=0.72,
        provenance="synthetic-demo",
    )
    cfg = FarmConfig(max_generations=10, max_worms=72, max_depth=7, founder_pairs=4, random_seed=42)
    farm = WormFarm(cfg)
    farm.seed(claim)
    state = farm.run()
    print(json.dumps(farm.report(), ensure_ascii=False, indent=2))
    print("\nGenetic birth sample:")
    for r in [x for x in farm.genetics.records if x.success][:12]:
        child = state.worms[r.child_id]
        pa, pb = state.worms[r.parent_a], state.worms[r.parent_b]
        print(f"- {child.id} gen={child.generation} sex={child.sex} parents=({pa.id}:{pa.generation},{pb.id}:{pb.generation})")


if __name__ == "__main__":
    main()
