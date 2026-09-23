from __future__ import annotations

from .models import FarmState, WormStatus
from .runtime import InvariantViolation


def validate_state(state: FarmState) -> None:
    ids = set(state.worms)
    if len(ids) != len(state.worms):
        raise InvariantViolation("duplicate worm ids")
    finding_ids = set(state.findings)
    if len(finding_ids) != len(state.findings):
        raise InvariantViolation("duplicate finding ids")

    for worm in state.worms.values():
        if not 0 <= worm.energy <= 1 or not 0 <= worm.integrity <= 1 or not 0 <= worm.reputation <= 1:
            raise InvariantViolation(f"bounded state violation: {worm.id}")
        if worm.generation < 0:
            raise InvariantViolation(f"negative worm generation: {worm.id}")
        if worm.depth < 0 or worm.max_depth < 0:
            raise InvariantViolation(f"depth violation: {worm.id}")
        if worm.status not in {s.value for s in WormStatus}:
            raise InvariantViolation(f"unknown worm status: {worm.id}")

    for finding in state.findings.values():
        if finding.worm_id not in ids:
            raise InvariantViolation(f"orphan finding: {finding.id}")
        if finding.depth < 0:
            raise InvariantViolation(f"negative finding depth: {finding.id}")
        for value in (finding.novelty, finding.information_gain, finding.anomaly,
                      finding.validity, finding.falsifiability, finding.independence,
                      finding.counterfactual_score, finding.robustness):
            if not 0 <= value <= 1:
                raise InvariantViolation(f"finding score out of bounds: {finding.id}")

    # Sexual-generation invariant is stronger than the mating gate.
    for _, (a, b) in getattr(state, "lineage_parents", {}).items():
        if a in state.worms and b in state.worms:
            if state.worms[a].generation == state.worms[b].generation:
                raise InvariantViolation("same-generation parent pair found")


def assert_reproducible(seed: int, run_factory, projector):
    a = run_factory(seed)
    b = run_factory(seed)
    pa, pb = projector(a), projector(b)
    if pa != pb:
        raise InvariantViolation("same-seed runs diverged")
