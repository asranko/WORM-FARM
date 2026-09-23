from __future__ import annotations
from dataclasses import dataclass
from typing import List

from .models import Claim, Finding, Worm, WormKind


@dataclass
class AttackContext:
    claim: Claim
    parent: Finding | None
    related_findings: List[Finding]
    pressure: dict | None = None
    genome: dict | None = None
    forbidden_zone_hits: List[str] | None = None
    generation: int = 0
    sequence: int = 0
    specialty_skill: float = 0.25
    specialty_name: str = "GENERALIST"


def _assumption_targets(claim: Claim) -> List[str]:
    if claim.assumptions:
        return claim.assumptions
    text = claim.text.lower()
    candidates = []
    if "therefore" in text or "because" in text:
        candidates.append("The causal bridge is valid.")
    candidates.extend([
        "The measured variables adequately represent the phenomenon.",
        "The candidate alternatives are sufficiently covered by the search.",
        "The current framing contains the complete solution space.",
    ])
    return candidates


def attack(worm: Worm, ctx: AttackContext) -> Finding:
    claim, parent = ctx.claim, ctx.parent
    g = ctx.genome or {}
    if ctx.specialty_name == "SHASE":
        title = "Influence dynamics audit"
        detail = (
            "Decompose the synthetic artifact into stimulus, appraisal, arousal, attention, "
            "framing pressure, social pressure, evidence support, and decision-pressure components; "
            "then test whether the apparent belief shift survives removal of the strongest mechanism."
        )
        novelty, gain, anomaly, validity, falsifiability, independence, actionability = .74, .90, .62, .88, .94, .92, .84
        pressure = ctx.pressure or {}
        noise = float(pressure.get("evidence_noise", 0.0))
        contradiction = float(pressure.get("contradiction_rate", 0.0))
        validity = max(.08, validity - .05*noise - .03*contradiction)
        fidelity = max(.05, min(1.0, .55*float(g.get("meta_cognition", .5)) + .45*ctx.specialty_skill))
        gain = min(1.0, gain + .05*fidelity)
        fid = f"F-{ctx.generation:04d}-{ctx.sequence:05d}-" + worm.id
        return Finding(
            id=fid, worm_id=worm.id, kind=worm.kind, depth=worm.depth,
            title=title, detail=detail, parent_finding_id=parent.id if parent else None,
            evidence_refs=list(claim.evidence), assumptions_targeted=_assumption_targets(claim),
            novelty=max(0.0, min(1.0, novelty + .04*float(g.get("novelty", .5)))),
            information_gain=gain, anomaly=anomaly, validity=validity,
            falsifiability=falsifiability, independence=independence, actionability=actionability,
            habit_break=.70,
            deception_resistance=min(1.0, .55*falsifiability + .25*float(g.get("meta_cognition", .5)) + .20*float(g.get("restraint", .5))),
            independent_path=f"{worm.id}:{worm.depth}:SHASE:{ctx.generation}",
        )
    specs = {
        WormKind.SCOUT: ("Unexamined boundary", "Probe what the framing does not attempt to distinguish.", .66, .80, .42, .75, .80, .58, .60),
        WormKind.BURROWER: ("Hidden assumption", "Descend into the dependency: {target}", .62, .86, .54, .76, .84, .61, .65),
        WormKind.BREACHER: ("Frame break", "Construct a materially different problem framing.", .90, .83, .62, .71, .78, .70, .68),
        WormKind.PROBER: ("Minimal discriminating test", "Identify the smallest synthetic test separating leading explanations.", .50, .92, .52, .86, .93, .75, .82),
        WormKind.MIRROR: ("Self-counterexample", "Try to falsify the strongest interpretation with an internally coherent countermodel.", .70, .86, .70, .83, .91, .74, .78),
        WormKind.CROSSWORM: ("Cross-layer dependency", "Search for a hidden dependency shared by apparently separate findings.", .80, .88, .66, .76, .86, .89, .80),
        WormKind.WORM_EATER: ("Cognitive parasite audit", "Attack repetitive reasoning, suspicion inflation, and circular support.", .36, .76, .38, .94, .97, .92, .70),
        WormKind.OMEGA: ("Colony meta-attack", "Attack the rules by which the colony decides what deserves another excavation.", .44, .93, .52, .95, .97, .94, .86),
    }
    title, detail, novelty, gain, anomaly, validity, falsifiability, independence, actionability = specs[worm.kind]
    if worm.kind == WormKind.BURROWER:
        targets = _assumption_targets(claim)
        detail = detail.format(target=targets[worm.depth % len(targets)])
    if parent is not None:
        validity = max(.20, validity - 0.018 * parent.depth)
        independence = max(.20, independence - 0.012 * parent.depth)

    pressure = ctx.pressure or {}
    noise = float(pressure.get("evidence_noise", 0.0))
    contradiction = float(pressure.get("contradiction_rate", 0.0))
    drift = float(pressure.get("concept_drift", 0.0))
    penetration = float(g.get("penetration", .5))
    skepticism = float(g.get("skepticism", .5))
    restraint = float(g.get("restraint", .5))
    validity = max(.08, validity - 0.06 * noise - 0.04 * contradiction + 0.03 * restraint)
    falsifiability = max(.08, falsifiability - 0.035 * drift + 0.03 * skepticism)
    skill = max(0.0, min(1.0, ctx.specialty_skill))
    validity = min(1.0, validity + 0.055 * skill)
    gain = min(1.0, gain + min(.12, 0.04 * worm.depth * penetration) + .035 * skill)
    actionability = min(1.0, actionability + .025 * skill)
    forbidden_hits = ctx.forbidden_zone_hits or []
    habit_break = min(1.0, .35 * bool(forbidden_hits) + .45 * (worm.kind in {WormKind.BREACHER, WormKind.WORM_EATER, WormKind.OMEGA}))
    fid = f"F-{ctx.generation:04d}-{ctx.sequence:05d}-{worm.id}"
    return Finding(
        id=fid, worm_id=worm.id, kind=worm.kind, depth=worm.depth,
        title=title, detail=detail, parent_finding_id=parent.id if parent else None,
        evidence_refs=list(claim.evidence), assumptions_targeted=_assumption_targets(claim),
        novelty=max(.0, min(1.0, novelty + 0.06 * float(g.get("novelty", .5)))),
        information_gain=gain, anomaly=anomaly, validity=validity,
        falsifiability=falsifiability, independence=independence, actionability=actionability,
        habit_break=habit_break,
        deception_resistance=min(1.0, .5*falsifiability + .3*float(g.get("meta_cognition", .5)) + .2*restraint),
        independent_path=f"{worm.id}:{worm.depth}:{worm.kind.value}:{ctx.generation}",
    )


def choose_children(parent: Finding) -> List[WormKind]:
    routing = {
        WormKind.BREACHER: [WormKind.PROBER, WormKind.MIRROR, WormKind.CROSSWORM],
        WormKind.CROSSWORM: [WormKind.MIRROR, WormKind.WORM_EATER, WormKind.BURROWER],
        WormKind.PROBER: [WormKind.BURROWER, WormKind.CROSSWORM, WormKind.MIRROR],
        WormKind.MIRROR: [WormKind.PROBER, WormKind.WORM_EATER, WormKind.BREACHER],
        WormKind.WORM_EATER: [WormKind.PROBER, WormKind.BURROWER, WormKind.SCOUT],
        WormKind.BURROWER: [WormKind.BREACHER, WormKind.PROBER, WormKind.CROSSWORM],
        WormKind.SCOUT: [WormKind.BURROWER, WormKind.BREACHER, WormKind.PROBER],
        WormKind.OMEGA: [WormKind.WORM_EATER, WormKind.MIRROR, WormKind.PROBER],
    }
    return routing[parent.kind]
