from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Tuple
import math
import random

from .models import Worm, WormKind, WormStatus
from .colony_os import SPECIALIST_BY_NAME


class ArenaMode(str, Enum):
    IDEA_DUEL = "IDEA_DUEL"
    ASSUMPTION_WAR = "ASSUMPTION_WAR"
    ANOMALY_HUNT = "ANOMALY_HUNT"
    PARADIGM_SIEGE = "PARADIGM_SIEGE"
    COUNTERFACTUAL_WAR = "COUNTERFACTUAL_WAR"
    MODEL_KILLER = "MODEL_KILLER"
    BLIND_SPOT_HUNT = "BLIND_SPOT_HUNT"
    EVIDENCE_DUEL = "EVIDENCE_DUEL"
    TIME_RAID = "TIME_RAID"
    UNKNOWN_FRONT = "UNKNOWN_FRONT"
    WORM_VS_WORM = "WORM_VS_WORM"
    COLONY_WAR = "COLONY_WAR"


MODE_BY_KIND = {
    WormKind.SCOUT: [ArenaMode.UNKNOWN_FRONT, ArenaMode.ANOMALY_HUNT, ArenaMode.BLIND_SPOT_HUNT],
    WormKind.BURROWER: [ArenaMode.ASSUMPTION_WAR, ArenaMode.MODEL_KILLER, ArenaMode.BLIND_SPOT_HUNT],
    WormKind.BREACHER: [ArenaMode.PARADIGM_SIEGE, ArenaMode.IDEA_DUEL, ArenaMode.MODEL_KILLER],
    WormKind.PROBER: [ArenaMode.TIME_RAID, ArenaMode.ANOMALY_HUNT, ArenaMode.EVIDENCE_DUEL],
    WormKind.MIRROR: [ArenaMode.COUNTERFACTUAL_WAR, ArenaMode.IDEA_DUEL, ArenaMode.MODEL_KILLER],
    WormKind.CROSSWORM: [ArenaMode.COLONY_WAR, ArenaMode.BLIND_SPOT_HUNT, ArenaMode.EVIDENCE_DUEL],
    WormKind.WORM_EATER: [ArenaMode.MODEL_KILLER, ArenaMode.ASSUMPTION_WAR, ArenaMode.BLIND_SPOT_HUNT],
    WormKind.OMEGA: [ArenaMode.COLONY_WAR, ArenaMode.PARADIGM_SIEGE, ArenaMode.COUNTERFACTUAL_WAR],
}


@dataclass(frozen=True)
class ArenaConfig:
    enabled: bool = True
    matches_per_generation: int = 14
    tournament_interval: int = 4
    tournament_slots: int = 16
    hostility: float = 0.88
    energy_cost: float = 0.09
    stress_gain: float = 0.09
    win_reward: float = 0.10
    loss_penalty: float = 0.075
    draw_reward: float = 0.025
    rating_k: float = 26.0
    rating_floor: float = 700.0
    rating_ceiling: float = 1800.0
    adaptation_gain: float = 0.028
    learning_transfer: float = 0.16
    max_recent_pairs: int = 96
    tournament_bonus: float = 0.06


@dataclass
class MatchRecord:
    generation: int
    match_id: str
    mode: str
    worm_a: str
    worm_b: str
    score_a: float
    score_b: float
    result: str
    intensity: float
    learning_artifacts: int = 0
    tournament: bool = False


@dataclass
class ArenaState:
    matches: List[MatchRecord] = field(default_factory=list)
    ratings: Dict[str, float] = field(default_factory=dict)
    recent_pairs: List[Tuple[str, str]] = field(default_factory=list)
    tournaments: int = 0
    tournament_wins: Dict[str, int] = field(default_factory=dict)
    generation_match_counts: Dict[str, int] = field(default_factory=dict)


class WormArena:
    """Closed synthetic arena for adversarial cognitive competition."""

    def __init__(self, config: ArenaConfig | None = None, seed: int = 42):
        self.config = config or ArenaConfig()
        self.rng = random.Random(seed + 1001)
        self.state = ArenaState()
        self._seq = 0

    def _rating(self, worm: Worm) -> float:
        return self.state.ratings.setdefault(worm.id, 1000.0)

    def _pair_key(self, a: Worm, b: Worm) -> Tuple[str, str]:
        return tuple(sorted((a.id, b.id)))

    def _specialty_affinity(self, a: Worm, b: Worm) -> float:
        # Cross-specialist conflicts are more informative; legacy kind remains a
        # secondary discriminator for backward compatibility.
        if getattr(a, "specialty", "GENERALIST") != getattr(b, "specialty", "GENERALIST"):
            return 1.0
        return 0.55 if a.kind != b.kind else 0.25

    def _select_pairs(self, worms: List[Worm], generation: int, limit: int) -> List[Tuple[Worm, Worm]]:
        active = [w for w in worms if w.status == WormStatus.ACTIVE.value and w.energy > 0.08]
        candidates: List[Tuple[float, Worm, Worm]] = []
        for i, a in enumerate(active):
            for b in active[i + 1:]:
                key = self._pair_key(a, b)
                repeat_penalty = 0.70 if key in self.state.recent_pairs[-self.config.max_recent_pairs:] else 0.0
                score = (
                    self._specialty_affinity(a, b)
                    + 0.20 * abs(a.arena_rating - b.arena_rating) / 1000.0
                    + 0.12 * abs(a.resilience - b.resilience)
                    + 0.10 * abs(a.adaptation - b.adaptation)
                    - repeat_penalty
                )
                candidates.append((score, a, b))
        candidates.sort(key=lambda x: (-x[0], x[1].id, x[2].id))
        out: List[Tuple[Worm, Worm]] = []
        used = set()
        for _, a, b in candidates:
            if len(out) >= limit:
                break
            if a.id in used or b.id in used:
                continue
            out.append((a, b)); used.update((a.id, b.id))
        return out

    def _mode(self, a: Worm, b: Worm, generation: int) -> ArenaMode:
        pool: list[ArenaMode] = []
        for w in (a, b):
            profile = SPECIALIST_BY_NAME.get(getattr(w, "specialty", ""))
            if profile:
                for raw in profile.preferred_modes:
                    try:
                        pool.append(ArenaMode(raw))
                    except ValueError:
                        continue
            pool.extend(MODE_BY_KIND.get(w.kind, [ArenaMode.WORM_VS_WORM]))
        # Deduplicate while preserving deterministic priority.
        unique = list(dict.fromkeys(pool)) or [ArenaMode.WORM_VS_WORM]
        idx = (generation + len(a.id) + len(b.id) + int(self._rating(a) + self._rating(b))) % len(unique)
        return unique[idx]

    def _score(self, worm: Worm, opponent: Worm, mode: ArenaMode, pressure: Dict[str, Any], traits: Dict[str, float] | None = None) -> float:
        g = traits or {}
        adversariality = float(g.get("adversariality", .5))
        skepticism = float(g.get("skepticism", .5))
        counterfactual = float(g.get("counterfactual", .5))
        falsifiability = float(g.get("falsifiability", .5))
        penetration = float(g.get("penetration", .5))
        restraint = float(g.get("restraint", .5))
        war_resistance = float(g.get("war_resistance", .5))
        specialty_skill = float(getattr(worm, "skill_state", {}).get(getattr(worm, "specialty", ""), .25))
        mode_bonus = {
            ArenaMode.IDEA_DUEL: .28 * adversariality + .18 * falsifiability,
            ArenaMode.ASSUMPTION_WAR: .32 * penetration + .20 * skepticism,
            ArenaMode.ANOMALY_HUNT: .30 * skepticism + .18 * penetration,
            ArenaMode.PARADIGM_SIEGE: .34 * adversariality + .22 * penetration,
            ArenaMode.COUNTERFACTUAL_WAR: .36 * counterfactual + .16 * restraint,
            ArenaMode.MODEL_KILLER: .30 * adversariality + .24 * falsifiability,
            ArenaMode.BLIND_SPOT_HUNT: .32 * skepticism + .20 * counterfactual,
            ArenaMode.EVIDENCE_DUEL: .34 * falsifiability + .18 * restraint,
            ArenaMode.TIME_RAID: .28 * penetration + .16 * war_resistance,
            ArenaMode.UNKNOWN_FRONT: .26 * skepticism + .20 * novelty(worm),
            ArenaMode.WORM_VS_WORM: .24 * adversariality + .20 * war_resistance,
            ArenaMode.COLONY_WAR: .22 * adversariality + .18 * restraint + .16 * war_resistance,
        }[mode]
        base = .44 + mode_bonus
        pressure_factor = .04 * float(pressure.get("peer_attack", 0.0)) + .03 * float(pressure.get("contradiction_rate", 0.0))
        readiness = .14 * worm.resilience + .10 * worm.adaptation + .08 * worm.reputation
        penalty = .16 * worm.stress + .10 * max(0.0, .55 - worm.energy)
        specialty_bonus = .08 * max(0.0, min(1.0, specialty_skill))
        return max(.0, min(1.0, base + pressure_factor + readiness + specialty_bonus - penalty + self.rng.uniform(-.035, .035)))

    def fight(self, a: Worm, b: Worm, generation: int, pressure: Dict[str, Any], tournament: bool = False, trait_lookup: Dict[str, Dict[str, float]] | None = None) -> MatchRecord:
        mode = self._mode(a, b, generation)
        intensity = max(0.25, min(1.0, self.config.hostility + .08 * float(pressure.get("hostility", 0.0)) + self.rng.uniform(-.05, .05)))
        trait_lookup = trait_lookup or {}
        sa = self._score(a, b, mode, pressure, trait_lookup.get(a.id))
        sb = self._score(b, a, mode, pressure, trait_lookup.get(b.id))
        margin = sa - sb
        # High resilience reduces collapse; high war resistance helps endure heavy fights.
        resist_a = .55 * a.resilience + .45 * float(getattr(a, "counterattack_skill", .5))
        resist_b = .55 * b.resilience + .45 * float(getattr(b, "counterattack_skill", .5))
        if abs(margin) < .035:
            result = "DRAW"
            a.arena_draws += 1; b.arena_draws += 1
        elif margin > 0:
            result = "A_WIN"
            a.arena_wins += 1; b.arena_losses += 1
        else:
            result = "B_WIN"
            b.arena_wins += 1; a.arena_losses += 1

        expected_a = 1.0 / (1.0 + 10 ** ((self._rating(b) - self._rating(a)) / 400.0))
        if result == "DRAW": actual_a = .5
        else: actual_a = 1.0 if result == "A_WIN" else 0.0
        delta = self.config.rating_k * (actual_a - expected_a) * (0.65 + .35 * intensity)
        self.state.ratings[a.id] = max(self.config.rating_floor, min(self.config.rating_ceiling, self._rating(a) + delta))
        self.state.ratings[b.id] = max(self.config.rating_floor, min(self.config.rating_ceiling, self._rating(b) - delta))
        a.arena_rating = self.state.ratings[a.id]; b.arena_rating = self.state.ratings[b.id]

        # Fierce competition imposes costs and also teaches. Never acts on real targets.
        a.energy = max(0.0, a.energy - self.config.energy_cost * intensity)
        b.energy = max(0.0, b.energy - self.config.energy_cost * intensity)
        a.stress = min(1.0, a.stress + self.config.stress_gain * intensity * (1.0 - resist_a))
        b.stress = min(1.0, b.stress + self.config.stress_gain * intensity * (1.0 - resist_b))
        a.rivalry_index = min(1.0, getattr(a, "rivalry_index", 0.0) + intensity * .025)
        b.rivalry_index = min(1.0, getattr(b, "rivalry_index", 0.0) + intensity * .025)
        if result == "A_WIN":
            a.reward_points += self.config.win_reward; a.reputation = min(1.0, a.reputation + self.config.win_reward*.25)
            b.penalty_points += self.config.loss_penalty; b.reputation = max(0.0, b.reputation - self.config.loss_penalty*.16)
            a.successful_adaptations += 1; b.failures += 1
            a.arena_streak = max(1, getattr(a, "arena_streak", 0) + 1); b.arena_streak = min(-1, getattr(b, "arena_streak", 0) - 1)
        elif result == "B_WIN":
            b.reward_points += self.config.win_reward; b.reputation = min(1.0, b.reputation + self.config.win_reward*.25)
            a.penalty_points += self.config.loss_penalty; a.reputation = max(0.0, a.reputation - self.config.loss_penalty*.16)
            b.successful_adaptations += 1; a.failures += 1
            b.arena_streak = max(1, getattr(b, "arena_streak", 0) + 1); a.arena_streak = min(-1, getattr(a, "arena_streak", 0) - 1)
        else:
            a.reward_points += self.config.draw_reward; b.reward_points += self.config.draw_reward
            a.adaptation = min(1.0, a.adaptation + self.config.adaptation_gain*.7)
            b.adaptation = min(1.0, b.adaptation + self.config.adaptation_gain*.7)
            a.arena_streak = 0; b.arena_streak = 0

        self._seq += 1
        rec = MatchRecord(
            generation=generation, match_id=f"M-{generation:04d}-{self._seq:05d}", mode=mode.value,
            worm_a=a.id, worm_b=b.id, score_a=round(sa, 6), score_b=round(sb, 6), result=result,
            intensity=round(intensity, 6), tournament=tournament,
        )
        self.state.matches.append(rec)
        key = self._pair_key(a, b)
        self.state.recent_pairs.append(key)
        self.state.recent_pairs = self.state.recent_pairs[-self.config.max_recent_pairs:]
        self.state.generation_match_counts[str(generation)] = self.state.generation_match_counts.get(str(generation), 0) + 1
        return rec

    def run_generation(self, worms: List[Worm], generation: int, pressure: Dict[str, Any], trait_lookup: Dict[str, Dict[str, float]] | None = None) -> List[MatchRecord]:
        if not self.config.enabled:
            return []
        selected = self._select_pairs(worms, generation, self.config.matches_per_generation)
        out = [self.fight(a, b, generation, pressure, tournament=False, trait_lookup=trait_lookup) for a, b in selected]
        if generation > 0 and generation % max(1, self.config.tournament_interval) == 0:
            out.extend(self._tournament(worms, generation, pressure, trait_lookup))
        return out

    def _tournament(self, worms: List[Worm], generation: int, pressure: Dict[str, Any], trait_lookup: Dict[str, Dict[str, float]] | None = None) -> List[MatchRecord]:
        ranked = sorted([w for w in worms if w.status == WormStatus.ACTIVE.value], key=lambda w: (self._rating(w), w.reputation, w.id), reverse=True)
        slots = min(self.config.tournament_slots, len(ranked))
        ranked = ranked[:slots]
        if len(ranked) < 4:
            return []
        self.state.tournaments += 1
        out: List[MatchRecord] = []
        round_roster = ranked
        while len(round_roster) > 1:
            nxt: List[Worm] = []
            for i in range(0, len(round_roster)-1, 2):
                rec = self.fight(round_roster[i], round_roster[i+1], generation, pressure, tournament=True, trait_lookup=trait_lookup)
                out.append(rec)
                winner = round_roster[i] if rec.result == "A_WIN" else round_roster[i+1] if rec.result == "B_WIN" else max((round_roster[i], round_roster[i+1]), key=lambda w: (self._rating(w), w.id))
                winner.reward_points += self.config.tournament_bonus
                nxt.append(winner)
            if len(round_roster) % 2:
                nxt.append(round_roster[-1])
            round_roster = nxt
        champion = round_roster[0]
        self.state.tournament_wins[champion.id] = self.state.tournament_wins.get(champion.id, 0) + 1
        return out


def novelty(worm: Worm) -> float:
    habits = getattr(worm, "cognitive_habits", {})
    return max(0.0, min(1.0, 0.65 - 0.25 * habits.get("novelty_chasing", 0.0) + 0.10 * getattr(worm, "adaptation", .2)))


__all__ = ["ArenaConfig", "ArenaState", "MatchRecord", "ArenaMode", "WormArena"]
