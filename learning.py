from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List
import random

from .models import Worm


@dataclass
class LearningConfig:
    enabled: bool = True
    transfer_rate: float = .16
    teaching_reward: float = .025
    counterteaching_reward: float = .018
    max_events: int = 12000


@dataclass
class LearningEvent:
    generation: int
    source: str
    receiver: str
    channel: str
    delta: float
    mechanism: str
    artifact: str


@dataclass
class LearningState:
    events: List[LearningEvent] = field(default_factory=list)
    channel_totals: Dict[str, float] = field(default_factory=dict)


class LearningEcology:
    """Multi-directional synthetic learning: peer, counter-peer, environment, lineage, meta."""

    CHANNELS = ("peer", "counterpeer", "environment", "lineage", "meta")

    def __init__(self, config: LearningConfig | None = None, seed: int = 42):
        self.config = config or LearningConfig()
        self.rng = random.Random(seed + 2003)
        self.state = LearningState()

    def _ensure(self, worm: Worm) -> None:
        if not worm.learning_channels:
            worm.learning_channels = {k: 0.0 for k in self.CHANNELS}

    def _apply(self, worm: Worm, channel: str, amount: float) -> None:
        self._ensure(worm)
        amount = max(-.25, min(.25, amount))
        worm.learning_channels[channel] = max(0.0, min(1.0, worm.learning_channels[channel] + amount))
        worm.adaptation = max(0.0, min(1.0, worm.adaptation + .55*amount))
        worm.resilience = max(0.0, min(1.0, worm.resilience + .30*amount))

    def transfer(self, source: Worm, receiver: Worm, generation: int, channel: str, artifact: str, intensity: float) -> LearningEvent | None:
        if not self.config.enabled or len(self.state.events) >= self.config.max_events:
            return None
        if channel not in self.CHANNELS:
            raise ValueError(f"unknown learning channel: {channel}")
        # Transfer is strongest when the receiver has a gap and source has demonstrated competence.
        src_strength = .35*source.reputation + .25*source.resilience + .20*source.arena_rating/1200.0 + .20*source.adaptation
        gap = max(0.0, 1.0 - receiver.learning_channels.get(channel, 0.0))
        delta = self.config.transfer_rate * intensity * src_strength * gap
        self._apply(receiver, channel, delta)
        source.reward_points += self.config.teaching_reward * intensity
        receiver.private_notes.append(f"learned:{channel}:{artifact}")
        receiver.private_notes = receiver.private_notes[-24:]
        ev = LearningEvent(generation, source.id, receiver.id, channel, round(delta, 6), "DIRECTED_TRANSFER", artifact)
        self.state.events.append(ev)
        self.state.channel_totals[channel] = self.state.channel_totals.get(channel, 0.0) + delta
        return ev

    def bidirectional(self, a: Worm, b: Worm, generation: int, mechanism: str, intensity: float) -> list[LearningEvent]:
        out: list[LearningEvent] = []
        if mechanism == "ARENA":
            e1 = self.transfer(a, b, generation, "peer", "opponent_pattern", intensity)
            e2 = self.transfer(b, a, generation, "counterpeer", "counterattack_pattern", intensity*.9)
            if e1: out.append(e1)
            if e2: out.append(e2)
        elif mechanism == "FAILURE":
            e = self.transfer(b, a, generation, "environment", "survival_lesson", intensity)
            if e: out.append(e)
        elif mechanism == "META":
            e = self.transfer(a, b, generation, "meta", "rule_failure", intensity)
            if e: out.append(e)
        return out

    def lineage_transfer(self, parent: Worm, child: Worm, generation: int) -> list[LearningEvent]:
        out: list[LearningEvent] = []
        if not self.config.enabled:
            return out
        for ch in ("lineage", "meta"):
            e = self.transfer(parent, child, generation, ch, f"heritage:{parent.id}", .70)
            if e: out.append(e)
        return out


__all__ = ["LearningConfig", "LearningState", "LearningEvent", "LearningEcology"]
