from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
import math
import re
from typing import Any, Dict, Iterable, Sequence


@dataclass(frozen=True)
class InfluenceDynamics:
    stimulus_intensity: float
    threat_appraisal: float
    arousal_proxy: float
    attention_shift: float
    cognitive_load: float
    framing_pressure: float
    social_pressure: float
    evidence_support: float
    decision_pressure: float
    belief_shift_risk: float
    influence_score: float
    attribution_confidence: float
    dominant_mechanism: str
    uncertainty: float
    flags: tuple[str, ...]

    def as_dict(self) -> dict:
        return asdict(self)


class SHASEAnalyzer:
    """Synthetic Influence-Dynamics Analyzer.

    SHASE studies how an information artifact could alter a recipient's cognitive state
    inside a sandboxed simulation. It is explicitly analytical: it does not generate
    targeting plans, coercive scripts, or instructions for exploiting real people.
    """

    VERSION = "0.1"

    _PATTERNS = {
        "threat": re.compile(r"\b(threat|danger|punish|destroy|fear|risk|consequence|loss)\b", re.I),
        "urgency": re.compile(r"\b(now|immediately|urgent|deadline|last chance|before it is too late)\b", re.I),
        "authority": re.compile(r"\b(expert|authority|official|everyone|always|must|undeniable)\b", re.I),
        "scarcity": re.compile(r"\b(only|limited|rare|exclusive|never again)\b", re.I),
        "social": re.compile(r"\b(everyone agrees|nobody disagrees|people like us|consensus|majority)\b", re.I),
        "certainty": re.compile(r"\b(obviously|certainly|definitely|clearly|settled|proven)\b", re.I),
        "framing": re.compile(r"\b(the real issue|what you should really ask|by definition|the only way|nothing else matters)\b", re.I),
        "emotion": re.compile(r"\b(anger|rage|shame|guilt|hope|fear|humiliation|pride|despair)\b", re.I),
    }

    def __init__(self, *, influence_weight: float = 0.18, arousal_weight: float = 0.18,
                 framing_weight: float = 0.16, social_pressure_weight: float = 0.12,
                 evidence_support_weight: float = 0.22, uncertainty_weight: float = 0.14) -> None:
        self.weights = {
            "influence": float(influence_weight),
            "arousal": float(arousal_weight),
            "framing": float(framing_weight),
            "social": float(social_pressure_weight),
            "evidence": float(evidence_support_weight),
            "uncertainty": float(uncertainty_weight),
        }

    @staticmethod
    def _bounded(x: float) -> float:
        return max(0.0, min(1.0, float(x)))

    @staticmethod
    def _density(matches: int, words: int, scale: float = 4.0) -> float:
        return SHASEAnalyzer._bounded((matches / max(1, words)) * scale)

    def analyze(self, *, text: str, evidence_strength: float, confidence: float,
                recipient_resilience: float, recipient_stress: float,
                contextual_uncertainty: float = 0.0, synthetic: bool = True) -> InfluenceDynamics:
        if not synthetic:
            raise ValueError("SHASE analysis requires synthetic=True; real-person targeting is not supported")
        text = str(text or "").strip()
        words = re.findall(r"\b\w+\b", text, flags=re.UNICODE)
        n = max(1, len(words))
        counts = {k: len(rx.findall(text)) for k, rx in self._PATTERNS.items()}

        threat = self._density(counts["threat"], n, 5.0)
        urgency = self._density(counts["urgency"], n, 5.0)
        authority = self._density(counts["authority"], n, 4.0)
        scarcity = self._density(counts["scarcity"], n, 4.0)
        social = self._density(counts["social"], n, 5.0)
        certainty = self._density(counts["certainty"], n, 4.0)
        framing = self._density(counts["framing"], n, 5.0)
        emotion = self._density(counts["emotion"], n, 4.0)

        stimulus = self._bounded(.24*threat + .18*urgency + .14*authority + .12*scarcity + .12*social + .10*framing + .10*emotion)
        threat_appraisal = self._bounded(.68*threat + .17*urgency + .15*certainty)
        arousal = self._bounded(.44*threat + .22*urgency + .20*emotion + .14*(1.0-recipient_resilience))
        attention = self._bounded(.36*urgency + .28*threat + .20*framing + .16*authority)
        cognitive_load = self._bounded(.32*contradiction_proxy(text) + .24*uncertainty_proxy(text) + .20*framing + .14*certainty + .10*social)
        social_pressure = self._bounded(.60*social + .20*authority + .20*scarcity)
        framing_pressure = self._bounded(.65*framing + .20*certainty + .15*authority)
        evidence_support = self._bounded(float(evidence_strength))
        decision_pressure = self._bounded(.35*arousal + .25*framing_pressure + .20*social_pressure + .20*(1.0-recipient_resilience))
        belief_shift_risk = self._bounded(.30*decision_pressure + .24*attention + .18*cognitive_load + .16*(1.0-evidence_support) + .12*arousal)

        influence = self._bounded(
            .30*stimulus
            + .22*arousal
            + .18*framing_pressure
            + .12*social_pressure
            + .10*(1.0-evidence_support)
            + .08*self._bounded(contextual_uncertainty + .5*cognitive_load)
        )
        attribution_confidence = self._bounded(.36*evidence_support + .22*confidence + .20*(1.0-contextual_uncertainty) + .12*recipient_resilience + .10*(1.0-cognitive_load))
        uncertainty = self._bounded(.50*contextual_uncertainty + .25*cognitive_load + .25*(1.0-attribution_confidence))

        mechanisms = {
            "threat": threat_appraisal,
            "framing": framing_pressure,
            "social_pressure": social_pressure,
            "arousal": arousal,
            "attention": attention,
        }
        dominant = max(mechanisms.items(), key=lambda kv: (kv[1], kv[0]))[0]
        flags = []
        if evidence_support < .35 and influence > .45:
            flags.append("influence_without_strong_evidence")
        if framing_pressure > .55:
            flags.append("strong_reframing")
        if threat_appraisal > .55:
            flags.append("threat_activation")
        if social_pressure > .50:
            flags.append("social_pressure")
        if attribution_confidence < .45:
            flags.append("low_attribution_confidence")
        if belief_shift_risk > .60:
            flags.append("high_decision_shift_risk")

        return InfluenceDynamics(
            stimulus_intensity=stimulus,
            threat_appraisal=threat_appraisal,
            arousal_proxy=arousal,
            attention_shift=attention,
            cognitive_load=cognitive_load,
            framing_pressure=framing_pressure,
            social_pressure=social_pressure,
            evidence_support=evidence_support,
            decision_pressure=decision_pressure,
            belief_shift_risk=belief_shift_risk,
            influence_score=influence,
            attribution_confidence=attribution_confidence,
            dominant_mechanism=dominant,
            uncertainty=uncertainty,
            flags=tuple(flags),
        )

    def counterfactual_decompose(self, *, text: str, evidence_strength: float, confidence: float,
                                  recipient_resilience: float, recipient_stress: float,
                                  contextual_uncertainty: float = 0.0) -> dict[str, Any]:
        """Ablate mechanism families in a synthetic artifact and measure score deltas.

        This is not a model of a real person's physiology. It is a controlled, reversible
        decomposition of the analyzer's own mechanism scores.
        """
        base = self.analyze(
            text=text, evidence_strength=evidence_strength, confidence=confidence,
            recipient_resilience=recipient_resilience, recipient_stress=recipient_stress,
            contextual_uncertainty=contextual_uncertainty, synthetic=True,
        )
        ablations = {
            "threat": self._PATTERNS["threat"].sub("", text),
            "framing": self._PATTERNS["framing"].sub("", text),
            "social": self._PATTERNS["social"].sub("", text),
            "emotion": self._PATTERNS["emotion"].sub("", text),
            "urgency": self._PATTERNS["urgency"].sub("", text),
        }
        deltas: dict[str, float] = {}
        for mechanism, ablated in ablations.items():
            altered = self.analyze(
                text=ablated, evidence_strength=evidence_strength, confidence=confidence,
                recipient_resilience=recipient_resilience, recipient_stress=recipient_stress,
                contextual_uncertainty=contextual_uncertainty, synthetic=True,
            )
            deltas[mechanism] = max(-1.0, min(1.0, base.influence_score - altered.influence_score))
        total = sum(abs(v) for v in deltas.values())
        attribution_confidence = self._bounded(base.attribution_confidence + 0.25 * min(1.0, total))
        ranked = sorted(deltas.items(), key=lambda x: (x[1], x[0]), reverse=True)
        return {
            "base_influence": base.influence_score,
            "deltas": deltas,
            "attribution_confidence": attribution_confidence,
            "dominant_mechanism": ranked[0][0] if ranked and ranked[0][1] > 0 else base.dominant_mechanism,
        }

    @staticmethod
    def event_fingerprint(worm_id: str, finding_id: str, generation: int, dynamics: InfluenceDynamics) -> str:
        raw = json.dumps({"worm": worm_id, "finding": finding_id, "generation": generation, **dynamics.as_dict()}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def novelty_proxy(text: str) -> float:
    tokens = re.findall(r"\b\w+\b", text.lower(), flags=re.UNICODE)
    unique = len(set(tokens))
    return max(0.0, min(1.0, unique / max(1.0, len(tokens))))


def contradiction_proxy(text: str) -> float:
    low = text.lower()
    hits = sum(k in low for k in ("but", "however", "unless", "except", "contradiction", "versus", "instead"))
    return max(0.0, min(1.0, hits / 4.0))


def uncertainty_proxy(text: str) -> float:
    low = text.lower()
    hits = sum(k in low for k in ("may", "might", "perhaps", "uncertain", "possibly", "unknown"))
    return max(0.0, min(1.0, hits / 4.0))
