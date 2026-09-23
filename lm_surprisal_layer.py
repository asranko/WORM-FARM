from __future__ import annotations
from pathlib import Path
from dataclasses import asdict
from typing import Iterable
import math, json
from .models import Finding
from .runtime import BrainAdapter

try:
    from .wormlm_micro.surprisal import WormLMSurprisalSensor
except Exception:
    WormLMSurprisalSensor = None

class LMNoveltyLayer:
    """Optional real-LM layer: turns a trained language model into a novelty signal.

    It is deliberately non-authoritative: it may raise anomaly/novelty, but it cannot
    prove truth or invalidate evidence. The core WORM audit remains the final judge.
    """
    def __init__(self, checkpoint: str | Path, vocab: str | Path):
        if WormLMSurprisalSensor is None:
            raise RuntimeError('WormLM artifacts require PyTorch and the wormlm_micro package')
        self.sensor = WormLMSurprisalSensor(checkpoint, vocab)

    def enrich(self, finding: Finding) -> Finding:
        text = f"{finding.title}. {finding.detail[:1600]}"
        result = self.sensor.score(text)
        finding.anomaly = max(0.0, min(1.0, max(finding.anomaly, result.novelty)))
        finding.independent_path = f"{finding.independent_path}|LM-SURPRISAL:{result.mean_nll:.4f}".strip('|')
        return finding

    def batch_enrich(self, findings: Iterable[Finding]) -> list[Finding]:
        return [self.enrich(f) for f in findings]
