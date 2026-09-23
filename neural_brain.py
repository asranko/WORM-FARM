from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import pickle
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import torch
from torch import nn

from .brain import BrainProtocolError, validate_proposal
from .models import Claim, Finding, Worm
from .safety import validate_text


@dataclass(frozen=True)
class NeuralBrainConfig:
    hidden_dim: int = 64
    claim_buckets: int = 48
    learning_rate: float = 1.5e-3
    weight_decay: float = 1e-4
    entropy_bonus: float = 0.012
    baseline_decay: float = 0.95
    temperature: float = 0.85
    seed: int = 42


ACTION_MODES = (
    "ASSUMPTION_BREAK",
    "BOUNDARY_HUNT",
    "CONTRADICTION",
    "COUNTERFACTUAL",
    "ALTERNATIVE",
    "EVIDENCE",
    "MODEL_KILLER",
    "TRANSFER",
)

# Fixed feature keys, kept deliberately compact so this remains a genuinely small model.
GENOME_KEYS = (
    "penetration", "skepticism", "novelty", "falsifiability", "independence",
    "persistence", "adversariality", "restraint", "pattern_sensitivity", "recovery",
    "counterfactual", "meta_cognition", "war_resistance", "memory", "curiosity",
)
STRATEGY_KEYS = (
    "penetration_drive", "novelty_drive", "evidence_demand", "counterfactual_drive",
    "adversarial_pressure", "search_breadth", "persistence", "recovery", "teaching",
    "restraint", "meta_reasoning", "transferability",
)
PRESSURE_KEYS = (
    "hostility", "noise", "contradiction_rate", "decoy_rate", "concept_drift",
    "resource_scarcity", "deadline_pressure", "peer_attack",
)


def _stable_bucket(text: str, modulo: int) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % modulo


def _hash_features(text: str, buckets: int) -> list[float]:
    words = [w for w in text.lower().split() if w]
    out = [0.0] * buckets
    if not words:
        return out
    for token in words[:96]:
        idx = _stable_bucket(token, buckets)
        out[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in out)) or 1.0
    return [v / norm for v in out]


def _safe_text(text: str, limit: int) -> str:
    text = str(text).strip()[:limit]
    if not text or not validate_text(text).allowed:
        return "synthetic cognitive artifact"
    return text


class _TinyPolicyNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, action_count: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.action_head = nn.Linear(hidden_dim, action_count)
        self.metric_head = nn.Linear(hidden_dim, 4)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        return self.action_head(h), torch.sigmoid(self.metric_head(h)), self.value_head(h).squeeze(-1)


class TorchNeuralBrain:
    """Small real PyTorch neural policy brain.

    It is intentionally not a language model. It is a trainable neural controller that
    conditions proposal mode and epistemic scores on claim text, genome, strategy, and
    environmental state. Text is rendered from the selected cognitive mode so the Farm can
    consume the existing structured BrainProposal contract safely.
    """

    name = "torch-neural-small"

    def __init__(self, config: NeuralBrainConfig | None = None) -> None:
        self.config = config or NeuralBrainConfig()
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
        torch.manual_seed(self.config.seed)
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(self.config.seed ^ 0x51A7)
        self.input_dim = self.config.claim_buckets + len(GENOME_KEYS) + len(STRATEGY_KEYS) + len(PRESSURE_KEYS) + 8 + 4
        self.net = _TinyPolicyNet(self.input_dim, self.config.hidden_dim, len(ACTION_MODES))
        self.optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay
        )
        self.baseline = 0.0
        self.calls = 0
        self.learn_steps = 0
        self._pending: Dict[str, dict] = {}

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.net.parameters())

    def _features(self, *, claim: Claim, worm: Worm, context: Dict[str, Any]) -> torch.Tensor:
        text = f"{claim.text} {' '.join(claim.assumptions[:8])} {worm.specialty} {worm.current_strategy}"
        values = _hash_features(text, self.config.claim_buckets)
        genome = context.get("genome", {})
        strategy = context.get("strategy_phenotype", {})
        pressure = context.get("pressure", {})
        values.extend(float(genome.get(k, 0.5)) for k in GENOME_KEYS)
        values.extend(float(strategy.get(k, 0.5)) for k in STRATEGY_KEYS)
        values.extend(float(pressure.get(k, 0.5)) for k in PRESSURE_KEYS)
        values.extend([
            max(0.0, min(1.0, worm.depth / max(1, worm.max_depth))),
            max(0.0, min(1.0, worm.energy)),
            max(0.0, min(1.0, worm.integrity)),
            max(0.0, min(1.0, worm.reputation)),
            max(0.0, min(1.0, worm.resilience)),
            max(0.0, min(1.0, worm.adaptation)),
            max(0.0, min(1.0, worm.stress)),
            max(0.0, min(1.0, worm.learning_capacity)),
        ])
        workspace = context.get("shared_workspace", [])
        research_events = context.get("research_events", {})
        accepted_ratio = float(research_events.get("accepted_ratio", 0.0)) if isinstance(research_events, dict) else 0.0
        equivocation = float(research_events.get("equivocation", 0.0)) if isinstance(research_events, dict) else 0.0
        mi = float(research_events.get("mi_bits", 0.0)) if isinstance(research_events, dict) else 0.0
        values.extend([
            max(0.0, min(1.0, len(workspace) / 8.0)),
            max(0.0, min(1.0, accepted_ratio)),
            max(0.0, min(1.0, equivocation)),
            max(0.0, min(1.0, mi / 2.0)),
        ])
        return torch.tensor(values, dtype=torch.float32).unsqueeze(0)

    @staticmethod
    def _mode_text(mode: str, claim: Claim, worm: Worm, scores: Dict[str, float]) -> tuple[str, str, list[str], str, str]:
        claim_text = _safe_text(claim.text, 220)
        if mode == "ASSUMPTION_BREAK":
            title = f"Assumption breach [{worm.specialty}]"
            hypothesis = f"The framing of '{claim_text}' depends on an assumption that may not be necessary."
            assumptions = ["The current framing contains all variables needed for the conclusion."]
            counter = "The omitted assumption may not materially change the conclusion."
            falsifier = "A controlled synthetic test removes the targeted assumption without changing the outcome."
        elif mode == "BOUNDARY_HUNT":
            title = f"Boundary fault [{worm.specialty}]"
            hypothesis = f"'{claim_text}' may fail at an untested boundary condition."
            assumptions = ["The observed behavior remains valid across the explored domain."]
            counter = "The apparent boundary may be an artifact of limited sampling."
            falsifier = "Stress testing across the boundary shows no material change in outcome."
        elif mode == "CONTRADICTION":
            title = f"Contradiction pocket [{worm.specialty}]"
            hypothesis = f"The claim '{claim_text}' contains a tension with another plausible explanation."
            assumptions = ["All relevant constraints are jointly consistent."]
            counter = "The tension could disappear after adding missing context."
            falsifier = "A consistent model explains both sides without residual contradiction."
        elif mode == "COUNTERFACTUAL":
            title = f"Counterfactual fracture [{worm.specialty}]"
            hypothesis = f"A minimal counterfactual change may alter the conclusion drawn from '{claim_text}'."
            assumptions = ["The current causal framing survives small counterfactual perturbations."]
            counter = "The conclusion may be insensitive to the proposed perturbation."
            falsifier = "Counterfactual simulation leaves the predicted outcome materially unchanged."
        elif mode == "ALTERNATIVE":
            title = f"Alternative route [{worm.specialty}]"
            hypothesis = f"A materially different framing may explain '{claim_text}' with fewer hidden assumptions."
            assumptions = ["The conventional framing spans the useful solution space."]
            counter = "The alternative may add complexity without explanatory gain."
            falsifier = "Independent comparison finds no improvement in explanation or prediction."
        elif mode == "EVIDENCE":
            title = f"Evidence stress point [{worm.specialty}]"
            hypothesis = f"The evidential chain behind '{claim_text}' may have a weak or dependent link."
            assumptions = ["The evidence sources are sufficiently independent for the conclusion."]
            counter = "The evidence may remain strong after dependency correction."
            falsifier = "Independent evidence reproduces the conclusion under the same controls."
        elif mode == "MODEL_KILLER":
            title = f"Model failure test [{worm.specialty}]"
            hypothesis = f"A deliberately harsh synthetic case may expose a failure mode in the model used for '{claim_text}'."
            assumptions = ["The current model has no relevant failure regime."]
            counter = "The adversarial case may be outside the intended domain."
            falsifier = "The model survives the constructed failure regime without degraded validity."
        else:
            title = f"Transfer fracture [{worm.specialty}]"
            hypothesis = f"A strategy effective elsewhere may expose a blind spot in the framing of '{claim_text}'."
            assumptions = ["Lessons from adjacent problem classes transfer poorly to this case."]
            counter = "Cross-domain transfer may introduce misleading analogies."
            falsifier = "Independent transfer attempts fail to improve explanation or prediction."
        # Scores are intentionally network-controlled, while text remains sandbox-safe and deterministic.
        return title, hypothesis, assumptions, counter, falsifier

    def propose(self, *, claim: Claim, worm: Worm, parent: Finding | None, context: Dict[str, Any]) -> Dict[str, Any]:
        x = self._features(claim=claim, worm=worm, context=context)
        logits, metrics, value = self.net(x)
        temperature = max(0.15, float(self.config.temperature))
        probs = torch.softmax(logits / temperature, dim=-1).squeeze(0)
        action = int(torch.multinomial(probs, 1, generator=self._generator).item())
        log_prob = torch.log(probs[action].clamp_min(1e-8))
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum()
        m = metrics.squeeze(0).detach().cpu().tolist()
        mode = ACTION_MODES[action]
        title, hypothesis, assumptions, counter, falsifier = self._mode_text(
            mode, claim, worm,
            {"novelty": m[0], "information_gain": m[1], "falsifiability": m[2], "independence": m[3]},
        )
        proposal = {
            "title": title,
            "core_hypothesis": hypothesis,
            "hidden_assumptions": assumptions,
            "counterargument": counter,
            "falsifier": falsifier,
            "novelty": float(m[0]),
            "information_gain": float(m[1]),
            "falsifiability": float(m[2]),
            "independence": float(m[3]),
        }
        validate_proposal(proposal)
        self._pending[worm.id] = {
            "features": x.detach(),
            "action": action,
            "log_prob": log_prob,
            "entropy": entropy,
            "value": float(value.item()),
            "target_metrics": torch.tensor(m, dtype=torch.float32).unsqueeze(0),
        }
        self.calls += 1
        return proposal

    def learn(self, *, worm_id: str, reward: float, finding: Finding | None = None) -> None:
        pending = self._pending.pop(worm_id, None)
        if pending is None:
            return
        # Reward is bounded to avoid unstable updates under harsh environments.
        r = max(-1.0, min(1.0, float(reward)))
        self.baseline = self.config.baseline_decay * self.baseline + (1.0 - self.config.baseline_decay) * r
        advantage = r - self.baseline
        logits, metrics, value = self.net(pending["features"])
        probs = torch.softmax(logits / max(0.15, float(self.config.temperature)), dim=-1)
        action = pending["action"]
        log_prob = torch.log(probs[0, action].clamp_min(1e-8))
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum()
        policy_loss = -advantage * log_prob - self.config.entropy_bonus * entropy
        value_loss = 0.20 * (value.squeeze(0) - torch.tensor(r, dtype=torch.float32)) ** 2
        metric_loss = torch.tensor(0.0)
        if finding is not None:
            target = torch.tensor([
                finding.novelty, finding.information_gain, finding.falsifiability, finding.independence
            ], dtype=torch.float32).unsqueeze(0)
            metric_loss = 0.10 * torch.nn.functional.mse_loss(metrics, target)
        loss = policy_loss + value_loss + metric_loss
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.optimizer.step()
        self.learn_steps += 1

    def runtime_state(self) -> Dict[str, Any]:
        buffer = io.BytesIO()
        torch.save({
            "model": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "baseline": self.baseline,
            "calls": self.calls,
            "learn_steps": self.learn_steps,
            "generator_state": self._generator.get_state(),
        }, buffer)
        return {
            "format": "torch-neural-brain-v2",
            "config": asdict(self.config),
            "blob": base64.b64encode(buffer.getvalue()).decode("ascii"),
        }

    def load_runtime_state(self, state: Dict[str, Any]) -> None:
        if not state:
            return
        if state.get("format") != "torch-neural-brain-v2":
            raise BrainProtocolError("unsupported torch neural brain state")
        cfg = NeuralBrainConfig(**state.get("config", {}))
        if cfg.hidden_dim != self.config.hidden_dim or cfg.claim_buckets != self.config.claim_buckets:
            raise BrainProtocolError("neural brain architecture mismatch on restore")
        raw = base64.b64decode(state["blob"].encode("ascii"))
        payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
        self.net.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.baseline = float(payload.get("baseline", 0.0))
        self.calls = int(payload.get("calls", 0))
        self.learn_steps = int(payload.get("learn_steps", 0))
        if payload.get("generator_state") is not None:
            self._generator.set_state(payload["generator_state"])

    def fingerprint(self) -> str:
        buffer = io.BytesIO()
        torch.save(self.net.state_dict(), buffer)
        return hashlib.sha256(buffer.getvalue()).hexdigest()[:16]
