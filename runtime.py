from __future__ import annotations

import hashlib
import json
import os
import tempfile
import base64
import pickle
from dataclasses import asdict, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Protocol, Sequence

from .models import Claim, Finding, FarmState, Worm, WormKind, WormStatus
from .genetics import CognitiveGenome


class InvariantViolation(RuntimeError):
    """Raised when a colony invariant is violated."""


class BrainAdapter(Protocol):
    """Pluggable brain interface for production LLM-backed worms."""

    name: str

    def propose(self, *, claim: Claim, worm: Worm, parent: Finding | None,
                context: Dict[str, Any]) -> Dict[str, Any]:
        """Return a validated proposal payload. No side effects are allowed."""


class RuleBasedBrain:
    """Deterministic fallback brain that satisfies the production proposal contract."""

    name = "rule-based"

    def propose(self, *, claim: Claim, worm: Worm, parent: Finding | None,
                context: Dict[str, Any]) -> Dict[str, Any]:
        workspace = context.get("shared_workspace", [])
        anchor = workspace[0] if workspace else None
        anchor_note = (
            f" The shared workspace contains an accepted signal from {anchor.get('specialty', 'another specialist')} requiring independent re-check."
            if isinstance(anchor, dict) else ""
        )
        dual = context.get("dual_mode_signal", {}) if isinstance(context.get("dual_mode_signal", {}), dict) else {}
        dual_note = ""
        if dual:
            dual_note = (
                f" DualLM selected {dual.get('selected_mode','AR')} mode; "
                f"uncertainty={float(dual.get('uncertainty', 0.0)):.3f}; "
                f"mode_score={float(dual.get('mode_score', 0.0)):.3f}."
            )
        selected_mode = str(dual.get("selected_mode", "BASE")).upper()
        ar_signal = max(0.0, min(1.0, float(dual.get("ar_signal", 0.0)))) if dual else 0.0
        diff_signal = max(0.0, min(1.0, float(dual.get("diff_signal", 0.0)))) if dual else 0.0
        if selected_mode == "DIFFUSION":
            mode_title = "Bidirectional refinement breach"
            core = (
                f"The framing '{claim.text[:220]}' may omit a boundary condition that becomes visible only when alternative token relationships are reconsidered."
            )
            novelty = 0.58 + 0.20 * diff_signal
            information_gain = 0.60 + 0.22 * diff_signal
            falsifiability = 0.74 + 0.10 * diff_signal
        else:
            mode_title = "Causal sequence breach"
            core = (
                f"The framing '{claim.text[:220]}' may omit a boundary condition whose effect depends on causal ordering and sequence constraints."
            )
            novelty = 0.52 + 0.12 * ar_signal
            information_gain = 0.55 + 0.16 * ar_signal
            falsifiability = 0.78 + 0.10 * ar_signal
        return {
            "title": f"{mode_title} [{worm.specialty}|g{worm.generation}|d{worm.depth}|{selected_mode}]",
            "core_hypothesis": core + anchor_note + dual_note,
            "hidden_assumptions": [
                "The current search framing captures the relevant variables."
            ],
            "counterargument": (
                "The apparent boundary may reflect missing evidence rather than a missing mechanism."
            ),
            "falsifier": (
                "A controlled synthetic test shows that removing the targeted assumption does not change the outcome."
            ),
            "novelty": min(1.0, novelty),
            "information_gain": min(1.0, information_gain),
            "falsifiability": min(1.0, falsifiability),
            "independence": 0.74,
        }


class EventLog:
    """Append-only hash-chained event log for auditability and crash diagnosis."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self._last_hash = "0" * 64

    @property
    def last_hash(self) -> str:
        return self._last_hash

    def append(self, event_type: str, generation: int, payload: Dict[str, Any]) -> dict[str, Any]:
        event = {
            "seq": len(self.events) + 1,
            "generation": generation,
            "type": event_type,
            "payload": payload,
            "prev_hash": self._last_hash,
        }
        encoded = json.dumps(event, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        event["hash"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        self._last_hash = event["hash"]
        self.events.append(event)
        return event

    def verify(self) -> None:
        prev = "0" * 64
        for index, event in enumerate(self.events, start=1):
            if event["seq"] != index or event["prev_hash"] != prev:
                raise InvariantViolation("event-chain continuity failure")
            body = {k: event[k] for k in ("seq", "generation", "type", "payload", "prev_hash")}
            encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            expected = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
            if expected != event["hash"]:
                raise InvariantViolation(f"event tampering detected at seq={index}")
            prev = event["hash"]

    def export(self) -> list[dict[str, Any]]:
        self.verify()
        return list(self.events)

    def import_events(self, events: Sequence[dict[str, Any]]) -> None:
        self.events = []
        self._last_hash = "0" * 64
        for event in events:
            body = {k: event[k] for k in ("seq", "generation", "type", "payload", "prev_hash")}
            encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            expected = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
            if expected != event.get("hash"):
                raise InvariantViolation("invalid imported event hash")
            if event["prev_hash"] != self._last_hash or event["seq"] != len(self.events) + 1:
                raise InvariantViolation("invalid imported event chain")
            self.events.append(dict(event))
            self._last_hash = event["hash"]


class CheckpointStore:
    """Atomic JSON checkpoint store. The checkpoint is self-describing and versioned."""

    VERSION = 1

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)

    def save(self, payload: Dict[str, Any]) -> None:
        wrapped = {"version": self.VERSION, "payload": payload}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(prefix=self.path.name + ".", dir=self.path.parent, text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(wrapped, fh, ensure_ascii=False, sort_keys=True, indent=2)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(temp, self.path)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)

    def load(self) -> Dict[str, Any]:
        with self.path.open("r", encoding="utf-8") as fh:
            wrapped = json.load(fh)
        if wrapped.get("version") != self.VERSION:
            raise ValueError(f"unsupported checkpoint version: {wrapped.get('version')}")
        return wrapped["payload"]


def enum_value(value: Any) -> Any:
    return value.value if isinstance(value, Enum) else value


def dataclass_to_dict(obj: Any) -> Any:
    if isinstance(obj, Enum):
        return obj.value
    if is_dataclass(obj):
        return {f.name: dataclass_to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, dict):
        return {str(k): dataclass_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [dataclass_to_dict(v) for v in obj]
    return obj


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / max(1, len(vals))


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


def pack_state(obj: Any) -> str:
    return base64.b64encode(pickle.dumps(obj, protocol=5)).decode("ascii")


def unpack_state(blob: str) -> Any:
    return pickle.loads(base64.b64decode(blob.encode("ascii")))
