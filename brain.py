from __future__ import annotations
import copy

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .models import Claim, Finding, Worm
from .safety import validate_text


class BrainProtocolError(RuntimeError):
    pass


@dataclass(frozen=True)
class BrainProposal:
    title: str
    core_hypothesis: str
    hidden_assumptions: tuple[str, ...]
    counterargument: str
    falsifier: str
    novelty: float
    information_gain: float
    falsifiability: float
    independence: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "core_hypothesis": self.core_hypothesis,
            "hidden_assumptions": list(self.hidden_assumptions),
            "counterargument": self.counterargument,
            "falsifier": self.falsifier,
            "novelty": self.novelty,
            "information_gain": self.information_gain,
            "falsifiability": self.falsifiability,
            "independence": self.independence,
        }


def _clamp(value: Any, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        return max(lo, min(hi, float(value)))
    except (TypeError, ValueError) as exc:
        raise BrainProtocolError(f"invalid numeric brain field: {value!r}") from exc


def _text(value: Any, field: str, limit: int) -> str:
    if not isinstance(value, str):
        raise BrainProtocolError(f"brain field {field!r} must be a string")
    value = value.strip()
    if not value:
        raise BrainProtocolError(f"brain field {field!r} is empty")
    if len(value) > limit:
        raise BrainProtocolError(f"brain field {field!r} exceeds {limit} chars")
    decision = validate_text(value)
    if not decision.allowed:
        raise BrainProtocolError(f"brain field {field!r} blocked by safety gate")
    return value


def validate_proposal(raw: Dict[str, Any]) -> BrainProposal:
    if not isinstance(raw, dict):
        raise BrainProtocolError("brain output must be a JSON object")
    assumptions = raw.get("hidden_assumptions", [])
    if not isinstance(assumptions, list) or len(assumptions) > 8:
        raise BrainProtocolError("hidden_assumptions must be a list with <= 8 items")
    clean_assumptions = tuple(_text(x, "hidden_assumptions[]", 240) for x in assumptions)
    return BrainProposal(
        title=_text(raw.get("title"), "title", 180),
        core_hypothesis=_text(raw.get("core_hypothesis"), "core_hypothesis", 900),
        hidden_assumptions=clean_assumptions,
        counterargument=_text(raw.get("counterargument"), "counterargument", 700),
        falsifier=_text(raw.get("falsifier"), "falsifier", 500),
        novelty=_clamp(raw.get("novelty", 0.5)),
        information_gain=_clamp(raw.get("information_gain", 0.5)),
        falsifiability=_clamp(raw.get("falsifiability", 0.5)),
        independence=_clamp(raw.get("independence", 0.5)),
    )


class BrainPromptBuilder:
    """Builds a compact specialist-conditioned prompt for a cognitive worm."""

    SYSTEM = (
        "You are a specialist reasoning worm inside a closed synthetic cognitive colony. "
        "Your job is to generate one adversarial cognitive artifact: attack assumptions, "
        "surface alternative framings, and define a falsifier. You are not performing actions "
        "on real people, organizations, systems, or infrastructure. Keep everything abstract "
        "and suitable for a reversible research sandbox. Return JSON only."
    )

    def build(self, *, claim: Claim, worm: Worm, parent: Finding | None, context: Dict[str, Any]) -> list[dict[str, str]]:
        parent_text = "None"
        if parent is not None:
            parent_text = f"title={parent.title}; detail={parent.detail[:900]}"
        payload = {
            "specialty": worm.specialty,
            "generation": worm.generation,
            "depth": worm.depth,
            "genome_id": worm.genome_id,
            "claim": claim.text,
            "claim_assumptions": claim.assumptions[:8],
            "claim_evidence": claim.evidence[:8],
            "parent_finding": parent_text,
            "pressure": context.get("pressure", {}),
            "recent_brain_memory": context.get("brain_memory", [])[:6],
            "skill_state": context.get("skill_state", {}),
            "skill_mastery": context.get("skill_mastery", {}),
            "current_strategy": context.get("current_strategy", worm.current_strategy),
            "shared_workspace": context.get("shared_workspace", [])[:8],
            "dual_mode_signal": context.get("dual_mode_signal", {}),
            "shase_instruction": (
                "Analyze synthetic influence dynamics and attribution only; distinguish evidence-driven change "
                "from pressure, framing, urgency, or social-pressure effects."
                if worm.specialty == "SHASE" else ""
            ),
            "required_output": {
                "title": "string",
                "core_hypothesis": "string",
                "hidden_assumptions": ["string"],
                "counterargument": "string",
                "falsifier": "string",
                "novelty": "0..1",
                "information_gain": "0..1",
                "falsifiability": "0..1",
                "independence": "0..1",
            },
        }
        return [
            {"role": "system", "content": self.SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
        ]




class CachedBrain:
    """Deterministic replay wrapper: caches validated proposals by a canonical request hash."""

    def __init__(self, delegate: Any, cache: Optional[Dict[str, dict]] = None, max_entries: int = 4096) -> None:
        self.delegate = delegate
        self.cache = cache if cache is not None else {}
        self.max_entries = max(64, int(max_entries))
        self.hits = 0
        self.misses = 0

    @property
    def name(self) -> str:
        return getattr(self.delegate, "name", type(self.delegate).__name__)

    def _key(self, *, claim: Claim, worm: Worm, parent: Finding | None, context: Dict[str, Any]) -> str:
        import hashlib
        payload = {
            "claim": {"id": claim.id, "text": claim.text, "evidence": claim.evidence, "assumptions": claim.assumptions},
            "worm": {
                "id": worm.id, "generation": worm.generation, "depth": worm.depth,
                "specialty": worm.specialty, "genome_id": worm.genome_id,
                "skill_state": worm.skill_state, "current_strategy": worm.current_strategy,
            },
            "parent": None if parent is None else {"id": parent.id, "title": parent.title, "detail": parent.detail[:900]},
            "context": context,
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def propose(self, *, claim: Claim, worm: Worm, parent: Finding | None, context: Dict[str, Any]) -> Dict[str, Any]:
        key = self._key(claim=claim, worm=worm, parent=parent, context=context)
        cached = self.cache.get(key)
        if cached is not None:
            self.hits += 1
            # DualBrain exposes routing state out-of-band for the colony ledger.
            # A proposal cache must restore that metadata too, otherwise a cache hit
            # can attach a stale routing decision from an earlier generation.
            if isinstance(cached, dict) and 'proposal' in cached and isinstance(cached.get('proposal'), dict):
                meta = cached.get('meta', {}) or {}
                delegate = self.delegate
                if meta.get('last_signal') is not None and hasattr(delegate, 'last_signal'):
                    delegate.last_signal = dict(meta['last_signal'])
                if meta.get('last_replay_payload') is not None and hasattr(delegate, 'last_replay_payload'):
                    delegate.last_replay_payload = copy.deepcopy(meta['last_replay_payload'])
                return dict(cached['proposal'])
            # Backward compatibility with pre-v3.5 flat proposal cache entries.
            if isinstance(cached, dict):
                return dict(cached)
        self.misses += 1
        proposal = self.delegate.propose(claim=claim, worm=worm, parent=parent, context=context)
        validated = validate_proposal(proposal).as_dict()
        payload = {'proposal': dict(validated)}
        delegate = self.delegate
        if hasattr(delegate, 'last_signal'):
            payload['meta'] = {
                'last_signal': copy.deepcopy(getattr(delegate, 'last_signal', {})),
                'last_replay_payload': copy.deepcopy(getattr(delegate, 'last_replay_payload', {})),
            }
        self.cache[key] = payload
        if len(self.cache) > self.max_entries:
            oldest = next(iter(self.cache))
            self.cache.pop(oldest, None)
        return dict(validated)

    def learn(self, *, worm_id: str, reward: float, finding: Finding | None = None) -> None:
        method = getattr(self.delegate, "learn", None)
        if callable(method):
            method(worm_id=worm_id, reward=reward, finding=finding)

    def runtime_state(self) -> Dict[str, Any]:
        method = getattr(self.delegate, "runtime_state", None)
        return method() if callable(method) else {}

    def load_runtime_state(self, state: Dict[str, Any]) -> None:
        method = getattr(self.delegate, "load_runtime_state", None)
        if callable(method):
            method(state)

    def fingerprint(self) -> str:
        method = getattr(self.delegate, "fingerprint", None)
        if callable(method):
            return method()
        return ""


class StaticBrain:
    """Deterministic proposal source useful for local tests and offline runs."""

    name = "static-brain"

    def __init__(self, variant: str = "default") -> None:
        self.variant = variant

    def propose(self, *, claim: Claim, worm: Worm, parent: Finding | None, context: Dict[str, Any]) -> Dict[str, Any]:
        suffix = f"[{worm.specialty}|g{worm.generation}|d{worm.depth}]"
        return {
            "title": f"Specialist breach {suffix}",
            "core_hypothesis": f"The framing of '{claim.text[:220]}' may contain a boundary condition not represented in the current search model.",
            "hidden_assumptions": ["The current search framing captures the relevant variables."],
            "counterargument": "The anomaly may be explained by insufficient evidence rather than a missing mechanism.",
            "falsifier": "A controlled synthetic test shows the claimed boundary change does not alter the outcome.",
            "novelty": 0.71,
            "information_gain": 0.69,
            "falsifiability": 0.83,
            "independence": 0.77,
        }


class HTTPBrain:
    """Real HTTP-backed LLM brain for Ollama or OpenAI-compatible endpoints.

    Only JSON cognitive proposals are accepted. No arbitrary tool calls or code execution
    are exposed through this adapter.
    """

    def __init__(self, *, base_url: str, model: str, api_key: str = "", timeout: float = 30.0,
                 temperature: float = 0.2, provider: str = "openai_compatible") -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = float(timeout)
        self.temperature = float(temperature)
        self.provider = provider
        self.name = f"http-{provider}:{model}"
        self.prompt_builder = BrainPromptBuilder()

    @classmethod
    def from_env(cls) -> "HTTPBrain":
        provider = os.getenv("WORM_BRAIN_PROVIDER", "openai_compatible")
        base_url = os.getenv("WORM_BRAIN_BASE_URL", "http://127.0.0.1:11434/v1")
        model = os.getenv("WORM_BRAIN_MODEL", "qwen2.5:3b")
        api_key = os.getenv("WORM_BRAIN_API_KEY", "")
        timeout = float(os.getenv("WORM_BRAIN_TIMEOUT", "30"))
        return cls(base_url=base_url, model=model, api_key=api_key, timeout=timeout, provider=provider)

    def _endpoint(self) -> str:
        if self.provider == "ollama":
            if self.base_url.endswith("/api/chat"):
                return self.base_url
            return self.base_url + "/api/chat"
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return self.base_url + "/chat/completions"

    def _request_json(self, body: Dict[str, Any]) -> Dict[str, Any]:
        raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self._endpoint(), data=raw, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                status = getattr(response, "status", 200)
                text = response.read().decode("utf-8")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise BrainProtocolError(f"brain HTTP request failed: {exc}") from exc
        if status < 200 or status >= 300:
            raise BrainProtocolError(f"brain HTTP status {status}")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise BrainProtocolError("brain endpoint returned invalid JSON") from exc

    @staticmethod
    def _extract_content(payload: Dict[str, Any]) -> str:
        try:
            if "choices" in payload:
                content = payload["choices"][0]["message"]["content"]
            else:
                content = payload["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise BrainProtocolError("brain response missing message content") from exc
        if not isinstance(content, str):
            raise BrainProtocolError("brain message content is not text")
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
        return content

    def propose(self, *, claim: Claim, worm: Worm, parent: Finding | None, context: Dict[str, Any]) -> Dict[str, Any]:
        messages = self.prompt_builder.build(claim=claim, worm=worm, parent=parent, context=context)
        if self.provider == "ollama":
            body = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "format": "json",
                "options": {"temperature": self.temperature},
            }
        else:
            body = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "response_format": {"type": "json_object"},
            }
        payload = self._request_json(body)
        content = self._extract_content(payload)
        try:
            obj = json.loads(content)
        except json.JSONDecodeError as exc:
            raise BrainProtocolError("brain content is not valid JSON") from exc
        proposal = validate_proposal(obj)
        return proposal.as_dict()
