from __future__ import annotations

import json
import random
from dataclasses import asdict
from hashlib import sha256
from typing import Any

import numpy as np

from worm_information_bus import Message, ShannonWorkspace, mutual_information_discrete, information_efficiency
from model_diff import diff_behavior_vectors

SPECIALTIES = [
    "ASSUMPTION_HUNTER", "ANOMALY_HUNTER", "CAUSALITY", "COUNTERFACTUAL",
    "EVIDENCE_FORENSICS", "PARADIGM_BREAKER", "BOUNDARY_HUNTER", "NOVELTY_FORGE",
    "RED_TEAM", "MODEL_KILLER", "META_CRITIC", "SYNTHESIZER"
]


def make_population(seed: int, generation: int, n: int = 24) -> list[Message]:
    rng = random.Random(seed + generation * 997)
    messages: list[Message] = []
    for i in range(n):
        specialty = SPECIALTIES[i % len(SPECIALTIES)]
        payload = tuple(round(rng.uniform(-1, 1), 3) for _ in range(8))
        messages.append(
            Message(
                worm_id=f"G{generation:04d}-W{i:03d}",
                generation=generation,
                specialty=specialty,
                payload=payload,
                confidence=round(rng.random(), 3),
                novelty=round(rng.random(), 3),
                criticality=round(rng.random(), 3),
                bits=32,
            )
        )
    return messages


def run_reference(seed: int = 42, generations: int = 8) -> dict[str, Any]:
    bus = ShannonWorkspace(capacity_bits=256, noise=0.08, seed=seed)
    generation_rows = []
    before_vectors = {s: [0.0] for s in SPECIALTIES}
    all_accepted: list[Message] = []

    for g in range(generations):
        messages = make_population(seed, g)
        decision = bus.route(messages)
        all_accepted.extend(decision.accepted)
        generation_rows.append({
            "generation": g,
            "name": ["GENESIS", "FIRST BURROW", "PRESSURE VEIL", "FAULTLINE", "BLACK ICE", "UNSEEN CAUSE", "DEEP SCAR", "NULL HORIZON"][g],
            "submitted": len(messages),
            "accepted": len(decision.accepted),
            "rejected": len(decision.rejected),
            "used_bits": decision.used_bits,
            "equivocation": round(decision.equivocation, 6),
        })
        # Behavioral profile for model diffing.
        for s in SPECIALTIES:
            vals = [m.novelty + m.confidence - 0.5 * m.criticality for m in messages if m.specialty == s]
            before_vectors[s] = vals or [0.0]

    labels_a = [int.from_bytes(sha256(m.worm_id.encode("utf-8")).digest()[:4], "big") % 4 for m in all_accepted]
    labels_b = [int((sum(m.payload) + 4) // 2) % 4 for m in all_accepted]
    mi = mutual_information_discrete(labels_a, labels_b)
    eff = information_efficiency(mi, 32)

    # Synthetic pre/post population for behavioral diff demonstration.
    after_vectors = {s: list(np.asarray(v) + 0.1 * (i + 1)) for i, (s, v) in enumerate(before_vectors.items())}
    diffs = diff_behavior_vectors(before_vectors, after_vectors)

    return {
        "seed": seed,
        "generations": generation_rows,
        "workspace": {
            "capacity_bits": bus.capacity_bits,
            "noise": bus.noise,
            "mutual_information_bits": round(mi, 6),
            "information_efficiency_bits_per_bit": round(eff, 6),
        },
        "model_diff": [asdict(d) for d in diffs],
        "accepted_messages": len(all_accepted),
    }


if __name__ == "__main__":
    print(json.dumps(run_reference(), indent=2))
