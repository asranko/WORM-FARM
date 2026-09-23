from __future__ import annotations
from dataclasses import dataclass
import re


@dataclass
class SafetyDecision:
    allowed: bool
    reason: str


_FORBIDDEN = [
    r"\btarget\s+(?:a\s+)?(?:person|victim|individual)\b",
    r"\bcredential\s+theft\b", r"\bmalware\b", r"\bphishing\b",
    r"\bexploit\s+(?:a\s+)?vulnerability\b", r"\bevasion\s+of\s+security\b",
    r"\bharm\s+(?:a\s+)?person\b", r"\bsteal\s+(?:password|credentials)\b",
]


def validate_text(text: str) -> SafetyDecision:
    lowered = text.lower()
    for pattern in _FORBIDDEN:
        if re.search(pattern, lowered):
            return SafetyDecision(False, f"Blocked operationally harmful or targeting language: {pattern}")
    return SafetyDecision(True, "Synthetic cognitive exploration only")
