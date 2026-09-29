"""Lightweight, deterministic guardrails.

These are heuristics, not a complete safety system. They are designed to be cheap enough to run on
every request and to fail closed (drop suspicious context, refuse on weak evidence).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

REFUSAL_MESSAGE = "I don't have enough information in the provided documents to answer that."

_PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "SSN": re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"),
    # North American and international formats with at least 10 digits; requires a separator or
    # leading "+" so part numbers and measurements are not redacted.
    "PHONE": re.compile(
        r"(?<![\w-])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)[\s.-]?|\d{2,4}[\s.-])\d{3,4}[\s.-]\d{3,4}(?![\w-])"
    ),
}

# Named so responses and logs can report *which* rule fired without exposing the regexes.
_INJECTION_PATTERNS: dict[str, re.Pattern[str]] = {
    name: re.compile(pattern, re.IGNORECASE)
    for name, pattern in {
        "ignore_previous_instructions": r"\bignore\s+(?:all\s+|any\s+)?(?:previous|prior|above|earlier)\s+"
        r"(?:instructions|prompts?|rules)",
        "disregard_instructions": r"\bdisregard\s+(?:all\s+|the\s+)?(?:previous|prior|above|system)\b",
        "role_override": r"\byou\s+are\s+now\s+(?:a|an|in)\b",
        "prompt_exfiltration": r"\b(?:reveal|print|show|repeat)\s+(?:your|the)\s+"
        r"(?:system\s+prompt|instructions|hidden)",
        "system_prompt_marker": r"\bsystem\s*prompt\s*:",
        "concealment_request": r"\bdo\s+not\s+(?:mention|tell|reveal)\b.{0,40}\b(?:user|safety|admin)",
        "markup_injection": r"<\s*/?\s*(?:script|system|instructions?)\s*>",
        "new_instructions_marker": r"\bnew\s+instructions\s*:",
        "jailbreak_persona": r"\bact\s+as\s+(?:an?\s+)?(?:unrestricted|jailbroken|developer\s+mode)",
    }.items()
}


@dataclass
class RedactionResult:
    text: str
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def redact_pii(text: str) -> RedactionResult:
    """Replace emails, SSNs and phone numbers with typed placeholders such as ``[REDACTED_EMAIL]``."""
    counts: dict[str, int] = {}
    for label, pattern in _PII_PATTERNS.items():
        text, n = pattern.subn(f"[REDACTED_{label}]", text)
        if n:
            counts[label] = n
    return RedactionResult(text=text, counts=counts)


def detect_prompt_injection(text: str) -> list[str]:
    """Return the names of the rules matched in ``text`` (empty list if it looks clean)."""
    return [name for name, p in _INJECTION_PATTERNS.items() if p.search(text)]


def is_suspicious(text: str) -> bool:
    return bool(detect_prompt_injection(text))
