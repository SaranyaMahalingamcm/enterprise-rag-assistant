"""Small text utilities shared by retrieval, generation and evaluation."""

from __future__ import annotations

import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_TABLE_SEP_RE = re.compile(r"^\|?[\s:|-]+\|?$")


def _normalize(token: str) -> str:
    """Very light plural folding ("spills" -> "spill", "switches" -> "switch"); codes are untouched."""
    if len(token) <= 3 or not token.isalpha():
        return token
    if token.endswith(("ches", "shes", "sses", "xes")):
        return token[:-2]
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    """Lowercase word tokens; keeps codes such as ``e101``, ``lc-500`` and ``1.5`` intact."""
    tokens = _TOKEN_RE.findall(text.lower())
    if drop_stopwords:
        tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
    return [_normalize(t) for t in tokens]


def _table_cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def split_sentences(text: str) -> list[str]:
    """Split markdown chunk text into sentence-like units.

    Soft-wrapped paragraph lines are re-joined, each list item is one unit, and table rows become
    ``"Header: value; Header: value"`` units. Blockquotes (used for document notices) are skipped.
    """
    blocks: list[tuple[str, bool]] = []  # (text, splittable into sentences)
    current: list[str] = []
    current_is_item = False
    header: list[str] | None = None
    lines = text.splitlines()

    def flush() -> None:
        if current:
            blocks.append((" ".join(current), not current_is_item))
            current.clear()

    for i, raw in enumerate(lines):
        line = raw.strip()
        if not line or line.startswith(">"):
            flush()
            continue
        if line.startswith("|"):
            flush()
            if _TABLE_SEP_RE.match(line):
                continue
            cells = _table_cells(line)
            nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
            if nxt.startswith("|") and _TABLE_SEP_RE.match(nxt):
                header = cells  # header row: remember it, do not emit
                continue
            if header and len(header) == len(cells):
                blocks.append(("; ".join(f"{h}: {c}" for h, c in zip(header, cells, strict=True)), False))
            else:
                blocks.append((", ".join(cells), False))
            continue
        header = None  # any non-table line ends the current table
        if _LIST_ITEM_RE.match(line):
            flush()
            current_is_item = True
            current.append(_LIST_ITEM_RE.sub("", line))
        else:
            if not current:
                current_is_item = False
            current.append(line)
    flush()

    units: list[str] = []
    for block, splittable in blocks:
        block = block.replace("**", "").strip()
        if not splittable:
            units.append(block)  # list items and table rows stay whole: they are self-contained facts
        else:
            units.extend(s.strip() for s in _SENTENCE_RE.split(block) if s.strip())
    return units
