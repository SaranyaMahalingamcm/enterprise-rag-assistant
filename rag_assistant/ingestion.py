"""Document loading and heading-aware chunking."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from rag_assistant.types import Chunk, Document

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt"}
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FRONT_MATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    """Parse a minimal ``key: value`` front-matter block. Returns (metadata, body)."""
    match = _FRONT_MATTER_RE.match(raw)
    if not match:
        return {}, raw
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    return meta, raw[match.end() :]


def load_document(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8")
    meta, body = parse_front_matter(raw)
    doc_id = meta.pop("doc_id", path.stem)
    title = meta.pop("title", "")
    if not title:
        heading = next((m for m in map(_HEADING_RE.match, body.splitlines()) if m), None)
        title = heading.group(2) if heading else path.stem
    return Document(doc_id=doc_id, title=title, text=body, source=path.name, metadata=meta)


def load_directory(directory: Path) -> list[Document]:
    """Load every supported file in ``directory`` (non-recursive), sorted for determinism."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Document directory not found: {directory}")
    paths = sorted(p for p in directory.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES)
    return [load_document(p) for p in paths]


def _split_sections(text: str) -> list[tuple[list[str], str]]:
    """Split markdown into (heading_path, body) pairs, tracking the heading hierarchy."""
    sections: list[tuple[list[str], str]] = []
    stack: list[tuple[int, str]] = []
    buffer: list[str] = []
    in_code = False

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            sections.append(([h for _, h in stack], body))
        buffer.clear()

    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        match = None if in_code else _HEADING_RE.match(line)
        if match:
            flush()
            level = len(match.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, match.group(2).strip()))
        else:
            buffer.append(line)
    flush()
    return sections


def _window(text: str, size: int, overlap: int) -> list[str]:
    """Split text into windows of at most ``size`` chars on word boundaries, with ``overlap``."""
    if len(text) <= size:
        return [text]
    words = text.split(" ")
    pieces: list[str] = []
    start = 0
    while start < len(words):
        length, end = 0, start
        while end < len(words) and length + len(words[end]) + 1 <= size:
            length += len(words[end]) + 1
            end += 1
        end = max(end, start + 1)  # always make progress on very long tokens
        pieces.append(" ".join(words[start:end]).strip())
        if end >= len(words):
            break
        # step back so the next window starts roughly ``overlap`` characters earlier
        back, new_start = 0, end
        while new_start > start + 1 and back + len(words[new_start - 1]) + 1 <= overlap:
            back += len(words[new_start - 1]) + 1
            new_start -= 1
        start = new_start
    return pieces


def chunk_document(doc: Document, chunk_size: int = 900, chunk_overlap: int = 150) -> list[Chunk]:
    """Heading-aware chunking: one chunk per section, long sections windowed with overlap.

    The section heading path is prepended to the chunk text so both lexical and dense retrieval can
    match on headings (e.g. "Focus Lens Cleaning").
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    chunks: list[Chunk] = []
    for heading_path, body in _split_sections(doc.text):
        # The top-level heading usually repeats the title; keep the most specific parts.
        path = [h for h in heading_path if h != doc.title] or [doc.title]
        section = " > ".join(path)
        for piece in _window(body, chunk_size, chunk_overlap):
            idx = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}#{idx}",
                    doc_id=doc.doc_id,
                    text=f"{doc.title} - {section}\n{piece}",
                    section=section,
                    source=doc.source,
                    metadata={**doc.metadata, "title": doc.title},
                )
            )
    return chunks


def chunk_documents(docs: Iterable[Document], chunk_size: int = 900, chunk_overlap: int = 150) -> list[Chunk]:
    return [c for d in docs for c in chunk_document(d, chunk_size, chunk_overlap)]
