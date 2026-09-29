from __future__ import annotations

import pytest

from rag_assistant.config import Settings
from rag_assistant.ingestion import chunk_document, load_directory, parse_front_matter
from rag_assistant.types import Document

MARKDOWN = """# Pump Manual

Intro paragraph.

## Startup

Step one. Step two.

### Warnings

Do not run dry.

## Shutdown

Turn it off.
"""


def test_front_matter_is_parsed_and_stripped() -> None:
    meta, body = parse_front_matter("---\ndoc_id: x1\ndoc_type: sop\n---\n# Title\nBody")
    assert meta == {"doc_id": "x1", "doc_type": "sop"}
    assert body.startswith("# Title")


def test_heading_aware_chunks_track_section_path() -> None:
    doc = Document(doc_id="pump", title="Pump Manual", text=MARKDOWN, source="pump.md")
    chunks = chunk_document(doc, chunk_size=500, chunk_overlap=50)
    sections = [c.section for c in chunks]
    assert sections == ["Pump Manual", "Startup", "Startup > Warnings", "Shutdown"]
    assert chunks[2].chunk_id == "pump#2"
    assert "Do not run dry." in chunks[2].text
    assert all(c.doc_id == "pump" and c.source == "pump.md" for c in chunks)


def test_long_sections_are_windowed_with_overlap() -> None:
    words = " ".join(f"word{i}" for i in range(400))
    doc = Document(doc_id="long", title="Long", text=f"## Body\n\n{words}", source="long.md")
    chunks = chunk_document(doc, chunk_size=300, chunk_overlap=60)
    assert len(chunks) > 1
    bodies = [c.text.split("\n", 1)[1] for c in chunks]
    assert all(len(b) <= 300 for b in bodies)
    # consecutive windows share words (overlap) and together cover the whole section
    assert set(bodies[0].split()) & set(bodies[1].split())
    assert bodies[-1].endswith("word399")


def test_invalid_overlap_rejected() -> None:
    doc = Document(doc_id="d", title="t", text="## A\ntext", source="d.md")
    with pytest.raises(ValueError):
        chunk_document(doc, chunk_size=100, chunk_overlap=100)


def test_sample_corpus_loads_with_metadata() -> None:
    docs = load_directory(Settings(_env_file=None).docs_dir)
    assert 10 <= len(docs) <= 15
    ids = {d.doc_id for d in docs}
    assert len(ids) == len(docs), "doc_ids must be unique"
    assert all("doc_type" in d.metadata for d in docs)
