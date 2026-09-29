from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_assistant.config import Settings
from rag_assistant.eval import metrics
from rag_assistant.eval.runner import DEFAULT_GOLDEN, load_golden_set, run


def test_retrieval_metrics() -> None:
    ranking = ["x", "a", "b"]
    assert metrics.hit_at_k(ranking, {"a"}, 1) == 0.0
    assert metrics.hit_at_k(ranking, {"a"}, 2) == 1.0
    assert metrics.reciprocal_rank(ranking, {"b"}) == pytest.approx(1 / 3)
    assert metrics.reciprocal_rank(ranking, {"z"}) == 0.0
    assert metrics.citation_precision(["a", "x"], {"a"}) == 0.5


def test_groundedness_ignores_citation_markers() -> None:
    assert metrics.groundedness("Clean the lens daily [1].", ["Clean the lens daily with IPA."]) == 1.0
    assert metrics.groundedness("Replace the turbine [1].", ["Clean the lens daily."]) == 0.0


def test_golden_set_is_well_formed() -> None:
    items = load_golden_set(DEFAULT_GOLDEN)
    assert 15 <= len(items) <= 20
    assert len({i.id for i in items}) == len(items)
    assert any(not i.answerable for i in items)
    assert all(i.expected_doc_ids for i in items if i.answerable)


def test_eval_run_writes_reports_and_meets_baseline(tmp_path: Path) -> None:
    report = run(DEFAULT_GOLDEN, tmp_path, Settings(_env_file=None))
    assert (tmp_path / "eval_report.md").read_text().startswith("# Evaluation Report")
    data = json.loads((tmp_path / "eval_results.json").read_text())
    assert data["summary"]["questions"] == len(report.rows)
    # regression guard for the offline baseline (synthetic data)
    assert report.summary["hit@5"] >= 0.9
    assert report.summary["refusal_accuracy_unanswerable"] >= 0.5  # near-domain refusals are a known gap
    off_topic = [r for r in report.rows if r["id"] in {"q19", "q20"}]
    assert all(r["refused"] for r in off_topic)
