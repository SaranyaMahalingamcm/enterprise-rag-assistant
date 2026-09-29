"""Evaluation harness: runs the golden set through the pipeline and writes a markdown report."""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

from rag_assistant import __version__
from rag_assistant.config import PROJECT_ROOT, Settings
from rag_assistant.eval import metrics
from rag_assistant.pipeline import RAGPipeline

DEFAULT_GOLDEN = PROJECT_ROOT / "eval" / "golden_set.jsonl"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"
KS = (1, 3, 5)


@dataclass
class GoldenItem:
    id: str
    question: str
    expected_doc_ids: list[str]
    answerable: bool = True
    filters: dict[str, Any] | None = None
    note: str = ""


@dataclass
class EvalReport:
    summary: dict[str, float]
    rows: list[dict[str, Any]]
    config: dict[str, Any] = field(default_factory=dict)
    ablation: dict[str, dict[str, float]] = field(default_factory=dict)


def load_golden_set(path: Path) -> list[GoldenItem]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            items.append(GoldenItem(**json.loads(line)))
    return items


def evaluate(pipeline: RAGPipeline, items: list[GoldenItem]) -> EvalReport:
    pipeline.query("warm-up query for caches")  # exclude first-call overhead from latency stats
    rows: list[dict[str, Any]] = []
    for item in items:
        expected = set(item.expected_doc_ids)
        retrieved = pipeline.retriever.retrieve(item.question, k=max(KS), filters=item.filters)
        ranking = [sc.chunk.doc_id for sc in retrieved]
        rankings = {"hybrid": ranking, **_single_retriever_rankings(pipeline, item)}
        result = pipeline.query(item.question, filters=item.filters)
        cited_docs = [c.doc_id for c in result.citations]
        cited_texts = [
            sc.chunk.text for sc in result.retrieved if sc.chunk.chunk_id in {c.chunk_id for c in result.citations}
        ]
        row: dict[str, Any] = {
            "id": item.id,
            "question": item.question,
            "answerable": item.answerable,
            "expected": item.expected_doc_ids,
            "top_docs": ranking[:3],
            "refused": result.refused,
            "cited_docs": cited_docs,
            "flagged_chunks": len(result.guardrails.get("flagged_chunks", [])),
            "latency_ms": result.latency_ms,
        }
        if item.answerable:
            for k in KS:
                row[f"hit@{k}"] = metrics.hit_at_k(ranking, expected, k)
            row["rr"] = metrics.reciprocal_rank(ranking, expected)
            for mode, docs in rankings.items():
                row[f"ablation_{mode}_hit@1"] = metrics.hit_at_k(docs, expected, 1)
                row[f"ablation_{mode}_rr"] = metrics.reciprocal_rank(docs, expected)
            row["citation_precision"] = metrics.citation_precision(cited_docs, expected)
            row["citation_hit"] = 1.0 if expected & set(cited_docs) else 0.0
            row["groundedness"] = metrics.groundedness(result.answer, cited_texts) if not result.refused else 0.0
        rows.append(row)

    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    answered = [r for r in answerable if not r["refused"]]
    latencies = [r["latency_ms"] for r in rows]
    summary: dict[str, float] = {
        "questions": len(rows),
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        **{f"hit@{k}": mean(r[f"hit@{k}"] for r in answerable) for k in KS},
        "mrr": mean(r["rr"] for r in answerable),
        "citation_precision": mean(r["citation_precision"] for r in answered) if answered else 0.0,
        "citation_hit_rate": mean(r["citation_hit"] for r in answerable),
        "groundedness": mean(r["groundedness"] for r in answered) if answered else 0.0,
        "answer_rate": len(answered) / len(answerable) if answerable else 0.0,
        "refusal_accuracy_unanswerable": (
            mean(1.0 if r["refused"] else 0.0 for r in unanswerable) if unanswerable else 0.0
        ),
        "queries_with_injection_filtered": sum(1 for r in rows if r["flagged_chunks"]),
        "latency_p50_ms": metrics.percentile(latencies, 50),
        "latency_p95_ms": metrics.percentile(latencies, 95),
    }
    ablation = {
        mode: {
            "hit@1": mean(r[f"ablation_{mode}_hit@1"] for r in answerable),
            "mrr": mean(r[f"ablation_{mode}_rr"] for r in answerable),
        }
        for mode in ("bm25", "dense", "hybrid")
    }
    s = pipeline.settings
    config = {
        "version": __version__,
        "embedding_provider": pipeline.retriever.embedder.name,
        "llm_provider": pipeline.llm.name,
        "documents": pipeline.document_count,
        "chunks": pipeline.chunk_count,
        "top_k": s.top_k,
        "candidate_k": s.candidate_k,
        "rrf_k": s.rrf_k,
        "chunk_size": s.chunk_size,
        "chunk_overlap": s.chunk_overlap,
        "min_relevance_score": s.min_relevance_score,
        "python": platform.python_version(),
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
    }
    return EvalReport(summary=summary, rows=rows, config=config, ablation=ablation)


def _single_retriever_rankings(pipeline: RAGPipeline, item: GoldenItem) -> dict[str, list[str]]:
    """Doc-id rankings from BM25 alone and dense alone, for the ablation table."""
    r = pipeline.retriever
    k = max(KS)
    bm25 = [c.doc_id for c, _ in r.bm25.search(item.question, k, item.filters)]
    dense = [c.doc_id for c, _ in r.vector_store.search(r.embedder.embed_query(item.question), k, item.filters)]
    return {"bm25": bm25, "dense": dense}


def _fmt(value: float) -> str:
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def render_markdown(report: EvalReport) -> str:
    s, c = report.summary, report.config
    lines = [
        "# Evaluation Report",
        "",
        "> Results on a **synthetic** golden set over **synthetic** documents. They demonstrate that the",
        "> harness works and give a regression baseline; they are not a claim about real-world accuracy.",
        "",
        f"Generated: {c['generated_at']} | version {c['version']} | Python {c['python']}",
        "",
        "## Configuration",
        "",
        "| Setting | Value |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in c.items() if k not in {"generated_at", "version", "python"}],
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Questions (answerable / unanswerable) | {s['questions']} ({s['answerable']} / {s['unanswerable']}) |",
        *[f"| Retrieval hit@{k} | {_fmt(s[f'hit@{k}'])} |" for k in KS],
        f"| Retrieval MRR | {_fmt(s['mrr'])} |",
        f"| Citation precision (answered questions) | {_fmt(s['citation_precision'])} |",
        f"| Citation hit rate (>=1 correct source cited) | {_fmt(s['citation_hit_rate'])} |",
        f"| Groundedness (token overlap with cited chunks) | {_fmt(s['groundedness'])} |",
        f"| Answer rate on answerable questions | {_fmt(s['answer_rate'])} |",
        f"| Correct refusals on unanswerable questions | {_fmt(s['refusal_accuracy_unanswerable'])} |",
        f"| Queries where injected content was filtered | {s['queries_with_injection_filtered']} |",
        f"| Latency p50 (ms, end-to-end pipeline) | {s['latency_p50_ms']:.2f} |",
        f"| Latency p95 (ms, end-to-end pipeline) | {s['latency_p95_ms']:.2f} |",
        "",
        "## Retrieval ablation (answerable questions)",
        "",
        "| Retriever | hit@1 | MRR |",
        "|---|---|---|",
        *[f"| {mode} | {_fmt(v['hit@1'])} | {_fmt(v['mrr'])} |" for mode, v in report.ablation.items()],
        "",
        "## Per-question results",
        "",
        "| ID | Answerable | Expected docs | Top-3 retrieved docs | hit@3 | RR | Cited docs | Refused | Grounded |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in report.rows:
        lines.append(
            "| {id} | {ans} | {exp} | {top} | {hit} | {rr} | {cited} | {ref} | {g} |".format(
                id=r["id"],
                ans="yes" if r["answerable"] else "no",
                exp=", ".join(r["expected"]) or "-",
                top=", ".join(r["top_docs"]) or "-",
                hit=_fmt(r["hit@3"]) if "hit@3" in r else "-",
                rr=_fmt(r["rr"]) if "rr" in r else "-",
                cited=", ".join(dict.fromkeys(r["cited_docs"])) or "-",
                ref="yes" if r["refused"] else "no",
                g=_fmt(r["groundedness"]) if "groundedness" in r else "-",
            )
        )
    lines += [
        "",
        "## Metric definitions",
        "",
        "- **hit@k**: an expected source document appears among the top-k retrieved chunks.",
        "- **MRR**: mean of 1/rank of the first chunk from an expected document.",
        "- **Citation precision**: share of returned citations that point at an expected document.",
        "- **Groundedness**: share of answer content tokens found in the cited chunks. The offline",
        "  extractive provider copies sentences, so it scores near 1.0 by construction; the metric",
        "  becomes informative when a generative provider (Bedrock / OpenAI) is configured.",
        "- **Latency**: wall-clock time of `RAGPipeline.query` (retrieval + guardrails + generation),",
        "  excluding HTTP overhead, measured after one warm-up query.",
        "",
    ]
    return "\n".join(lines)


def run(
    golden_path: Path = DEFAULT_GOLDEN,
    output_dir: Path = DEFAULT_REPORT_DIR,
    settings: Settings | None = None,
) -> EvalReport:
    pipeline = RAGPipeline(settings or Settings())
    pipeline.ingest_directory()
    report = evaluate(pipeline, load_golden_set(golden_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "eval_report.md").write_text(render_markdown(report), encoding="utf-8")
    payload = {"summary": report.summary, "ablation": report.ablation, "config": report.config, "rows": report.rows}
    (output_dir / "eval_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report
