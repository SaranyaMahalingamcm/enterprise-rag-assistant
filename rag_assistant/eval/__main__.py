"""CLI: ``python -m rag_assistant.eval [--golden PATH] [--output-dir DIR]``."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from rag_assistant.eval.runner import DEFAULT_GOLDEN, DEFAULT_REPORT_DIR, run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline on a golden set.")
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN, help="Golden set JSONL file.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR, help="Where to write reports.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.ERROR)  # keep CLI output readable
    report = run(args.golden, args.output_dir)
    s = report.summary
    print(f"Evaluated {s['questions']} questions -> {args.output_dir / 'eval_report.md'}")
    for key in (
        "hit@1",
        "hit@3",
        "hit@5",
        "mrr",
        "citation_precision",
        "citation_hit_rate",
        "groundedness",
        "answer_rate",
        "refusal_accuracy_unanswerable",
    ):
        print(f"  {key:<32} {s[key]:.3f}")
    print(f"  {'latency_p50_ms':<32} {s['latency_p50_ms']:.2f}")
    print(f"  {'latency_p95_ms':<32} {s['latency_p95_ms']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
