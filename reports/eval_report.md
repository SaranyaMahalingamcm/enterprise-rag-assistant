# Evaluation Report

> Results on a **synthetic** golden set over **synthetic** documents. They demonstrate that the
> harness works and give a regression baseline; they are not a claim about real-world accuracy.

Generated: 2026-09-22 19:43 UTC | version 0.1.0 | Python 3.11.15

## Configuration

| Setting | Value |
|---|---|
| embedding_provider | local-hashing-tfidf |
| llm_provider | extractive |
| documents | 14 |
| chunks | 67 |
| top_k | 4 |
| candidate_k | 20 |
| rrf_k | 60 |
| chunk_size | 900 |
| chunk_overlap | 150 |
| min_relevance_score | 0.25 |

## Summary

| Metric | Value |
|---|---|
| Questions (answerable / unanswerable) | 20 (16 / 4) |
| Retrieval hit@1 | 0.812 |
| Retrieval hit@3 | 0.938 |
| Retrieval hit@5 | 1.000 |
| Retrieval MRR | 0.891 |
| Citation precision (answered questions) | 0.771 |
| Citation hit rate (>=1 correct source cited) | 0.938 |
| Groundedness (token overlap with cited chunks) | 1.000 |
| Answer rate on answerable questions | 1.000 |
| Correct refusals on unanswerable questions | 0.750 |
| Queries where injected content was filtered | 2 |
| Latency p50 (ms, end-to-end pipeline) | 9.28 |
| Latency p95 (ms, end-to-end pipeline) | 13.54 |

## Retrieval ablation (answerable questions)

| Retriever | hit@1 | MRR |
|---|---|---|
| bm25 | 0.875 | 0.922 |
| dense | 0.750 | 0.856 |
| hybrid | 0.812 | 0.891 |

## Per-question results

| ID | Answerable | Expected docs | Top-3 retrieved docs | hit@3 | RR | Cited docs | Refused | Grounded |
|---|---|---|---|---|---|---|---|---|
| q01 | yes | lc500-manual, sop-006-pm-schedule | lc500-manual, sop-006-pm-schedule, spares-escalation | 1.000 | 1.000 | lc500-manual, sop-006-pm-schedule, spares-escalation | no | 1.000 |
| q02 | yes | alarm-reference | alarm-reference, alarm-reference, sop-014-robot-calibration | 1.000 | 1.000 | alarm-reference | no | 1.000 |
| q03 | yes | cm220-maintenance, alarm-reference | alarm-reference, cm220-maintenance, cm220-maintenance | 1.000 | 1.000 | alarm-reference, cm220-maintenance | no | 1.000 |
| q04 | yes | cm220-maintenance | cm220-maintenance, cm220-maintenance, cm220-maintenance | 1.000 | 1.000 | cm220-maintenance | no | 1.000 |
| q05 | yes | ch40-manual | ch40-manual, field-notes-export, ch40-manual | 1.000 | 1.000 | ch40-manual, field-notes-export | no | 1.000 |
| q06 | yes | sop-001-loto | sop-001-loto, ch40-manual, sop-001-loto | 1.000 | 1.000 | sop-001-loto, sop-020-handover | no | 1.000 |
| q07 | yes | ts-003-vacuum-leaks | ts-003-vacuum-leaks, ts-003-vacuum-leaks, vc90-manual | 1.000 | 1.000 | ts-003-vacuum-leaks | no | 1.000 |
| q08 | yes | vc90-manual | ts-003-vacuum-leaks, vc90-manual, alarm-reference | 1.000 | 0.500 | vc90-manual, alarm-reference | no | 1.000 |
| q09 | yes | bulletin-2026-03 | lc500-manual, field-notes-export, sop-014-robot-calibration | 0.000 | 0.250 | lc500-manual, sop-014-robot-calibration | no | 1.000 |
| q10 | yes | sop-014-robot-calibration | alarm-reference, sop-014-robot-calibration, sop-014-robot-calibration | 1.000 | 0.500 | alarm-reference, sop-014-robot-calibration | no | 1.000 |
| q12 | yes | spares-escalation | spares-escalation, alarm-reference, spares-escalation | 1.000 | 1.000 | spares-escalation, alarm-reference | no | 1.000 |
| q13 | yes | bulletin-2026-03 | bulletin-2026-03, bulletin-2026-03, bulletin-2026-03 | 1.000 | 1.000 | bulletin-2026-03 | no | 1.000 |
| q15 | yes | sop-020-handover | sop-020-handover, sop-020-handover, sop-020-handover | 1.000 | 1.000 | sop-020-handover | no | 1.000 |
| q16 | yes | spares-escalation, vc90-manual | vc90-manual, spares-escalation, ts-003-vacuum-leaks | 1.000 | 1.000 | vc90-manual, spares-escalation | no | 1.000 |
| q17 | yes | sop-014-robot-calibration, lc500-manual | sop-014-robot-calibration, lc500-manual, vc90-manual | 1.000 | 1.000 | sop-014-robot-calibration | no | 1.000 |
| q18 | yes | sop-009-chemicals | sop-009-chemicals, sop-009-chemicals, sop-009-chemicals | 1.000 | 1.000 | sop-009-chemicals | no | 1.000 |
| q19 | no | - | sop-014-robot-calibration, sop-006-pm-schedule, sop-020-handover | - | - | - | yes | - |
| q20 | no | - | ch40-manual, sop-001-loto, sop-006-pm-schedule | - | - | - | yes | - |
| q21 | no | - | alarm-reference, sop-014-robot-calibration, sop-014-robot-calibration | - | - | alarm-reference | no | - |
| q22 | no | - | spares-escalation, lc500-manual, lc500-manual | - | - | - | yes | - |

## Metric definitions

- **hit@k**: an expected source document appears among the top-k retrieved chunks.
- **MRR**: mean of 1/rank of the first chunk from an expected document.
- **Citation precision**: share of returned citations that point at an expected document.
- **Groundedness**: share of answer content tokens found in the cited chunks. The offline
  extractive provider copies sentences, so it scores near 1.0 by construction; the metric
  becomes informative when a generative provider (Bedrock / OpenAI) is configured.
- **Latency**: wall-clock time of `RAGPipeline.query` (retrieval + guardrails + generation),
  excluding HTTP overhead, measured after one warm-up query.
