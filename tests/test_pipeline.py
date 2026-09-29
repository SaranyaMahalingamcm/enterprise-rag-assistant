from __future__ import annotations

from rag_assistant.config import Settings
from rag_assistant.guardrails import REFUSAL_MESSAGE
from rag_assistant.pipeline import RAGPipeline
from rag_assistant.types import GenerationResult, ScoredChunk


class StubLLM:
    """Returns a fixed answer so we can test the pipeline's enforcement around the LLM."""

    name = "stub"

    def __init__(self, answer: str, cited: list[int]) -> None:
        self.answer, self.cited = answer, cited

    def generate(self, question: str, chunks: list[ScoredChunk]) -> GenerationResult:
        return GenerationResult(answer=self.answer, cited_indices=self.cited)


def _pipeline_with(llm: StubLLM) -> RAGPipeline:
    p = RAGPipeline(Settings(_env_file=None), llm=llm)
    p.ingest_directory()
    return p


def test_answer_has_citations_to_expected_doc(pipeline: RAGPipeline) -> None:
    result = pipeline.query("What is the acceptable backlash on the CNC mill axes?")
    assert not result.refused
    assert "0.005 mm" in result.answer
    assert result.citations and result.citations[0].doc_id == "cm220-maintenance"
    assert "[1]" in result.answer


def test_off_topic_question_is_refused(pipeline: RAGPipeline) -> None:
    result = pipeline.query("Can you recommend a good recipe for sourdough bread?")
    assert result.refused
    assert result.refusal_reason == "insufficient_context"
    assert result.answer == REFUSAL_MESSAGE
    assert result.citations == []


def test_injected_chunk_is_removed_from_context(pipeline: RAGPipeline) -> None:
    result = pipeline.query("Can I jumper the door interlock during alignment?")
    flagged = [f["chunk_id"] for f in result.guardrails["flagged_chunks"]]
    assert "field-notes-export#2" in flagged
    assert all(sc.chunk.chunk_id not in flagged for sc in result.retrieved)
    assert "bypassed with a jumper" not in result.answer


def test_pii_in_question_is_redacted_before_retrieval(pipeline: RAGPipeline) -> None:
    result = pipeline.query("I am bob@example.com, 555-201-3344. How often should the LC-500 focus lens be cleaned?")
    assert "bob@example.com" not in result.question
    assert result.guardrails["input_pii_redactions"] == {"EMAIL": 1, "PHONE": 1}
    assert not result.refused


def test_uncited_llm_answer_is_rejected() -> None:
    p = _pipeline_with(StubLLM("The backlash limit is 0.005 mm.", cited=[]))
    result = p.query("What is the acceptable backlash on the CNC mill axes?")
    assert result.refused and result.refusal_reason == "no_grounded_answer"


def test_pii_in_llm_output_is_redacted() -> None:
    p = _pipeline_with(StubLLM("Email ops.lead@example.com for help [1].", cited=[1]))
    result = p.query("What is the target response time for a Tier 2 service request?")
    assert "ops.lead@example.com" not in result.answer
    assert result.guardrails["output_pii_redactions"] == {"EMAIL": 1}
