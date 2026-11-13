"""Artificial corpus only: these tests contain no medical guidance."""
import json
import pytest
from medical_chatbot.rag import DocumentChunk, IntakeRag, retrieve
from medical_chatbot.schemas.conversation import Conversation, Message
from medical_chatbot.schemas.clinical_state import ClinicalCase, ClinicalState
from medical_chatbot.agents.intake.graph import IntakeSession, build_intake_graph, run_turn
from scripts.ingest_documents import ingest

TEXT = "Synthetic headache intake example for software testing only. This is not medical guidance."


def corpus(tmp_path, reviewed=True):
    chunk = DocumentChunk(chunk_id="test1", source_id="synthetic", title="Synthetic headache",
        version="test-v1", content=TEXT, reviewed_by="test-fixture" if reviewed else None,
        reviewed_at="2026-11-01T10:00:00+07:00" if reviewed else None)
    path = tmp_path / "corpus.jsonl"
    path.write_text(chunk.model_dump_json() + "\n", encoding="utf-8")
    return str(path)


def context():
    case = ClinicalCase(case_id="c1", chief_complaint="headache",
        present_illness={"onset": "yesterday", "duration": "two days", "course": "unchanged"})
    return (Conversation(conversation_id="test", patient_id="synthetic",
                         clinical_state=ClinicalState(current_case_id="c1", cases=[case])),
            Message(message_id="u1", role="USER", content="headache unchanged", timestamp="2026-11-08T10:00:00+07:00"))


def answer(**changes):
    return dict(recommendation="SAFE", reason="Synthetic supported outcome",
        evidence=[{"chunk_id": "test1", "quote": TEXT}],
        patient_evidence=[{"message_id": "u1", "source_text": "headache"}]) | changes


def test_missing_corpus_does_not_call_provider():
    def fail(_):
        pytest.fail("No corpus must not call Gemini")
    result = IntakeRag(generate_json=fail).assess(*context(), "c1")
    assert result.status == "UNAVAILABLE" and result.risk.result is None


def test_unreviewed_corpus_cannot_authorize_clinical_risk(tmp_path):
    def fail(_):
        pytest.fail("Unreviewed documents must not be sent for clinical assessment")
    result = IntakeRag(corpus(tmp_path, reviewed=False), generate_json=fail).assess(*context(), "c1")
    assert result.status == "INSUFFICIENT" and result.risk.result is None


def test_supported_result_keeps_source_version_and_exact_citation(tmp_path):
    result = IntakeRag(corpus(tmp_path), generate_json=lambda _: json.dumps(answer())).assess(*context(), "c1")
    assert result.status == "READY"
    assert result.risk.assessment_source == "RAG"
    assert result.risk.evidence_ids == ["test1"]
    assert result.sources[0].version == "test-v1"


@pytest.mark.parametrize("changes", [
    {"evidence": [{"chunk_id": "invented", "quote": TEXT}]},
    {"evidence": [{"chunk_id": "test1", "quote": "Completely invented guideline quotation."}]},
    {"patient_evidence": [{"message_id": "unknown", "source_text": "headache"}]},
    {"patient_evidence": []}, {"emergency": True},
    {"requirements": [{"key": "fever", "target": "present_illness.associated_symptoms", "priority": "CRITICAL",
                       "reason": "Synthetic example", "evidence_ids": ["fake"]}]},
])
def test_invalid_or_fabricated_grounding_is_rejected(tmp_path, changes):
    result = IntakeRag(corpus(tmp_path), generate_json=lambda _: json.dumps(answer(**changes))).assess(*context(), "c1")
    assert result.status == "FAILED"
    assert result.risk.result is None


def test_uncertain_supported_red_flag_becomes_blocking_requirement_not_patient_fact(tmp_path):
    raw = answer(recommendation="UNCERTAIN", patient_evidence=[], red_flags_to_check=[{
        "key": "synthetic_flag", "target": "present_illness.associated_symptoms", "priority": "HIGH",
        "reason": "Synthetic intake question", "evidence_ids": ["test1"]}])
    result = IntakeRag(corpus(tmp_path), generate_json=lambda _: json.dumps(raw)).assess(*context(), "c1")
    assert result.status == "READY"
    assert result.risk.result is None
    assert result.requirements[0].priority == "CRITICAL"
    assert result.requirements[0].blocking


def test_ingest_replaces_source_version_and_retrieval_is_traceable(tmp_path):
    source, output = tmp_path / "source.md", tmp_path / "corpus.jsonl"
    source.write_text(TEXT * 3, encoding="utf-8")
    count = ingest(source, output, source_id="s1", title="headache", version="v1", chunk_size=120, overlap=20)
    assert count > 1
    ingest(source, output, source_id="s1", title="headache", version="v2")
    chunks = retrieve(str(output), "headache")
    assert chunks and all(c.version == "v2" for c in chunks)
    assert all(c.source_id == "s1" for c in chunks)
    with pytest.raises(ValueError):
        ingest(source, source, source_id="s1", title="test", version="v2")


def test_explicit_specialist_request_uses_reviewed_rag_before_completion(tmp_path):
    conversation, user = context()
    def extract(_):
        return json.dumps({"case_result": {"resolution": "OLD_CASE", "target_case_id": "c1"}, "need_medical_rag": True, "rag_reason": "Specialist question support"})
    def screen(_):
        return json.dumps({"result": "SAFE", "reason": "Synthetic screen",
            "evidence": [{"message_id": "u1", "source_text": "headache"}]})
    graph = build_intake_graph(extract_json=extract, risk_json=screen,
        rag_service=IntakeRag(corpus(tmp_path), generate_json=lambda _: json.dumps(answer())))
    result = run_turn(graph, IntakeSession(conversation=conversation), user)
    assert result.output.next_action == "PASS_TO_AGENT_2"
    assert result.output.risk.assessment_source == "RAG"
    assert result.rag.status == "READY"
    assert result.session.conversation.clinical_state.cases[0].workflow.status == "WAITING_AGENT_2"


def test_rag_requirement_blocks_pass_and_is_asked(tmp_path):
    conversation, user = context()
    raw = answer(recommendation="UNCERTAIN", patient_evidence=[], requirements=[{
        "key": "synthetic_flag", "target": "present_illness.associated_symptoms", "priority": "CRITICAL",
        "reason": "Synthetic question", "evidence_ids": ["test1"]}])
    graph = build_intake_graph(
        extract_json=lambda _: json.dumps({"case_result": {"resolution": "OLD_CASE", "target_case_id": "c1"}, "need_medical_rag": True, "rag_reason": "Specialist question support"}),
        risk_json=lambda _: json.dumps({"result": "SAFE", "reason": "Synthetic screen", "evidence": [{"message_id": "u1", "source_text": "headache"}]}),
        question_json=lambda p: json.dumps({"question": "Synthetic question?", "scope_ids": [s["id"] for s in p["approved_scopes"]]}),
        rag_service=IntakeRag(corpus(tmp_path), generate_json=lambda _: json.dumps(raw)))
    result = run_turn(graph, IntakeSession(conversation=conversation), user)
    assert result.output.next_action == "ASK_MORE"
    assert result.session.conversation.messages[-1].asked_requirements[0].key == "synthetic_flag"


def test_grounded_danger_uses_rag_origin_and_preserves_evidence(tmp_path):
    conversation, user = context()
    graph = build_intake_graph(
        extract_json=lambda _: json.dumps({"case_result": {"resolution": "OLD_CASE", "target_case_id": "c1"}, "need_medical_rag": True, "rag_reason": "Specialist question support"}),
        risk_json=lambda _: json.dumps({"result": "SAFE", "reason": "Synthetic screen", "evidence": [{"message_id": "u1", "source_text": "headache"}]}),
        rag_service=IntakeRag(corpus(tmp_path), generate_json=lambda _: json.dumps(answer(recommendation="DANGEROUS", emergency=True))))
    result = run_turn(graph, IntakeSession(conversation=conversation), user)
    assert result.output.next_action == "EMERGENCY"
    assert result.output.risk.assessment_source == "RAG"
    assert result.output.risk.evidence_ids == ["test1"]


def test_conflicting_requirement_policy_fails_rag_instead_of_crashing_turn(tmp_path):
    raw = answer(requirements=[{"key": "onset", "target": "present_illness.onset", "priority": "CRITICAL",
                                "reason": "Synthetic conflicting catalog", "evidence_ids": ["test1"]}])
    result = IntakeRag(corpus(tmp_path), generate_json=lambda _: json.dumps(raw)).assess(*context(), "c1")
    assert result.status == "FAILED" and result.risk.result is None


def test_ingestion_review_metadata_must_be_complete(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text(TEXT, encoding="utf-8")
    with pytest.raises(ValueError):
        ingest(source, tmp_path / "corpus.jsonl", source_id="s", title="test", version="v1", reviewed_by="test-fixture")


def test_uncertain_screen_can_use_grounded_rag_questions_without_default_safe(tmp_path):
    conversation, user = context()
    conversation.clinical_state.cases[0].present_illness.course = None
    raw = answer(recommendation="UNCERTAIN", patient_evidence=[], red_flags_to_check=[{
        "key": "synthetic_flag", "target": "present_illness.associated_symptoms", "priority": "HIGH",
        "reason": "Synthetic intake check", "evidence_ids": ["test1"]}])
    graph = build_intake_graph(
        extract_json=lambda _: json.dumps({"case_result": {"resolution": "OLD_CASE", "target_case_id": "c1"}, "need_medical_rag": True, "rag_reason": "Specialist question support"}),
        risk_json=lambda _: json.dumps({"result": "UNCERTAIN", "reason": "Need grounded questions"}),
        question_json=lambda p: json.dumps({"question": "Synthetic question?", "scope_ids": [s["id"] for s in p["approved_scopes"]]}),
        rag_service=IntakeRag(corpus(tmp_path), generate_json=lambda _: json.dumps(raw)))
    result = run_turn(graph, IntakeSession(conversation=conversation), user)
    assert result.output.next_action == "ASK_MORE"
    assert result.output.risk.result is None
    assert result.session.conversation.messages[-1].asked_requirements[0].priority == "CRITICAL"
