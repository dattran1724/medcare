"""Agent 1 Atlas RAG integration tests using synthetic search results."""
import json

import pytest

from medical_chatbot.agents.intake.graph import IntakeSession, build_intake_graph, run_turn
from medical_chatbot.rag import IntakeRag
from medical_chatbot.schemas.clinical_state import ClinicalCase, ClinicalState
from medical_chatbot.schemas.conversation import Conversation, Message

TEXT = "Synthetic source passage for testing citation grounding only."


def chunk():
    return {"chunk_id": "c1", "text": TEXT, "source": "guide.md", "pages": [3],
            "heading_path": ["Section A"]}


def context():
    case = ClinicalCase(case_id="case1", chief_complaint="headache")
    conversation = Conversation(conversation_id="conv", patient_id="synthetic",
        clinical_state=ClinicalState(current_case_id="case1", cases=[case]))
    message = Message(message_id="u1", role="USER", content="headache",
                      timestamp="2026-11-08T10:00:00+07:00")
    return conversation, message


def rag_answer(**changes):
    return {"requirements": [{
        "key": "symptom_context", "target": "present_illness.associated_symptoms",
        "priority": "HIGH", "reason": "Synthetic question support",
        "evidence_ids": ["c1"]}],
        "red_flags_to_check": [],
        "evidence": [{"chunk_id": "c1", "quote": TEXT}]} | changes


def test_atlas_fields_are_passed_to_gemini_and_citation_is_checked():
    conversation, message = context()
    captured = {}
    def generate(payload):
        captured.update(payload)
        return json.dumps(rag_answer())
    result = IntakeRag(search_fn=lambda query: [chunk()], generate_json=generate).assess(
        conversation, message, "case1")
    assert result.status == "READY"
    assert set(captured["documents"][0]) == {
        "chunk_id", "text", "source", "pages", "heading_path"}
    assert result.sources[0].source == "guide.md"
    assert result.requirements[0].key == "symptom_context"


@pytest.mark.parametrize("search_fn", [lambda _: [], lambda _: (_ for _ in ()).throw(ConnectionError())])
def test_no_atlas_evidence_or_search_error_does_not_call_gemini(search_fn):
    conversation, message = context()
    def fail(_):
        pytest.fail("Gemini must not run without retrieved Atlas evidence")
    result = IntakeRag(search_fn=search_fn, generate_json=fail).assess(conversation, message, "case1")
    assert result.status in {"UNAVAILABLE", "FAILED"}


@pytest.mark.parametrize("answer", [
    {"requirements": [], "red_flags_to_check": [],
     "evidence": [{"chunk_id": "c1", "quote": TEXT}]},
    {"requirements": [{"key": "x", "target": "present_illness.associated_symptoms",
       "priority": "HIGH", "reason": "Unsupported", "evidence_ids": ["missing"]}],
     "red_flags_to_check": [], "evidence": [{"chunk_id": "c1", "quote": TEXT}]},
    {"requirements": [{"key": "x", "target": "present_illness.associated_symptoms",
       "priority": "HIGH", "reason": "Unsupported", "evidence_ids": ["c1"]}],
     "red_flags_to_check": [], "evidence": [{"chunk_id": "c1", "quote": "fabricated quote"}]},
])
def test_insufficient_or_fabricated_evidence_never_becomes_ready(answer):
    conversation, message = context()
    result = IntakeRag(search_fn=lambda _: [chunk()],
        generate_json=lambda _: json.dumps(answer)).assess(conversation, message, "case1")
    assert result.status in {"INSUFFICIENT", "FAILED"}


def graph_with(service):
    def extract(payload):
        return json.dumps({"case_result": {"resolution": "NEW_CASE", "new_problem": "headache"},
                           "need_medical_rag": True, "rag_reason": "Specialist question support"})
    def screen(payload):
        msg = payload["new_message"]
        return json.dumps({"result": "SAFE", "reason": "Synthetic screening",
            "evidence": [{"message_id": msg["message_id"], "source_text": msg["content"]}]})
    return build_intake_graph(extract_json=extract, risk_json=screen,
                              question_json=lambda _: "{}", rag_service=service)


@pytest.mark.parametrize("search_fn,expected", [
    (lambda _: [], "HANDOFF_DOCTOR"),
    (lambda _: (_ for _ in ()).throw(ConnectionError()), "HANDOFF_DOCTOR"),
])
def test_agent1_hands_off_when_atlas_has_no_evidence_or_fails(search_fn, expected):
    service = IntakeRag(search_fn=search_fn)
    conversation, _ = context()
    result = run_turn(graph_with(service), IntakeSession(conversation=conversation),
        Message(message_id="u2", role="USER", content="headache detail",
                timestamp="2026-11-08T11:00:00+07:00"))
    assert result.output.next_action == expected
    assert result.session.conversation.messages[-1].role == "AI"


def test_agent1_keeps_rag_conditions_and_skips_atlas_when_not_requested():
    class NoSearch:
        def assess(self, *args):
            pytest.fail("RAG must be skipped when screening and extraction do not request it")
    def extract(payload):
        return json.dumps({"case_result": {"resolution": "NEW_CASE", "new_problem": "headache"}})
    def screen(payload):
        msg = payload["new_message"]
        return json.dumps({"result": "SAFE", "reason": "Synthetic screening",
            "evidence": [{"message_id": msg["message_id"], "source_text": msg["content"]}]})
    graph = build_intake_graph(extract_json=extract, risk_json=screen,
        question_json=lambda _: "{}", rag_service=NoSearch())
    conversation, _ = context()
    result = run_turn(graph, IntakeSession(conversation=conversation),
        Message(message_id="u2", role="USER", content="headache detail",
                timestamp="2026-11-08T11:00:00+07:00"))
    assert result.rag is None and result.rag_skip_reason == "not_requested"
