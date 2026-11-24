"""End-to-end graph/API tests with injected providers, never real API calls."""
import json
import pytest
from fastapi.testclient import TestClient
from medical_chatbot.api import create_app
from medical_chatbot.agents.intake.graph import IntakeSession, build_intake_graph, run_turn
from medical_chatbot.schemas.conversation import Conversation, Message


@pytest.fixture(autouse=True)
def isolate_runtime_database(monkeypatch):
    monkeypatch.setenv("MONGODB_URI", "")
    monkeypatch.setenv("MEDCARE_DB_PATH", ":memory:")
    monkeypatch.delenv("MEDCARE_RAG_PATH", raising=False)


def extraction(payload):
    message = payload["new_message"]
    cases = payload["clinical_state"]["cases"]
    resolution = ({"resolution": "OLD_CASE", "target_case_id": cases[0]["case_id"]} if cases else
                  {"resolution": "NEW_CASE", "new_problem": "headache"})
    facts = []
    for field, value in [("duration", "two days"), ("onset", "yesterday"), ("course", "unchanged")]:
        if value in message["content"]:
            facts.append({"field": f"present_illness.{field}", "key": field, "value": value,
                          "status": "PRESENT", "source": {"message_id": message["message_id"], "source_text": value}})
    return json.dumps({"case_result": resolution, "case_facts": facts})


def screening(payload, result="SAFE", emergency=False, **changes):
    message = payload["new_message"]
    return json.dumps(dict(result=result, reason="Synthetic screening result", emergency=emergency,
        evidence=[{"message_id": message["message_id"], "source_text": message["content"]}]) | changes)


def question(payload):
    return json.dumps({"question": "Bạn có thể mô tả thêm thời điểm bắt đầu và diễn biến không?",
                       "scope_ids": [s["id"] for s in payload["approved_scopes"]]})


def graph(**overrides):
    return build_intake_graph(**(dict(extract_json=extraction, risk_json=screening,
                                      question_json=question) | overrides))


def session():
    return IntakeSession(conversation=Conversation(conversation_id="conv", patient_id="synthetic"))


def message(text="headache for two days", id="u1"):
    return Message(message_id=id, role="USER", content=text, timestamp="2026-11-06T10:00:00+07:00")


def test_graph_multiple_turns_persists_history_and_question_metadata():
    original = session()
    flow = graph()
    first = run_turn(flow, original, message())
    assert first.output.next_action == "ASK_MORE"
    assert not original.conversation.messages
    history = first.session.conversation.messages
    assert [m.role for m in history] == ["USER", "AI"]
    assert [r.key for r in history[-1].asked_requirements] == ["onset", "course"]
    second = run_turn(flow, first.session, message("started yesterday and is unchanged", "u2"))
    assert len(second.session.conversation.messages) == 4
    assert second.output.response_text
    assert second.session.conversation.messages[-1].content == second.output.response_text
    assert second.output.next_action == "PASS_TO_AGENT_2"
    assert second.rag is None
    case = second.session.conversation.clinical_state.cases[0]
    assert case.present_illness.onset == "yesterday"
    assert case.present_illness.course == "unchanged"
    assert case.response_mode == "AI_ACTIVE"  # A request is not doctor acceptance.


def test_unanswered_question_after_correction_keeps_intake_open():
    def extract(payload):
        data = json.loads(extraction(payload))
        current = payload["new_message"]
        if current["message_id"] == "u2":
            data["case_facts"] = [dict(field="present_illness.onset", key="onset",
                value="this morning", status="PRESENT", change_type="CORRECTION",
                source=dict(message_id="u2", source_text=current["content"]))]
        return json.dumps(data)

    flow = graph(extract_json=extract)
    first = run_turn(flow, session(), message("headache started yesterday"))
    second = run_turn(flow, first.session,
                      message("Correction, I meant this morning", "u2"))
    case = second.session.conversation.clinical_state.cases[0]
    assert case.present_illness.onset == "this morning"
    assert case.progress_history[-1].type == "CORRECTION"
    assert case.workflow.status == "COLLECTING"
    assert second.output.next_action == "ASK_MORE"
    assert {r.key for r in second.session.conversation.messages[-1].asked_requirements} == {"duration", "course"}
    third = run_turn(flow, second.session, message("headache unchanged", "u3"))
    assert third.output.next_action == "ASK_MORE"
    assert third.session.conversation.clinical_state.cases[0].present_illness.course == "unchanged"


def test_default_database_and_paginated_history_survive_restart(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MEDCARE_DB_PATH")
    with TestClient(create_app(graph())) as client:
        cid = client.post("/conversations", json={"patient_id": "synthetic"}).json()["conversation"]["conversation_id"]
        client.post(f"/conversations/{cid}/messages", json={"message_id": "u1", "content": "headache"})
        assert client.get("/health").json()["storage"] == "sqlite"
    assert (tmp_path / "agent1.sqlite3").exists()
    with TestClient(create_app(graph())) as client:
        url = f"/conversations/{cid}/messages"
        assert client.get(url, params={"limit": 1}).json()[0]["role"] == "USER"
        assert client.get(url, params={"offset": 1, "limit": 1}).json()[0]["role"] == "AI"
        assert client.get(url, params={"offset": 2}).json() == []
        assert client.get(url, params={"limit": 201}).status_code == 422
        assert client.get(url, params={"offset": -1}).status_code == 422
        assert client.get("/conversations/missing/messages").status_code == 404


def test_risk_runs_first_and_danger_survives_extraction_failure():
    calls = []
    def assess(payload):
        calls.append("risk")
        return screening(payload, result="DANGEROUS", emergency=True)
    def fail(_):
        calls.append("extract")
        raise ValueError("secret")
    result = run_turn(graph(risk_json=assess, extract_json=fail), session(), message())
    assert calls == ["risk", "extract"]
    assert result.output.next_action == "EMERGENCY"
    assert result.output.risk.result == "DANGEROUS"
    assert result.processing_errors[0].stage == "extraction"
    assert result.session.conversation.messages[0].role == "USER"


@pytest.mark.parametrize("risk_json", [lambda _: "bad JSON", lambda p: screening(p, result="UNCERTAIN"),
    lambda p: screening(p, needs_external_evidence=True),
    lambda p: screening(p, evidence=[{"message_id": "missing", "source_text": "invented"}])])
def test_risk_uncertainty_and_invalid_evidence_never_default_safe(risk_json):
    result = run_turn(graph(risk_json=risk_json), session(), message())
    if result.output.risk.assessment_status == "FAILED":
        assert result.output.status == "FAILED" and result.output.next_action is None
    elif result.rag is not None:
        assert result.output.next_action == "HANDOFF_DOCTOR"
    else:
        assert result.output.next_action == "ASK_MORE"
        assert result.output.risk.result is None


def test_question_failure_preserves_user_and_data_without_marking_asked():
    result = run_turn(graph(question_json=lambda _: "invalid"), session(), message())
    assert result.output.status == "FAILED"
    assert result.output.error.stage == "question"
    assert len(result.session.conversation.messages) == 1
    assert all(r.asked_count == 0 for r in result.session.conversation.clinical_state.cases[0].missing_information)


def test_ambiguous_case_keeps_facts_pending_and_asks_llm_clarification():
    def ambiguous(payload):
        data = json.loads(extraction(payload))
        data["case_result"] = {"resolution": "UNCERTAIN"}
        return json.dumps(data)
    result = run_turn(graph(extract_json=ambiguous), session(), message())
    assert result.output.next_action == "ASK_MORE"
    assert result.session.pending
    assert not result.session.conversation.clinical_state.cases


def test_doctor_mode_suppresses_routine_reply_but_screening_still_runs():
    initial = run_turn(graph(), session(), message()).session
    initial.conversation.clinical_state.cases[0].response_mode = "DOCTOR_ACTIVE"
    calls = []
    def assess(payload):
        calls.append(1)
        return screening(payload)
    result = run_turn(graph(risk_json=assess), initial, message("started yesterday", "u2"))
    assert result.output.status == "NO_REPLY"
    assert calls == [1]
    assert result.session.conversation.messages[-1].role == "USER"


def test_rag_request_is_not_silently_ignored():
    def rag(payload):
        data = json.loads(extraction(payload)) | {"need_medical_rag": True, "rag_reason": "Need evidence"}
        return json.dumps(data)
    assert run_turn(graph(extract_json=rag), session(), message()).output.next_action == "HANDOFF_DOCTOR"


def test_existing_danger_is_not_downgraded_by_later_safe_assessment():
    first = run_turn(graph(risk_json=lambda p: screening(p, result="DANGEROUS", emergency=True)), session(), message())
    second = run_turn(graph(), first.session, message("started yesterday", "u2"))
    assert second.output.next_action == "EMERGENCY"
    assert second.output.risk.result == "DANGEROUS"


def test_local_api_idempotency_conflict_and_state_read():
    calls = []
    def extract(payload):
        calls.append(1)
        return extraction(payload)
    client = TestClient(create_app(graph(extract_json=extract)))
    assert client.get("/health").json()["storage"] == "memory"
    created = client.post("/conversations", json={"patient_id": "p"})
    assert created.status_code == 201
    id = created.json()["conversation"]["conversation_id"]
    url = f"/conversations/{id}/messages"
    body = {"message_id": "u1", "content": "headache for two days"}
    first = client.post(url, json=body)
    assert first.status_code == 200
    assert first.json()["output"]["next_action"] == "ASK_MORE"
    assert client.post(url, json=body).json() == first.json()
    assert calls == [1]
    assert client.post(url, json=body | {"content": "different"}).status_code == 409
    assert len(client.get(f"/conversations/{id}").json()["conversation"]["messages"]) == 2
    assert client.post(url, json=body | {"role": "DOCTOR"}).status_code == 422
    assert client.post("/conversations/unknown/messages", json=body).status_code == 404


def test_api_unexpected_failure_is_sanitized_and_not_committed():
    class BrokenGraph:
        def invoke(self, state):
            state["session"].conversation.patient_id = "mutated"
            raise ValueError("secret")
    client = TestClient(create_app(BrokenGraph()))
    id = client.post("/conversations", json={"patient_id": "p"}).json()["conversation"]["conversation_id"]
    response = client.post(f"/conversations/{id}/messages", json={"message_id": "u1", "content": "text"})
    assert response.status_code == 503 and "secret" not in response.text
    assert client.get(f"/conversations/{id}").json()["conversation"]["patient_id"] == "p"


def test_pending_doctor_handoff_does_not_restart_normal_questions():
    initial = run_turn(graph(risk_json=lambda p: screening(p, result="DANGEROUS")), session(), message("headache since yesterday for two days unchanged")).session
    assert initial.conversation.clinical_state.cases[0].workflow.status == "WAITING_DOCTOR"
    result = run_turn(graph(), initial, message("still headache", "u2"))
    assert result.output.status == "NO_REPLY"


def test_explicit_doctor_request_routes_to_handoff():
    def assess(payload):
        m = payload["new_message"]
        return screening(payload, doctor_request={"message_id": m["message_id"], "source_text": "doctor"})
    result = run_turn(graph(risk_json=assess), session(), message("headache and want a doctor"))
    assert result.output.next_action == "HANDOFF_DOCTOR"


def test_confirmed_pending_replays_original_source_and_is_removed():
    def extract(payload):
        if payload["new_message"]["message_id"] == "u1":
            raw = json.loads(extraction(payload))
            raw["case_result"] = {"resolution": "UNCERTAIN"}
            return json.dumps(raw)
        pending = payload["pending_facts"][0]
        return json.dumps({"case_result": {"resolution": "NEW_CASE", "new_problem": "headache"},
            "pending_bindings": [{"source_message_id": "u1", "field": pending["field"], "key": pending["key"],
                "confirmation_source": {"message_id": "u2", "source_text": "yes"}}]})
    workflow = graph(extract_json=extract)
    first = run_turn(workflow, session(), message())
    assert len(first.session.pending) == 1 and not first.session.conversation.clinical_state.cases
    second = run_turn(workflow, first.session, message("yes the headache", "u2"))
    assert not second.session.pending
    assert second.session.conversation.clinical_state.cases[0].present_illness.duration == "two days"
    duration = next(f for f in second.session.facts if f.field.value == "present_illness.duration")
    assert duration.source.message_id == "u1"
    assert duration.observed_at == message().timestamp
    assert first.session.pending  # Original input is unchanged.


@pytest.mark.parametrize("bad", ["fake_fact", "fake_source", "ambiguous", "duplicate"])
def test_pending_cannot_be_bound_without_valid_reference_and_confirmation(bad):
    from medical_chatbot.agents.intake.input_processor import process_input, InputProcessingError
    from medical_chatbot.schemas.outputs import FactProposal
    original = message()
    conversation = Conversation(conversation_id="c", patient_id="p", messages=[original])
    pending = FactProposal(field="present_illness.duration", key="duration", value="two days", status="PRESENT",
        source={"message_id": "u1", "source_text": "two days"})
    binding = {"source_message_id": "u1", "field": "present_illness.duration", "key": "duration",
               "confirmation_source": {"message_id": "u2", "source_text": "yes"}}
    resolution = {"resolution": "NEW_CASE", "new_problem": "headache"}
    if bad == "fake_fact":
        binding["key"] = "invented"
    elif bad == "fake_source":
        binding["confirmation_source"]["source_text"] = "not in message"
    elif bad == "ambiguous":
        resolution = {"resolution": "UNCERTAIN"}
    bindings = [binding, binding] if bad == "duplicate" else [binding]
    with pytest.raises(InputProcessingError):
        process_input(conversation, message("yes headache", "u2"), pending=[pending],
            generate_json=lambda _: json.dumps({"case_result": resolution, "pending_bindings": bindings}))


def test_llm_screening_does_not_write_official_clinical_safe():
    result = run_turn(graph(), session(), message())
    assert result.output.risk.assessment_source == "LLM_SCREENING"
    assert result.session.conversation.clinical_state.cases[0].risk_assessment.level == "NOT_ASSESSED"


def test_intake_completion_does_not_require_safe_assessment():
    from medical_chatbot.schemas.outputs import Agent1Output
    output = Agent1Output(status="COMPLETED", next_action="PASS_TO_AGENT_2", chief_complaint_known=True,
        case_result={"resolution": "OLD_CASE", "target_case_id": "c1"})
    assert output.risk.result is None
    assert output.risk.assessment_status == "NOT_ASSESSED"


def test_dizziness_facets_do_not_overwrite_or_conflict_across_turns():
    turns = [
        ("Tôi 30 tuổi, thỉnh thoảng chóng mặt.", [("frequency", "thỉnh thoảng")]),
        ("Bắt đầu khoảng một tuần trước.", [("onset", "khoảng một tuần trước")]),
        ("Mỗi lần kéo dài vài phút.", [("episode_duration", "vài phút")]),
        ("Mức độ không thay đổi.", [("course", "không thay đổi")]),
    ]
    def extract(payload):
        user = payload["new_message"]
        index = int(user["message_id"][1:]) - 1
        cases = payload["clinical_state"]["cases"]
        resolution = (dict(resolution="OLD_CASE", target_case_id=cases[0]["case_id"]) if cases
                      else dict(resolution="NEW_CASE", new_problem="chóng mặt"))
        return json.dumps(dict(case_result=resolution, case_facts=[dict(
            field=f"present_illness.{field}", key="chóng mặt", value=value, status="PRESENT",
            source=dict(message_id=user["message_id"], source_text=user["content"]))
            for field, value in turns[index][1]]))
    current = session()
    flow = graph(extract_json=extract)
    for index, (text, _) in enumerate(turns, 1):
        result = run_turn(flow, current, message(text, f"u{index}"))
        assert result.output.next_action == "ASK_MORE"
        assert not result.processing_errors and not result.session.conflicts
        current = result.session
    illness = current.conversation.clinical_state.cases[0].present_illness
    assert illness.onset == "khoảng một tuần trước"
    assert illness.frequency == "thỉnh thoảng"
    assert illness.episode_duration == "vài phút"
    assert illness.course == "không thay đổi"
    assert illness.duration is None  # No invented total duration from relative onset.
    assert [r.key for r in current.conversation.messages[-1].asked_requirements] == ["duration"]
    assert all(f.key == f.field.value.rsplit(".", 1)[-1] for f in current.facts)


def test_invalid_risk_quote_reports_rule_and_notice_without_fake_ai_question():
    result = run_turn(graph(risk_json=lambda p: screening(p,
        evidence=[dict(message_id="u1", source_text="not an exact patient quote")])), session(), message())
    assert result.output.status == "FAILED"
    assert result.output.error.message == "Response rule failed: risk_evidence_quote_not_exact"
    assert result.user_notice and result.rag_skip_reason == "risk_failed"
    assert len(result.session.conversation.messages) == 1
    assert all(r.asked_count == 0 for c in result.session.conversation.clinical_state.cases for r in c.missing_information)


def test_provider_screening_uses_original_message_text_not_model_quotes(monkeypatch):
    from medical_chatbot.agents.intake import risk_assessor
    user = message("synthetic danger statement")
    def provider(payload, *, system_instruction, response_schema):
        assert "evidence_message_ids" in response_schema["properties"]
        assert "evidence" not in response_schema["properties"]
        return json.dumps(dict(result="DANGEROUS", reason="Synthetic signal",
            evidence_message_ids=[user.message_id], emergency=True, doctor_requested=True))
    monkeypatch.setattr(risk_assessor, "gemini_generate", provider)
    decision, risk = risk_assessor.assess_message(session().conversation, user)
    assert risk.result == "DANGEROUS" and risk.emergency
    assert risk.signals[0].source.source_text == user.content
    assert decision.doctor_request.source_text == user.content


@pytest.mark.parametrize("reference", ["unknown", "ai-message"])
def test_provider_screening_rejects_unknown_and_nonuser_references(monkeypatch, reference):
    from medical_chatbot.agents.intake import risk_assessor
    conversation = session().conversation
    conversation.messages.append(Message(message_id="ai-message", role="AI", content="Synthetic AI text",
        timestamp="2026-11-06T09:00:00+07:00"))
    monkeypatch.setattr(risk_assessor, "gemini_generate", lambda *a, **k: json.dumps(dict(
        result="SAFE", reason="Synthetic screen", evidence_message_ids=[reference])))
    decision, risk = risk_assessor.assess_message(conversation, message())
    assert decision is None and risk.assessment_status == "FAILED"
    assert "risk_evidence_message_not_user_or_unknown" in risk.error.message


def test_older_state_loads_with_new_optional_fields():
    from medical_chatbot.schemas.clinical_state import ClinicalState
    old = dict(current_case_id="c1", cases=[dict(case_id="c1", present_illness=dict(onset="yesterday"))])
    state = ClinicalState.model_validate(old)
    assert state.cases[0].present_illness.frequency is None
    assert state.cases[0].present_illness.episode_duration is None


def test_extraction_schema_error_has_safe_location_and_is_not_duplicated():
    def extract(payload):
        data = json.loads(extraction(payload))
        data["patient_facts"] = [dict(field="patient_info.age", key="age", value="PRIVATE_INVALID_AGE",
            status="PRESENT", source=dict(message_id="u1", source_text="headache"))]
        return json.dumps(data)
    result = run_turn(graph(extract_json=extract), session(), message())
    assert result.output.status == "FAILED"
    assert len(result.processing_errors) == 1
    error = result.processing_errors[0]
    assert error.code == "VALIDATION" and "facts" in error.message
    assert "PRIVATE_INVALID_AGE" not in result.model_dump_json()
    assert result.rag_skip_reason == "extraction_failed"
    assert result.session.conversation.clinical_state.cases == []


def test_extraction_provider_error_retains_provider_category():
    from medical_chatbot.agents.intake.question_generator import QuestionGenerationError
    def fail(_):
        raise QuestionGenerationError("Gemini request failed (HTTP 429)")
    result = run_turn(graph(extract_json=fail), session(), message())
    assert result.output.error.code == "PROVIDER"
    assert "429" in result.output.error.message
    assert len(result.processing_errors) == 1
    assert result.rag_skip_reason == "extraction_failed"


def test_rejected_fact_rolls_back_other_accepted_facts_in_same_turn():
    def extract(payload):
        data = json.loads(extraction(payload))
        data["case_facts"].append(dict(field="clinical_data.vitals", key="workflow",
            value="two days", status="PRESENT", source=dict(message_id="u1", source_text="two days")))
        return json.dumps(data)
    result = run_turn(graph(extract_json=extract), session(), message())
    assert result.output.status == "FAILED"
    assert result.session.facts == []
    assert result.session.conversation.clinical_state.cases == []
    assert len(result.session.conversation.messages) == 1


def test_complete_intake_with_unknown_risk_does_not_call_rag_or_infer_safe():
    class UnexpectedRag:
        def assess(self, *args):
            raise AssertionError("Basic intake completion should not call RAG")
    result = run_turn(graph(risk_json=lambda p: screening(p, result="UNCERTAIN"),
                            rag_service=UnexpectedRag()), session(),
                      message("headache yesterday two days unchanged"))
    assert result.output.next_action == "PASS_TO_AGENT_2"
    assert result.output.risk.result is None
    assert result.session.conversation.clinical_state.cases[0].risk_assessment.level == "NOT_ASSESSED"


@pytest.mark.parametrize("rag_status,expected", [
    ("FAILED", "HANDOFF_DOCTOR"), ("UNAVAILABLE", "HANDOFF_DOCTOR"),
    ("INSUFFICIENT", "HANDOFF_DOCTOR")])
def test_specialist_rag_failure_or_missing_knowledge_hands_off(rag_status, expected):
    from medical_chatbot.rag import RagResult
    from medical_chatbot.schemas.risk import ProcessingError
    class StubRag:
        def assess(self, *args):
            return RagResult(status=rag_status, error=ProcessingError(stage="rag", code="TIMEOUT",
                message="Synthetic timeout", retryable=True) if rag_status == "FAILED" else None)
    def extract(payload):
        return json.dumps(json.loads(extraction(payload)) | dict(
            need_medical_rag=True, rag_reason="Specialist follow-up needed"))
    result = run_turn(graph(extract_json=extract, rag_service=StubRag()), session(), message())
    assert result.output.next_action == expected
    case = result.session.conversation.clinical_state.cases[0]
    assert case.workflow.status == "WAITING_DOCTOR"


def test_unknown_answer_remains_visible_when_other_intake_items_complete():
    def extract(payload):
        data = json.loads(extraction(payload))
        if payload["new_message"]["message_id"] == "u2":
            case_id = payload["clinical_state"]["current_case_id"]
            data["unavailable_answers"] = [dict(key="onset", target_case_id=case_id,
                status="UNKNOWN", source=dict(message_id="u2", source_text="do not know"))]
        return json.dumps(data)
    flow = graph(extract_json=extract)
    first = run_turn(flow, session(), message("headache two days unchanged"))
    second = run_turn(flow, first.session, message("I do not know", "u2"))
    assert second.output.next_action == "PASS_TO_AGENT_2"
    assert second.output.requirements[0].status == "UNKNOWN"
    assert second.session.conversation.clinical_state.cases[0].present_illness.onset is None


def retry_client(question_provider=None, risk_provider=screening, extract_provider=extraction):
    workflow = graph(question_json=question_provider or question, risk_json=risk_provider,
                     extract_json=extract_provider)
    client = TestClient(create_app(workflow))
    id = client.post("/conversations", json={"patient_id": "test"}).json()["conversation"]["conversation_id"]
    return client, f"/conversations/{id}/messages"


def test_question_retry_does_not_repeat_extraction_or_user_message():
    calls = {"extract": 0, "risk": 0, "question": 0}
    def extract(payload):
        calls["extract"] += 1
        return extraction(payload)
    def assess(payload):
        calls["risk"] += 1
        return screening(payload)
    def ask(payload):
        calls["question"] += 1
        if calls["question"] == 1:
            raise TimeoutError("test")
        return question(payload)
    client, url = retry_client(ask, assess, extract)
    first = client.post(url, json={"message_id": "u1", "content": "headache for two days"})
    assert first.json()["output"]["status"] == "FAILED"
    retry_url = url + "/u1/retry"
    second = client.post(retry_url, json={"retry_id": "r1"})
    assert second.status_code == 200
    assert second.json()["output"]["next_action"] == "ASK_MORE"
    assert calls == {"extract": 1, "risk": 1, "question": 2}
    history = second.json()["session"]["conversation"]["messages"]
    assert [m["role"] for m in history] == ["USER", "AI"]
    assert client.post(retry_url, json={"retry_id": "r1"}).json() == second.json()
    assert calls["question"] == 2
    assert client.post(retry_url, json={"retry_id": "r2"}).status_code == 409


def test_retry_limit_and_stale_retry_are_rejected():
    client, url = retry_client(lambda _: "invalid")
    client.post(url, json={"message_id": "u1", "content": "headache for two days"})
    for id in ["r1", "r2"]:
        assert client.post(url + "/u1/retry", json={"retry_id": id}).status_code == 200
    assert client.post(url + "/u1/retry", json={"retry_id": "r3"}).status_code == 409
    client.post(url, json={"message_id": "u2", "content": "started yesterday"})
    assert client.post(url + "/u1/retry", json={"retry_id": "r4"}).status_code == 409


def test_screening_failure_can_retry_without_duplicate_history():
    calls = []
    def assess(payload):
        calls.append(1)
        return "bad JSON" if len(calls) == 1 else screening(payload)
    client, url = retry_client(risk_provider=assess)
    first = client.post(url, json={"message_id": "u1", "content": "headache for two days"}).json()
    assert first["processing_errors"][0]["stage"] == "risk"
    result = client.post(url + "/u1/retry", json={"retry_id": "r1"}).json()
    assert result["output"]["next_action"] == "ASK_MORE"
    assert len(result["session"]["conversation"]["messages"]) == 2


def test_danger_signal_cannot_be_erased_by_retry():
    client, url = retry_client(risk_provider=lambda p: screening(p, result="DANGEROUS", emergency=True),
                               extract_provider=lambda _: "invalid")
    first = client.post(url, json={"message_id": "u1", "content": "headache for two days"})
    assert first.json()["output"]["next_action"] == "EMERGENCY"
    assert client.post(url + "/u1/retry", json={"retry_id": "r1"}).status_code == 409


def test_old_pending_temporal_fact_cannot_overwrite_newer_answer():
    def extract(payload):
        if payload["new_message"]["message_id"] == "u1":
            data = json.loads(extraction(payload))
            data["case_result"] = {"resolution": "UNCERTAIN"}
            data["case_facts"][0]["change_type"] = "TEMPORAL_CHANGE"
            data["case_facts"][0]["source"]["source_text"] = "now two days"
            return json.dumps(data)
        return json.dumps({"case_result": {"resolution": "NEW_CASE", "new_problem": "headache"},
            "case_facts": [{"field": "present_illness.duration", "key": "duration", "value": "three days",
                "status": "PRESENT", "source": {"message_id": "u2", "source_text": "three days"}}],
            "pending_bindings": [{"source_message_id": "u1", "field": "present_illness.duration", "key": "duration",
                "confirmation_source": {"message_id": "u2", "source_text": "yes"}}]})
    workflow = graph(extract_json=extract)
    first = run_turn(workflow, session(), message("headache now two days"))
    second = run_turn(workflow, first.session, message("yes headache three days", "u2"))
    assert second.session.conversation.clinical_state.cases[0].present_illness.duration == "three days"
    assert second.session.conflicts


def test_doctor_review_allows_follow_up_without_stale_provisional_danger():
    from datetime import timedelta
    from medical_chatbot.clinical.case_manager import DoctorContext, prepare_doctor_handoff, confirm_doctor_handoff
    from medical_chatbot.schemas.clinical_state import DoctorHandoffContent
    first = run_turn(graph(risk_json=lambda p: screening(p, result="DANGEROUS")), session(), message()).session
    case = first.conversation.clinical_state.cases[0]
    case.response_mode = "DOCTOR_ACTIVE"
    at = first.conversation.messages[-1].timestamp + timedelta(seconds=1)
    first.conversation.messages.append(Message(message_id="doctor1", role="DOCTOR",
        content="Reviewed; follow up on progression", timestamp=at, case_ids=[case.case_id]))
    doctor = DoctorContext(doctor_id="doctor1")
    draft = prepare_doctor_handoff(first.conversation, case.case_id, doctor)
    first.conversation = confirm_doctor_handoff(first.conversation, draft,
        DoctorHandoffContent(summary="Reviewed", follow_up="Ask onset/course"), doctor,
        action="RETURN_TO_AI", confirmed_at=at + timedelta(seconds=1))
    result = run_turn(graph(), first, Message(message_id="u2", role="USER", content="started yesterday",
                                             timestamp=at + timedelta(seconds=2)))
    assert result.output.next_action == "ASK_MORE"
    assert not result.session.risk_by_case


def test_new_raw_emergency_is_not_hidden_by_old_nonemergency_when_extraction_fails():
    initial = run_turn(graph(risk_json=lambda p: screening(p, result="DANGEROUS")), session(), message()).session
    result = run_turn(graph(risk_json=lambda p: screening(p, result="DANGEROUS", emergency=True),
                            extract_json=lambda _: "invalid"), initial, message("new emergency", "u2"))
    assert result.output.next_action == "EMERGENCY"


def test_unmapped_danger_is_not_forgotten_on_the_next_safe_screen():
    initial = run_turn(graph(risk_json=lambda p: screening(p, result="DANGEROUS", emergency=True),
                             extract_json=lambda _: "invalid"), session(), message()).session
    result = run_turn(graph(), initial, message("headache for two days", "u2"))
    assert result.output.next_action == "EMERGENCY"


def test_mapped_user_history_is_included_in_doctor_review():
    from datetime import timedelta
    from medical_chatbot.clinical.case_manager import DoctorContext, prepare_doctor_handoff
    result = run_turn(graph(), session(), message()).session
    case = result.conversation.clinical_state.cases[0]
    case.response_mode = "DOCTOR_ACTIVE"
    result.conversation.messages.append(Message(message_id="doctor1", role="DOCTOR", content="Reviewed",
        timestamp=result.conversation.messages[-1].timestamp + timedelta(seconds=1), case_ids=[case.case_id]))
    draft = prepare_doctor_handoff(result.conversation, case.case_id, DoctorContext(doctor_id="d1"))
    assert any(m.message_id == "u1" for m in draft.messages)


def test_switching_cases_does_not_bypass_unreviewed_patient_danger():
    initial = run_turn(graph(risk_json=lambda p: screening(p, result="DANGEROUS")), session(), message()).session
    def different_case(_):
        return json.dumps({"case_result": {"resolution": "NEW_CASE", "new_problem": "stomach"}})
    result = run_turn(graph(extract_json=different_case), initial, message("stomach problem", "u2"))
    assert len(result.session.conversation.clinical_state.cases) == 2
    assert result.output.next_action == "HANDOFF_DOCTOR"
    assert result.output.risk.result == "DANGEROUS"
    assert result.session.conversation.clinical_state.cases[-1].risk_assessment.level == "NOT_ASSESSED"


def test_sqlite_restart_preserves_session_history_and_cached_response(tmp_path):
    path = str(tmp_path / "agent1.sqlite3")
    with TestClient(create_app(graph(), database_path=path)) as client:
        id = client.post("/conversations", json={"patient_id": "p"}).json()["conversation"]["conversation_id"]
        url = f"/conversations/{id}/messages"
        body = {"message_id": "u1", "content": "headache for two days"}
        response = client.post(url, json=body)
        assert response.status_code == 200
        saved = client.get(f"/conversations/{id}").json()
        assert client.get("/health").json()["storage"] == "sqlite"
    def fail(_):
        pytest.fail("A cached request must not invoke the provider")
    with TestClient(create_app(graph(extract_json=fail, risk_json=fail, question_json=fail), database_path=path)) as client:
        assert client.get(f"/conversations/{id}").json() == saved
        assert client.post(url, json=body).json() == response.json()
        assert client.post(url, json=body | {"content": "different"}).status_code == 409


def test_sqlite_question_retry_survives_restart_without_duplicate_history(tmp_path):
    path = str(tmp_path / "agent1.sqlite3")
    with TestClient(create_app(graph(question_json=lambda _: "invalid"), database_path=path)) as client:
        id = client.post("/conversations", json={"patient_id": "p"}).json()["conversation"]["conversation_id"]
        url = f"/conversations/{id}/messages"
        first = client.post(url, json={"message_id": "u1", "content": "headache for two days"})
        assert first.json()["output"]["status"] == "FAILED"
    with TestClient(create_app(graph(), database_path=path)) as client:
        response = client.post(url + "/u1/retry", json={"retry_id": "r1"})
        assert response.status_code == 200
        assert response.json()["output"]["next_action"] == "ASK_MORE"
        assert [m["role"] for m in response.json()["session"]["conversation"]["messages"]] == ["USER", "AI"]
    with TestClient(create_app(graph(), database_path=path)) as client:
        assert client.post(url + "/u1/retry", json={"retry_id": "r1"}).json() == response.json()


def test_sqlite_retry_attempt_limit_survives_restart(tmp_path):
    path = str(tmp_path / "agent1.sqlite3")
    with TestClient(create_app(graph(question_json=lambda _: "invalid"), database_path=path)) as client:
        id = client.post("/conversations", json={"patient_id": "p"}).json()["conversation"]["conversation_id"]
        url = f"/conversations/{id}/messages"
        client.post(url, json={"message_id": "u1", "content": "headache for two days"})
        assert client.post(url + "/u1/retry", json={"retry_id": "r1"}).status_code == 200
    with TestClient(create_app(graph(question_json=lambda _: "invalid"), database_path=path)) as client:
        assert client.post(url + "/u1/retry", json={"retry_id": "r2"}).status_code == 200
    with TestClient(create_app(graph(), database_path=path)) as client:
        assert client.post(url + "/u1/retry", json={"retry_id": "r3"}).status_code == 409


def test_sqlite_transaction_rolls_back_all_namespace_updates(tmp_path):
    from medical_chatbot.storage import LocalStore
    store = LocalStore(str(tmp_path / "agent1.sqlite3"))
    try:
        with pytest.raises(ValueError):
            with store.transaction():
                store.put("sessions", "c1", {"state": "new"})
                store.put("replies", ("c1", "m1"), {"result": "new"})
                raise ValueError("simulate failure before commit")
        with store.transaction():
            assert store.get("sessions", "c1") is None
            assert store.get("replies", ("c1", "m1")) is None
    finally:
        store.close()


def test_memory_storage_remains_isolated_across_app_restarts():
    with TestClient(create_app(graph(), database_path=":memory:")) as client:
        id = client.post("/conversations", json={"patient_id": "p"}).json()["conversation"]["conversation_id"]
    with TestClient(create_app(graph(), database_path=":memory:")) as client:
        assert client.get(f"/conversations/{id}").status_code == 404
        assert client.get("/health").json()["storage"] == "memory"


def test_database_config_reads_environment(tmp_path, monkeypatch):
    path = tmp_path / "configured.sqlite3"
    monkeypatch.setenv("MEDCARE_DB_PATH", str(path))
    with TestClient(create_app(graph())) as client:
        assert client.get("/health").json()["storage"] == "sqlite"
    assert path.exists()


def test_configured_mongodb_is_selected_without_silent_sqlite_fallback(monkeypatch):
    from medical_chatbot.storage import create_store
    monkeypatch.setenv("MONGODB_URI", "mongodb://example.invalid")
    monkeypatch.setenv("MONGODB_DATABASE", "synthetic")
    store = create_store()
    assert store.backend == "mongodb" and store._client is None  # No network on construction.
    store.close()


def test_explicit_sqlite_path_overrides_mongodb_for_offline_tests(monkeypatch):
    from medical_chatbot.storage import create_store
    monkeypatch.setenv("MONGODB_URI", "mongodb://example.invalid")
    store = create_store(":memory:")
    try:
        assert store.backend == "memory"
    finally:
        store.close()


def test_two_api_instances_share_idempotent_message_commit(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from time import sleep
    calls = []
    def extract(payload):
        calls.append(1)
        sleep(0.03)
        return extraction(payload)
    path = str(tmp_path / "shared.sqlite3")
    with TestClient(create_app(graph(extract_json=extract), database_path=path)) as first:
        with TestClient(create_app(graph(extract_json=extract), database_path=path)) as second:
            id = first.post("/conversations", json={"patient_id": "p"}).json()["conversation"]["conversation_id"]
            url = f"/conversations/{id}/messages"
            body = {"message_id": "u1", "content": "headache for two days"}
            with ThreadPoolExecutor(max_workers=2) as executor:
                responses = list(executor.map(lambda client: client.post(url, json=body), [first, second]))
            assert all(response.status_code == 200 for response in responses)
            assert responses[0].json() == responses[1].json()
            assert len(calls) == 1


def test_invalid_saved_session_returns_sanitized_storage_error(tmp_path):
    import sqlite3
    path = str(tmp_path / "agent1.sqlite3")
    with TestClient(create_app(graph(), database_path=path)) as client:
        id = client.post("/conversations", json={"patient_id": "p"}).json()["conversation"]["conversation_id"]
        with sqlite3.connect(path) as connection:
            connection.execute("UPDATE agent_data SET value=? WHERE namespace=? AND key=?",
                (json.dumps({"secret": "patient content"}), "sessions", json.dumps(id)))
        response = client.get(f"/conversations/{id}")
        assert response.status_code == 503
        assert "secret" not in response.text and "patient content" not in response.text
