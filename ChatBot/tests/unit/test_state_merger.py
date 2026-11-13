import pytest
from pydantic import ValidationError
from medical_chatbot.clinical.state_merger import update_clinical_state
from medical_chatbot.schemas.clinical_state import ClinicalCase, ClinicalState
from medical_chatbot.schemas.conversation import Message
from medical_chatbot.schemas.outputs import InputProcessorOutput
from datetime import datetime
from medical_chatbot.clinical.case_manager import (
    DoctorContext, prepare_doctor_handoff, confirm_doctor_handoff,
)
from medical_chatbot.schemas.clinical_state import DoctorHandoffContent
from medical_chatbot.schemas.conversation import Conversation
from medical_chatbot.requirements.engine import evaluate_requirements, record_unavailable_answer, mark_question_sent
from medical_chatbot.schemas.facts import Fact
from medical_chatbot.schemas.requirements import Requirement
from medical_chatbot.schemas.risk import RiskResult


def run(state=None, text="headache for two days", field="present_illness.duration",
        value="two days", status="PRESENT", change="NONE", resolution="OLD_CASE",
        facts=(), conflicts=(), source=None, key="duration"):
    state = state if state is not None else ClinicalState(cases=[ClinicalCase(case_id="c1")])
    proposal = dict(field=field, key=key, value=value, status=status,
                    change_type=change, source=source or dict(message_id="m1", source_text=text))
    case_result = dict(resolution=resolution)
    if resolution == "OLD_CASE":
        case_result["target_case_id"] = "c1"
    if resolution == "NEW_CASE":
        case_result["new_problem"] = "headache"
    output = InputProcessorOutput.model_validate(dict(case_result=case_result,
        **{"patient_facts" if field.startswith(("patient_info.", "medical_history.")) else "case_facts": [proposal]}))
    message = Message(message_id="m1", role="USER", content=text, timestamp="2026-10-30T10:00:00+07:00")
    return update_clinical_state(state, message, output, facts=facts, conflicts=conflicts)


def test_update_is_pure_and_creates_new_case_with_source():
    before = ClinicalState()
    result = run(before, resolution="NEW_CASE")
    assert before.cases == []
    case = result.state.cases[0]
    assert case.chief_complaint == "headache"
    assert case.present_illness.duration == "two days"
    assert result.state.current_case_id == case.case_id
    assert all(f.target_case_id == case.case_id for f in result.facts)


def test_unknown_case_does_not_receive_facts():
    result = run(resolution="UNCERTAIN")
    assert len(result.pending) == 1
    assert result.state.cases[0].present_illness.duration is None
    assert result.facts == []


def test_foreign_case_rejected_without_mutation():
    state = ClinicalState()
    with pytest.raises(ValueError, match="does not belong"):
        run(state)
    assert state.cases == []


def test_wrong_source_is_rejected_without_creating_case():
    result = run(ClinicalState(), resolution="NEW_CASE", source=dict(message_id="other", source_text="headache"))
    assert len(result.rejected) == 1
    assert result.state.cases == []


def test_patient_update_does_not_require_case():
    result = run(ClinicalState(), field="patient_info.age", value=30, resolution="UNCERTAIN")
    assert result.state.patient_info.age == 30
    assert result.facts[0].target_case_id is None


def test_list_dedup_and_explicit_denial():
    first = run(field="present_illness.associated_symptoms", key="fever", value="fever")
    second = run(first.state, field="present_illness.associated_symptoms", key="fever", value="fever", facts=first.facts)
    assert second.state.cases[0].present_illness.associated_symptoms == ["fever"]
    denied = run(second.state, field="present_illness.associated_symptoms", key="fever", value=None,
                 status="ABSENT", change="TEMPORAL_CHANGE", text="now no fever", facts=second.facts)
    assert denied.state.cases[0].present_illness.associated_symptoms == []
    assert denied.facts[-1].status == "ABSENT"
    assert denied.state.cases[0].progress_history[-1].old_value == "fever"


def test_uncertainty_never_becomes_confirmed_value():
    result = run(status="UNCERTAIN")
    assert result.state.cases[0].present_illness.duration is None
    assert len(result.pending) == 1


def test_ambiguous_change_keeps_value_and_can_be_clarified():
    first = run()
    second = run(first.state, value="three days", facts=first.facts)
    assert second.state.cases[0].present_illness.duration == "two days"
    assert len(second.conflicts) == 1
    resolved = run(second.state, value="three days", text="correction: three days",
                   change="CORRECTION", facts=second.facts, conflicts=second.conflicts)
    assert resolved.state.cases[0].present_illness.duration == "three days"
    assert resolved.conflicts == []


def test_change_flag_alone_is_not_evidence():
    first = run()
    second = run(first.state, value="three days", change="CORRECTION", facts=first.facts)
    assert len(second.conflicts) == 1


def test_scalar_conflict_cannot_be_bypassed_with_different_key():
    first = run()
    second = run(first.state, value="three days", key="another_key", facts=first.facts)
    assert len(second.conflicts) == 1


def test_vitals_merge_by_key():
    first = run(field="clinical_data.vitals", key="temperature", value=38.5)
    second = run(first.state, field="clinical_data.vitals", key="pulse", value=80, facts=first.facts)
    assert second.state.cases[0].clinical_data.vitals == {"temperature": 38.5, "pulse": 80}


def test_untracked_list_denial_requires_clarification():
    state = ClinicalState(cases=[ClinicalCase(case_id="c1", present_illness={"associated_symptoms": ["fever"]})])
    result = run(state, field="present_illness.associated_symptoms", key="fever", value=None, status="ABSENT")
    assert result.state.cases[0].present_illness.associated_symptoms == ["fever"]
    assert len(result.conflicts) == 1


@pytest.mark.parametrize("value", [{"workflow": "PASS"}, ["nested"]])
def test_open_data_cannot_carry_nested_patches(value):
    result = run(field="clinical_data.other", key="misc", value=value)
    assert len(result.rejected) == 1
    assert result.state.cases[0].clinical_data.other == []


def test_extractor_cannot_write_workflow():
    with pytest.raises(ValidationError):
        run(field="workflow.next_action", value="PASS_TO_AGENT_2")


def test_reopens_case_preserves_doctor_and_invalidates_old_safe():
    state = ClinicalState(cases=[ClinicalCase(case_id="c1", status="RESOLVED", response_mode="DOCTOR_ACTIVE",
        risk_assessment={"level": "SAFE"})])
    result = run(state)
    case = result.state.cases[0]
    assert case.status == "ACTIVE"
    assert case.response_mode == "DOCTOR_ACTIVE"
    assert case.risk_assessment.level == "NOT_ASSESSED"
    assert state.cases[0].status == "RESOLVED"


def test_dangerous_is_never_downgraded():
    state = ClinicalState(cases=[ClinicalCase(case_id="c1", risk_assessment={"level": "DANGEROUS", "danger_signs": ["signal"]})])
    assert run(state).state.cases[0].risk_assessment.level == "DANGEROUS"


# Doctor review/handoff tests share this existing clinical test module.


def doctor_review():
    conversation = Conversation(
        conversation_id="conv1", patient_id="patient1",
        clinical_state=ClinicalState(cases=[ClinicalCase(
            case_id="c1", response_mode="DOCTOR_ACTIVE",
            workflow={"status": "WAITING_DOCTOR", "next_action": "HANDOFF_DOCTOR"})]),
        messages=[Message(message_id="doctor1", role="DOCTOR", content="Please monitor symptoms.",
                          timestamp="2026-11-02T10:00:00+07:00", case_ids=["c1"])],
    )
    doctor = DoctorContext(doctor_id="d1")
    draft = prepare_doctor_handoff(conversation, "c1", doctor)
    return conversation, doctor, draft


def confirm(conversation, doctor, draft, action="RETURN_TO_AI", **changes):
    return confirm_doctor_handoff(
        conversation, draft, DoctorHandoffContent(summary="Doctor reviewed summary", follow_up="Ask about changes"),
        doctor, action=action,
        confirmed_at=changes.get("confirmed_at", datetime.fromisoformat("2026-11-02T11:00:00+07:00")),
    )


def test_prepare_draft_does_not_change_official_state():
    conversation, doctor, draft = doctor_review()
    case = conversation.clinical_state.cases[0]
    assert case.doctor_handoff is None
    assert case.response_mode == "DOCTOR_ACTIVE"
    assert draft.content.summary == "Please monitor symptoms."
    draft.content.summary = "Edited draft"
    assert case.doctor_handoff is None
    assert conversation.messages[0].content == "Please monitor symptoms."


@pytest.mark.parametrize("action,status,workflow", [
    ("RETURN_TO_AI", "ACTIVE", "COLLECTING"), ("CLOSE_CASE", "RESOLVED", "COMPLETED"),
])
def test_confirm_saves_reviewed_content_and_changes_mode(action, status, workflow):
    original, doctor, draft = doctor_review()
    result = confirm(original, doctor, draft, action)
    case = result.clinical_state.cases[0]
    assert case.doctor_handoff.summary == "Doctor reviewed summary"
    assert case.doctor_handoff.follow_up == "Ask about changes"
    assert case.doctor_handoff.doctor_id == "d1"
    assert case.doctor_handoff.source_message_ids == ["doctor1"]
    assert case.response_mode == "AI_ACTIVE"
    assert case.status == status
    assert case.workflow.status == workflow
    assert case.workflow.next_action is None
    assert result.messages[-1].role == "SYSTEM"
    assert all(m.role != "AI" for m in result.messages)
    assert original.clinical_state.cases[0].doctor_handoff is None
    assert original.clinical_state.cases[0].response_mode == "DOCTOR_ACTIVE"


@pytest.mark.parametrize("change", ["new_message", "edited_message", "state"])
def test_stale_review_requires_refresh(change):
    conversation, doctor, draft = doctor_review()
    if change == "new_message":
        conversation.messages.append(Message(message_id="new", role="USER", content="New symptom",
            timestamp="2026-11-02T10:30:00+07:00", case_ids=["c1"]))
    elif change == "edited_message":
        conversation.messages[0].content = "Updated instructions"
    else:
        conversation.clinical_state.patient_info.age = 40
    with pytest.raises(ValueError, match="changed"):
        confirm(conversation, doctor, draft)
    assert conversation.clinical_state.cases[0].response_mode == "DOCTOR_ACTIVE"


def test_wrong_doctor_cannot_confirm_someone_elses_draft():
    conversation, doctor, draft = doctor_review()
    for actor in [DoctorContext(doctor_id="d2")]:
        with pytest.raises(PermissionError):
            confirm(conversation, actor, draft)


def test_handoff_requires_doctor_mode_and_doctor_messages():
    conversation, doctor, draft = doctor_review()
    conversation.messages = []
    with pytest.raises(ValueError, match="No doctor"):
        prepare_doctor_handoff(conversation, "c1", doctor)
    conversation.clinical_state.cases[0].response_mode = "AI_ACTIVE"
    with pytest.raises(ValueError, match="doctor control"):
        prepare_doctor_handoff(conversation, "c1", doctor)


@pytest.mark.parametrize("action,time", [
    ("INVALID", "2026-11-02T11:00:00+07:00"),
    ("RETURN_TO_AI", "2026-11-02T11:00:00"),
    ("RETURN_TO_AI", "2026-11-02T09:00:00+07:00"),
])
def test_invalid_confirmation_leaves_original_unchanged(action, time):
    conversation, doctor, draft = doctor_review()
    with pytest.raises(ValueError):
        confirm(conversation, doctor, draft, action, confirmed_at=datetime.fromisoformat(time))
    assert conversation.clinical_state.cases[0].doctor_handoff is None


def test_confirmation_cannot_be_replayed_and_preserves_danger():
    conversation, doctor, _ = doctor_review()
    conversation.clinical_state.cases[0].risk_assessment.level = "DANGEROUS"
    conversation.clinical_state.cases[0].risk_assessment.danger_signs = ["Existing signal"]
    draft = prepare_doctor_handoff(conversation, "c1", doctor)
    result = confirm(conversation, doctor, draft)
    assert result.clinical_state.cases[0].risk_assessment.level == "DANGEROUS"
    with pytest.raises(ValueError):
        confirm(result, doctor, draft)


def test_handoff_cannot_reference_user_messages():
    conversation, doctor, draft = doctor_review()
    result = confirm(conversation, doctor, draft)
    result.messages[0].role = "USER"
    with pytest.raises(ValidationError):
        Conversation.model_validate(result.model_dump())




def intake_state(complete=False):
    case = ClinicalCase(case_id="c1", chief_complaint="headache")
    if complete:
        case.present_illness.onset = "yesterday"
        case.present_illness.duration = "one day"
        case.present_illness.course = "unchanged"
    return ClinicalState(current_case_id="c1", cases=[case])


def extra(key="fever", priority="HIGH", blocking=True, **changes):
    return Requirement.model_validate(dict(key=key, target="present_illness.associated_symptoms",
        target_case_id="c1", source="COMPLAINT", reason="Clarify symptom", priority=priority,
        blocking=blocking) | changes)


def symptom_fact(key="fever", status="ABSENT", value=None, case_id="c1"):
    return Fact(fact_id=key, field="present_illness.associated_symptoms", key=key,
        status=status, value=value, target_case_id=case_id,
        source=dict(message_id="m", source_text="No fever"))


def test_engine_partial_answers_clarifies_missing_items_without_handoff():
    state = intake_state()
    first = evaluate_requirements(state)
    assert [r.key for r in first.selected] == ["onset", "duration"]
    assert state.cases[0].missing_information == []
    first = mark_question_sent(first, "Test question")
    first.state.cases[0].present_illness.onset = "yesterday"
    second = evaluate_requirements(first.state)
    assert [r.key for r in second.selected] == ["course"]
    second = mark_question_sent(second, "Test question")
    third = evaluate_requirements(second.state)
    assert third.status == "ASK_MORE"
    assert {r.key for r in third.selected} == {"duration", "course"}
    assert third.question is None
    assert {r.key for r in third.state.cases[0].missing_information} == {"duration", "course"}


def test_critical_tier_blocks_high_even_when_already_asked():
    first = evaluate_requirements(intake_state(), additional=[extra(priority="CRITICAL")])
    assert [r.key for r in first.selected] == ["fever"]
    first = mark_question_sent(first, "Test question")
    second = evaluate_requirements(first.state)
    assert second.status == "ASK_MORE"
    assert [r.key for r in second.selected] == ["fever"]


def test_same_target_requires_distinct_semantic_keys_and_denial_counts():
    result = evaluate_requirements(intake_state(True),
        additional=[extra(), extra(key="cough")], facts=[symptom_fact()])
    assert [r.key for r in result.selected] == ["cough"]


def test_nonempty_list_is_not_answer_to_every_symptom():
    state = intake_state(True)
    state.cases[0].present_illness.associated_symptoms = ["cough"]
    result = evaluate_requirements(state, additional=[extra()])
    assert [r.key for r in result.selected] == ["fever"]


def test_foreign_fact_does_not_satisfy_requirement():
    result = evaluate_requirements(intake_state(True), additional=[extra()],
        facts=[symptom_fact(case_id="other")])
    assert result.status == "ASK_MORE"


def test_uncertain_fact_does_not_satisfy_requirement():
    result = evaluate_requirements(intake_state(True), additional=[extra()],
        facts=[symptom_fact(status="UNCERTAIN")])
    assert result.status == "ASK_MORE"


@pytest.mark.parametrize("status", ["UNKNOWN", "DECLINED", "UNAVAILABLE"])
def test_unanswerable_item_is_skipped_while_other_information_is_collected(status):
    first = evaluate_requirements(intake_state())
    updated = record_unavailable_answer(first.state, "c1", "onset", status)
    result = evaluate_requirements(updated)
    assert result.status == "ASK_MORE"
    assert "onset" not in [r.key for r in result.selected]
    assert result.question is None
    assert first.state.cases[0].missing_information[0].status == "MISSING"


def test_optional_and_nonapplicable_requirements_do_not_block():
    result = evaluate_requirements(intake_state(True), additional=[
        extra(priority="OPTIONAL", blocking=False), extra(key="cough", applicable=False)])
    assert result.status == "READY_FOR_REVIEW"
    assert result.question is None


def test_safe_is_not_inferred_and_even_safe_is_not_pass():
    state = intake_state(True)
    state.cases[0].risk_assessment.level = "SAFE"
    assert evaluate_requirements(state).status == "READY_FOR_REVIEW"
    safe = RiskResult(assessment_status="ASSESSED", result="SAFE", reason="External assessment")
    assert evaluate_requirements(state, risk=safe).status == "READY_FOR_REVIEW"


def test_engine_does_not_convert_technical_risk_failure_into_referral():
    risk = RiskResult(assessment_status="FAILED", error=dict(stage="risk", code="TIMEOUT", message="Timeout"))
    assert evaluate_requirements(intake_state(), risk=risk).status == "ASK_MORE"


def test_danger_blocks_routine_questions():
    state = intake_state()
    state.cases[0].risk_assessment.level = "DANGEROUS"
    state.cases[0].risk_assessment.danger_signs = ["signal"]
    assert evaluate_requirements(state).status == "HANDOFF_DOCTOR"


def test_doctor_mode_suppresses_questions_without_changing_mode():
    state = intake_state()
    state.cases[0].response_mode = "DOCTOR_ACTIVE"
    result = evaluate_requirements(state)
    assert result.status == "NO_REPLY"
    assert result.question is None
    assert result.state.cases[0].response_mode == "DOCTOR_ACTIVE"


def test_case_clarification_precedes_requirements():
    assert evaluate_requirements(ClinicalState()).status == "CLARIFY_CASE"
    assert evaluate_requirements(intake_state(), case_uncertain=True).status == "CLARIFY_CASE"


def test_conflict_clarification_precedes_requirements():
    merged = run()
    conflicting = run(merged.state, value="three days", facts=merged.facts)
    conflicting.state.current_case_id = "c1"
    result = evaluate_requirements(conflicting.state, conflicts=conflicting.conflicts)
    assert result.status == "CLARIFY_CONFLICT"
    assert result.selected == []


def test_duplicate_requirements_deduplicate_or_reject_conflicting_policy():
    result = evaluate_requirements(intake_state(True), additional=[extra(), extra()])
    assert len(result.selected) == 1
    with pytest.raises(ValueError, match="conflicting"):
        evaluate_requirements(intake_state(True), additional=[extra(), extra(priority="CRITICAL")])
    with pytest.raises(ValueError, match="another case"):
        evaluate_requirements(intake_state(True), additional=[extra(target_case_id="other")])


def test_doctors_need_no_case_allowlist():
    conversation, _, _ = doctor_review()
    draft = prepare_doctor_handoff(conversation, "c1", DoctorContext(doctor_id="another_doctor"))
    assert draft.doctor_id == "another_doctor"


def test_resolved_case_does_not_restart_questions_without_new_intake():
    state = intake_state()
    state.cases[0].status = "RESOLVED"
    result = evaluate_requirements(state)
    assert result.status == "NO_REPLY"
    assert result.question is None


from medical_chatbot.agents.intake.question_generator import generate_question, QuestionGenerationError
import json


def fake_question(payload):
    return json.dumps({"question": "Câu hỏi do model giả lập tạo ra?",
                       "scope_ids": [s["id"] for s in payload["approved_scopes"]]})


def test_llm_receives_selected_scopes_and_context_without_mutation():
    result = evaluate_requirements(intake_state())
    snapshot = result.model_dump()
    def respond(payload):
        assert payload["case"]["chief_complaint"] == "headache"
        assert [s["requirement"]["key"] for s in payload["approved_scopes"]] == ["onset", "duration"]
        return fake_question(payload)
    assert generate_question(result, generate_json=respond)
    assert result.model_dump() == snapshot
    assert all(r.asked_count == 0 for r in result.selected)


@pytest.mark.parametrize("response", [
    "not json", "{}", '{"question":" ","scope_ids":["r0","r1"]}',
    '{"question":"test?","scope_ids":["unexpected"]}',
    '{"question":"test?","scope_ids":["r0"]}',
    '{"question":"test?","scope_ids":["r0","r1","r1"]}',
])
def test_bad_llm_response_does_not_mark_requirements_asked(response):
    result = evaluate_requirements(intake_state())
    with pytest.raises(QuestionGenerationError):
        generate_question(result, generate_json=lambda _: response)
    assert all(r.asked_count == 0 for r in result.selected)
    assert evaluate_requirements(result.state).status == "ASK_MORE"


def test_llm_timeout_has_no_template_fallback_or_secret_leak():
    def timeout(_):
        raise TimeoutError("secret request contents")
    with pytest.raises(QuestionGenerationError) as caught:
        generate_question(evaluate_requirements(intake_state()), generate_json=timeout)
    assert "secret" not in str(caught.value)


def test_delivery_tracking_is_separate_from_generation():
    result = evaluate_requirements(intake_state())
    question = generate_question(result, generate_json=fake_question)
    sent = mark_question_sent(result, question)
    assert all(r.asked_count == 1 for r in sent.selected)
    assert all(r.asked_count == 0 for r in result.selected)
    with pytest.raises(ValueError):
        mark_question_sent(sent, question)


def test_no_llm_for_doctor_mode_or_waiting_risk():
    def fail(_):
        pytest.fail("Unexpected LLM call")
    state = intake_state()
    state.cases[0].response_mode = "DOCTOR_ACTIVE"
    assert generate_question(evaluate_requirements(state), generate_json=fail) is None
    assert generate_question(evaluate_requirements(intake_state(True)), generate_json=fail) is None


def test_clarification_uses_llm_instead_of_fixed_question():
    result = evaluate_requirements(ClinicalState())
    assert result.question is None
    assert generate_question(result, generate_json=fake_question)


def test_gemini_sdk_request_configuration_and_client_cleanup(monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace
    from medical_chatbot.agents.intake.question_generator import gemini_generate
    captured = {}
    class Client:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.models = SimpleNamespace(generate_content=self.request)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            captured["closed"] = True
        def request(self, **kwargs):
            captured["request"] = kwargs
            return SimpleNamespace(text='{"question":"test?","scope_ids":["r0"]}')
    google = ModuleType("google")
    google.genai = SimpleNamespace(Client=Client)
    dotenv = ModuleType("dotenv")
    dotenv.load_dotenv = lambda **kwargs: None
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "dotenv", dotenv)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    monkeypatch.setenv("GEMINI_TIMEOUT_MS", "10000")
    assert gemini_generate({"approved_scopes": []})
    assert captured["client"]["http_options"] == {"timeout": 10000, "retry_options": {
        "attempts": 2, "initial_delay": 1.0, "max_delay": 2.0,
        "http_status_codes": [500, 502, 503, 504]}}
    assert captured["request"]["model"] == "test-model"
    assert captured["request"]["config"]["response_mime_type"] == "application/json"
    assert captured["closed"]


from medical_chatbot.agents.intake.input_processor import process_input, InputProcessingError, ExtractionOutput
from medical_chatbot.requirements.engine import make_question_message


def extraction_context():
    result = evaluate_requirements(intake_state())
    question = make_question_message(result, "Khi nào bắt đầu, kéo dài bao lâu?", message_id="q1",
                                     timestamp=datetime.fromisoformat("2026-11-02T10:00:00+07:00"))
    sent = mark_question_sent(result, question.content)
    conversation = Conversation(conversation_id="conv", patient_id="p", clinical_state=sent.state,
                                messages=[question])
    message = Message(message_id="u1", role="USER", content="two days",
                      timestamp="2026-11-02T10:01:00+07:00")
    return conversation, message


def extracted_duration():
    return {"case_result": {"resolution": "OLD_CASE", "target_case_id": "c1"},
            "case_facts": [{"field": "present_illness.duration", "key": "duration", "value": "two days",
                "status": "PRESENT", "source": {"message_id": "u1", "source_text": "two days"}}]}


def test_extract_short_answer_uses_question_context_and_merges_partial_answer():
    conversation, message = extraction_context()
    before = conversation.model_dump()
    def respond(payload):
        assert payload["last_question"]["message_id"] == "q1"
        assert [r["key"] for r in payload["last_question"]["asked_requirements"]] == ["onset", "duration"]
        return json.dumps(extracted_duration())
    extracted = process_input(conversation, message, generate_json=respond)
    merged = update_clinical_state(conversation.clinical_state, message, extracted)
    assert merged.state.cases[0].present_illness.duration == "two days"
    assert merged.state.cases[0].present_illness.onset is None
    assert conversation.model_dump() == before


@pytest.mark.parametrize("error", ["wrong_case", "wrong_source_id", "fake_excerpt", "control_field", "bad_json"])
def test_bad_extraction_is_rejected_without_mutation(error):
    conversation, message = extraction_context()
    data = extracted_duration()
    if error == "wrong_case":
        data["case_result"]["target_case_id"] = "foreign"
    elif error == "wrong_source_id":
        data["case_facts"][0]["source"]["message_id"] = "q1"
    elif error == "fake_excerpt":
        data["case_facts"][0]["source"]["source_text"] = "invented"
    elif error == "control_field":
        data["case_facts"][0]["field"] = "response_mode"
    raw = "not json" if error == "bad_json" else json.dumps(data)
    with pytest.raises(InputProcessingError):
        process_input(conversation, message, generate_json=lambda _: raw)
    assert conversation.clinical_state.cases[0].present_illness.duration is None


def test_explicit_unknown_answer_is_recorded_without_referral():
    conversation, message = extraction_context()
    message.content = "I do not remember"
    raw = {"case_result": {"resolution": "OLD_CASE", "target_case_id": "c1"},
           "unavailable_answers": [{"key": "onset", "target_case_id": "c1", "status": "UNKNOWN",
               "source": {"message_id": "u1", "source_text": "do not remember"}}]}
    extracted = process_input(conversation, message, generate_json=lambda _: json.dumps(raw))
    merged = update_clinical_state(conversation.clinical_state, message, extracted)
    assert evaluate_requirements(merged.state).status == "ASK_MORE"


def test_unasked_refusal_cannot_mark_another_requirement_unavailable():
    conversation, message = extraction_context()
    raw = {"case_result": {"resolution": "OLD_CASE", "target_case_id": "c1"},
           "unavailable_answers": [{"key": "course", "target_case_id": "c1", "status": "DECLINED",
               "source": {"message_id": "u1", "source_text": "two days"}}]}
    with pytest.raises(InputProcessingError):
        process_input(conversation, message, generate_json=lambda _: json.dumps(raw))


def test_new_case_extraction_and_ambiguous_case_keep_routing_in_code():
    conversation = Conversation(conversation_id="conv", patient_id="p")
    message = Message(message_id="u1", role="USER", content="headache",
                      timestamp="2026-11-02T10:01:00+07:00")
    raw = {"case_result": {"resolution": "NEW_CASE", "new_problem": "headache"},
           "case_facts": [{"field": "chief_complaint", "key": "chief_complaint", "value": "headache",
               "status": "PRESENT", "source": {"message_id": "u1", "source_text": "headache"}}]}
    extracted = process_input(conversation, message, generate_json=lambda _: json.dumps(raw))
    assert update_clinical_state(conversation.clinical_state, message, extracted).state.cases
    raw["case_result"] = {"resolution": "UNCERTAIN"}
    extracted = process_input(conversation, message, generate_json=lambda _: json.dumps(raw))
    result = update_clinical_state(conversation.clinical_state, message, extracted)
    assert not result.state.cases and len(result.pending) == 1


def test_extraction_provider_failure_does_not_leak_raw_text():
    conversation, message = extraction_context()
    def fail(_):
        raise RuntimeError("secret patient text")
    with pytest.raises(InputProcessingError) as caught:
        process_input(conversation, message, generate_json=fail)
    assert "secret" not in str(caught.value)


def test_question_metadata_is_only_allowed_on_ai_messages():
    conversation, message = extraction_context()
    with pytest.raises(ValidationError):
        Message.model_validate(message.model_dump() | {"asked_requirements": conversation.messages[0].asked_requirements})


def test_extraction_schema_has_no_recursive_json_value_reference():
    schema = ExtractionOutput.model_json_schema()
    assert "JsonValue" not in schema.get("$defs", {})
    assert "facts" in schema["properties"]
    assert "patient_facts" not in schema["properties"]
    assert "case_facts" not in schema["properties"]


def test_real_provider_scope_failure_is_classified_by_code(monkeypatch):
    from medical_chatbot.agents.intake import input_processor
    conversation = Conversation(conversation_id="conv", patient_id="synthetic")
    message = Message(message_id="msg_001", role="USER", content="Tôi 30 tuổi, thỉnh thoảng bị chóng mặt.",
                      timestamp="2026-11-13T10:00:00+07:00")
    # Reproduces the structure observed from Gemini: all facts in patient_facts.
    facts = [dict(field=field, key=key, value=value, status="PRESENT",
                  source=dict(message_id="msg_001", source_text=text)) for field, key, value, text in [
        ("patient_info.age", "age", 30, "Tôi 30 tuổi"),
        ("chief_complaint", "chief_complaint", "chóng mặt", "chóng mặt"),
        ("present_illness.frequency", "frequency", "thỉnh thoảng", "thỉnh thoảng")]]
    raw = dict(case_result=dict(resolution="NEW_CASE", target_case_id=None, new_problem="chóng mặt"),
               patient_facts=facts, case_facts=[], need_medical_rag=False, rag_reason=None)
    legacy = process_input(conversation, message, generate_json=lambda _: json.dumps(raw))
    assert [p.field.value for p in legacy.patient_facts] == ["patient_info.age"]
    assert len(legacy.case_facts) == 2

    def provider(payload, *, system_instruction, response_schema):
        assert "facts" in response_schema["properties"]
        assert "patient_facts" not in response_schema["properties"]
        return json.dumps({"case_result": raw["case_result"], "facts": facts})
    monkeypatch.setattr(input_processor, "gemini_generate", provider)
    extracted = process_input(conversation, message)
    merged = update_clinical_state(conversation.clinical_state, message, extracted)
    assert merged.state.patient_info.age == 30
    assert merged.state.cases[0].chief_complaint == "chóng mặt"
    assert merged.state.cases[0].present_illness.frequency == "thỉnh thoảng"
    assert not merged.rejected


def test_flat_extraction_still_rejects_invalid_sources_and_rag_flags():
    conversation, message = extraction_context()
    data = extracted_duration()
    data["facts"] = data.pop("case_facts")
    data["facts"][0]["source"]["source_text"] = "invented"
    with pytest.raises(InputProcessingError) as caught:
        process_input(conversation, message, generate_json=lambda _: json.dumps(data))
    assert "fact_source_not_current_message_or_exact_excerpt" in str(caught.value)
    data["facts"][0]["source"]["source_text"] = message.content
    data["rag_reason"] = "reason without flag"
    with pytest.raises(InputProcessingError) as caught:
        process_input(conversation, message, generate_json=lambda _: json.dumps(data))
    assert "response: rag_reason_mismatch" in str(caught.value)


def test_mixed_wire_formats_are_rejected():
    conversation, message = extraction_context()
    data = extracted_duration() | {"facts": []}
    with pytest.raises(InputProcessingError):
        process_input(conversation, message, generate_json=lambda _: json.dumps(data))


def test_duplicated_onset_cannot_silently_satisfy_total_duration():
    conversation, message = extraction_context()
    message.content = "Tôi bắt đầu bị khoảng một tuần trước."
    def raw(field, value, quote):
        return dict(field=f"present_illness.{field}", key=field, value=value, status="PRESENT",
                    source=dict(message_id=message.message_id, source_text=quote))
    data = dict(case_result=dict(resolution="OLD_CASE", target_case_id="c1"), facts=[
        raw("onset", "một tuần trước", message.content),
        raw("duration", "một tuần trước", message.content)])
    result = process_input(conversation, message, generate_json=lambda _: json.dumps(data))
    assert [p.field.value for p in result.case_facts] == ["present_illness.onset"]
    # Independently stated total duration must still be accepted.
    message.content += " Tổng thời gian là một tuần."
    data["facts"][1] = raw("duration", "một tuần", "Tổng thời gian là một tuần.")
    result = process_input(conversation, message, generate_json=lambda _: json.dumps(data))
    assert len(result.case_facts) == 2


def test_temperature_change_keeps_current_value_and_keyed_history():
    first = run(field="clinical_data.vitals", key="temperature", value=38)
    second = run(first.state, field="clinical_data.vitals", key="temperature", value=39,
                 text="now temperature is 39", change="TEMPORAL_CHANGE", facts=first.facts)
    case = second.state.cases[0]
    assert case.clinical_data.vitals["temperature"] == 39
    event = case.progress_history[-1]
    assert (event.key, event.old_value, event.new_value, event.type) == ("temperature", 38, 39, "TEMPORAL_CHANGE")
    assert first.state.cases[0].clinical_data.vitals["temperature"] == 38


def test_unknown_word_does_not_count_as_now_temporal_evidence():
    first = run(field="clinical_data.vitals", key="temperature", value=38)
    second = run(first.state, field="clinical_data.vitals", key="temperature", value=39,
                 text="unknown temperature 39", change="TEMPORAL_CHANGE", facts=first.facts)
    assert second.conflicts
    assert second.state.cases[0].clinical_data.vitals["temperature"] == 38


def test_older_message_cannot_replace_latest_value_with_temporal_cue():
    first = run(field="clinical_data.vitals", key="temperature", value=39)
    latest = first.facts[0].model_copy(update={"observed_at": datetime.fromisoformat("2026-10-30T11:00:00+07:00")})
    older = run(first.state, field="clinical_data.vitals", key="temperature", value=38,
                text="now temperature 38", change="TEMPORAL_CHANGE", facts=[latest])
    assert older.state.cases[0].clinical_data.vitals["temperature"] == 39
    assert older.conflicts


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_clinical_values_are_rejected(value):
    with pytest.raises(ValidationError):
        run(field="clinical_data.vitals", key="temperature", value=value)
