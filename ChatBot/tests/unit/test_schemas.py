import pytest
from pathlib import Path
from pydantic import ValidationError
from medical_chatbot.schemas.clinical_state import ClinicalCase, ClinicalState
from medical_chatbot.schemas.conversation import Conversation, Message
from medical_chatbot.schemas.facts import Fact
from medical_chatbot.schemas.outputs import Agent1Input, Agent1Output, CaseResolution, InputProcessorOutput
from medical_chatbot.schemas.requirements import Requirement
from medical_chatbot.schemas.risk import RiskResult


def test_documented_json_example_is_valid():
    path = Path(__file__).resolve().parents[2] / "docs" / "clinical_state.example.json"
    state = ClinicalState.model_validate_json(path.read_text(encoding="utf-8"))
    assert state.patient_info.age is None
    assert state.cases[0].risk_assessment.level == "NOT_ASSESSED"


def fact(**changes):
    data = dict(fact_id="f1", field="present_illness.associated_symptoms", key="fever",
                value="sốt", status="PRESENT", target_case_id="c1",
                source=dict(message_id="m1", source_text="sốt"))
    return Fact.model_validate(data | changes)


def message(**changes):
    return Message.model_validate(dict(message_id="m1", role="USER", content="Tôi sốt",
                                       timestamp="2026-10-19T10:00:00+07:00") | changes)


def requirement(**changes):
    return Requirement.model_validate(dict(key="fever", target="present_illness.associated_symptoms",
        target_case_id="c1", priority="HIGH", source="CORE", reason="Cần hỏi", blocking=True) | changes)


def test_conversation_roundtrip_and_input():
    state = ClinicalState(cases=[ClinicalCase(case_id="c1", chief_complaint="S?t")], current_case_id="c1")
    conversation = Conversation(conversation_id="conv1", patient_id="p1", messages=[message()], clinical_state=state)
    assert Conversation.model_validate_json(conversation.model_dump_json()) == conversation
    assert Agent1Input(conversation=conversation, user_message=message(), current_time=message().timestamp)


def test_unknown_absent_and_uncertain_are_distinct():
    assert ClinicalState().patient_info.age is None
    assert fact(status="ABSENT", value=None).status == "ABSENT"
    assert fact(status="UNCERTAIN", value=None).status == "UNCERTAIN"
    assert RiskResult().result is None


@pytest.mark.parametrize("changes", [
    {"field": "workflow.mode"}, {"status": "UNKNOWN"}, {"value": None},
    {"status": "ABSENT"}, {"target_case_id": None},
    {"source": {"message_id": "m1", "source_text": " "}},
    {"field": "patient_info.age", "target_case_id": None, "value": "30"},
    {"field": "patient_info.age", "target_case_id": None, "value": -1},
    {"field": "patient_info.age", "target_case_id": None, "value": True},
    {"observed_at": "2026-10-19T10:00:00"}, {"workflow": "PASS"},
    {"value": []}, {"value": 42}, {"value": " "},
])
def test_invalid_facts(changes):
    with pytest.raises(ValidationError):
        fact(**changes)


def test_case_and_state_references():
    for data in [dict(current_case_id="missing"),
                 dict(cases=[dict(case_id="c1"), dict(case_id="c1")]),
                 dict(patient_info=dict(age="30")),
                 dict(patient_facts=[]), dict(medical_history=dict(conditions="asthma"))]:
        with pytest.raises(ValidationError):
            ClinicalState.model_validate(data)



@pytest.mark.parametrize("changes", [{"priority": "CRITICAL", "blocking": False},
                                      {"priority": "OPTIONAL"}, {"target_case_id": None}])
def test_invalid_requirement(changes):
    with pytest.raises(ValidationError):
        requirement(**changes)


@pytest.mark.parametrize("data", [
    {"result": "SAFE"}, {"assessment_status": "ASSESSED", "result": "SAFE"},
    {"assessment_status": "FAILED"},
    {"assessment_status": "ASSESSED", "result": "DANGEROUS", "reason": "Nguy cơ"},
    {"assessment_status": "ASSESSED", "result": "SAFE", "reason": "Đã đánh giá", "emergency": True},
])
def test_invalid_risk(data):
    with pytest.raises(ValidationError):
        RiskResult(**data)


def test_processing_failure_and_no_reply():
    error = dict(stage="risk", code="TIMEOUT", message="Timeout", retryable=True)
    assert RiskResult(assessment_status="FAILED", error=error).result is None
    assert Agent1Output(status="FAILED", error=error).next_action is None
    assert Agent1Output(status="NO_REPLY").response_text is None
    with pytest.raises(ValidationError):
        Agent1Output(status="FAILED", error=error, next_action="PASS_TO_AGENT_2")


def test_pass_guards():
    safe = RiskResult(assessment_source="RAG", assessment_status="ASSESSED", result="SAFE", reason="Đã đánh giá")
    data = dict(status="COMPLETED", next_action="PASS_TO_AGENT_2", chief_complaint_known=True,
                case_result=CaseResolution(resolution="OLD_CASE", target_case_id="c1"), risk=safe)
    assert Agent1Output(**data)
    assert Agent1Output(**(data | dict(risk=RiskResult())))
    assert Agent1Output(**(data | dict(requirements=[requirement(status="DECLINED")])))
    for changes in [dict(chief_complaint_known=False),
                    dict(unresolved_conflict_count=1), dict(requirements=[requirement()])]:
        with pytest.raises(ValidationError):
            Agent1Output(**(data | changes))


def test_processor_scope_and_rag_reason():
    assert InputProcessorOutput(case_result=CaseResolution(resolution="NEW_CASE", new_problem="Đau đầu"))
    with pytest.raises(ValidationError):
        CaseResolution(resolution="OLD_CASE")
    with pytest.raises(ValidationError):
        InputProcessorOutput(case_result=CaseResolution(resolution="UNCERTAIN"), need_medical_rag=True)
    proposal = fact().model_dump(exclude={"fact_id", "target_case_id", "observed_at"})
    pending = InputProcessorOutput(case_result=CaseResolution(resolution="UNCERTAIN"), case_facts=[proposal])
    assert pending.case_facts[0].key == "fever"
    with pytest.raises(ValidationError):
        InputProcessorOutput(case_result=CaseResolution(resolution="UNCERTAIN"), patient_facts=[proposal])


def test_history_source_and_message_validation():
    for msg in [dict(timestamp="2026-10-19T10:00:00"), dict(role="BOT"), dict(content=" ")]:
        with pytest.raises(ValidationError):
            message(**msg)
    with pytest.raises(ValidationError):
        Conversation(conversation_id="conv", patient_id="p", messages=[message(), message()])


def test_mutable_defaults_are_independent_and_execution_is_separate():
    first, second = ClinicalState(), ClinicalState()
    first.cases.append(ClinicalCase(case_id="c1"))
    assert second.cases == []
    with pytest.raises(ValidationError):
        ClinicalState(attempt_count=1)


def test_dangerous_requires_escalation_and_emergency_is_separate():
    risk = RiskResult(assessment_status="ASSESSED", result="DANGEROUS", reason="Cần đánh giá",
                      signals=[dict(description="Dấu hiệu đã ghi nhận", source=fact().source)])
    assert not risk.emergency
    assert Agent1Output(status="COMPLETED", risk=risk, next_action="HANDOFF_DOCTOR",
                        response_text="Cần bác sĩ đánh giá trực tiếp")
    for action in ["ASK_MORE", "PASS_TO_AGENT_2", "EMERGENCY"]:
        with pytest.raises(ValidationError):
            Agent1Output(status="COMPLETED", risk=risk, next_action=action, response_text="Thông báo")


def test_proposals_reject_model_assigned_ids():
    proposal = fact().model_dump(exclude={"fact_id", "target_case_id", "observed_at"})
    with pytest.raises(ValidationError):
        InputProcessorOutput(case_result=CaseResolution(resolution="NEW_CASE", new_problem="Sốt"),
                             case_facts=[proposal | {"fact_id": "model-generated"}])


def test_all_public_contracts_have_json_schema():
    for schema in [Fact, ClinicalCase, ClinicalState, Message, Conversation,
                   Requirement, RiskResult, Agent1Input, Agent1Output, InputProcessorOutput]:
        assert schema.model_json_schema()["additionalProperties"] is False


def test_exact_state_shape_and_roundtrip():
    state = ClinicalState(cases=[ClinicalCase(case_id="case_001", response_mode="DOCTOR_ACTIVE")])
    data = state.model_dump(mode="json")
    assert set(data) == {"patient_info", "medical_history", "current_case_id", "cases"}
    assert data["patient_info"] == {"age": None, "sex": None, "pregnancy_status": None}
    case = data["cases"][0]
    assert set(case) == {"case_id", "chief_complaint", "status", "response_mode", "present_illness", "clinical_data", "progress_history", "risk_assessment", "missing_information", "workflow", "doctor_handoff"}
    assert case["present_illness"]["character"] is None
    assert case["risk_assessment"]["level"] == "NOT_ASSESSED"
    assert case["progress_history"] == []
    assert ClinicalState.model_validate_json(state.model_dump_json()) == state


def test_populated_state_preserves_plain_values():
    state = ClinicalState.model_validate({
        "patient_info": {"age": 30},
        "medical_history": {"conditions": ["hen"]},
        "current_case_id": "c1",
        "cases": [{"case_id": "c1", "chief_complaint": "ho",
            "present_illness": {"associated_symptoms": ["sốt"]},
            "clinical_data": {"vitals": {"nhiet_do": 38.5}},
            "progress_history": [{"field": "present_illness.severity", "old_value": "nhẹ", "new_value": "vừa", "time": "2026-10-19T10:00:00+07:00", "type": "TEMPORAL_CHANGE"}]}]})
    assert state.patient_info.age == 30
    assert state.cases[0].clinical_data.vitals["nhiet_do"] == 38.5


@pytest.mark.parametrize("data", [
    {"present_illness": {"character": []}},
    {"progress_history": [{"field": None, "old_value": None, "new_value": None, "time": None, "type": "TEMPORAL_CHANGE"}]},
    {"risk_assessment": {"level": "SAFE", "requires_emergency_care": True}},
    {"risk_assessment": {"level": "DANGEROUS"}},
    {"response_mode": "INVALID"},
    {"workflow": {"next_action": "INVALID"}},
])
def test_invalid_case_data(data):
    with pytest.raises(ValidationError):
        ClinicalCase.model_validate({"case_id": "c1"} | data)
