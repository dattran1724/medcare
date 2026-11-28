"""One LangGraph invocation per message; no database or Agent 2 implementation."""
from uuid import uuid4
from datetime import datetime, timezone
from collections.abc import Callable
import os
from langgraph.graph import StateGraph, START, END

from medical_chatbot.schemas.conversation import Message
from medical_chatbot.schemas.outputs import Agent1Output, InputProcessorOutput, CaseResolution, FactProposal
from medical_chatbot.schemas.risk import ProcessingError, processing_error
from medical_chatbot.schemas.risk import RiskResult
from medical_chatbot.clinical.state_merger import update_clinical_state
from medical_chatbot.requirements.engine import evaluate_requirements, mark_question_sent, make_question_message
from .input_processor import process_input
from .question_generator import generate_question
from .risk_assessor import assess_message
from medical_chatbot.graph.state import IntakeSession, TurnResult, Agent1GraphState
from medical_chatbot.rag import IntakeRag

DANGER_WARNING = "Bạn đang có dấu hiệu nguy hiểm nên cần được thăm khám trực tiếp. Trong lúc chờ bác sĩ, AI có thể tiếp tục ghi nhận thông tin nhưng không thay thế việc khám trực tiếp."


def _stronger_danger(current, previous):
    """Retain danger across turns, but never hide a new emergency."""
    if previous and previous.result == "DANGEROUS" and (
        current.result != "DANGEROUS" or (previous.emergency and not current.emergency)
    ):
        return previous
    return current


def _ask(session, intake, risk, case_result, generate_json, *, source_rejected=False,
         danger_warning=None):
    question = generate_question(intake, history=session.conversation.messages,
                                 conflicts=session.conflicts, generate_json=generate_json)
    if question is None:
        raise ValueError("Question was not generated")
    message_id = session.conversation.messages[-1].message_id
    selected_fields = {requirement.target for requirement in intake.selected}
    recorded_other_facts = any(fact.source.message_id == message_id and fact.field not in selected_fields
                               for fact in session.facts)
    if source_rejected:
        question = f"Mình đã ghi nhận phần thông tin xác thực được; một số chi tiết chưa được cập nhật. {question}"
    elif recorded_other_facts:
        question = f"Mình đã ghi nhận thông tin bạn vừa chia sẻ. {question}"
    if danger_warning:
        question = f"{danger_warning}\n\n{question}"
    message = make_question_message(intake, question, message_id=f"ai_{uuid4().hex}",
        timestamp=max(session.conversation.messages[-1].timestamp, datetime.now(timezone.utc)))
    sent = mark_question_sent(intake, question)
    output = Agent1Output(status="COMPLETED", risk=risk, case_result=case_result,
        requirements=next((c.missing_information for c in sent.state.cases
            if c.case_id == sent.state.current_case_id), []),
        next_action="ASK_MORE", response_text=question, unresolved_conflict_count=len(session.conflicts),
        risk_warning=danger_warning)
    session.conversation.clinical_state = sent.state
    session.conversation.messages.append(message)
    session.pending_question = None
    session.pending_question_for = None
    session.pending_question_risk = None
    session.pending_question_case = None
    session.question_attempts = 0
    return output


def retry_question(session: IntakeSession, message_id: str, *, question_json=None) -> TurnResult:
    """Retry only wording, preserving accepted facts and never duplicating USER data."""
    session = IntakeSession.model_validate(session.model_dump())
    intake = session.pending_question
    if (intake is None or session.pending_question_for != message_id
            or not session.conversation.messages or session.conversation.messages[-1].message_id != message_id
            or session.conversation.clinical_state != intake.state or session.question_attempts >= 3):
        raise ValueError("No current retryable question, state changed, or attempt limit reached")
    session.question_attempts += 1
    risk = session.pending_question_risk or RiskResult()
    try:
        output = _ask(session, intake, risk, session.pending_question_case, question_json)
    except Exception as error:
        failure = processing_error("question", error)
        failure.retryable = session.question_attempts < 3
        output = Agent1Output(status="FAILED", risk=risk, error=failure)
    return TurnResult(session=session, output=output,
                      processing_errors=[output.error] if output.error else [],
                      user_notice="Chưa tạo được câu hỏi. Bạn có thể thử lại." if output.error else None)


def build_intake_graph(*, extract_json: Callable | None = None,
                       question_json: Callable | None = None, risk_json: Callable | None = None,
                       rag_service: IntakeRag | None = None):
    def screen(state):
        decision, risk = assess_message(state["session"].conversation, state["message"], generate_json=risk_json)
        return {"screening": decision, "risk": risk}

    def merge(state):
        session = state["session"].model_copy(deep=True)
        session.pending_question = None
        session.pending_question_for = None
        session.pending_question_risk = None
        session.pending_question_case = None
        session.question_attempts = 0
        rollback = session.model_copy(deep=True)
        try:
            # A confirmed doctor review supersedes old provisional screening,
            # not an official RAG/doctor risk finding or newly reported danger.
            history = {m.message_id: m for m in session.conversation.messages}
            for reviewed in session.conversation.clinical_state.cases:
                previous = session.risk_by_case.get(reviewed.case_id)
                if (reviewed.doctor_handoff and previous and previous.assessment_source == "LLM_SCREENING"
                        and previous.signals and all(
                            signal.source.message_id in history
                            and history[signal.source.message_id].timestamp <= reviewed.doctor_handoff.confirmed_at
                            for signal in previous.signals)):
                    session.risk_by_case.pop(reviewed.case_id)
                unbound = session.unbound_risk
                if (reviewed.doctor_handoff and unbound and unbound.assessment_source == "LLM_SCREENING"
                        and unbound.signals and all(
                            signal.source.message_id in history
                            and reviewed.case_id in history[signal.source.message_id].case_ids
                            and history[signal.source.message_id].timestamp <= reviewed.doctor_handoff.confirmed_at
                            for signal in unbound.signals)):
                    session.unbound_risk = None
            extracted = process_input(session.conversation, state["message"], pending=session.pending, generate_json=extract_json)
            old_ids = {c.case_id for c in session.conversation.clinical_state.cases}
            if extracted.case_result.resolution == "NEW_CASE" and not any(p.field.value == "chief_complaint" for p in extracted.case_facts):
                problem = extracted.case_result.new_problem
                extracted.case_facts.append(FactProposal(field="chief_complaint", key="chief_complaint",
                    value=problem, status="PRESENT", source={"message_id": state["message"].message_id, "source_text": problem}))
            merged = update_clinical_state(session.conversation.clinical_state, state["message"], extracted,
                                           facts=session.facts, conflicts=session.conflicts)
            bound_id = (extracted.case_result.target_case_id if extracted.case_result.resolution == "OLD_CASE" else
                merged.state.current_case_id if merged.state.current_case_id not in old_ids else None)
            # Replay only individually confirmed pending facts, retaining their original source/time.
            for binding in extracted.pending_bindings:
                identity = (binding.source_message_id, binding.field, binding.key)
                proposal = next(p for p in session.pending if (p.source.message_id, p.field, p.key) == identity)
                original = next(m for m in session.conversation.messages if m.message_id == binding.source_message_id)
                replay = InputProcessorOutput(case_result=CaseResolution(resolution="OLD_CASE", target_case_id=bound_id),
                    case_facts=[proposal.model_copy(update={"change_type": "NONE"})])
                replayed = update_clinical_state(merged.state, original.model_copy(update={"case_ids": []}), replay,
                                                 facts=merged.facts, conflicts=merged.conflicts)
                if replayed.rejected or replayed.pending:
                    raise ValueError("Pending fact could not be bound")
                merged.state, merged.facts, merged.conflicts = replayed.state, replayed.facts, replayed.conflicts
                if bound_id not in original.case_ids:
                    original.case_ids.append(bound_id)
                session.pending = [p for p in session.pending if (p.source.message_id, p.field, p.key) != identity]
            session.conversation.clinical_state = merged.state
            session.facts, session.conflicts = merged.facts, merged.conflicts
            # Retain unbound facts; never automatically attach old pending data to a new case.
            for proposal in merged.pending:
                if proposal not in session.pending:
                    session.pending.append(proposal)
            if merged.rejected:
                # Never commit a partial merge as a successful extraction.
                raise ValueError("Merger rejected proposed facts")
            return {"session": session, "extracted": extracted, "extraction_failed": False, "extraction_error": None,
                    "rag_required": extracted.need_medical_rag, "bound_case_id": bound_id}
        except Exception as error:
            return {"session": rollback, "extracted": None, "extraction_failed": True, "rag_required": False,
                    "extraction_error": processing_error("extraction", error, validating=True)}

    def respond(state):
        session, message, risk = state["session"], state["message"], state["risk"]
        conversation = session.conversation
        clinical = conversation.clinical_state
        extracted = state.get("extracted")
        case = next((c for c in clinical.cases if c.case_id == clinical.current_case_id), None)
        # Mapping failures must not attach raw data/risk to the previous current case.
        bound = next((c for c in clinical.cases if c.case_id == state.get("bound_case_id")), None)
        rag = state.get("rag")
        rag_ready = rag is not None and rag.status == "READY"
        if bound:
            message = message.model_copy(update={"case_ids": [bound.case_id]})
        elif risk.result == "DANGEROUS":
            session.unbound_risk = _stronger_danger(risk, session.unbound_risk).model_copy(deep=True)
        if bound and risk.result == "DANGEROUS":
            previous = session.risk_by_case.get(bound.case_id)
            if previous is None or not previous.emergency or risk.emergency:
                session.risk_by_case[bound.case_id] = risk.model_copy(deep=True)
        bound_assessment = risk
        # Changing complaint must not bypass an unresolved patient danger signal.
        for previous in session.risk_by_case.values():
            risk = _stronger_danger(risk, previous)
        risk = _stronger_danger(risk, session.unbound_risk)
        # LLM screening is a provisional signal, not a RAG/doctor-approved clinical conclusion.
        if bound and bound_assessment.result == "DANGEROUS" and bound_assessment.assessment_source in {"RAG", "DOCTOR"}:
            bound.risk_assessment.level = "DANGEROUS"
            bound.risk_assessment.danger_signs = [s.description for s in bound_assessment.signals]
            bound.risk_assessment.requires_emergency_care = bound_assessment.emergency
            bound.risk_assessment.requires_in_person_evaluation = True
        elif bound and bound_assessment.result == "SAFE" and bound_assessment.assessment_source in {"RAG", "DOCTOR"} and bound.risk_assessment.level != "DANGEROUS" and not (
            state["screening"] and state["screening"].needs_external_evidence
        ):
            bound.risk_assessment.level = "SAFE"
        conversation.messages.append(message)
        output = None
        previous_danger = case and case.risk_assessment.level == "DANGEROUS"
        doctor_pending = bool(case and case.workflow.status == "WAITING_DOCTOR")
        danger_pending = bool(doctor_pending and risk.result == "DANGEROUS")
        if case and case.response_mode == "DOCTOR_ACTIVE":
            output = Agent1Output(status="NO_REPLY", risk=risk)
        elif (risk.result == "DANGEROUS" or previous_danger) and not doctor_pending:
            emergency = risk.emergency or bool(case and case.risk_assessment.requires_emergency_care)
            # Legacy case summaries without risk provenance cannot create a new
            # sourced emergency assessment. Normal sessions retain it above.
            emergency = bool(emergency and risk.emergency)
            output = Agent1Output(status="COMPLETED", risk=risk,
                next_action="EMERGENCY" if emergency else "HANDOFF_DOCTOR",
                response_text=("Có dấu hiệu cần trợ giúp y tế khẩn cấp. Hãy tìm trợ giúp y tế ngay; không chờ phản hồi trong cuộc trò chuyện."
                    if emergency else "Thông tin hiện tại cần bác sĩ đánh giá trực tiếp. Hệ thống đã ghi nhận yêu cầu chuyển bác sĩ; chưa có xác nhận tiếp nhận."))
        elif state["screening"] and state["screening"].doctor_request and not doctor_pending:
            output = Agent1Output(status="COMPLETED", risk=risk, next_action="HANDOFF_DOCTOR",
                response_text="Đã ghi nhận yêu cầu gặp bác sĩ của bạn; chưa có xác nhận bác sĩ tiếp nhận.")
        elif danger_pending and (state["extraction_failed"] or risk.assessment_status == "FAILED"):
            response = f"{DANGER_WARNING}\n\nMình chưa xử lý được đầy đủ tin nhắn này. Bạn có thể gửi lại hoặc bổ sung thông tin trong lúc chờ bác sĩ."
            output = Agent1Output(status="COMPLETED", risk=risk, next_action="CONTINUE_AI",
                response_text=response, risk_warning=DANGER_WARNING)
        elif state["extraction_failed"] or risk.assessment_status == "FAILED":
            error = state["extraction_error"] if state["extraction_failed"] else risk.error
            if (state["extraction_failed"] and risk.assessment_status == "ASSESSED"
                    and risk.result == "SAFE" and error
                    and "fact_source_not_current_message_or_exact_excerpt" in error.message):
                response = ("Mình chưa thể xác nhận chính xác một phần thông tin bạn gửi nên chưa cập nhật phần đó. "
                            "Bạn vui lòng diễn đạt lại thông tin bằng câu ngắn hơn để mình tiếp tục nhé.")
                output = Agent1Output(status="COMPLETED", risk=risk, next_action="ASK_MORE",
                                      response_text=response)
                conversation.messages.append(Message(message_id=f"ai_{uuid4().hex}", role="AI",
                    content=response, timestamp=max(message.timestamp, datetime.now(timezone.utc)),
                    case_ids=message.case_ids))
            else:
                output = Agent1Output(status="FAILED", risk=risk, error=error or ProcessingError(
                    stage="routing", code="PROVIDER", message="Message processing failed", retryable=True))
        elif rag and rag.status in {"UNAVAILABLE", "INSUFFICIENT", "FAILED"} and not doctor_pending:
            output = Agent1Output(status="COMPLETED", risk=risk, next_action="HANDOFF_DOCTOR",
                response_text="Tài liệu hiện có chưa đủ để tiếp tục thu thập thông tin chuyên môn. Đã ghi nhận yêu cầu bác sĩ xem xét; chưa có xác nhận tiếp nhận.")
        else:
            intake = evaluate_requirements(clinical, facts=session.facts, conflicts=session.conflicts, risk=risk,
                case_uncertain=bound is None, additional=rag.requirements if rag_ready else (),
                doctor_handoff_pending=doctor_pending)
            conversation.clinical_state = intake.state
            try:
                if intake.status in {"ASK_MORE", "CLARIFY_CASE", "CLARIFY_CONFLICT"}:
                    output = _ask(session, intake, risk, extracted.case_result, question_json,
                                  source_rejected=extracted.source_rejected_count > 0,
                                  danger_warning=DANGER_WARNING if danger_pending else None)
                elif intake.status == "HANDOFF_DOCTOR":
                    output = Agent1Output(status="COMPLETED", risk=risk, next_action="HANDOFF_DOCTOR",
                        response_text="Có dấu hiệu nguy hiểm cần bác sĩ xem xét. Đã ghi nhận yêu cầu chuyển bác sĩ; chưa có xác nhận tiếp nhận.")
                elif intake.status == "READY_FOR_REVIEW" and extracted.source_rejected_count:
                    response = ("Mình đã ghi nhận phần thông tin xác thực được; một số chi tiết chưa được cập nhật. "
                                "Bạn vui lòng diễn đạt lại phần đó để mình hoàn tất thu thập nhé.")
                    output = Agent1Output(status="COMPLETED", risk=risk, next_action="ASK_MORE",
                                          response_text=response)
                    conversation.messages.append(Message(message_id=f"ai_{uuid4().hex}", role="AI",
                        content=response, timestamp=max(message.timestamp, datetime.now(timezone.utc)),
                        case_ids=message.case_ids))
                elif intake.status == "READY_FOR_REVIEW" and doctor_pending:
                    response = (f"{DANGER_WARNING}\n\n" if danger_pending else "") + (
                        "Mình đã ghi nhận thông tin bổ sung. Hội thoại vẫn đang chờ bác sĩ tiếp nhận.")
                    output = Agent1Output(status="COMPLETED", risk=risk, next_action="CONTINUE_AI",
                        response_text=response, risk_warning=DANGER_WARNING if danger_pending else None)
                elif intake.status == "READY_FOR_REVIEW":
                    output = Agent1Output(status="COMPLETED", risk=risk, chief_complaint_known=bool(bound.chief_complaint),
                        case_result=CaseResolution(resolution="OLD_CASE", target_case_id=bound.case_id),
                        requirements=next(c for c in intake.state.cases if c.case_id == bound.case_id).missing_information,
                        unresolved_conflict_count=sum(c.proposed.target_case_id in (None, bound.case_id) for c in session.conflicts),
                        next_action="PASS_TO_AGENT_2",
                        response_text="Đã ghi nhận thông tin ban đầu của bạn. Thông tin đã sẵn sàng cho bước đánh giá tiếp theo.")
                    target = next(c for c in conversation.clinical_state.cases if c.case_id == bound.case_id)
                    target.workflow.status = "WAITING_AGENT_2"
                    target.workflow.next_action = "PASS_TO_AGENT_2"
                else:
                    output = Agent1Output(status="NO_REPLY", risk=risk)
            except Exception as error:
                if intake.status in {"ASK_MORE", "CLARIFY_CASE", "CLARIFY_CONFLICT"}:
                    session.pending_question = intake.model_copy(deep=True)
                    session.pending_question_for = message.message_id
                    session.pending_question_risk = risk.model_copy(deep=True)
                    session.pending_question_case = extracted.case_result.model_copy(deep=True)
                    session.question_attempts = 1
                    failure = processing_error("question", error)
                else:
                    # Previously every response error was mislabeled as question failure.
                    # Do not create a question retry for a failed handoff/result validation.
                    failure = processing_error("routing", error)
                output = Agent1Output(status="FAILED", risk=risk, error=failure)
        if output.next_action in {"HANDOFF_DOCTOR", "EMERGENCY"}:
            current = next((c for c in conversation.clinical_state.cases if c.case_id == conversation.clinical_state.current_case_id), None)
            if current and bound:
                current.workflow.status = "WAITING_DOCTOR"
                current.workflow.next_action = output.next_action
        elif doctor_pending and case:
            case.workflow.status = "WAITING_DOCTOR"
        if output.next_action in {"HANDOFF_DOCTOR", "EMERGENCY", "PASS_TO_AGENT_2", "CONTINUE_AI"} and output.response_text:
            conversation.messages.append(Message(message_id=f"ai_{uuid4().hex}", role="AI",
                content=output.response_text, timestamp=max(message.timestamp, datetime.now(timezone.utc)),
                case_ids=[bound.case_id] if bound else []))
        # Validate the entire updated snapshot before committing any storage change.
        return {"session": IntakeSession.model_validate(session.model_dump()), "output": output}

    def retrieve_evidence(state):
        case = next((c for c in state["session"].conversation.clinical_state.cases
                     if c.case_id == state.get("bound_case_id")), None)
        saved_risk = (state["session"].risk_by_case.get(case.case_id) if case else None)
        danger_pending = bool((saved_risk and saved_risk.result == "DANGEROUS")
                              or state["session"].unbound_risk and state["session"].unbound_risk.result == "DANGEROUS")
        reason = ("extraction_failed" if state["extraction_failed"] else
                  "risk_failed" if state["risk"].assessment_status == "FAILED" else
                  "case_unbound" if case is None else
                  "doctor_mode" if case.response_mode == "DOCTOR_ACTIVE" else None)
        if reason:
            return {"rag": None, "rag_skip_reason": reason}
        # Retrieve only for a concrete specialist question need, not to certify SAFE.
        # Danger and explicit doctor requests route directly without delaying on RAG.
        if danger_pending or state["risk"].result == "DANGEROUS" or (state["screening"] and state["screening"].doctor_request):
            return {"rag": None, "rag_skip_reason": "direct_escalation"}
        requested = state["rag_required"] or bool(
            state["screening"] and state["screening"].needs_external_evidence)
        if not requested:
            return {"rag": None, "rag_skip_reason": "not_requested"}
        service = rag_service or IntakeRag()
        return {"rag": service.assess(state["session"].conversation, state["message"], case.case_id), "rag_skip_reason": None}

    graph = StateGraph(Agent1GraphState)
    graph.add_node("screen_raw_message", screen)
    graph.add_node("extract_and_merge", merge)
    graph.add_node("route_and_respond", respond)
    graph.add_node("retrieve_medical_evidence", retrieve_evidence)
    graph.add_edge(START, "screen_raw_message")
    graph.add_edge("screen_raw_message", "extract_and_merge")
    graph.add_edge("extract_and_merge", "retrieve_medical_evidence")
    graph.add_edge("retrieve_medical_evidence", "route_and_respond")
    graph.add_edge("route_and_respond", END)
    compiled = graph.compile()
    # Keep provider injection on the compiled graph for isolated retry tests.
    compiled.question_provider = question_json
    return compiled


def run_turn(graph, session: IntakeSession, message: Message) -> TurnResult:
    session = IntakeSession.model_validate(session.model_dump())
    message = Message.model_validate(message.model_dump())
    if message.role != "USER" or any(m.message_id == message.message_id for m in session.conversation.messages):
        raise ValueError("Expected a new USER message")
    result = graph.invoke({"session": session, "message": message})
    errors = []
    if result["risk"].error:
        errors.append(result["risk"].error)
    if result.get("extraction_error"):
        errors.append(result["extraction_error"])
    if result["output"].error and result["output"].error not in errors:
        errors.append(result["output"].error)
    rag = result.get("rag")
    if rag and rag.error:
        errors.append(rag.error)
    return TurnResult(session=result["session"], output=result["output"], processing_errors=errors,
                      rag=rag, rag_skip_reason=result.get("rag_skip_reason"),
                      user_notice="Chưa xử lý xong tin nhắn. Bạn có thể thử lại trước khi gửi tin nhắn mới."
                          if result["output"].status == "FAILED" else None)
