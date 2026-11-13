"""Doctor review and handoff functions; no UI, storage, or LLM integration."""

from datetime import datetime
from hashlib import sha256
from uuid import uuid4

from medical_chatbot.schemas.base import NonEmpty, Schema
from medical_chatbot.schemas.clinical_state import DoctorHandoff, DoctorHandoffContent
from medical_chatbot.schemas.conversation import Conversation, Message
from medical_chatbot.schemas.outputs import DoctorHandoffDraft


class DoctorContext(Schema):
    """Trusted server context, supplied after authenticating the doctor role.

    Never construct this from a browser request body or model output.
    This module does not implement authentication itself.
    """

    doctor_id: NonEmpty


def _doctor_case(conversation, case_id):
    case = next((c for c in conversation.clinical_state.cases if c.case_id == case_id), None)
    if case is None:
        raise ValueError("Case does not belong to this conversation")
    if case.response_mode != "DOCTOR_ACTIVE":
        raise ValueError("Case is not under doctor control")
    return case


def _snapshot(conversation: Conversation) -> str:
    # Conservative: any history or clinical-state change requires a fresh review.
    return sha256(conversation.model_dump_json().encode("utf-8")).hexdigest()


def prepare_doctor_handoff(
    conversation: Conversation, case_id: str, doctor: DoctorContext,
) -> DoctorHandoffDraft:
    """Open review data without changing state/mode.

    The initial summary quotes doctor messages; it is NOT an AI-generated medical
    summary. A future summarizer may replace draft.content for doctor review.
    Keep the returned draft server-side; the browser submits edited content only.
    """
    conversation = Conversation.model_validate(conversation.model_dump())
    doctor = DoctorContext.model_validate(doctor.model_dump())
    _doctor_case(conversation, case_id)
    messages = [m for m in conversation.messages if case_id in m.case_ids]
    doctor_messages = [m for m in messages if m.role == "DOCTOR"]
    if not doctor_messages:
        raise ValueError("No doctor messages to review")
    return DoctorHandoffDraft(
        conversation_id=conversation.conversation_id, case_id=case_id,
        doctor_id=doctor.doctor_id, snapshot=_snapshot(conversation), messages=messages,
        content=DoctorHandoffContent(summary="\n\n".join(m.content for m in doctor_messages)),
    )


def confirm_doctor_handoff(
    conversation: Conversation,
    draft: DoctorHandoffDraft,
    content: DoctorHandoffContent,
    doctor: DoctorContext,
    *,
    action: str,
    confirmed_at: datetime,
) -> Conversation:
    """Confirm edits and change mode together on a copy.

    The caller must atomically save the result against the same conversation
    version used here. An in-memory snapshot check alone cannot prevent a DB race.
    Draft/snapshot must come from trusted server storage, not be accepted from UI.
    """
    updated = Conversation.model_validate(conversation.model_dump())
    draft = DoctorHandoffDraft.model_validate(draft.model_dump())
    content = DoctorHandoffContent.model_validate(content.model_dump())
    doctor = DoctorContext.model_validate(doctor.model_dump())
    case = _doctor_case(updated, draft.case_id)
    if draft.doctor_id != doctor.doctor_id or draft.conversation_id != updated.conversation_id:
        raise PermissionError("Draft belongs to a different doctor or conversation")
    if draft.snapshot != _snapshot(updated):
        raise ValueError("Conversation changed; refresh the draft and review new data")
    sources = [m.message_id for m in updated.messages if m.role == "DOCTOR" and case.case_id in m.case_ids]
    handoff = DoctorHandoff(**content.model_dump(), doctor_id=doctor.doctor_id,
                            source_message_ids=sources, confirmed_at=confirmed_at, action=action)
    if any(m.timestamp > handoff.confirmed_at for m in updated.messages):
        raise ValueError("Confirmation time cannot precede conversation messages")
    case.doctor_handoff = handoff
    case.response_mode = "AI_ACTIVE"
    case.status = "RESOLVED" if action == "CLOSE_CASE" else "ACTIVE"
    case.workflow.status = "COMPLETED" if action == "CLOSE_CASE" else "COLLECTING"
    case.workflow.next_action = None
    # A handoff does not certify SAFE or remove a DANGEROUS assessment.
    if case.risk_assessment.level == "SAFE":
        case.risk_assessment.level = "NOT_ASSESSED"
    updated.messages.append(Message(
        message_id=f"handoff_{uuid4().hex}", role="SYSTEM",
        content=f"Doctor {doctor.doctor_id} confirmed {action} for {case.case_id}.",
        timestamp=handoff.confirmed_at, case_ids=[case.case_id],
    ))
    return Conversation.model_validate(updated.model_dump())
