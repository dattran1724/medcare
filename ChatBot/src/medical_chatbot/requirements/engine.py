"""Minimal intake requirements and next questions; no PASS routing."""
from collections.abc import Sequence
from datetime import datetime
from typing import Literal
from pydantic import Field

from medical_chatbot.clinical.state_merger import PendingConflict
from medical_chatbot.schemas.base import Schema
from medical_chatbot.schemas.clinical_state import ClinicalState
from medical_chatbot.schemas.facts import Fact
from medical_chatbot.schemas.requirements import Requirement
from medical_chatbot.schemas.risk import RiskResult
from medical_chatbot.schemas.conversation import Message
from .question_selector import select_questions
from .satisfaction import CORE_FIELDS, is_satisfied


class RequirementResult(Schema):
    state: ClinicalState
    status: Literal["ASK_MORE", "CLARIFY_CASE", "CLARIFY_CONFLICT",
                    "HANDOFF_DOCTOR", "READY_FOR_REVIEW", "NO_REPLY"]
    selected: list[Requirement] = Field(default_factory=list)
    question: str | None = None


def record_unavailable_answer(state: ClinicalState, case_id: str, key: str, status: str) -> ClinicalState:
    """Record explicit unknown/refusal; this does not infer intent from free text."""
    if status not in {"UNKNOWN", "DECLINED", "UNAVAILABLE"}:
        raise ValueError("Expected UNKNOWN, DECLINED or UNAVAILABLE")
    updated = ClinicalState.model_validate(state.model_dump())
    case = next((c for c in updated.cases if c.case_id == case_id), None)
    if case is None:
        raise ValueError("Unknown case")
    matches = [r for r in case.missing_information if r.key == key]
    if len(matches) != 1:
        raise ValueError("Requirement key must identify one missing item")
    matches[0].status = status
    return updated


def evaluate_requirements(
    state: ClinicalState, *, facts: Sequence[Fact] = (),
    conflicts: Sequence[PendingConflict] = (), additional: Sequence[Requirement] = (),
    risk: RiskResult | None = None, case_uncertain: bool = False,
) -> RequirementResult:
    """Pure update. Save result.state to retain question tracking.

    Metadata, additional requirements and risk must be trusted application results.
    Core requirements apply to the active intake case; this is a starter catalog,
    not a comprehensive medical policy. This function never returns PASS.
    """
    updated = ClinicalState.model_validate(state.model_dump())
    case = next((c for c in updated.cases if c.case_id == updated.current_case_id), None)
    risk = RiskResult.model_validate((risk or RiskResult()).model_dump())
    if case and (case.response_mode == "DOCTOR_ACTIVE" or case.status == "RESOLVED"):
        return RequirementResult(state=updated, status="NO_REPLY")
    if case:
        case.workflow.next_action = None
    if risk.result == "DANGEROUS" or (
        case and case.risk_assessment.level == "DANGEROUS"
    ):
        return RequirementResult(state=updated, status="HANDOFF_DOCTOR")
    if case_uncertain or case is None:
        return RequirementResult(state=updated, status="CLARIFY_CASE")
    if any(c.proposed.target_case_id in (None, case.case_id) for c in conflicts):
        return RequirementResult(state=updated, status="CLARIFY_CONFLICT")
    requirements = [Requirement(key=key, target=target, target_case_id=case.case_id,
        priority="HIGH", source="CORE", reason=f"Initial intake: {key}", blocking=True)
        for key, target in CORE_FIELDS.items()]
    previous = {(r.target_case_id, r.key): r for r in case.missing_information}
    for raw in additional:
        item = Requirement.model_validate(raw.model_dump())
        if item.target_case_id not in (None, case.case_id):
            raise ValueError("Requirement belongs to another case")
        requirements.append(item)
    requirements.extend(r for r in case.missing_information if r.source != "CORE")
    unique = {}
    for item in requirements:
        identity = (item.target_case_id, item.key)
        if identity in unique:
            existing = unique[identity]
            if (existing.target, existing.priority, existing.blocking, existing.applicable) != (
                item.target, item.priority, item.blocking, item.applicable
            ):
                raise ValueError("Duplicate requirement key has conflicting target/policy")
            continue
        old = previous.get(identity)
        if old and old.target != item.target:
            raise ValueError("Tracked requirement changed target")
        item = item.model_copy(deep=True)
        item.asked_count = old.asked_count if old else 0
        item.status = old.status if old else "MISSING"
        if is_satisfied(updated, item, facts):
            item.status = "SATISFIED"
        elif item.status == "SATISFIED":
            item.status = "MISSING"
        unique[identity] = item
    case.missing_information = [r for r in unique.values() if r.applicable and r.status != "SATISFIED"]
    selected = select_questions(case.missing_information)
    if selected:
        case.workflow.status = "COLLECTING"
        case.workflow.next_action = "ASK_MORE"
        return RequirementResult(state=updated, status="ASK_MORE", selected=selected)
    case.workflow.next_action = None
    return RequirementResult(state=updated, status="READY_FOR_REVIEW")


def mark_question_sent(result: RequirementResult, question: str) -> RequirementResult:
    """Call after successful delivery, not after selection or failed generation."""
    if result.status not in {"ASK_MORE", "CLARIFY_CASE", "CLARIFY_CONFLICT"}:
        raise ValueError("This result does not request a question")
    if not question.strip():
        raise ValueError("Question cannot be blank")
    updated = RequirementResult.model_validate(result.model_dump())
    if updated.question is not None:
        raise ValueError("Question was already marked sent")
    updated.question = question.strip()
    for item in updated.selected:
        matches = [r for c in updated.state.cases for r in c.missing_information
                   if (r.target_case_id, r.key) == (item.target_case_id, item.key)]
        if len(matches) != 1 or matches[0].asked_count != item.asked_count:
            raise ValueError("Selected requirement does not match state")
        matches[0].asked_count += 1
        item.asked_count += 1
    return updated


def make_question_message(result: RequirementResult, question: str, *,
                          message_id: str, timestamp: datetime) -> Message:
    """Build history metadata; building does not send or increment asked_count."""
    if result.status not in {"ASK_MORE", "CLARIFY_CASE", "CLARIFY_CONFLICT"}:
        raise ValueError("Result does not request a question")
    case_id = result.state.current_case_id
    return Message(message_id=message_id, role="AI", content=question, timestamp=timestamp,
                   case_ids=[case_id] if case_id else [],
                   asked_requirements=[r.model_copy(deep=True) for r in result.selected])
