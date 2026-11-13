"""Conversation session metadata and transient LangGraph execution state."""
from typing import TypedDict
from pydantic import Field
from medical_chatbot.schemas.base import Schema
from medical_chatbot.schemas.conversation import Conversation, Message
from medical_chatbot.schemas.outputs import Agent1Output, InputProcessorOutput, FactProposal, CaseResolution
from medical_chatbot.schemas.facts import Fact
from medical_chatbot.schemas.risk import RiskResult, ProcessingError
from medical_chatbot.clinical.state_merger import PendingConflict
from medical_chatbot.agents.intake.risk_assessor import ScreeningDecision
from medical_chatbot.requirements.engine import RequirementResult
from medical_chatbot.rag import RagResult


class IntakeSession(Schema):
    conversation: Conversation
    facts: list[Fact] = Field(default_factory=list)
    conflicts: list[PendingConflict] = Field(default_factory=list)
    pending: list[FactProposal] = Field(default_factory=list)
    risk_by_case: dict[str, RiskResult] = Field(default_factory=dict)
    unbound_risk: RiskResult | None = None
    pending_question: RequirementResult | None = None
    pending_question_for: str | None = None
    pending_question_risk: RiskResult | None = None
    pending_question_case: CaseResolution | None = None
    question_attempts: int = 0


class TurnResult(Schema):
    session: IntakeSession
    output: Agent1Output
    processing_errors: list[ProcessingError] = Field(default_factory=list)
    rag: RagResult | None = None
    rag_skip_reason: str | None = None
    user_notice: str | None = None

class Agent1GraphState(TypedDict, total=False):
    session: IntakeSession
    message: Message
    screening: ScreeningDecision | None
    risk: RiskResult
    extracted: InputProcessorOutput | None
    extraction_failed: bool
    extraction_error: ProcessingError | None
    rag_skip_reason: str | None
    rag_required: bool
    bound_case_id: str | None
    output: Agent1Output
    rag: RagResult | None
