from typing import Literal, Self

from pydantic import AwareDatetime, Field, JsonValue, model_validator
from pydantic_core import PydanticCustomError

from .base import NonEmpty, Schema
from .conversation import Conversation, Message
from .clinical_state import DoctorHandoffContent
from .facts import Fact, FactField, FactSource, FactStatus
from .requirements import Requirement
from .risk import ProcessingError, RiskResult


class DoctorHandoffDraft(Schema):
    """Review-only data. Never interpreted as an approved case conclusion."""

    conversation_id: NonEmpty
    case_id: NonEmpty
    doctor_id: NonEmpty
    snapshot: NonEmpty
    messages: list[Message]
    content: DoctorHandoffContent


class CaseResolution(Schema):
    resolution: Literal["OLD_CASE", "NEW_CASE", "UNCERTAIN"]
    target_case_id: NonEmpty | None = None
    new_problem: NonEmpty | None = None

    @model_validator(mode="after")
    def check_resolution(self) -> Self:
        if (self.resolution == "OLD_CASE") != (self.target_case_id is not None):
            raise PydanticCustomError("case_target_mismatch", "Only OLD_CASE requires a target ID")
        if (self.resolution == "NEW_CASE") != (self.new_problem is not None):
            raise PydanticCustomError("case_problem_mismatch", "Only NEW_CASE requires a new problem")
        return self


class FactProposal(Schema):
    """No fact ID or case ID authority is granted to the extractor."""

    field: FactField
    key: NonEmpty
    value: JsonValue = None
    status: FactStatus
    source: FactSource
    change_type: Literal["NONE", "CORRECTION", "TEMPORAL_CHANGE", "AMBIGUOUS"] = "NONE"

    @model_validator(mode="after")
    def validate_fact_value(self) -> Self:
        if self.field.is_scalar:
            self.key = self.field.value.rsplit(".", 1)[-1]
        Fact(fact_id="validation-only", field=self.field, key=self.key,
             value=self.value, status=self.status, source=self.source,
             target_case_id="validation-only" if self.field.is_case_field else None)
        return self


class UnavailableAnswer(Schema):
    key: NonEmpty
    target_case_id: NonEmpty | None = None
    status: Literal["UNKNOWN", "DECLINED", "UNAVAILABLE"]
    source: FactSource


class PendingBinding(Schema):
    source_message_id: NonEmpty
    field: FactField
    key: NonEmpty
    confirmation_source: FactSource


class InputProcessorOutput(Schema):
    case_result: CaseResolution
    patient_facts: list[FactProposal] = Field(default_factory=list)
    case_facts: list[FactProposal] = Field(default_factory=list)
    need_medical_rag: bool = False
    rag_reason: NonEmpty | None = None
    unavailable_answers: list[UnavailableAnswer] = Field(default_factory=list)
    pending_bindings: list[PendingBinding] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_proposals(self) -> Self:
        if self.need_medical_rag != (self.rag_reason is not None):
            raise PydanticCustomError("rag_reason_mismatch", "RAG flag and reason must agree")
        if any(p.field.is_case_field for p in self.patient_facts):
            raise PydanticCustomError("patient_fact_scope_mismatch", "patient_facts must have patient scope")
        if any(not p.field.is_case_field for p in self.case_facts):
            raise PydanticCustomError("case_fact_scope_mismatch", "case_facts must have case scope")
        return self


class Agent1Input(Schema):
    conversation: Conversation
    user_message: Message
    current_time: AwareDatetime

    @model_validator(mode="after")
    def check_message(self) -> Self:
        if self.user_message.role != "USER":
            raise ValueError("Agent 1 requires a USER message")
        cases = {c.case_id for c in self.conversation.clinical_state.cases}
        if set(self.user_message.case_ids) - cases:
            raise ValueError("Incoming message refers to unknown cases")
        for old in self.conversation.messages:
            if old.message_id == self.user_message.message_id and old != self.user_message:
                raise ValueError("Message ID reused with different content")
        return self


class Agent1Output(Schema):
    status: Literal["COMPLETED", "FAILED", "NO_REPLY"]
    case_result: CaseResolution | None = None
    risk: RiskResult = Field(default_factory=RiskResult)
    requirements: list[Requirement] = Field(default_factory=list)
    unresolved_conflict_count: int = Field(default=0, ge=0)
    chief_complaint_known: bool = False
    next_action: Literal["ASK_MORE", "PASS_TO_AGENT_2", "HANDOFF_DOCTOR", "EMERGENCY"] | None = None
    response_text: NonEmpty | None = None
    error: ProcessingError | None = None

    @model_validator(mode="after")
    def check_outcome(self) -> Self:
        if self.status == "FAILED":
            if self.error is None or self.next_action is not None or self.response_text is not None:
                raise ValueError("FAILED requires error and no business action/reply")
        elif self.error is not None:
            raise ValueError("Processing errors require FAILED")
        elif self.status == "NO_REPLY":
            if self.next_action is not None or self.response_text is not None:
                raise ValueError("NO_REPLY cannot contain an action or reply")
        elif self.next_action is None:
            raise ValueError("COMPLETED requires an action")
        if self.next_action in {"ASK_MORE", "HANDOFF_DOCTOR", "EMERGENCY"} and not self.response_text:
            raise ValueError("This action requires a user-facing message")
        if self.next_action == "EMERGENCY" and not self.risk.emergency:
            raise ValueError("EMERGENCY requires an emergency risk assessment")
        if self.next_action in {"ASK_MORE", "PASS_TO_AGENT_2"} and self.risk.result == "DANGEROUS":
            raise ValueError("DANGEROUS requires escalation")
        if self.next_action == "PASS_TO_AGENT_2":
            if (self.risk.assessment_status == "FAILED" or not self.chief_complaint_known
                or self.unresolved_conflict_count
                or self.case_result is None or self.case_result.resolution != "OLD_CASE"
                or any(r.applicable and r.blocking and r.status == "MISSING" for r in self.requirements)):
                raise ValueError("PASS prerequisites are not satisfied")
        return self
