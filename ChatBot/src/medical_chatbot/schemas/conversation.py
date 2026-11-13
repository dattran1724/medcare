from typing import Literal, Self

from pydantic import AwareDatetime, Field, model_validator

from .base import NonEmpty, Schema
from .clinical_state import ClinicalState
from .requirements import Requirement


class Message(Schema):
    message_id: NonEmpty
    role: Literal["USER", "AI", "DOCTOR", "SYSTEM"]
    content: NonEmpty
    timestamp: AwareDatetime
    case_ids: list[NonEmpty] = Field(default_factory=list)
    asked_requirements: list[Requirement] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_question_context(self) -> Self:
        if self.asked_requirements and self.role != "AI":
            raise ValueError("Only AI messages may carry asked requirements")
        identities = [(r.target_case_id, r.key) for r in self.asked_requirements]
        if len(identities) != len(set(identities)):
            raise ValueError("Duplicate asked requirements")
        if any(r.target_case_id is not None and r.target_case_id not in self.case_ids
               for r in self.asked_requirements):
            raise ValueError("Asked requirement must belong to message cases")
        return self


class Conversation(Schema):
    conversation_id: NonEmpty
    patient_id: NonEmpty
    messages: list[Message] = Field(default_factory=list)
    clinical_state: ClinicalState = Field(default_factory=ClinicalState)

    @model_validator(mode="after")
    def check_references(self) -> Self:
        messages = {m.message_id: m for m in self.messages}
        if len(messages) != len(self.messages):
            raise ValueError("Duplicate message IDs")
        cases = {c.case_id for c in self.clinical_state.cases}
        if any(set(m.case_ids) - cases for m in self.messages):
            raise ValueError("Message refers to an unknown case")
        for case in self.clinical_state.cases:
            if case.doctor_handoff:
                for message_id in case.doctor_handoff.source_message_ids:
                    source = messages.get(message_id)
                    if source is None or source.role != "DOCTOR" or case.case_id not in source.case_ids:
                        raise ValueError("Handoff sources must be doctor messages belonging to this case")
        return self
