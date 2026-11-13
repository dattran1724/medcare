from typing import Literal, Self

from pydantic import Field, model_validator

from .base import NonEmpty, Schema
from .facts import FactField


class Requirement(Schema):
    key: NonEmpty
    target: FactField
    target_case_id: NonEmpty | None = None
    priority: Literal["CRITICAL", "HIGH", "OPTIONAL"]
    source: Literal["CORE", "COMPLAINT", "INTAKE_RULE", "RAG", "AGENT_2"]
    reason: NonEmpty
    applicable: bool = True
    blocking: bool
    status: Literal["MISSING", "SATISFIED", "UNKNOWN", "DECLINED", "UNAVAILABLE"] = "MISSING"
    asked_count: int = Field(default=0, ge=0)
    evidence_ids: list[NonEmpty] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_policy(self) -> Self:
        if self.priority == "CRITICAL" and not self.blocking:
            raise ValueError("CRITICAL must block PASS")
        if self.priority == "OPTIONAL" and self.blocking:
            raise ValueError("OPTIONAL cannot block PASS")
        if self.target.is_case_field != (self.target_case_id is not None):
            raise ValueError("Requirement target scope does not match case ID")
        return self
