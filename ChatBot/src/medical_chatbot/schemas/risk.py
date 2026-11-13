from typing import Literal, Self
import logging

from pydantic import Field, ValidationError, model_validator

from .base import NonEmpty, Schema
from .facts import FactSource


class ProcessingError(Schema):
    stage: NonEmpty
    code: Literal["VALIDATION", "TIMEOUT", "PROVIDER", "INTERNAL"]
    message: NonEmpty
    retryable: bool = False


class ResponseRuleError(ValueError):
    """Only code-owned rule names are passed here; never provider/user strings."""

    def __init__(self, rule: str):
        self.processing_error = ProcessingError(stage="validation", code="VALIDATION",
            message=f"Response rule failed: {rule}", retryable=True)
        super().__init__(rule)


def processing_error(stage: str, error: Exception, *, validating: bool = False) -> ProcessingError:
    """Report categories/field paths, never exception text or patient/provider payloads."""
    code = "VALIDATION" if validating else "INTERNAL"
    detail = "Processing failed"
    if isinstance(error, ValidationError):
        issues = error.errors(include_input=False, include_context=False, include_url=False)
        # Locations can contain arbitrary JSON keys: expose only schema field names/indices.
        allowed = {"facts", "patient_facts", "case_facts", "value", "field", "key", "status", "source", "message_id",
                   "source_text", "case_result", "resolution", "new_problem", "target_case_id", "rag_reason",
                   "need_medical_rag", "pending_bindings", "unavailable_answers", "change_type", "evidence",
                   "doctor_request", "result", "reason", "emergency", "needs_external_evidence"}
        fields = "; ".join(f"{'.'.join(str(p) if isinstance(p, int) or p in allowed else '?' for p in e['loc'])[:120] or 'response'}: {e['type']}"
                           for e in issues[:3])
        code, detail = "VALIDATION", "Invalid structured response: " + fields
    elif hasattr(error, "processing_error"):
        return error.processing_error.model_copy(update={"stage": stage})
    elif validating:
        detail = "Response failed source, case or answer consistency checks"
    logging.getLogger(__name__).warning("stage=%s code=%s exception_type=%s", stage, code, type(error).__name__)
    return ProcessingError(stage=stage, code=code, message=detail, retryable=True)


class RiskSignal(Schema):
    description: NonEmpty
    source: FactSource


class RiskResult(Schema):
    assessment_source: Literal["UNSPECIFIED", "LLM_SCREENING", "RAG", "DOCTOR"] = "UNSPECIFIED"
    evidence_ids: list[NonEmpty] = Field(default_factory=list)
    assessment_status: Literal["NOT_ASSESSED", "ASSESSED", "FAILED"] = "NOT_ASSESSED"
    result: Literal["SAFE", "DANGEROUS"] | None = None
    reason: NonEmpty | None = None
    signals: list[RiskSignal] = Field(default_factory=list)
    emergency: bool = False
    error: ProcessingError | None = None

    @model_validator(mode="after")
    def check_assessment(self) -> Self:
        if self.assessment_status == "ASSESSED":
            if self.result is None or self.reason is None or self.error is not None:
                raise ValueError("Assessment requires result/reason and no processing error")
            if self.result == "DANGEROUS" and not self.signals:
                raise ValueError("DANGEROUS requires sourced signals")
        elif self.result is not None or self.reason is not None or self.signals or self.emergency:
            raise ValueError("Unassessed/failed risk cannot claim clinical conclusions")
        if (self.assessment_status == "FAILED") != (self.error is not None):
            raise ValueError("Only FAILED requires a processing error")
        if self.emergency and self.result != "DANGEROUS":
            raise ValueError("Emergency requires DANGEROUS")
        return self
