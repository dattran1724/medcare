"""Clinical data follows the agreed grouped JSON structure."""
from typing import Literal, Self
from pydantic import AwareDatetime, Field, JsonValue, StrictInt, model_validator
from .base import NonEmpty, Schema
from .requirements import Requirement

class PatientInfo(Schema):
    age: StrictInt | None = Field(default=None, ge=0, le=130)
    sex: NonEmpty | None = None
    pregnancy_status: NonEmpty | None = None

class MedicalHistory(Schema):
    conditions: list[NonEmpty] = Field(default_factory=list)
    surgical_history: list[NonEmpty] = Field(default_factory=list)
    current_medications: list[NonEmpty] = Field(default_factory=list)
    allergies: list[NonEmpty] = Field(default_factory=list)
    family_history: list[NonEmpty] = Field(default_factory=list)

class PresentIllness(Schema):
    onset: NonEmpty | None = Field(default=None, description="When symptoms first started, not sudden versus gradual onset")
    duration: NonEmpty | None = Field(default=None, description="Overall duration of the problem, not the length of each episode")
    episode_duration: NonEmpty | None = None
    frequency: NonEmpty | None = None
    course: NonEmpty | None = Field(default=None, description="Evolution over time: improving, worsening, unchanged; not frequency")
    location: NonEmpty | None = None
    character: NonEmpty | None = None
    severity: NonEmpty | None = None
    associated_symptoms: list[NonEmpty] = Field(default_factory=list)
    aggravating_factors: list[NonEmpty] = Field(default_factory=list)
    relieving_factors: list[NonEmpty] = Field(default_factory=list)

class ClinicalData(Schema):
    vitals: dict[str, JsonValue] = Field(default_factory=dict)
    lab_results: list[JsonValue] = Field(default_factory=list)
    imaging: list[JsonValue] = Field(default_factory=list)
    other: list[JsonValue] = Field(default_factory=list)

class ProgressEvent(Schema):
    field: NonEmpty
    key: NonEmpty | None = None
    old_value: JsonValue
    new_value: JsonValue
    time: AwareDatetime
    type: Literal["TEMPORAL_CHANGE", "CORRECTION"]

class CaseRisk(Schema):
    level: Literal["NOT_ASSESSED", "SAFE", "DANGEROUS"] = "NOT_ASSESSED"
    danger_signs: list[NonEmpty] = Field(default_factory=list)
    requires_emergency_care: bool = False
    requires_in_person_evaluation: bool = False

    @model_validator(mode="after")
    def check_risk(self) -> Self:
        if self.level != "DANGEROUS" and (self.danger_signs or self.requires_emergency_care or self.requires_in_person_evaluation):
            raise ValueError("Danger signals/escalation require DANGEROUS")
        if self.level == "DANGEROUS" and not self.danger_signs:
            raise ValueError("DANGEROUS requires danger signals")
        return self

class Workflow(Schema):
    status: Literal["COLLECTING", "WAITING_AGENT_2", "WAITING_DOCTOR", "CONSULTING", "COMPLETED"] = "COLLECTING"
    next_action: Literal["ASK_MORE", "PASS_TO_AGENT_2", "HANDOFF_DOCTOR", "EMERGENCY"] | None = None


class DoctorHandoffContent(Schema):
    summary: NonEmpty
    conclusion: NonEmpty | None = None
    instructions: NonEmpty | None = None
    follow_up: NonEmpty | None = None


class DoctorHandoff(DoctorHandoffContent):
    doctor_id: NonEmpty
    source_message_ids: list[NonEmpty] = Field(min_length=1)
    confirmed_at: AwareDatetime
    action: Literal["RETURN_TO_AI", "CLOSE_CASE"]


class ClinicalCase(Schema):
    case_id: NonEmpty
    chief_complaint: NonEmpty | None = None
    status: Literal["ACTIVE", "RESOLVED"] = "ACTIVE"
    response_mode: Literal["AI_ACTIVE", "DOCTOR_ACTIVE"] = "AI_ACTIVE"
    present_illness: PresentIllness = Field(default_factory=PresentIllness)
    clinical_data: ClinicalData = Field(default_factory=ClinicalData)
    progress_history: list[ProgressEvent] = Field(default_factory=list)
    risk_assessment: CaseRisk = Field(default_factory=CaseRisk)
    missing_information: list[Requirement] = Field(default_factory=list)
    workflow: Workflow = Field(default_factory=Workflow)
    doctor_handoff: DoctorHandoff | None = None

    @model_validator(mode="after")
    def check_requirements(self) -> Self:
        if any(r.target_case_id not in (None, self.case_id) for r in self.missing_information):
            raise ValueError("Requirement belongs to a different case")
        return self

class ClinicalState(Schema):
    patient_info: PatientInfo = Field(default_factory=PatientInfo)
    medical_history: MedicalHistory = Field(default_factory=MedicalHistory)
    current_case_id: NonEmpty | None = None
    cases: list[ClinicalCase] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_cases(self) -> Self:
        ids = [case.case_id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate case IDs")
        if self.current_case_id is not None and self.current_case_id not in ids:
            raise ValueError("Current case must exist")
        return self
