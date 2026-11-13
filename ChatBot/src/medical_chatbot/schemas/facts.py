from enum import StrEnum
from typing import Self

from pydantic import AwareDatetime, JsonValue, model_validator
from pydantic_core import PydanticCustomError

from .base import NonEmpty, Schema


class FactStatus(StrEnum):
    PRESENT = "PRESENT"
    ABSENT = "ABSENT"
    UNCERTAIN = "UNCERTAIN"


class FactField(StrEnum):
    AGE = "patient_info.age"
    SEX = "patient_info.sex"
    PREGNANCY = "patient_info.pregnancy_status"
    CONDITIONS = "medical_history.conditions"
    SURGERIES = "medical_history.surgical_history"
    MEDICATIONS = "medical_history.current_medications"
    ALLERGIES = "medical_history.allergies"
    FAMILY_HISTORY = "medical_history.family_history"
    CHIEF_COMPLAINT = "chief_complaint"
    ONSET = "present_illness.onset"
    DURATION = "present_illness.duration"
    EPISODE_DURATION = "present_illness.episode_duration"
    FREQUENCY = "present_illness.frequency"
    COURSE = "present_illness.course"
    LOCATION = "present_illness.location"
    CHARACTER = "present_illness.character"
    SEVERITY = "present_illness.severity"
    ASSOCIATED_SYMPTOMS = "present_illness.associated_symptoms"
    AGGRAVATING_FACTORS = "present_illness.aggravating_factors"
    RELIEVING_FACTORS = "present_illness.relieving_factors"
    VITALS = "clinical_data.vitals"
    LABS = "clinical_data.lab_results"
    IMAGING = "clinical_data.imaging"
    OTHER = "clinical_data.other"

    @property
    def is_case_field(self) -> bool:
        return not self.value.startswith(("patient_info.", "medical_history."))

    @property
    def is_scalar(self) -> bool:
        return self.value.startswith("patient_info.") or self in {
            self.CHIEF_COMPLAINT, self.ONSET, self.DURATION, self.EPISODE_DURATION,
            self.FREQUENCY, self.COURSE, self.LOCATION, self.CHARACTER, self.SEVERITY}


# Shared by extraction and question wording: one meaning per field.
INTAKE_FIELD_GUIDANCE = {
    "present_illness.onset": "When it started. 'Bắt đầu một tuần trước' answers onset; do not ask sudden/gradual instead.",
    "present_illness.duration": "Overall duration of the problem, e.g. 'bị trong một tuần'; never duration of each episode.",
    "present_illness.episode_duration": "Length of each episode, e.g. 'mỗi lần vài phút'.",
    "present_illness.frequency": "How often episodes occur, e.g. 'thỉnh thoảng', 'ba lần mỗi ngày'.",
    "present_illness.course": "Change over time, e.g. 'không thay đổi', 'nặng dần'; not intermittent frequency.",
}


class FactSource(Schema):
    message_id: NonEmpty
    source_text: NonEmpty


class Fact(Schema):
    fact_id: NonEmpty
    field: FactField
    key: NonEmpty
    value: JsonValue = None
    status: FactStatus
    source: FactSource
    target_case_id: NonEmpty | None = None
    observed_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def check_scope_and_value(self) -> Self:
        if self.field.is_case_field != (self.target_case_id is not None):
            raise PydanticCustomError("fact_scope_mismatch", "Case facts require a target; patient facts cannot have one")
        if self.status == FactStatus.PRESENT and self.value is None:
            raise PydanticCustomError("present_fact_missing_value", "PRESENT requires a value")
        if self.status == FactStatus.ABSENT and self.value is not None:
            raise PydanticCustomError("absent_fact_has_value", "ABSENT uses key to name the denied fact and value=null")
        if self.field == FactField.AGE and self.value is not None:
            if type(self.value) is not int or not 0 <= self.value <= 130:
                raise PydanticCustomError("invalid_age", "Age must be an integer between 0 and 130")
        text_fields = {
            FactField.SEX, FactField.PREGNANCY, FactField.CONDITIONS,
            FactField.SURGERIES, FactField.MEDICATIONS, FactField.ALLERGIES,
            FactField.FAMILY_HISTORY, FactField.CHIEF_COMPLAINT, FactField.ONSET,
            FactField.DURATION, FactField.EPISODE_DURATION, FactField.FREQUENCY,
            FactField.COURSE, FactField.LOCATION, FactField.CHARACTER,
            FactField.SEVERITY, FactField.ASSOCIATED_SYMPTOMS,
            FactField.AGGRAVATING_FACTORS, FactField.RELIEVING_FACTORS,
        }
        if self.field in text_fields and self.value is not None:
            if not isinstance(self.value, str) or not self.value.strip():
                raise PydanticCustomError("fact_requires_text", "This field requires a nonempty textual value per fact")
        return self
