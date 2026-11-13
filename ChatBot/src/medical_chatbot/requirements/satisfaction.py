"""Match the requested fact/key, not just a nonempty target collection."""
from collections.abc import Sequence
from medical_chatbot.schemas.clinical_state import ClinicalState
from medical_chatbot.schemas.facts import Fact
from medical_chatbot.schemas.requirements import Requirement

CORE_FIELDS = {
    "chief_complaint": "chief_complaint",
    "onset": "present_illness.onset",
    "duration": "present_illness.duration",
    "course": "present_illness.course",
}


def is_satisfied(state: ClinicalState, requirement: Requirement, facts: Sequence[Fact]) -> bool:
    value = state
    if requirement.target_case_id:
        value = next(c for c in state.cases if c.case_id == requirement.target_case_id)
    for part in requirement.target.value.split("."):
        value = getattr(value, part)
    match = next((f for f in reversed(facts)
                  if f.field == requirement.target and (f.field.is_scalar or f.key == requirement.key)
                  and f.target_case_id == requirement.target_case_id), None)
    if match:
        if match.status == "UNCERTAIN":
            return False
        if match.status == "ABSENT":
            return isinstance(value, (list, dict))
        if isinstance(value, list):
            return match.value in value
        if isinstance(value, dict):
            return value.get(requirement.key) == match.value
        return value is not None and value == match.value
    # Keep this fallback for older persisted sessions without matching fact metadata.
    # It is not redundant: new field/key normalization does not rewrite old records.
    return (requirement.source == "CORE"
            and CORE_FIELDS.get(requirement.key) == requirement.target.value
            and value is not None and not isinstance(value, (list, dict)))
