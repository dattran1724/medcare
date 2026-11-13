"""Pure intake updates. Callers retain the returned facts/conflicts between turns."""

from collections.abc import Sequence
from uuid import uuid4
import re

from pydantic import Field, JsonValue

from medical_chatbot.schemas.base import NonEmpty, Schema
from medical_chatbot.schemas.clinical_state import ClinicalCase, ClinicalState, ProgressEvent
from medical_chatbot.schemas.conversation import Message
from medical_chatbot.schemas.facts import Fact, FactStatus
from medical_chatbot.schemas.outputs import FactProposal, InputProcessorOutput


class RejectedFact(Schema):
    proposal: FactProposal
    reason: NonEmpty


class PendingConflict(Schema):
    proposed: Fact
    old_value: JsonValue
    reason: NonEmpty


class UpdateResult(Schema):
    state: ClinicalState
    facts: list[Fact] = Field(default_factory=list)
    conflicts: list[PendingConflict] = Field(default_factory=list)
    pending: list[FactProposal] = Field(default_factory=list)
    rejected: list[RejectedFact] = Field(default_factory=list)


def _location(state: ClinicalState, fact: Fact):
    root = state
    if fact.target_case_id is not None:
        root = next(c for c in state.cases if c.case_id == fact.target_case_id)
    parts = fact.field.value.split(".")
    for part in parts[:-1]:
        root = getattr(root, part)
    return root, parts[-1]


def _identity(fact: Fact, collection: bool):
    return fact.target_case_id, fact.field, fact.key if collection else None


def _has_change_evidence(proposal: FactProposal) -> bool:
    """Conservative explicit cues, not an inference from arrival order."""
    text = proposal.source.source_text.casefold()
    cues = {
        "CORRECTION": ("correction", "i meant", "nhầm", "sửa lại"),
        "TEMPORAL_CHANGE": ("now", "today", "earlier", "bây giờ", "hiện tại", "hôm nay", "đã hết"),
    }
    return any(re.search(r"(?<!\w)" + re.escape(cue) + r"(?!\w)", text)
               for cue in cues.get(proposal.change_type, ()))


def update_clinical_state(
    state: ClinicalState,
    message: Message,
    extracted: InputProcessorOutput,
    *,
    facts: Sequence[Fact] = (),
    conflicts: Sequence[PendingConflict] = (),
) -> UpdateResult:
    """Return a new state; never mutate input. No extraction, storage or risk assessment.

    ``facts`` and ``conflicts`` must be the metadata returned by the previous call.
    Raw untrusted data must first pass Message/InputProcessorOutput.model_validate.
    Source matching validates provenance, not the medical meaning of the text.
    """
    state = ClinicalState.model_validate(state.model_dump())
    message = Message.model_validate(message.model_dump())
    extracted = InputProcessorOutput.model_validate(extracted.model_dump())
    if message.role != "USER":
        raise ValueError("Intake updates require a USER message")
    result = UpdateResult(state=state, facts=[f.model_copy(deep=True) for f in facts],
                          conflicts=[c.model_copy(deep=True) for c in conflicts])
    resolution = extracted.case_result
    target = None
    if resolution.resolution == "OLD_CASE":
        target = next((c for c in state.cases if c.case_id == resolution.target_case_id), None)
        if target is None:
            raise ValueError("Target case does not belong to this state")
    if set(message.case_ids) - {c.case_id for c in state.cases}:
        raise ValueError("Message refers to an unknown case")
    unavailable = []
    for answer in extracted.unavailable_answers:
        if (target is None or answer.target_case_id not in (None, target.case_id)
                or answer.source.message_id != message.message_id
                or answer.source.source_text not in message.content):
            raise ValueError("Unavailable answer has an invalid source or case")
        matches = [r for r in target.missing_information if r.key == answer.key
                   and r.target_case_id == answer.target_case_id]
        if len(matches) != 1:
            raise ValueError("Unavailable answer must identify a missing requirement")
        if any(p.field == matches[0].target and p.key == answer.key
               for p in extracted.patient_facts + extracted.case_facts):
            raise ValueError("Cannot supply a fact and an unavailable answer for the same requirement")
        if answer.key in {a.key for a in unavailable}:
            raise ValueError("Duplicate unavailable answer")
        unavailable.append(answer)

    # Check sources before creating a case or changing any patient data.
    valid = []
    for proposal in extracted.patient_facts + extracted.case_facts:
        if (proposal.source.message_id != message.message_id
                or proposal.source.source_text not in message.content):
            result.rejected.append(RejectedFact(proposal=proposal, reason="Source does not match this message"))
        else:
            valid.append(proposal)
    if resolution.resolution == "NEW_CASE" and any(p.field.is_case_field for p in valid):
        if resolution.new_problem not in message.content:
            raise ValueError("New problem must be an excerpt of the message")
        target = ClinicalCase(case_id=f"case_{uuid4().hex}")
        state.cases.append(target)
        if not any(p.field.value == "chief_complaint" for p in valid):
            valid.insert(0, FactProposal(field="chief_complaint", key="chief_complaint",
                value=resolution.new_problem, status="PRESENT",
                source={"message_id": message.message_id, "source_text": resolution.new_problem}))
    if target is not None:
        state.current_case_id = target.case_id

    for proposal in valid:
        if proposal.field.is_case_field and target is None:
            result.pending.append(proposal)
            continue
        fact = Fact(fact_id=f"fact_{uuid4().hex}", field=proposal.field, key=proposal.key,
                    value=proposal.value, status=proposal.status, source=proposal.source,
                    target_case_id=target.case_id if proposal.field.is_case_field else None,
                    observed_at=message.timestamp)
        if fact.status == FactStatus.UNCERTAIN:
            result.pending.append(proposal)
            continue
        parent, name = _location(state, fact)
        current = getattr(parent, name)
        collection = isinstance(current, (list, dict))
        identity = _identity(fact, collection)
        previous = next((f for f in reversed(result.facts) if _identity(f, collection) == identity), None)
        pending_conflicts = [c for c in result.conflicts if _identity(c.proposed, collection) == identity]
        if pending_conflicts and not _has_change_evidence(proposal):
            result.pending.append(proposal)
            continue
        # Reject non-scalar vitals and nested control fields hidden in open data.
        if isinstance(current, dict) and fact.status == FactStatus.PRESENT:
            if fact.key in {"workflow", "response_mode", "risk_assessment", "case_id", "status", "doctor_handoff"}:
                result.rejected.append(RejectedFact(proposal=proposal, reason="Control keys are not clinical data"))
                continue
            if not isinstance(fact.value, (str, int, float)) or isinstance(fact.value, bool):
                result.rejected.append(RejectedFact(proposal=proposal, reason="A vital must be one scalar value"))
                continue
        if isinstance(fact.value, (list, dict)):
            result.rejected.append(RejectedFact(proposal=proposal, reason="Open clinical entries must be scalar; nested patches are not allowed"))
            continue
        old = (current.get(fact.key) if isinstance(current, dict) else
               previous.value if isinstance(current, list) and previous else
               None if isinstance(current, list) else current)
        changed = previous is not None and (previous.status != fact.status or previous.value != fact.value)
        changed = changed or (old is not None and old != fact.value)
        if changed and previous and previous.observed_at and fact.observed_at < previous.observed_at:
            result.conflicts.append(PendingConflict(proposed=fact, old_value=old,
                reason="Older message cannot overwrite the latest accepted value"))
            continue
        if isinstance(current, list) and fact.status == FactStatus.ABSENT and previous is None and current:
            # A plain list has no semantic keys: never guess which element to remove.
            old = list(current)
            result.conflicts.append(PendingConflict(proposed=fact, old_value=old,
                reason="Cannot remove an untracked list entry; supply its original fact metadata"))
            continue
        if changed and not _has_change_evidence(proposal):
            result.conflicts.append(PendingConflict(proposed=fact, old_value=old,
                                                    reason="Change requires explicit correction or temporal evidence"))
            continue
        if isinstance(current, list):
            updated = list(current)
            if changed and previous and previous.value in updated:
                updated.remove(previous.value)
            if fact.status == FactStatus.PRESENT and fact.value not in updated:
                updated.append(fact.value)
        elif isinstance(current, dict):
            updated = dict(current)
            if fact.status == FactStatus.ABSENT:
                updated.pop(fact.key, None)
            else:
                updated[fact.key] = fact.value
        else:
            updated = fact.value if fact.status == FactStatus.PRESENT else None
        # Validate on a candidate before accepting any field or metadata update.
        candidate = state.model_copy(deep=True)
        candidate_parent, candidate_name = _location(candidate, fact)
        setattr(candidate_parent, candidate_name, updated)
        candidate = ClinicalState.model_validate(candidate.model_dump())
        state = candidate
        result.state = state
        if fact.target_case_id:
            case = next(c for c in state.cases if c.case_id == fact.target_case_id)
            case.status = "ACTIVE"
            if changed:
                case.progress_history.append(ProgressEvent(
                    field=fact.field.value, key=fact.key, old_value=old, new_value=fact.value,
                    time=message.timestamp, type=proposal.change_type))
        # New medical input invalidates an earlier SAFE conclusion, including
        # patient-level updates. Never downgrade DANGEROUS or change doctor mode.
        for case in state.cases:
            if case.risk_assessment.level == "SAFE":
                case.risk_assessment.level = "NOT_ASSESSED"
        result.facts.append(fact)
        result.conflicts = [c for c in result.conflicts if _identity(c.proposed, collection) != identity]
    for answer in unavailable:
        case = next(c for c in state.cases if c.case_id == resolution.target_case_id)
        requirement = next(r for r in case.missing_information if r.key == answer.key
                           and r.target_case_id == answer.target_case_id)
        requirement.status = answer.status
    result.state = ClinicalState.model_validate(state.model_dump())
    return result
