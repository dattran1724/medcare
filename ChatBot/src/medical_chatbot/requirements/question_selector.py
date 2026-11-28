from collections.abc import Sequence
from medical_chatbot.schemas.requirements import Requirement


def select_questions(requirements: Sequence[Requirement], limit: int = 2) -> list[Requirement]:
    """Follow up unanswered questions before moving to a new requirement."""
    if limit < 1:
        raise ValueError("Question limit must be positive")
    missing = [r for r in requirements if r.applicable and r.blocking and r.status == "MISSING"]
    for priority in ("CRITICAL", "HIGH"):
        tier = [r for r in missing if r.priority == priority]
        if tier:
            pending = [r for r in tier if r.asked_count > 0]
            if pending:
                return sorted(pending, key=lambda r: r.asked_count)[:limit]
            return tier[:limit]
    return []
