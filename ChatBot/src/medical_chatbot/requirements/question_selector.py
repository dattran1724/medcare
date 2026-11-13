from collections.abc import Sequence
from medical_chatbot.schemas.requirements import Requirement


def select_questions(requirements: Sequence[Requirement], limit: int = 2) -> list[Requirement]:
    """Prioritize unasked items; clarify unanswered items without inferring refusal."""
    if limit < 1:
        raise ValueError("Question limit must be positive")
    missing = [r for r in requirements if r.applicable and r.blocking and r.status == "MISSING"]
    for priority in ("CRITICAL", "HIGH"):
        tier = [r for r in missing if r.priority == priority]
        if tier:
            unasked = [r for r in tier if r.asked_count == 0]
            return (unasked or sorted(tier, key=lambda r: r.asked_count))[:limit]
    return []
