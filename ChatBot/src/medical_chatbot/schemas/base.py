from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints

NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Schema(BaseModel):
    """Reject undeclared fields at every external contract boundary."""

    model_config = ConfigDict(extra="forbid", validate_default=True, allow_inf_nan=False)
