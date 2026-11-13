"""Contrat commun de revue humaine, vérifié par le graphe et la persistance."""

from datetime import datetime, timedelta
from typing import Literal

from pydantic import Field, field_validator

from ..diagnostic.schemas import StrictModel

REVIEW_OUTCOMES = {
    "approve": "approved_documentary_proposal",
    "reject": "rejected",
    "request_revision": "revision_requested",
    "request_information": "information_requested",  # Compatibilité avec les dossiers existants.
}


def validate_utc_timestamp(value: str) -> str:
    """Exiger un horodatage serveur explicite en UTC, sans modifier les dates capteurs."""
    timestamp = datetime.fromisoformat(value)
    if timestamp.utcoffset() != timedelta(0):
        raise ValueError("Un horodatage UTC explicite est requis.")
    return value


class HumanReview(StrictModel):
    """Une soumission explicite liée à la proposition affichée, jamais produite par le LLM."""
    proposal_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    decision: Literal["approve", "reject", "request_revision", "request_information"]
    reviewer: str = Field(min_length=2, max_length=200)
    comment: str = Field(min_length=1, max_length=2000)
    acknowledged: Literal[True]

    @field_validator("reviewer", "comment")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("Identifiant du validateur et commentaire doivent être renseignés.")
        return value.strip()

    @field_validator("acknowledged", mode="before")
    @classmethod
    def explicit_acknowledgment(cls, value):
        if value is not True:
            raise ValueError("Une confirmation explicite est requise.")
        return value


class ReviewReceipt(StrictModel):
    submission: HumanReview
    recorded_at_utc: str

    @field_validator("recorded_at_utc")
    @classmethod
    def utc_timestamp(cls, value):
        return validate_utc_timestamp(value)
