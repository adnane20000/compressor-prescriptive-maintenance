"""État partagé sérialisable et contrats des entrées et de la revue humaine."""

import math
from datetime import datetime
from typing import Literal, TypedDict

from pydantic import Field, model_validator

from ..diagnostic.schemas import StrictModel
from ..ingestion.schema import REQUIRED_COLUMNS
from ..maintenance.review import HumanReview  # Ancien chemin public conservé.

MAX_SAMPLES = 6000


class WorkflowRequest(StrictModel):
    """Une fenêtre bornée avec contexte passé ; aucune donnée temps réel implicite."""
    samples: list[dict] = Field(min_length=1, max_length=MAX_SAMPLES)
    data_context: Literal["historical_replay", "synthetic_test"]
    question: str = Field(default="Quels documents décrivent Oil_temperature, Motor_current, TP2 et TP3 ?", min_length=5, max_length=4000)

    @model_validator(mode="after")
    def validate_samples(self):
        for row in self.samples:
            if set(row) != set(REQUIRED_COLUMNS):
                raise ValueError("Fournir timestamp et les 15 vrais signaux, sans index exporté.")
            timestamp = row["timestamp"]
            if not isinstance(timestamp, str) or datetime.fromisoformat(timestamp).tzinfo is not None:
                raise ValueError("Les dates MetroPT doivent être des chaînes ISO sans fuseau inventé.")
            for name, value in row.items():
                if name != "timestamp" and value is not None:
                    if type(value) not in (int, float) or not math.isfinite(value):
                        raise ValueError("Les mesures doivent être numériques finies ou absentes (null).")
        return self


class JournalEntry(TypedDict):
    sequence: int
    node: str
    status: str
    timestamp_utc: str


class WorkflowState(TypedDict):
    """Uniquement des primitives JSON ; clients, modèles et clés restent hors état."""
    run_id: str
    request: dict
    phase: str
    route: Literal["normal", "anomaly", "insufficient", "error"] | None
    ml_event: dict | None
    measurements: dict
    monitoring_summary: dict | None
    retrieval: dict | None
    diagnostic: dict | None
    proposal: dict | None
    review: dict | None
    review_error: str | None
    decision: dict | None
    errors: list[dict]
    journal: list[JournalEntry]
