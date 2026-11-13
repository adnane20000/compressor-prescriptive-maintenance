"""Décisions locales idempotentes, indépendantes des checkpoints LangGraph."""

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from pydantic import model_validator

from ..diagnostic.schemas import StrictModel


class DecisionRecord(StrictModel):
    run_id: str
    recorded_at_utc: str
    outcome: Literal["closed_without_alert", "approved_documentary_proposal", "rejected", "information_requested"]
    human_validated: bool
    review: dict | None
    proposal_id: str | None
    ml_event: dict | None
    simulation: bool | None
    diagnosis_confirmed: Literal[False] = False
    intervention_authorized: Literal[False] = False
    physical_action_triggered: Literal[False] = False

    @model_validator(mode="after")
    def consistent_decision(self):
        if self.human_validated != (self.outcome == "approved_documentary_proposal"):
            raise ValueError("Seule une approbation humaine documentaire peut être validée.")
        if self.outcome == "closed_without_alert":
            if self.review is not None or self.proposal_id is not None:
                raise ValueError("Une clôture sans alerte ne contient pas de validation humaine.")
        elif self.review is None or self.proposal_id is None:
            raise ValueError("Une décision de revue exige sa soumission et sa proposition.")
        return self


class DecisionStore:
    """Un enregistrement par dossier ; une reprise ne crée pas de deuxième décision."""
    def __init__(self, path: str | Path):
        self.path = Path(path)

    @contextmanager
    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=10)
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("CREATE TABLE IF NOT EXISTS workflow_decisions (run_id TEXT PRIMARY KEY, record_json TEXT NOT NULL)")
            with connection:
                yield connection
        finally:
            connection.close()

    def get(self, run_id: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute("SELECT record_json FROM workflow_decisions WHERE run_id=?", (run_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def save(self, record: DecisionRecord) -> dict:
        data = record.model_dump(mode="json")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute("SELECT record_json FROM workflow_decisions WHERE run_id=?", (record.run_id,)).fetchone()
            if existing:
                previous = json.loads(existing[0])
                without_date = lambda item: {key: value for key, value in item.items() if key != "recorded_at_utc"}
                if without_date(previous) != without_date(data):
                    raise ValueError("Une décision différente existe déjà pour ce dossier.")
                return previous
            connection.execute("INSERT INTO workflow_decisions VALUES (?, ?)",
                               (record.run_id, json.dumps(data, ensure_ascii=False, allow_nan=False)))
        return data
