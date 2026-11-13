"""Décisions locales idempotentes, indépendantes des checkpoints LangGraph."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import field_validator, model_validator

from ..diagnostic.schemas import StrictModel
from ..maintenance.review import REVIEW_OUTCOMES, ReviewReceipt, validate_utc_timestamp


class DecisionRecord(StrictModel):
    run_id: str
    recorded_at_utc: str
    outcome: Literal["closed_without_alert", "approved_documentary_proposal", "rejected", "revision_requested", "information_requested"]
    human_validated: bool
    review: dict | None
    proposal_id: str | None
    ml_event: dict | None
    simulation: bool | None
    validator_id: str | None = None
    comments: str | None = None
    reviewed_at_utc: str | None = None
    diagnosis_confirmed: Literal[False] = False
    intervention_authorized: Literal[False] = False
    physical_action_triggered: Literal[False] = False

    @field_validator("recorded_at_utc")
    @classmethod
    def utc_timestamp(cls, value):
        return validate_utc_timestamp(value)

    @model_validator(mode="after")
    def consistent_decision(self):
        if self.human_validated != (self.outcome == "approved_documentary_proposal"):
            raise ValueError("Seule une approbation humaine documentaire peut être validée.")
        if self.outcome == "closed_without_alert":
            if any(value is not None for value in (self.review, self.proposal_id, self.validator_id, self.comments, self.reviewed_at_utc)):
                raise ValueError("Une clôture sans alerte ne contient pas de validation humaine.")
        elif self.review is None or self.proposal_id is None:
            raise ValueError("Une décision de revue exige sa soumission et sa proposition.")
        else:
            receipt = ReviewReceipt.model_validate(self.review)
            submission = receipt.submission
            if submission.proposal_id != self.proposal_id or REVIEW_OUTCOMES[submission.decision] != self.outcome:
                raise ValueError("La décision doit correspondre à la soumission et à la proposition revues.")
            if datetime.fromisoformat(self.recorded_at_utc) < datetime.fromisoformat(receipt.recorded_at_utc):
                raise ValueError("La décision ne peut pas précéder la soumission humaine.")
            derived = {"validator_id": submission.reviewer, "comments": submission.comment, "reviewed_at_utc": receipt.recorded_at_utc}
            for name, value in derived.items():
                if getattr(self, name) is not None and getattr(self, name) != value:
                    raise ValueError("Les métadonnées de validation doivent correspondre à la revue explicite.")
                setattr(self, name, value)
            self.review = receipt.model_dump(mode="json")
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
            connection.execute("CREATE TABLE IF NOT EXISTS workflow_decisions (run_id TEXT PRIMARY KEY, record_json TEXT NOT NULL, decision TEXT, decided_at_utc TEXT, validator_id TEXT, comments TEXT)")
            columns = {row[1] for row in connection.execute("PRAGMA table_info(workflow_decisions)")}
            missing = {"decision", "decided_at_utc", "validator_id", "comments"} - columns
            if missing:
                # Migration transactionnelle des décisions précédentes, sans modifier leur contenu métier.
                with connection:
                    connection.execute("BEGIN IMMEDIATE")
                    for name in sorted(missing):
                        connection.execute(f"ALTER TABLE workflow_decisions ADD COLUMN {name} TEXT")
                    for run_id, raw in connection.execute("SELECT run_id, record_json FROM workflow_decisions"):
                        record = DecisionRecord.model_validate_json(raw)
                        if record.run_id != run_id:
                            raise ValueError("Identifiant incohérent dans une décision locale.")
                        connection.execute("UPDATE workflow_decisions SET decision=?, decided_at_utc=?, validator_id=?, comments=? WHERE run_id=?",
                                           (record.outcome, record.recorded_at_utc, record.validator_id, record.comments, run_id))
            with connection:
                yield connection
        finally:
            connection.close()

    def get(self, run_id: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute("SELECT record_json, decision, decided_at_utc, validator_id, comments FROM workflow_decisions WHERE run_id=?", (run_id,)).fetchone()
        if not row:
            return None
        record = DecisionRecord.model_validate_json(row[0])
        if record.run_id != run_id or (record.outcome, record.recorded_at_utc, record.validator_id, record.comments) != row[1:]:
            raise ValueError("Métadonnées SQLite incohérentes avec la décision validée.")
        return record.model_dump(mode="json")

    def save(self, record: DecisionRecord) -> dict:
        # model_copy()/model_construct() ne valident pas : revérifier avant toute écriture.
        record = DecisionRecord.model_validate(record.model_dump(mode="json"))
        data = record.model_dump(mode="json")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute("SELECT record_json FROM workflow_decisions WHERE run_id=?", (record.run_id,)).fetchone()
            if existing:
                previous = DecisionRecord.model_validate_json(existing[0]).model_dump(mode="json")
                without_date = lambda item: {key: value for key, value in item.items() if key != "recorded_at_utc"}
                if without_date(previous) != without_date(data):
                    raise ValueError("Une décision différente existe déjà pour ce dossier.")
                return previous
            connection.execute("INSERT INTO workflow_decisions (run_id, record_json, decision, decided_at_utc, validator_id, comments) VALUES (?, ?, ?, ?, ?, ?)",
                               (record.run_id, json.dumps(data, ensure_ascii=False, allow_nan=False), record.outcome,
                                record.recorded_at_utc, record.validator_id, record.comments))
        return data
