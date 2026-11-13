"""Nœuds bornés : ML, recherche, diagnostic, proposition et interruption humaine."""

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from langgraph.types import interrupt
from pydantic import ValidationError

from ..diagnostic.schemas import AnomalyEvent, DiagnosticResult
from ..maintenance.proposal import prepare_proposal
from ..storage.decisions import DecisionRecord, DecisionStore
from .state import HumanReview, WorkflowRequest, WorkflowState

LOGGER = logging.getLogger("metropt.workflow")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class WorkflowPersistenceError(RuntimeError):
    """La finalisation peut être reprise depuis le dernier checkpoint, sans nouveau LLM."""
    def __init__(self, run_id):
        self.run_id = run_id
        super().__init__(f"Décision non enregistrée pour {run_id} : reprendre la finalisation depuis le checkpoint.")


@dataclass
class WorkflowServices:
    """Dépendances applicatives de confiance, exclues de l'état persistant."""
    monitor: Callable
    retrieve: Callable
    diagnose: Callable
    decisions: DecisionStore


class WorkflowNodes:
    def __init__(self, services: WorkflowServices):
        self.services = services

    def _journal(self, state, node, status):
        entry = {"sequence": len(state["journal"]) + 1, "node": node, "status": status, "timestamp_utc": utc_now()}
        LOGGER.info("%s", json.dumps({"run_id": state["run_id"], **entry}, ensure_ascii=False))
        return state["journal"] + [entry]

    def _error(self, state, node, code):
        # Les messages des services peuvent contenir des secrets ou du texte PDF : codes fixes seulement.
        return state["errors"] + [{"node": node, "code": code}]

    def monitor_node(self, state: WorkflowState):
        try:
            monitored = self.services.monitor(WorkflowRequest.model_validate(state["request"]))
            event = AnomalyEvent.model_validate(monitored["event"])
            return {"ml_event": {**monitored["event"], **event.model_dump(mode="json")},
                    "measurements": monitored["measurements"], "monitoring_summary": monitored["summary"],
                    "phase": "monitored", "journal": self._journal(state, "monitor_node", event.status)}
        except Exception:
            return {"phase": "monitor_error", "errors": self._error(state, "monitor_node", "ml_unavailable_or_invalid"),
                    "journal": self._journal(state, "monitor_node", "error")}

    def anomaly_router(self, state: WorkflowState):
        event = state["ml_event"]
        if state["phase"] == "monitor_error":
            route = "error"
        elif not event or event["status"] != "scored":
            route = "insufficient"
        else:
            route = "anomaly" if event["is_anomaly"] else "normal"
        return {"route": route, "journal": self._journal(state, "anomaly_router", route)}

    def retrieval_node(self, state: WorkflowState):
        try:
            result = self.services.retrieve(state["request"]["question"])
            retrieved = result.to_dict()
            ready = retrieved["status"] == "passages_found" and bool(retrieved["passages"]) and not retrieved["safety_blocked"]
            return {"retrieval": retrieved, "phase": "retrieved" if ready else "sources_missing",
                    "journal": self._journal(state, "retrieval_node", "passages_found" if ready else "information_insuffisante")}
        except Exception:
            return {"phase": "retrieval_error", "errors": self._error(state, "retrieval_node", "retrieval_unavailable_or_invalid"),
                    "journal": self._journal(state, "retrieval_node", "error")}

    def diagnosis_node(self, state: WorkflowState):
        try:
            result = self.services.diagnose(state["ml_event"], state["measurements"], state["retrieval"]["passages"])
            validated = DiagnosticResult.model_validate(result.model_dump(mode="json"))
            errors = state["errors"] if validated.etat_generation == "ok" else self._error(state, "diagnosis_node", validated.etat_generation)
            return {"diagnostic": validated.model_dump(mode="json"), "phase": "diagnosed", "errors": errors,
                    "journal": self._journal(state, "diagnosis_node", validated.etat_generation)}
        except Exception:
            return {"phase": "diagnosis_error", "errors": self._error(state, "diagnosis_node", "diagnostic_unavailable_or_invalid"),
                    "journal": self._journal(state, "diagnosis_node", "error")}

    def planning_node(self, state: WorkflowState):
        blockers = [error["code"] for error in state["errors"]]
        if state["route"] in ("insufficient", "error"):
            blockers.append("Mesures ou historique insuffisants pour une analyse documentée.")
        if not state["retrieval"] or not state["retrieval"]["passages"]:
            blockers.append("information insuffisante : aucun passage documentaire exploitable.")
        if state["retrieval"] and state["retrieval"]["safety_blocked"]:
            blockers.append("Demande à risque refusée par les garde-fous.")
        proposal = prepare_proposal(state["run_id"], state["diagnostic"], blockers=blockers)
        journal = self._journal(state, "planning_node", "proposed" if proposal.approval_allowed else "blocked")
        journal = self._journal({**state, "journal": journal}, "human_review_node", "awaiting_review")
        return {"proposal": proposal.model_dump(mode="json"), "phase": "awaiting_human", "journal": journal}

    def human_review_node(self, state: WorkflowState):
        # Aucun effet externe avant interrupt : LangGraph réexécute ce nœud à la reprise.
        submission = interrupt({"run_id": state["run_id"], "proposal": state["proposal"],
                                "ml_event": state["ml_event"], "diagnostic": state["diagnostic"],
                                "diagnostic_status": state["diagnostic"]["etat_generation"] if state["diagnostic"] else None,
                                "allowed_decisions": ["approve", "reject", "request_information"] if state["proposal"]["approval_allowed"] else ["reject", "request_information"],
                                "previous_error": state["review_error"],
                                "notice": "Revue documentaire uniquement. Aucune intervention autorisée ou déclenchée."})
        try:
            review = HumanReview.model_validate(submission)
            if review.proposal_id != state["proposal"]["proposal_id"]:
                raise ValueError("Proposition différente.")
            if review.decision == "approve" and not state["proposal"]["approval_allowed"]:
                raise ValueError("Dossier bloqué.")
        except (ValidationError, ValueError):
            return {"review": None, "review_error": "Soumission invalide, proposition différente ou approbation bloquée.",
                    "phase": "awaiting_human", "journal": self._journal(state, "human_review_node", "invalid_review")}
        return {"review": {"submission": review.model_dump(), "recorded_at_utc": utc_now()},
                "review_error": None, "phase": "reviewed",
                "journal": self._journal(state, "human_review_node", review.decision)}

    def finalize_node(self, state: WorkflowState):
        normal = state["route"] == "normal"
        if normal:
            outcome = "closed_without_alert"
        else:
            review = HumanReview.model_validate(state["review"]["submission"])
            if review.proposal_id != state["proposal"]["proposal_id"] or (review.decision == "approve" and not state["proposal"]["approval_allowed"]):
                raise ValueError("Revue non conforme à la proposition.")
            outcome = {"approve": "approved_documentary_proposal", "reject": "rejected", "request_information": "information_requested"}[review.decision]
        record = DecisionRecord(run_id=state["run_id"], recorded_at_utc=utc_now(), outcome=outcome,
                                human_validated=outcome == "approved_documentary_proposal", review=None if normal else state["review"],
                                proposal_id=None if normal else state["proposal"]["proposal_id"], ml_event=state["ml_event"],
                                simulation=state["diagnostic"]["simulation"] if state["diagnostic"] else None)
        try:
            saved = self.services.decisions.save(record)
        except (sqlite3.Error, OSError, ValueError):
            LOGGER.error("workflow_finalization_failed run_id=%s code=decision_not_persisted", state["run_id"])
            raise WorkflowPersistenceError(state["run_id"]) from None
        return {"decision": saved, "phase": "finalized", "journal": self._journal(state, "finalize_node", outcome)}
