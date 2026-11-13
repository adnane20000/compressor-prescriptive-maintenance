"""Préparation documentaire d'une intervention ; aucune procédure physique générée."""

import hashlib
import json
from typing import Literal

from ..diagnostic.schemas import CHECKS, CheckCode, DiagnosticResult, StrictModel


class PreparationAction(StrictModel):
    code: CheckCode
    description: str
    nature: Literal["revue documentaire ou analyse de données"] = "revue documentaire ou analyse de données"


class MaintenanceProposal(StrictModel):
    proposal_id: str
    run_id: str
    status: Literal["proposition non validée"] = "proposition non validée"
    scope: Literal["préparation documentaire avant intervention"] = "préparation documentaire avant intervention"
    approval_allowed: bool
    simulation: bool | None
    actions: list[PreparationAction]
    documentary_references: list[dict]
    missing_information: list[str]
    blockers: list[str]
    physical_procedure: None = None
    intervention_authorized: Literal[False] = False
    human_validation_required: Literal[True] = True


def prepare_proposal(run_id: str, diagnostic: dict | None, *, blockers: list[str]) -> MaintenanceProposal:
    """Reprendre seulement le catalogue de contrôles ; citer le contexte sans en déduire une procédure."""
    result = DiagnosticResult.model_validate(diagnostic) if diagnostic else None
    allowed = bool(result and result.etat_generation == "ok" and result.sources and not blockers)
    codes = [check.code for check in result.controles_recommandes] if result else ["review_sensor_data", "request_equipment_manual", "request_qualified_safety_review"]
    content = {
        "run_id": run_id, "approval_allowed": allowed, "simulation": result.simulation if result else None,
        "actions": [PreparationAction(code=code, description=CHECKS[code]).model_dump() for code in dict.fromkeys(codes)],
        "documentary_references": [{"source_id": source.id, "file_name": source.file_name, "page": source.page,
                                    "pdf_sha256": source.pdf_sha256, "applicability": source.applicability,
                                    "equipment_reference": source.equipment_reference} for source in result.sources] if result else [],
        "missing_information": [entry.description for entry in result.informations_manquantes] if result else ["Diagnostic documenté indisponible."],
        "blockers": list(dict.fromkeys(blockers + ([] if allowed else ["Dossier insuffisant : approbation documentaire bloquée."]))),
    }
    # L'identifiant lie la décision humaine au contenu exact présenté pour ce dossier.
    identifier = hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    return MaintenanceProposal(proposal_id=identifier, **content)
