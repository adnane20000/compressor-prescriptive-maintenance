"""Ablations hors ligne et dix scénarios, sans appel réseau ni action physique.

A et B sont des adaptateurs d'évaluation : mêmes mesures et contrat de sélection,
mais sans routage ML, proposition ou stockage de revue. C utilise le vrai graphe.
Le moniteur synthétique injecté teste le routage, jamais la performance du ML.
"""

import json
import platform
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import partial
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np
import pymupdf
from pydantic import ValidationError

from ..diagnostic import generate_diagnostic
from ..diagnostic.grounding import GroundingError, normalize_measurements, verify_sources
from ..diagnostic.providers import LLMApiError, LLMResponseError, MockProvider
from ..diagnostic.schemas import DiagnosticDraft, ObservedFact
from ..diagnostic.service import _assemble_draft
from ..ingestion.schema import SIGNAL_NAMES
from ..orchestration import open_workflow
from ..orchestration.nodes import WorkflowServices
from ..rag import search_documents
from ..rag.documents import file_sha256
from ..rag.index import index_documents
from ..storage.decisions import DecisionStore

CONFIGURATIONS = {"A": "LLM seul — adaptateur sans ML ni RAG", "B": "LLM + RAG — sans graphe ni revue persistante",
                  "C": "ML + RAG + LangGraph + revue humaine"}


@dataclass(frozen=True)
class Scenario:
    identifier: str
    label: str
    requirement: str
    question: str = "Que decrit Motor_current dans le cas exploration ?"
    normal: bool = False
    review: str | None = None


SCENARIOS = (
    Scenario("normal", "Fonctionnement normal simulé", "normal_without_llm", normal=True),
    Scenario("anomaly", "Anomalie simulée", "documented_response"),
    Scenario("air_leak", "Suspicion de fuite, non confirmée", "documented_response", question="Que decrit Motor_current dans le cas exploration avec une suspicion de fuite d'air ?"),
    Scenario("no_documents", "Documentation pertinente absente", "honest_abstention", question="Quel est le couple de serrage exact de la culasse MetroPT-3 ?"),
    Scenario("contradiction", "Deux documents pédagogiques contradictoires", "conflict_exposed", question="Que decrit TP3 dans le cas divergence ?"),
    Scenario("api_failure", "Erreur API injectée", "api_failure_visible"),
    Scenario("uncertain", "Diagnostic incertain", "uncertainty_explicit"),
    Scenario("human_rejection", "Refus humain simulé", "rejection_recorded", review="reject"),
    Scenario("human_approval", "Approbation humaine simulée", "explicit_approval_only", review="approve"),
    Scenario("resume", "Reprise d'une interruption persistante", "resume_without_recomputation", review="request_revision"),
)


class CountingMock(MockProvider):
    """Compter les appels mock ; injecter un échec sans jamais contacter une API."""
    def __init__(self, *, fail=False):
        self.calls, self.fail = 0, fail

    def complete(self, payload):
        self.calls += 1
        if self.fail:
            raise LLMApiError("Échec API injecté — simulation, aucun appel distant.")
        return super().complete(payload)


class FixtureEncoder:
    """Vecteurs constants SYNTHÉTIQUES : exercice du stockage et des filtres seulement."""
    signature = {"model": "evaluation-synthetic-constant-v1", "dimension": 4}

    def encode_documents(self, texts):
        return np.tile(np.array([1, 0, 0, 0], dtype="float32"), (len(texts), 1))

    def encode_query(self, question):
        return np.array([1, 0, 0, 0], dtype="float32")


def create_fixture_corpus(directory):
    """Écrire des cas pédagogiques, sans source officielle ni procédure inventée."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    texts = {
        "cas_SYNTHETIQUE.pdf": "SIMULATION PEDAGOGIQUE exploration. Motor_current et Oil_temperature sont des mesures. Dans ce cas fictif une fuite d'air reste une hypothese non confirmee. Aucune procedure physique ni applicabilite MetroPT n'est etablie.",
        "divergence_A_SYNTHETIQUE.pdf": "SIMULATION PEDAGOGIQUE divergence. Dans le meme cas fictif, une baisse de TP3 implique une fuite d'air. Cette affirmation est une donnee de test contradictoire, pas un savoir industriel.",
        "divergence_B_SYNTHETIQUE.pdf": "SIMULATION PEDAGOGIQUE divergence. Dans le meme cas fictif, une baisse de TP3 exclut une fuite d'air. Cette affirmation est une donnee de test contradictoire, pas un savoir industriel.",
    }
    registry = {"version": 1, "documents": {}}
    for name, text in texts.items():
        with pymupdf.open() as document:
            page = document.new_page()
            page.insert_textbox(pymupdf.Rect(40, 40, 550, 750), text, fontsize=11)
            document.save(root / name)
        registry["documents"][name] = {"authority": "pedagogical", "document_type": "educational_note",
            "applicability": "general_reference", "sha256": file_sha256(root / name),
            "publisher": "Fixture synthétique de campagne — aucune provenance industrielle"}
    (root / "sources.json").write_text(json.dumps(registry, ensure_ascii=False), encoding="utf-8")
    return root


def measurement_facts(measurements):
    """Construire les faits sans score ML pour les deux ablations A et B."""
    return [ObservedFact(id=f"F{i}", origin="measurement", description="Mesure fournie pour l'évaluation.",
                        value=measure.value, sensor=name, unit=measure.unit, aggregation=measure.aggregation)
            for i, (name, measure) in enumerate(normalize_measurements(measurements).items(), 1) if measure.value is not None]


def run_ablation(measurements, question, passages, *, documents_directory, provider, configuration):
    """Appliquer le même fournisseur et l'assembleur validé, sans créer d'événement ML.

    Cette enveloppe d'évaluation conserve les observations indépendamment du LLM.
    Elle ne devient pas une nouvelle API de diagnostic de l'application.
    """
    if configuration not in ("A", "B"):
        raise ValueError("Ablation A ou B attendue.")
    facts = measurement_facts(measurements)
    sources = verify_sources(passages, documents_directory=documents_directory) if configuration == "B" else []
    result = {"etat_generation": "information_insuffisante", "faits_observes": [f.model_dump() for f in facts],
              "sources": [s.model_dump() for s in sources], "arguments_documentaires": [], "hypotheses_de_defaillance": [],
              "niveau_incertitude": "très élevé", "informations_manquantes": ["documentary_evidence"] if not sources else [],
              "diagnostic_confirme": False, "autorisation_intervention": False, "simulation": provider.simulation}
    if configuration == "B" and not sources:
        return result
    try:
        raw = provider.complete({"question": question, "event": None, "facts": [f.model_dump() for f in facts],
                                 "sources": [s.model_dump() for s in sources]})
        if not isinstance(raw, str) or len(raw) > 32_000:
            raise LLMResponseError("Réponse invalide.")
        draft = DiagnosticDraft.model_validate_json(raw)
        arguments, hypotheses = _assemble_draft(draft, sources, facts)
        result.update(etat_generation="ok" if sources else "information_insuffisante",
                      arguments_documentaires=[a.model_dump() for a in arguments],
                      hypotheses_de_defaillance=[h.model_dump() for h in hypotheses],
                      niveau_incertitude=draft.uncertainty,
                      informations_manquantes=list(dict.fromkeys(result["informations_manquantes"] + draft.missing_information)))
    except LLMApiError:
        result["etat_generation"] = "erreur_api"
    except (LLMResponseError, ValidationError, GroundingError):
        result["etat_generation"] = "reponse_invalide"
    return result


def audit_diagnostic(diagnostic, measurements):
    """Auditer des propriétés vérifiables, sans note de justesse clinique."""
    if diagnostic is None:
        return {"facts_preserved": None, "literal_citations": 0, "citation_count": 0, "fidelity": None,
                "no_confirmed_cause": True, "uncertainty_explicit": None, "hypothesis_count": 0}
    expected = {name: measure.value for name, measure in normalize_measurements(measurements).items() if measure.value is not None}
    actual = {fact["sensor"]: fact["value"] for fact in diagnostic["faits_observes"] if fact["origin"] == "measurement"}
    sources = {source["id"]: source for source in diagnostic["sources"]}
    arguments = diagnostic["arguments_documentaires"]
    valid = sum(argument["source_id"] in sources and argument["excerpt"] in sources[argument["source_id"]]["passage"]
                for argument in arguments)
    return {"facts_preserved": actual == expected, "literal_citations": valid, "citation_count": len(arguments),
            "fidelity": valid / len(arguments) if arguments else None,
            "no_confirmed_cause": diagnostic["diagnostic_confirme"] is False and not diagnostic["autorisation_intervention"]
                and all(hypothesis["status"] == "non confirmée" for hypothesis in diagnostic["hypotheses_de_defaillance"]),
            "uncertainty_explicit": diagnostic["niveau_incertitude"] in ("élevé", "très élevé"),
            "hypothesis_count": len(diagnostic["hypotheses_de_defaillance"])}


def synthetic_request(scenario):
    """Sept minutes fabriquées ; le moniteur injecte séparément sa décision.

    La pression descend seulement dans le cas de suspicion de fuite. Cela reste
    un stimulus de test ambigu, jamais une signature validée de fuite d'air.
    """
    values = {name: 1.0 for name in SIGNAL_NAMES}
    values.update(TP2=8.0, TP3=9.0,
                  Oil_temperature=50.0 if scenario.normal else 70.0, Motor_current=4.0 if scenario.normal else 7.0)
    if scenario.identifier == "uncertain":
        values["Oil_temperature"] = None
    samples = []
    for index in range(43):
        row = {"timestamp": (datetime(2020, 1, 1) + timedelta(seconds=10 * index)).isoformat(), **values}
        if scenario.identifier == "air_leak":
            row["TP3"] = 9.0 - 3.0 * index / 42
        samples.append(row)
    return {"samples": samples, "data_context": "synthetic_test", "question": scenario.question}


def execute_scenario(scenario, configuration, *, documents, index, encoder, directory):
    """Exécuter un cas et conserver ses critères, comptages et latence observée."""
    provider = CountingMock(fail=scenario.identifier == "api_failure")
    request = synthetic_request(scenario)
    measures = {name: request["samples"][-1][name] for name in ("TP2", "TP3", "Oil_temperature", "Motor_current")}
    calls = {"monitor": 0, "retrieve": 0}
    def retrieve(question):
        calls["retrieve"] += 1
        return search_documents(question, index_directory=index, encoder=encoder, top_k=3)
    def monitor(_request):
        calls["monitor"] += 1
        return {"event": {"available_at": request["samples"][-1]["timestamp"], "model_id": "INJECTED-SYNTHETIC-NOT-A-TRAINED-MODEL",
                "anomaly_score": 0.2 if scenario.normal else 0.8, "threshold": 0.65, "is_anomaly": not scenario.normal,
                "data_context": "synthetic_test", "status": "scored"}, "measurements": measures,
                "summary": {"scope": "Décision injectée pour vérifier le graphe, pas la performance ML"}}
    state, diagnostic, retrieval = None, None, None
    resumed, explicit_before, stored, preapproved, recomputed = False, False, None, False, False
    started = perf_counter()
    if configuration in ("A", "B"):
        retrieval = retrieve(scenario.question) if configuration == "B" else None
        diagnostic = run_ablation(measures, scenario.question, retrieval.passages if retrieval else [],
            documents_directory=documents, provider=provider, configuration=configuration)
    else:
        services = WorkflowServices(monitor=monitor, retrieve=retrieve,
            diagnose=partial(generate_diagnostic, documents_directory=documents, provider=provider),
            decisions=DecisionStore(Path(directory) / "decisions.sqlite"))
        with open_workflow(directory=directory, services=services) as runner:
            state = runner.start(request)
            diagnostic, retrieval = state["diagnostic"], state["retrieval"]
            preapproved = bool((state["decision"] or {}).get("human_validated"))
            explicit_before = bool(state["interrupts"]) and state["review"] is None and state["decision"] is None
            before = (calls.copy(), provider.calls)
        if scenario.review:
            with open_workflow(directory=directory, services=services) as runner:
                restored = runner.get_state(state["run_id"])
                resumed = bool(restored["interrupts"]) and restored["proposal"] == state["proposal"] and restored["diagnostic"] == diagnostic
                submission = {"proposal_id": restored["proposal"]["proposal_id"], "decision": scenario.review,
                    "reviewer": "VALIDATEUR-SIMULE-CAMPAGNE", "comment": "Décision injectée sur fixture SYNTHETIQUE uniquement.", "acknowledged": True}
                state = runner.review(state["run_id"], submission)
                stored = runner.decisions.get(state["run_id"])
            recomputed = before != (calls.copy(), provider.calls)
    duration = perf_counter() - started
    audit = audit_diagnostic(diagnostic, measures)
    sources = diagnostic["sources"] if diagnostic else []
    arguments = diagnostic["arguments_documentaires"] if diagnostic else []
    cited = {argument["source_id"] for argument in arguments}
    conflict_sources = [source for source in sources if source["file_name"].startswith("divergence_")]
    # Oracle explicite : les deux affirmations contradictoires doivent être exposées
    # comme arguments et signalées insuffisantes. Leur seule présence dans sources
    # ou une incertitude générique ne démontre pas une détection de contradiction.
    conflict_exposed = len(conflict_sources) == 2 and all(source["id"] in cited for source in conflict_sources)
    conflict_exposed = conflict_exposed and bool(diagnostic and "documentary_evidence" in [
        item.get("code") if isinstance(item, dict) else item for item in diagnostic["informations_manquantes"]])
    status = diagnostic["etat_generation"] if diagnostic else None
    checks = {
        "normal_without_llm": bool(state and state["route"] == "normal" and provider.calls == calls["retrieve"] == 0
                                   and stored is None and not preapproved and state["decision"]["outcome"] == "closed_without_alert"),
        "documented_response": bool(status == "ok" and arguments and audit["uncertainty_explicit"]),
        "honest_abstention": bool((status == "information_insuffisante") or (state and state["phase"] == "awaiting_human"
                                  and not state["proposal"]["approval_allowed"] and not diagnostic)),
        "conflict_exposed": conflict_exposed,
        "api_failure_visible": bool(status == "erreur_api" and not arguments and provider.calls == 1),
        "uncertainty_explicit": bool(audit["uncertainty_explicit"] and diagnostic["informations_manquantes"]),
        "rejection_recorded": bool(stored and stored["outcome"] == "rejected" and not stored["human_validated"]),
        "explicit_approval_only": bool(explicit_before and not preapproved and stored and stored["human_validated"]
                                       and stored["review"]["submission"]["acknowledged"] is True),
        "resume_without_recomputation": bool(resumed and not recomputed and stored and stored["outcome"] == "revision_requested"),
    }
    unsupported = (configuration != "C" and scenario.requirement in ("normal_without_llm", "rejection_recorded", "explicit_approval_only", "resume_without_recomputation")) or (configuration == "A" and scenario.requirement in ("documented_response", "conflict_exposed"))
    passed = bool(checks[scenario.requirement] and audit["no_confirmed_cause"] and audit["facts_preserved"] is not False)
    return {"scenario": scenario.identifier, "label": scenario.label, "configuration": configuration,
        "data_kind": "synthetic_test", "llm_kind": "mock_with_injected_error" if provider.fail else "deterministic_mock",
        "raw_samples": len(request["samples"]), "synthetic_tp3_start": request["samples"][0]["TP3"],
        "synthetic_tp3_end": request["samples"][-1]["TP3"], "missing_oil_temperature": measures["Oil_temperature"] is None,
        "ml_kind": "injected_test_double" if configuration == "C" else "absent", "requirement": scenario.requirement,
        "status": "not_supported" if unsupported else "passed" if passed else "failed", "requirement_met": passed,
        "latency_seconds": duration, "calls": {**calls, "mock": provider.calls, "real_llm_api": 0},
        "diagnostic_status": status, "audit": audit, "conflict_exposed": conflict_exposed,
        "phase": state["phase"] if state else None, "paused_before_review": explicit_before,
        "proposal_approval_allowed": (state["proposal"] or {}).get("approval_allowed") if state else None,
        "approved_before_review": preapproved, "review_submitted": bool(scenario.review and configuration == "C"),
        "decision": stored["outcome"] if stored else None, "resumed": resumed, "recomputed_on_resume": recomputed,
        "physical_action_triggered": bool(stored and stored["physical_action_triggered"]),
        "references": [{"file": source["file_name"], "page": source["page"], "sha256": source["pdf_sha256"]} for source in sources]}


def summarize(rows):
    """Taux de traitement sur tous les cas ; NA conservé pour les métriques vides."""
    summary = []
    for configuration in CONFIGURATIONS:
        group = [row for row in rows if row["configuration"] == configuration]
        if not group:
            continue
        count = len(group)
        quoted = sum(row["audit"]["citation_count"] for row in group)
        valid = sum(row["audit"]["literal_citations"] for row in group)
        pending = [row for row in group if row["scenario"] != "normal"] if configuration == "C" else []
        resume = [row for row in group if row["scenario"] == "resume"] if configuration == "C" else []
        facts = [row for row in group if row["audit"]["facts_preserved"] is not None]
        uncertainty = [row for row in group if row["audit"]["uncertainty_explicit"] is not None]
        summary.append({"configuration": configuration, "runs": count,
            "passed": sum(row["status"] == "passed" for row in group),
            "failed": sum(row["status"] == "failed" for row in group),
            "not_supported": sum(row["status"] == "not_supported" for row in group),
            "scenario_handled_rate": sum(row["status"] == "passed" for row in group) / count,
            "latency_median_seconds": float(np.median([row["latency_seconds"] for row in group])),
            "latency_p95_seconds": float(np.percentile([row["latency_seconds"] for row in group], 95)),
            "literal_fidelity": valid / quoted if quoted else None, "citation_count": quoted,
            "hypothesis_count": sum(row["audit"]["hypothesis_count"] for row in group),
            "facts_checked": len(facts),
            "facts_preserved_rate": sum(row["audit"]["facts_preserved"] for row in facts) / len(facts) if facts else None,
            "uncertainty_checked": len(uncertainty),
            "explicit_uncertainty_rate": sum(row["audit"]["uncertainty_explicit"] for row in uncertainty) / len(uncertainty) if uncertainty else None,
            "automatic_approvals": sum(row["approved_before_review"] for row in group),
            "human_review_opportunities": len(pending) if configuration == "C" else None,
            "explicit_submissions": sum(row["review_submitted"] for row in group) if configuration == "C" else None,
            "human_review_compliance": sum(row["paused_before_review"] and not row["approved_before_review"] and not row["physical_action_triggered"] for row in pending) / len(pending) if pending else None,
            "persistent_resume_success_rate": sum(row["resumed"] and not row["recomputed_on_resume"] for row in resume) / len(resume) if resume else None,
            "real_llm_api_calls": sum(row["calls"]["real_llm_api"] for row in group),
            "diagnostic_clinical_quality": None, "precision": None, "recall": None, "f1": None})
    return summary


def run_campaign(directory, *, repeats=3):
    """Exécuter 10 × 3 × répétitions ; initialisation exclue des latences par cas."""
    if not 1 <= repeats <= 20:
        raise ValueError("Choisir entre 1 et 20 répétitions.")
    run = Path(directory) / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8])
    documents = create_fixture_corpus(run / "documents_SYNTHETIQUES")
    encoder = FixtureEncoder()
    index = run / "index_SYNTHETIQUE"
    index_documents(documents, index_directory=index, encoder=encoder)
    rows = []
    for repeat in range(repeats):
        for position, scenario in enumerate(SCENARIOS):
            # Rotation pour réduire un biais de première configuration, sans hasard.
            order = ("A", "B", "C")
            order = order[(repeat + position) % 3:] + order[:(repeat + position) % 3]
            for configuration in order:
                row = execute_scenario(scenario, configuration, documents=documents, index=index, encoder=encoder,
                    directory=run / f"{repeat}_{scenario.identifier}_{configuration}")
                row["repeat"] = repeat + 1
                rows.append(row)
    return {"protocol_version": "metropt-evaluation-v1", "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {"python": platform.python_version(), "system": platform.system(), "machine": platform.machine(),
                    "libraries": {name: version(name) for name in ("pandas", "numpy", "scikit-learn", "pydantic", "langgraph", "langgraph-checkpoint-sqlite", "chromadb", "sentence-transformers", "PyMuPDF", "plotly", "streamlit")}},
        "repeats": repeats, "scenario_count": len(SCENARIOS), "configurations": CONFIGURATIONS,
        "scope": "Comparaison de contrats et de capacités avec mock, pas un benchmark de LLM ni un classement clinique.",
        "ml_performance_evaluated": False, "real_llm_api_called": False,
        "rows": rows, "summary": summarize(rows)}
