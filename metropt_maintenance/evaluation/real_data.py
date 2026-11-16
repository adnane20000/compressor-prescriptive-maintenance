"""Évaluations réelles locales : prédictions en flux, recherche E5, graphe avec mock."""

import json
from dataclasses import asdict
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from ..diagnostic import generate_diagnostic
from ..ml.evaluation import DOCUMENTED_PERIODS
from ..ml.model import load_model
from ..orchestration import open_workflow
from ..orchestration.monitoring import load_measurement_window, monitor_measurements
from ..orchestration.nodes import WorkflowServices
from ..rag import search_documents
from ..rag.documents import file_sha256
from ..rag.embeddings import default_encoder
from ..rag.index import read_manifest
from ..storage.decisions import DecisionStore
from .campaign import CountingMock, audit_diagnostic, run_ablation


def reference_scores(expected, returned, *, k):
    """Recall des pages annotées : |pages pertinentes retrouvées| / |pages annotées|.

    Les annotations peuvent être incomplètes ou lister des pages alternatives.
    Ce rappel est donc distinct de hit@k, qui n'exige qu'une page pertinente.
    """
    if k < 1:
        raise ValueError("k doit être positif.")
    relevant = {(ref["file"], ref["page"]) for ref in expected}
    retrieved = {(ref["file"], ref["page"]) for ref in returned[:k]}
    return {"recall": len(relevant & retrieved) / len(relevant) if relevant else None,
            "hit": bool(relevant & retrieved) if relevant else None,
            "relevant_pages_found": len(relevant & retrieved), "annotated_pages": len(relevant)}


def evaluate_real_retrieval(cases, *, index_directory, encoder=None):
    """Mesurer la recherche, sans génération ; ne pas ajuster les seuils au benchmark."""
    if not cases:
        raise ValueError("Questions annotées requises.")
    encoder = encoder or default_encoder()
    manifest = read_manifest(index_directory)
    if manifest is None:
        raise ValueError("Index réel absent.")
    inventory = {entry["relative_path"] for entry in manifest["inventory"]}
    for case in cases:
        if type(case.get("should_abstain")) is not bool or not case.get("question"):
            raise ValueError("Question et statut d'abstention requis.")
        if not case["should_abstain"] and not case.get("expected_references"):
            raise ValueError("Références annotées requises.")
        if any(ref["file"] not in inventory for ref in case.get("expected_references", [])):
            raise ValueError("Référence annotée absente du corpus.")
    start = perf_counter()
    if hasattr(encoder, "prepare"):
        encoder.prepare()
    loading_seconds = perf_counter() - start
    rows = []
    for case in cases:
        started = perf_counter()
        result = search_documents(case["question"], index_directory=index_directory, encoder=encoder, top_k=5)
        elapsed = perf_counter() - started
        if result.reason in ("index_missing", "corpus_changed_reindex_required", "no_extractable_text"):
            raise ValueError("Corpus non prêt : aucune métrique de qualité calculée.")
        references = [{"file": passage.relative_path, "page": passage.page} for passage in result.passages]
        row = {**case, "status": result.status, "reason": result.reason, "returned_references": references,
               "latency_seconds": elapsed, "scores": {str(k): reference_scores(case.get("expected_references", []), references, k=k) for k in (1, 3, 5)}}
        rows.append(row)
    answerable = [row for row in rows if not row["should_abstain"]]
    unanswerable = [row for row in rows if row["should_abstain"]]
    scores = {str(k): {"macro_annotated_page_recall": float(np.mean([row["scores"][str(k)]["recall"] for row in answerable])) if answerable else None,
                      "hit_rate": float(np.mean([row["scores"][str(k)]["hit"] for row in answerable])) if answerable else None} for k in (1, 3, 5)}
    return {"question_count": len(rows), "answerable_count": len(answerable), "unanswerable_count": len(unanswerable),
        "scores": scores, "unanswerable_abstention_rate": sum(row["status"] == "information_insuffisante" for row in unanswerable) / len(unanswerable) if unanswerable else None,
        "encoder_loading_seconds": loading_seconds, "encoder": manifest["encoder"], "inventory": manifest["inventory"],
        "indexed_pages": manifest["page_count"], "indexed_chunks": manifest["chunk_count"], "cases": rows,
        "annotation_limits": "Annotations préexistantes non indépendantes, parfois pages alternatives ; recall calculé sur toutes les pages listées, sans prétendre couvrir toutes les pages pertinentes du corpus. Questions déjà utilisées pendant le développement. Pas de généralisation."}


def evaluate_saved_predictions(path, *, chunksize=20_000, periods=DOCUMENTED_PERIODS):
    """Recalculer les statistiques des prédictions figées, sans entraînement ni seuil nouveau."""
    counts, comparisons = {}, [{**asdict(period), "scored_windows": 0, "unscored_windows": 0, "alert_windows": 0} for period in periods]
    first_normal = first_alert = None
    outside_scored = outside_alerts = 0
    for block in pd.read_csv(path, chunksize=chunksize, parse_dates=["timestamp", "window_start"]):
        scored = block.status.eq("scored")
        if not block.is_anomaly.dropna().isin([True, False]).all():
            raise ValueError("Indicateur d'anomalie non booléen dans les prédictions.")
        alerts = block.is_anomaly.eq(True) & scored
        if (alerts != (scored & block.anomaly_score.gt(block.threshold))).any():
            raise ValueError("Prédictions incohérentes avec score > seuil.")
        for split, group in block.groupby("split"):
            row = counts.setdefault(split, {"windows": 0, "scored_windows": 0, "alert_windows": 0})
            row["windows"] += len(group)
            row["scored_windows"] += int(scored.loc[group.index].sum())
            row["alert_windows"] += int(alerts.loc[group.index].sum())
        test = block.split.eq("test")
        if first_normal is None and (test & scored & ~alerts).any():
            first_normal = block.loc[test & scored & ~alerts, "timestamp"].iloc[0].isoformat()
        if first_alert is None and (test & alerts).any():
            first_alert = block.loc[test & alerts, "timestamp"].iloc[0].isoformat()
        known = pd.Series(False, index=block.index)
        for period, row in zip(periods, comparisons):
            overlaps = test & block.window_start.lt(pd.Timestamp(period.end) + pd.Timedelta(minutes=1)) & block.timestamp.gt(pd.Timestamp(period.start))
            known |= overlaps
            row["scored_windows"] += int((overlaps & scored).sum())
            row["unscored_windows"] += int((overlaps & ~scored).sum())
            row["alert_windows"] += int((overlaps & alerts).sum())
        outside_scored += int((test & scored & ~known).sum())
        outside_alerts += int((test & alerts & ~known).sum())
    for row in counts.values():
        row["alert_rate"] = row["alert_windows"] / row["scored_windows"] if row["scored_windows"] else None
    observable = [row for row in comparisons if row["scored_windows"]]
    for row in comparisons:
        expected = int((pd.Timestamp(row["end"]) + pd.Timedelta(minutes=1) - pd.Timestamp(row["start"])).total_seconds() / 60)
        row["expected_minutes"] = expected
        row["observation_coverage"] = row["scored_windows"] / expected
        row["alert_fraction_of_scored_minutes"] = row["alert_windows"] / row["scored_windows"] if row["scored_windows"] else None
    return {"prediction_sha256": file_sha256(path), "counts_by_split": counts, "documented_periods": comparisons,
        "period_coverage": {"with_observations": len(observable), "with_at_least_one_alert": sum(row["alert_windows"] > 0 for row in observable),
                            "fraction": sum(row["alert_windows"] > 0 for row in observable) / len(observable) if observable else None},
        "outside_documented_periods": {"scored_windows": outside_scored, "alert_windows": outside_alerts, "label": "non étiqueté"},
        "representative_timestamps": {"model_not_flagged": first_normal, "model_flagged": first_alert},
        "precision": None, "recall": None, "f1": None,
        "limits": "Pas de vérité terrain binaire exhaustive. Couverture des quatre périodes rapportées et fractions de minutes alertées : mesures descriptives, pas recall de diagnostic de fuite. Hors période = non étiqueté. Aucun réentraînement ni ajustement de seuil."}


def evaluate_real_workflows(*, directory, csv_path, model_path, index_directory, documents_directory,
                            timestamps, repeats=3):
    """Comparer les architectures sur les mêmes moyennes réelles ; C re-score réellement.

    Les cas sont choisis selon le modèle, sans étiquette normale/panne confirmée.
    Les latences excluent la lecture commune de la fenêtre et la préparation des
    moyennes fournies à A/B. Aucun humain réel n'est simulé dans ces dossiers.
    """
    rows = []
    for case, timestamp in timestamps.items():
        if timestamp is None:
            continue
        end = pd.Timestamp(timestamp) + pd.Timedelta(seconds=30)
        start = pd.Timestamp(timestamp) - pd.Timedelta(minutes=7)
        question = "Selon le DOE, comment les fuites affectent-elles un système d'air comprimé ?"
        prepared = perf_counter()
        request = load_measurement_window(csv_path, start.isoformat(), end.isoformat(), data_context="historical_replay", question=question)
        reference = monitor_measurements(request, model_path=str(model_path))
        preparation_seconds = perf_counter() - prepared
        measures = reference["measurements"]
        for repeat in range(repeats):
            order = ("A", "B", "C")
            order = order[repeat % 3:] + order[:repeat % 3]
            for configuration in order:
                provider, calls = CountingMock(), {"monitor": 0, "retrieve": 0}
                def retrieve(query):
                    calls["retrieve"] += 1
                    return search_documents(query, index_directory=index_directory)
                def monitor(value):
                    calls["monitor"] += 1
                    return monitor_measurements(value, model_path=str(model_path))
                started = perf_counter()
                state, resumed, approved = None, None, False
                if configuration in ("A", "B"):
                    retrieval = retrieve(question) if configuration == "B" else None
                    diagnostic = run_ablation(measures, question, retrieval.passages if retrieval else [], documents_directory=documents_directory,
                                               provider=provider, configuration=configuration)
                else:
                    folder = Path(directory) / f"{case}_{repeat}"
                    services = WorkflowServices(monitor=monitor, retrieve=retrieve,
                        diagnose=partial(generate_diagnostic, documents_directory=documents_directory, provider=provider),
                        decisions=DecisionStore(folder / "decisions.sqlite"))
                    with open_workflow(directory=folder, services=services) as runner:
                        state = runner.start(request)
                        diagnostic = state["diagnostic"]
                    latency_before_reopen = perf_counter() - started
                    before = (calls.copy(), provider.calls)
                    with open_workflow(directory=folder, services=services) as runner:
                        restored = runner.get_state(state["run_id"])
                        resumed = restored["proposal"] == state["proposal"] and restored["diagnostic"] == diagnostic and before == (calls.copy(), provider.calls)
                        approved = bool((restored["decision"] or {}).get("human_validated"))
                elapsed = latency_before_reopen if configuration == "C" else perf_counter() - started
                rows.append({"case": case, "configuration": configuration, "repeat": repeat + 1,
                    "data_kind": "historical_replay", "llm_kind": "deterministic_mock", "raw_samples": len(request.samples),
                    "available_at": reference["event"]["available_at"], "model_id": reference["event"]["model_id"],
                    "measured_anomaly_score": reference["event"]["anomaly_score"], "threshold": reference["event"]["threshold"],
                    "is_anomaly_by_model": reference["event"]["is_anomaly"], "common_input_preparation_seconds": preparation_seconds,
                    "latency_seconds": elapsed, "calls": {**calls, "mock": provider.calls, "real_llm_api": 0},
                    "audit": audit_diagnostic(diagnostic, measures), "diagnostic_status": diagnostic["etat_generation"] if diagnostic else None,
                    "route": state["route"] if state else None, "awaiting_human": bool(state and state["interrupts"]),
                    "checkpoint_reopened_consistently": resumed, "review_submitted": False, "human_validated": approved,
                    "references": [{"file": source["file_name"], "page": source["page"], "sha256": source["pdf_sha256"]} for source in diagnostic["sources"]] if diagnostic else []})
    return {"rows": rows, "selection": "Première minute test scorée sans alerte et première minute test alertée dans les prédictions figées. Ces sélections ne constituent pas une vérité terrain.",
            "latency_scope": "Temps applicatif local hors préparation commune ; mock sans inférence générative distante. Réouverture du checkpoint vérifiée après mesure. Une première exécution peut inclure des caches froids."}


def run_real_evaluation(*, directory, csv_path="data/raw/MetroPT3(AirCompressor).csv", model_path="models/isolation_forest/baseline.joblib",
                        predictions_path="data/local/ml_baseline/predictions.csv", index_directory="data/local/rag", documents_directory="documents", repeats=3):
    """Ne calculer une section que si ses artefacts existent ; signaler les absences."""
    result = {"data_kind": "historical_replay_and_real_pdfs", "llm_kind": "deterministic_mock", "errors": [], "ml": None, "retrieval": None, "workflows": None}
    if Path(predictions_path).is_file():
        result["ml"] = evaluate_saved_predictions(predictions_path)
    else:
        result["errors"].append("saved_predictions_missing")
    if Path(index_directory, "index.json").is_file():
        cases = []
        for name in ("rag_uci_questions.json", "rag_reference_questions.json"):
            fixture_path = Path(__file__).resolve().parents[2] / "tests/fixtures" / name
            benchmark = json.loads(fixture_path.read_text())
            cases.extend({**case, "annotation_file": name, "annotation_file_sha256": file_sha256(fixture_path),
                          "annotation_original_scope": benchmark["scope"]} for case in benchmark["cases"])
        result["retrieval"] = evaluate_real_retrieval(cases, index_directory=index_directory)
    else:
        result["errors"].append("rag_index_missing")
    if result["ml"] and result["retrieval"] and Path(csv_path).is_file() and Path(model_path).is_file():
        model = load_model(model_path)
        result["artifacts"] = {"csv_sha256": file_sha256(csv_path), "model_sha256": file_sha256(model_path),
                               "model_id": model.model_id, "threshold": model.threshold, "training_metadata": model.metadata}
        result["workflows"] = evaluate_real_workflows(directory=directory, csv_path=csv_path, model_path=model_path,
            index_directory=index_directory, documents_directory=documents_directory, timestamps=result["ml"]["representative_timestamps"], repeats=repeats)
    else:
        result["errors"].append("real_workflow_prerequisites_missing")
    return result
