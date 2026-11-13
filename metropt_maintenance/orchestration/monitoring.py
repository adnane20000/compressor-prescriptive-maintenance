"""Adaptation des mesures historiques aux variables causales de la baseline existante."""

import pandas as pd

from ..ingestion.csv_reader import iter_chunks
from ..ingestion.schema import SIGNAL_NAMES
from ..ml.features import aggregate_minutes, engineer_features
from ..ml.model import load_model, predict_anomaly
from ..ml.workflow import prediction_to_record
from .state import MAX_SAMPLES, WorkflowRequest


def load_measurement_window(path, start: str, end: str, *, data_context: str, question: str,
                            max_samples: int = MAX_SAMPLES) -> WorkflowRequest:
    """Lire un intervalle inclusif par blocs et refuser un dépassement, sans trier.

    Le CSV n'est pas indexé : son préfixe est parcouru jusqu'au bloc contenant la
    fin demandée. Seuls les échantillons sélectionnés sont conservés en mémoire.
    """
    lower, upper = pd.Timestamp(start), pd.Timestamp(end)
    if any(pd.isna(value) or value.tzinfo is not None for value in (lower, upper)) or lower >= upper:
        raise ValueError("Choisir un début et une fin valides, ordonnés et sans fuseau.")
    if not 1 <= max_samples <= MAX_SAMPLES:
        raise ValueError("La limite doit être comprise entre 1 et 6000 mesures.")
    rows, previous = [], None
    for frame in iter_chunks(path, chunksize=10_000):
        if frame.empty:
            continue
        dates = frame["timestamp"]
        if dates.isna().any() or not dates.is_monotonic_increasing or dates.duplicated().any() or (previous is not None and dates.iloc[0] <= previous):
            raise ValueError("Fenêtre refusée : dates invalides, répétées ou non chronologiques dans la portion lue.")
        previous = dates.iloc[-1]
        selected = frame.loc[dates.between(lower, upper)]
        if len(rows) + len(selected) > max_samples:
            raise ValueError("Trop de mesures : réduire l'intervalle ; aucune troncature silencieuse.")
        for row in selected.itertuples(index=False, name=None):
            timestamp, *values = row
            rows.append({"timestamp": timestamp.isoformat(), **{name: float(value) if pd.notna(value) else None for name, value in zip(SIGNAL_NAMES, values)}})
        if dates.iloc[-1] >= upper:
            break
    return WorkflowRequest(samples=rows, data_context=data_context, question=question)


def monitor_measurements(request: WorkflowRequest, *, model_path: str) -> dict:
    """Scorer uniquement la dernière minute close ; le reste fournit son contexte.

    Aucun entraînement ni recalibration. Une minute ouverte est écartée et un trou
    réinitialise les variables temporelles, comme dans la baseline d'origine.
    """
    model = load_model(model_path)
    frame = pd.DataFrame(request.samples)
    features = engineer_features(aggregate_minutes([frame]))
    predictions = predict_anomaly(model, features)
    record = prediction_to_record(predictions.iloc[-1].to_dict(), model_id=model.model_id)
    record["data_context"] = request.data_context
    return {"event": record, "measurements": record["observed_signals"],
            "summary": {"raw_samples": len(frame), "closed_windows": len(features),
                        "monitored_window": record["window_start"],
                        "available_at": record["available_at"],
                        "scope": "dernière minute close ; fenêtres précédentes utilisées comme contexte",
                        "final_unclosed_raw_rows": features.attrs["final_unclosed_raw_rows"]}}
