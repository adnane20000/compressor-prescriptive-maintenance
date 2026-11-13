"""Exécution reproductible sur CSV réel et sorties structurées pour futurs agents."""

import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from ..ingestion.schema import SOURCE_URL
from .evaluation import evaluate_predictions
from .features import FEATURE_COLUMNS, build_features
from .model import ModelConfig, fit_baseline, predict_anomaly, save_model
from .report import summarize_scores, write_ml_report


@dataclass
class RunResult:
    metrics: dict
    model_path: Path
    report_path: Path


def prediction_to_record(prediction: dict, *, model_id: str, source_sha256: str | None = None) -> dict:
    """Contrat JSON sans ambiguïté entre alerte statistique et diagnostic."""
    score = prediction["anomaly_score"]
    flag = prediction["is_anomaly"]
    evidence = {}
    for name, unit in (("Oil_temperature", "°C"), ("Motor_current", "A"), ("TP2", "bar"), ("TP3", "bar")):
        value = prediction[f"{name}_mean"]
        evidence[name] = {"aggregation": "minute_mean", "value": float(value) if pd.notna(value) else None, "unit": unit}
    return {
        "schema_version": "anomaly-observation-v1", "model_id": model_id,
        "source_file_sha256": source_sha256, "data_context": "historical_replay",
        "window_start": pd.Timestamp(prediction["window_start"]).isoformat(),
        "available_at": pd.Timestamp(prediction["timestamp"]).isoformat(),
        "status": prediction["status"], "anomaly_score": float(score) if pd.notna(score) else None,
        "partition": prediction.get("split", "inference"),
        "threshold": float(prediction["threshold"]), "is_anomaly": bool(flag) if pd.notna(flag) else None,
        "score_is_failure_probability": False, "mode_interpreted": prediction["mode"],
        "mode_mixed": bool(prediction["mode_mixed"]), "sample_count": int(prediction["sample_count"]),
        "missing_fraction": float(prediction["missing_fraction"]), "observed_signals": evidence,
        "cause": None, "diagnosis_confirmed": False,
        "requires_human_review": bool(flag) if pd.notna(flag) else False,
        "interpretation": "Écart statistique à la référence apprise ; aucune cause identifiée.",
    }


def run_baseline(
    path: str | Path, *, output: str | Path = "data/local/ml_baseline",
    model_directory: str | Path = "models/isolation_forest", config: ModelConfig | None = None,
    chunksize: int = 50_000,
) -> RunResult:
    """Enchaîner les opérations sans utiliser les périodes externes pour apprendre."""
    path, output = Path(path), Path(output)
    config = config or ModelConfig()
    features = build_features(path, chunksize=chunksize)
    model = fit_baseline(features, config)
    predictions = predict_anomaly(model, features)
    # Les annotations documentées entrent seulement ici, après fixation du seuil.
    metrics = evaluate_predictions(predictions)
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    provenance = {"source_reference": SOURCE_URL, "csv_path": str(path.resolve()),
                  "csv_sha256": digest.hexdigest(), "csv_bytes": path.stat().st_size}
    model.metadata["source_provenance"] = provenance
    model_path = save_model(model, model_directory)
    metrics.update(
        model_id=model.model_id, threshold=model.threshold, protocol=asdict(config),
        provenance=provenance, feature_columns=list(FEATURE_COLUMNS),
        feature_version=model.feature_version, raw_rows_read=features.attrs["raw_rows_read"],
        final_unclosed_raw_rows=features.attrs["final_unclosed_raw_rows"],
        total_observed_windows=len(features), unscored_windows=int(predictions["status"].ne("scored").sum()),
        unscored_by_status={key: int(value) for key, value in predictions.loc[predictions["status"].ne("scored"), "status"].value_counts().items()},
        mode_mixed_windows=int(features["mode_mixed"].sum()),
        versions={"python": platform.python_version(), "sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        agent_alert_export_scope="test_only",
    )
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / "predictions.csv", index=False)
    with (output / "agent_alerts.jsonl").open("w", encoding="utf-8") as target:
        alerts = predictions.loc[predictions["is_anomaly"].fillna(False) & predictions["split"].eq("test")]
        for values in alerts.itertuples(index=False, name=None):
            record = prediction_to_record(dict(zip(alerts.columns, values)), model_id=model.model_id, source_sha256=digest.hexdigest())
            target.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    report = write_ml_report(metrics, summarize_scores(predictions), output)
    return RunResult(metrics, model_path, report)
