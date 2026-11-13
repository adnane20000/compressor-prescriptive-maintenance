"""Isolation Forest ajusté sur le passé, seuil fixé sur une calibration séparée."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

from .features import FEATURE_COLUMNS, FEATURE_VERSION


@dataclass(frozen=True)
class ModelConfig:
    train_end: str = "2020-02-22 00:00:00"
    calibration_end: str = "2020-03-01 00:00:00"
    calibration_tail_fraction: float = 0.01
    n_estimators: int = 200
    max_samples: int = 256
    random_state: int = 42
    max_train_windows: int = 50_000
    min_partition_windows: int = 128

    def __post_init__(self):
        dates = [pd.Timestamp(self.train_end), pd.Timestamp(self.calibration_end)]
        if any(pd.isna(date) or date.tzinfo is not None or date != date.floor("min") for date in dates) or dates[0] >= dates[1]:
            raise ValueError("Bornes de séparation invalides : dates sans fuseau, alignées à la minute et ordonnées.")
        if not 0 < self.calibration_tail_fraction < 0.5:
            raise ValueError("La fraction de queue de calibration doit être comprise entre 0 et 0,5.")
        if min(self.n_estimators, self.max_samples, self.max_train_windows, self.min_partition_windows) < 2:
            raise ValueError("Paramètres d'entraînement insuffisants.")
        if self.max_train_windows < self.min_partition_windows:
            raise ValueError("Le plafond d'entraînement est inférieur au minimum requis.")


@dataclass
class BaselineModel:
    preprocessor: SimpleImputer
    estimator: IsolationForest
    threshold: float
    config: ModelConfig
    model_id: str
    metadata: dict
    feature_version: str = FEATURE_VERSION


def split_names(timestamps: pd.Series, config: ModelConfig) -> pd.Series:
    """Une fenêtre fermant à la borne appartient encore à la période précédente."""
    return pd.Series(np.select(
        [timestamps.le(pd.Timestamp(config.train_end)), timestamps.le(pd.Timestamp(config.calibration_end))],
        ["train", "calibration"], default="test",
    ), index=timestamps.index, name="split")


def _validate_features(frame: pd.DataFrame) -> None:
    required = {*FEATURE_COLUMNS, "timestamp", "window_start", "mode", "mode_mixed", "eligible", "history_ready", "sample_count", "missing_fraction"}
    if required - set(frame):
        raise ValueError("Variables ML manquantes ; utiliser build_features() ou engineer_features().")
    dates = frame["timestamp"]
    if frame.empty or not frame.index.is_unique or dates.isna().any() or not dates.is_monotonic_increasing or dates.duplicated().any():
        raise ValueError("Les fenêtres ML doivent être non vides et strictement chronologiques.")


def fit_baseline(frame: pd.DataFrame, config: ModelConfig | None = None) -> BaselineModel:
    """Ne jamais ajuster l'imputation ou la forêt sur la calibration ou le test."""
    config = config or ModelConfig()
    _validate_features(frame)
    splits = split_names(frame["timestamp"], config)
    train = frame.loc[splits.eq("train") & frame["eligible"]]
    calibration = frame.loc[splits.eq("calibration") & frame["eligible"]]
    test = frame.loc[splits.eq("test") & frame["eligible"]]
    if min(len(train), len(calibration), len(test)) < config.min_partition_windows:
        raise ValueError("Pas assez de fenêtres éligibles dans entraînement, calibration ou test.")
    if len(train) > config.max_train_windows:
        chosen = np.sort(np.random.default_rng(config.random_state).choice(len(train), config.max_train_windows, replace=False))
        fitting = train.iloc[chosen]
    else:
        fitting = train
    values = fitting[list(FEATURE_COLUMNS)].replace([np.inf, -np.inf], np.nan)
    preprocessor = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    transformed = preprocessor.fit_transform(values)
    estimator = IsolationForest(
        n_estimators=config.n_estimators, max_samples=min(config.max_samples, len(fitting)),
        contamination="auto", random_state=config.random_state, n_jobs=1,
    ).fit(transformed)
    cal_values = preprocessor.transform(calibration[list(FEATURE_COLUMNS)].replace([np.inf, -np.inf], np.nan))
    calibration_scores = -estimator.score_samples(cal_values)
    threshold = float(np.quantile(calibration_scores, 1 - config.calibration_tail_fraction, method="higher"))
    metadata = {
        "feature_columns": list(FEATURE_COLUMNS), "feature_version": FEATURE_VERSION,
        "feature_contract": {"window_seconds": 60, "past_windows": 5, "minimum_raw_samples": 3,
                             "maximum_missing_fraction": 0.2, "gaps_reset_history": True,
                             "final_unclosed_window_discarded": True, "availability": "fin de minute historique"},
        "config": asdict(config), "sklearn_version": sklearn.__version__,
        "train_windows_available": len(train), "train_windows_fitted": len(fitting),
        "calibration_windows": len(calibration), "test_windows_available": len(test),
        "train_first": train["timestamp"].min().isoformat(), "train_last": train["timestamp"].max().isoformat(),
        "calibration_first": calibration["timestamp"].min().isoformat(), "calibration_last": calibration["timestamp"].max().isoformat(),
        "threshold": threshold, "threshold_method": "quantile higher des scores de calibration ; alerte strictement au-dessus",
        "calibration_alert_rate": float(np.mean(calibration_scores > threshold)),
        "score_definition": "-IsolationForest.score_samples ; plus élevé = plus inhabituel ; aucune probabilité de panne",
        "scaling": "aucun : les coupures de la forêt n'exigent pas de normalisation affine",
        "imputation": "médianes et indicateurs de valeurs absentes ajustés seulement sur le sous-ensemble d'entraînement",
        "all_missing_train_features": values.columns[values.isna().all()].tolist(),
    }
    digest = hashlib.sha256(json.dumps({key: value for key, value in metadata.items() if key != "test_windows_available"}, sort_keys=True).encode())
    digest.update(pd.util.hash_pandas_object(values, index=False).to_numpy().tobytes())
    digest.update(pd.util.hash_pandas_object(calibration[list(FEATURE_COLUMNS)], index=False).to_numpy().tobytes())
    return BaselineModel(preprocessor, estimator, threshold, config, digest.hexdigest()[:20], metadata)


def predict_anomaly(model: BaselineModel, features: pd.DataFrame, *, batch_size: int = 20_000) -> pd.DataFrame:
    """Scorer des fenêtres causales ; ne jamais retourner de probabilité ou cause.

    Entrée : sortie de build_features(), avec au moins cinq minutes de contexte
    précédent pour une inférence à chaud. Aucun fit n'est effectué ici.
    Les fenêtres non éligibles ont un score et un statut d'anomalie absents.
    """
    _validate_features(features)
    if model.feature_version != FEATURE_VERSION or batch_size < 1:
        raise ValueError("Version des variables incompatible ou batch_size invalide.")
    columns = ["timestamp", "window_start", "mode", "mode_mixed", "sample_count", "missing_fraction"]
    columns += [f"{name}_mean" for name in ("Oil_temperature", "Motor_current", "TP2", "TP3")]
    result = features[columns].copy()
    result["split"] = split_names(features["timestamp"], model.config)
    result["anomaly_score"] = np.nan
    result["threshold"] = model.threshold
    result["is_anomaly"] = pd.array([pd.NA] * len(result), dtype="boolean")
    result["status"] = np.where(features["history_ready"], "insufficient_data", "insufficient_history")
    selected = features.index[features["eligible"]]
    for offset in range(0, len(selected), batch_size):
        indices = selected[offset:offset + batch_size]
        matrix = features.loc[indices, list(FEATURE_COLUMNS)].replace([np.inf, -np.inf], np.nan)
        scores = -model.estimator.score_samples(model.preprocessor.transform(matrix))
        result.loc[indices, "anomaly_score"] = scores
        result.loc[indices, "is_anomaly"] = scores > model.threshold
        result.loc[indices, "status"] = "scored"
    return result


def save_model(model: BaselineModel, directory: str | Path) -> Path:
    """Conserver forêt, imputation, seuil et contrat des variables dans un artefact."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "baseline.joblib"
    joblib.dump(model, target, compress=3)
    manifest = {**model.metadata, "model_id": model.model_id,
                "artifact_sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
    (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return target


def load_model(path: str | Path) -> BaselineModel:
    """Charger uniquement un artefact local de confiance produit par ce projet."""
    model = joblib.load(path)
    if not isinstance(model, BaselineModel) or model.feature_version != FEATURE_VERSION:
        raise ValueError("Artefact ML incompatible.")
    if model.metadata["sklearn_version"] != sklearn.__version__:
        raise ValueError("Version scikit-learn différente de celle de l'entraînement ; recréer l'artefact.")
    return model
