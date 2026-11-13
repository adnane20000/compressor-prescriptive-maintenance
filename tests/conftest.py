"""Fabriques de petits CSV SYNTHÉTIQUES ; aucune mesure industrielle réelle."""

import csv
from datetime import datetime, timedelta

import pytest

from metropt_maintenance.ingestion.schema import REQUIRED_COLUMNS, SIGNAL_NAMES


@pytest.fixture
def make_csv(tmp_path):
    """Écrire les vrais noms de colonnes avec des valeurs fabriquées pour les tests."""
    def factory(overrides=None, *, with_index=True, filename="synthetic.csv"):
        overrides = [{} for _ in range(6)] if overrides is None else overrides
        path = tmp_path / filename
        fields = ([""] if with_index else []) + list(REQUIRED_COLUMNS)
        with path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(fields)
            for position, changes in enumerate(overrides):
                row = {name: 1.0 for name in SIGNAL_NAMES}
                row.update({
                    "timestamp": (datetime(2020, 2, 1) + timedelta(seconds=10 * position)).isoformat(sep=" "),
                    "TP2": 8.0, "TP3": 9.0, "Oil_temperature": 50.0,
                    "Motor_current": 0.04, "COMP": 1.0, "DV_eletric": 0.0,
                })
                row.update(changes)
                writer.writerow(([position * 10] if with_index else []) + [row[name] for name in REQUIRED_COLUMNS])
        return path
    return factory


@pytest.fixture
def ml_measurements():
    """Trois heures SYNTHÉTIQUES à six mesures/minute, jamais des données MetroPT réelles."""
    import numpy as np
    import pandas as pd
    dates = pd.date_range("2020-02-01", periods=1081, freq="10s")
    rng = np.random.default_rng(7)
    data = {"timestamp": dates}
    for name in SIGNAL_NAMES:
        data[name] = np.ones(len(dates))
    for name, level in [("TP2", 8), ("TP3", 9), ("H1", 9), ("Reservoirs", 9), ("Oil_temperature", 50), ("Motor_current", 4), ("DV_pressure", 0)]:
        data[name] = level + rng.normal(0, 0.05, len(dates))
    data["DV_eletric"] = np.zeros(len(dates))
    return pd.DataFrame(data)


@pytest.fixture
def ml_features(ml_measurements):
    from metropt_maintenance.ml.features import aggregate_minutes, engineer_features
    return engineer_features(aggregate_minutes([ml_measurements]))


@pytest.fixture
def ml_config():
    from metropt_maintenance.ml.model import ModelConfig
    return ModelConfig(train_end="2020-02-01 01:00:00", calibration_end="2020-02-01 02:00:00",
                       n_estimators=12, max_samples=32, min_partition_windows=8)
