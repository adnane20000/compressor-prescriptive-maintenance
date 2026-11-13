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
