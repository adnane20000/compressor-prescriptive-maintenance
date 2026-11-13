"""Schéma réel, sélection des colonnes et lecture bornée en mémoire."""

import numpy as np
import pandas as pd
import pytest

from metropt_maintenance.ingestion.csv_reader import inspect_header, iter_chunks, iter_raw_chunks, load_preview
from metropt_maintenance.ingestion.schema import REQUIRED_COLUMNS, SIGNALS, SIGNAL_NAMES, schema_records


def test_official_schema_has_only_fifteen_signals():
    assert len(SIGNALS) == 15
    assert {signal.kind for signal in SIGNALS} == {"analogique", "numérique"}
    assert sum(signal.kind == "analogique" for signal in SIGNALS) == 7
    assert "DV_eletric" in SIGNAL_NAMES and "Towers" in SIGNAL_NAMES
    assert "timestamp" not in SIGNAL_NAMES and "index" not in SIGNAL_NAMES
    assert next(row for row in schema_records() if row["name"] == "Oil_temperature")["unit"] == "°C"


@pytest.mark.parametrize("with_index", [True, False])
def test_manual_csv_with_or_without_index(make_csv, with_index):
    path = make_csv(with_index=with_index)
    header = inspect_header(path)
    assert ("Unnamed: 0" in header) == with_index
    chunks = list(iter_chunks(path, chunksize=2))
    assert [len(chunk) for chunk in chunks] == [2, 2, 2]
    assert list(chunks[0]) == list(REQUIRED_COLUMNS)
    assert str(chunks[0]["timestamp"].dtype) == "datetime64[ns]"
    assert chunks[0]["TP2"].dtype == np.dtype("float64")
    assert len(load_preview(path, rows=3)) == 3


def test_reader_passes_projection_and_row_limit_to_pandas(make_csv, monkeypatch):
    path = make_csv()
    calls = []
    original = pd.read_csv
    def tracked(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(pd, "read_csv", tracked)
    frames = list(iter_chunks(path, chunksize=2, columns=("timestamp", "TP2"), max_rows=3))
    assert sum(map(len, frames)) == 3
    assert list(frames[0]) == ["timestamp", "TP2"]
    assert calls[0]["chunksize"] == 2 and calls[0]["nrows"] == 3
    assert calls[0]["usecols"] == ("timestamp", "TP2")


def test_invalid_numeric_and_dates_are_not_imputed(make_csv):
    path = make_csv([{"TP2": "inf", "TP3": "bad", "timestamp": "invalid", "Oil_temperature": ""}])
    row = load_preview(path).iloc[0]
    assert pd.isna(row["timestamp"])
    assert all(pd.isna(row[name]) for name in ("TP2", "TP3", "Oil_temperature"))


def test_invalid_headers_and_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        inspect_header(tmp_path / "absent.csv")
    path = tmp_path / "bad.csv"
    for content, message in [("", "vide"), ("timestamp,TP2\n", "manquantes"), ("timestamp,timestamp\n", "dupliqués")]:
        path.write_text(content)
        with pytest.raises(ValueError, match=message):
            inspect_header(path)


@pytest.mark.parametrize("options", [{"chunksize": 0}, {"max_rows": 0}, {"columns": []}, {"columns": ["invented"]}])
def test_invalid_read_options(make_csv, options):
    with pytest.raises(ValueError):
        list(iter_raw_chunks(make_csv(), **options))


def test_preview_limit_and_unknown_typed_column(make_csv):
    path = make_csv()
    with pytest.raises(ValueError):
        load_preview(path, rows=10_001)
    with pytest.raises(ValueError):
        list(iter_chunks(path, columns=("Unnamed: 0",)))


def test_extra_field_is_not_silently_used_as_a_pandas_index(make_csv):
    path = make_csv([{}])
    lines = path.read_text().splitlines()
    path.write_text(lines[0] + "\n" + lines[1] + ",unexpected\n")
    with pytest.raises(ValueError, match="Structure CSV|Error tokenizing"):
        list(iter_raw_chunks(path))
