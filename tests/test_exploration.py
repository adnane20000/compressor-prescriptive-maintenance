"""Qualité exhaustive, frontières des blocs et statistiques numériques."""

import math

import pytest

from metropt_maintenance.ingestion.exploration import _median_from_counts, profile_csv
from collections import Counter


def test_profile_counts_duplicates_across_chunks_without_using_export_index(make_csv):
    rows = [
        {},
        {"timestamp": "2020-02-01 00:00:10", "Oil_temperature": "", "TP2": "bad"},
        {"timestamp": "2020-02-01 00:00:00"},
        {"timestamp": "invalid", "COMP": 2, "TP3": "inf"},
        {"timestamp": "2020-02-01 00:01:00", "Motor_current": 4},
    ]
    result = profile_csv(make_csv(rows), chunksize=2)
    assert result["rows_analyzed"] == 5
    assert result["signal_count"] == 15
    assert len(result["csv_columns"]) == 17
    assert result["duplicate_measurement_rows"] == 1
    assert result["duplicate_timestamps"] == 1
    assert result["missing_raw"]["Oil_temperature"] == 1
    assert result["invalid_values"]["TP2"] == 1
    assert result["invalid_values"]["TP3"] == 1
    assert result["invalid_values"]["timestamp"] == 1
    assert result["non_binary_values"]["COMP"] == 1
    assert result["time_intervals"]["backwards_intervals"] == 1
    assert result["time_intervals"]["count"] == 2  # Ne relie pas les dates à travers NaT.
    assert sum(result["mode_counts"].values()) == 5
    assert result["failure_labels_created"] is False
    assert len(result["file_sha256"]) == 64


def test_results_are_independent_of_chunk_boundaries(make_csv):
    path = make_csv([{"TP2": value} for value in [1, 2, 3, 4, 5, 6]])
    small, large = profile_csv(path, chunksize=1), profile_csv(path, chunksize=5)
    assert small["time_intervals"] == large["time_intervals"]
    assert small["duplicate_measurement_rows"] == large["duplicate_measurement_rows"]
    tp2 = next(row for row in small["statistics"] if row["column"] == "TP2")
    assert tp2["mean"] == pytest.approx(3.5)
    assert tp2["std_sample"] == pytest.approx(math.sqrt(3.5))
    for left, right in zip(small["statistics"], large["statistics"]):
        assert left["mean"] == pytest.approx(right["mean"])
        assert left["std_sample"] == pytest.approx(right["std_sample"])
    assert small["time_intervals"]["median_seconds"] == 10.0


def test_prefix_is_explicit_and_does_not_hash_the_rest_of_the_file(make_csv):
    result = profile_csv(make_csv(), chunksize=2, max_rows=3)
    assert result["scope"] == "prefix_limit" and result["rows_analyzed"] == 3
    assert result["file_sha256"] is None
    assert result["date_max"] == "2020-02-01T00:00:20"


def test_empty_csv_and_all_invalid_dates(make_csv):
    with pytest.raises(ValueError, match="aucune mesure"):
        profile_csv(make_csv([]))
    result = profile_csv(make_csv([{"timestamp": "bad", "TP2": ""}]))
    assert result["date_min"] is None and result["date_max"] is None
    assert result["time_intervals"]["median_seconds"] is None
    assert next(row for row in result["statistics"] if row["column"] == "TP2")["mean"] is None


def test_duplicate_time_is_not_necessarily_a_duplicate_measurement(make_csv):
    result = profile_csv(make_csv([{}, {"timestamp": "2020-02-01 00:00:00", "TP2": 10}]), chunksize=1)
    assert result["duplicate_timestamps"] == 1
    assert result["duplicate_measurement_rows"] == 0
    assert result["time_intervals"]["zero_intervals"] == 1


def test_large_gap_and_temporary_database_cleanup(make_csv, tmp_path):
    temp = tmp_path / "scratch"
    temp.mkdir()
    result = profile_csv(make_csv([{}, {"timestamp": "2020-02-01 01:00:00"}]), chunksize=1, temp_dir=temp)
    assert result["time_intervals"]["gaps_above_threshold"] == 1
    assert result["time_intervals"]["max_seconds"] == 3600
    assert list(temp.iterdir()) == []
    with pytest.raises(ValueError):
        profile_csv(make_csv(), gap_threshold_seconds=0)


@pytest.mark.parametrize("values,expected", [({}, None), ({9: 2, 10: 2}, 9.5), ({9: 1, 10: 4}, 10)])
def test_weighted_median(values, expected):
    assert _median_from_counts(Counter(values)) == expected
