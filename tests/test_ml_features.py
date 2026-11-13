"""Causalité, indépendance aux blocs et qualité des fenêtres d'une minute."""

import numpy as np
import pandas as pd
import pytest

from metropt_maintenance.ml.features import FEATURE_COLUMNS, aggregate_minutes, build_features, engineer_features


def test_minute_features_are_chunk_invariant_and_end_labeled(ml_measurements):
    full = aggregate_minutes([ml_measurements])
    chunked = aggregate_minutes(ml_measurements.iloc[i:i + 17] for i in range(0, len(ml_measurements), 17))
    pd.testing.assert_frame_equal(full, chunked)
    assert full["timestamp"].iloc[0] == pd.Timestamp("2020-02-01 00:01:00")
    assert full["sample_count"].eq(6).all()
    assert len(full) == 180
    assert full.attrs["final_unclosed_raw_rows"] == 1
    assert all(full["timestamp"] - full["window_start"] == pd.Timedelta(minutes=1))


def test_features_never_use_future_samples(ml_measurements):
    before = engineer_features(aggregate_minutes([ml_measurements]))
    changed = ml_measurements.copy()
    cutoff = pd.Timestamp("2020-02-01 01:00:00")
    changed.loc[changed["timestamp"] >= cutoff, "Oil_temperature"] = 9999
    after = engineer_features(aggregate_minutes([changed]))
    columns = list(FEATURE_COLUMNS)
    pd.testing.assert_frame_equal(before.loc[before["timestamp"] <= cutoff, columns],
                                  after.loc[after["timestamp"] <= cutoff, columns])
    assert before.loc[5, "Oil_temperature_past5_mean"] == pytest.approx(before.loc[:4, "Oil_temperature_mean"].mean())
    assert before.loc[5, "Oil_temperature_delta1"] == pytest.approx(before.loc[5, "Oil_temperature_mean"] - before.loc[4, "Oil_temperature_mean"])


def test_history_is_reset_after_missing_minute(ml_measurements):
    missing = ml_measurements[~ml_measurements["timestamp"].between("2020-02-01 00:10:00", "2020-02-01 00:10:59")]
    features = engineer_features(aggregate_minutes([missing]))
    reset = features.index[features["timestamp"].eq(pd.Timestamp("2020-02-01 00:12:00"))][0]
    assert features.loc[reset:reset + 4, "history_ready"].eq(False).all()
    assert pd.isna(features.loc[reset, "Oil_temperature_lag1"])
    assert features.loc[reset + 5, "history_ready"]


def test_binary_invalid_and_missing_are_quality_not_automatic_alerts(ml_measurements):
    measurements = ml_measurements.copy()
    mask = measurements["timestamp"].between("2020-02-01 00:10:00", "2020-02-01 00:10:59")
    measurements.loc[mask, ["TP2", "TP3", "Oil_temperature", "COMP"]] = np.nan
    measurements.loc[mask, "DV_eletric"] = 2
    features = engineer_features(aggregate_minutes([measurements]))
    row = features.loc[features["timestamp"].eq(pd.Timestamp("2020-02-01 00:11:00"))].iloc[0]
    assert row["missing_fraction"] == pytest.approx(5 / 15)
    assert not row["eligible"]
    assert row["mode"] == "indéterminé"


def test_nonchronological_duplicate_or_bad_dates_are_refused(ml_measurements):
    for frame in [ml_measurements.iloc[::-1], pd.concat([ml_measurements.iloc[:5], ml_measurements.iloc[4:10]])]:
        with pytest.raises(ValueError, match="chronologiques|ordre strict"):
            aggregate_minutes([frame])
    frame = ml_measurements.copy()
    frame.loc[0, "timestamp"] = pd.NaT
    with pytest.raises(ValueError):
        aggregate_minutes([frame])
    with pytest.raises(ValueError, match="entre blocs"):
        aggregate_minutes([ml_measurements.iloc[:5], ml_measurements.iloc[4:10]])
    with pytest.raises(ValueError, match="Aucune minute"):
        aggregate_minutes([ml_measurements.iloc[:3]])


def test_build_features_reads_manual_csv_and_skips_partial_start(ml_measurements, tmp_path):
    path = tmp_path / "synthetic.csv"
    ml_measurements.iloc[1:].to_csv(path, index=False)
    features = build_features(path, chunksize=17)
    assert features.loc[0, "partial_start"]
    assert not features.loc[0, "eligible"]
    assert len(FEATURE_COLUMNS) == 43


def test_sparse_minute_is_not_eligible(ml_measurements):
    frame = ml_measurements.copy()
    within = frame["timestamp"].between("2020-02-01 00:10:00", "2020-02-01 00:10:59")
    frame = frame.loc[~within | frame["timestamp"].eq(pd.Timestamp("2020-02-01 00:10:00"))]
    features = engineer_features(aggregate_minutes([frame]))
    row = features.loc[features["timestamp"].eq(pd.Timestamp("2020-02-01 00:11:00"))].iloc[0]
    assert row["sample_count"] == 1 and not row["eligible"]
