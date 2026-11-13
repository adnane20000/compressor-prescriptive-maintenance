"""Prétraitement entraînement seul, seuil calibration seul et persistance fidèle."""

import json

import numpy as np
import pandas as pd
import pytest

from metropt_maintenance.ml.features import FEATURE_COLUMNS
from metropt_maintenance.ml.model import ModelConfig, fit_baseline, load_model, predict_anomaly, save_model, split_names


def test_chronological_split_includes_only_closed_windows(ml_features, ml_config):
    names = split_names(ml_features["timestamp"], ml_config)
    assert names.iloc[59] == "train"
    assert names.iloc[60] == "calibration"
    assert names.iloc[119] == "calibration"
    assert names.iloc[120] == "test"


def test_changing_test_does_not_change_preprocessing_model_or_threshold(ml_features, ml_config):
    first = fit_baseline(ml_features, ml_config)
    changed = ml_features.copy()
    test = split_names(changed["timestamp"], ml_config).eq("test")
    changed.loc[test, list(FEATURE_COLUMNS)] = 1e6
    second = fit_baseline(changed, ml_config)
    np.testing.assert_equal(first.preprocessor.statistics_, second.preprocessor.statistics_)
    assert first.threshold == second.threshold and first.model_id == second.model_id
    assert first.estimator.estimators_[0].tree_.threshold.tolist() == second.estimator.estimators_[0].tree_.threshold.tolist()
    predictions = predict_anomaly(second, changed)
    assert predictions.loc[test, "is_anomaly"].notna().all()
    assert predictions.loc[:4, "is_anomaly"].isna().all()
    assert predictions.loc[:4, "anomaly_score"].isna().all()


def test_calibration_does_not_fit_the_preprocessor_and_threshold_is_quantile(ml_features, ml_config):
    original = fit_baseline(ml_features, ml_config)
    changed = ml_features.copy()
    calibration = split_names(changed["timestamp"], ml_config).eq("calibration")
    changed.loc[calibration, "Oil_temperature_mean"] = 1e6
    model = fit_baseline(changed, ml_config)
    np.testing.assert_equal(original.preprocessor.statistics_, model.preprocessor.statistics_)
    assert original.estimator.estimators_[0].tree_.threshold.tolist() == model.estimator.estimators_[0].tree_.threshold.tolist()
    matrix = model.preprocessor.transform(changed.loc[calibration, list(FEATURE_COLUMNS)])
    scores = -model.estimator.score_samples(matrix)
    assert model.threshold == np.quantile(scores, 0.99, method="higher")
    assert np.mean(scores > model.threshold) <= 0.01


def test_imputation_medians_use_training_values_only(ml_features, ml_config):
    changed = ml_features.copy()
    changed.loc[8, "TP2_mean"] = np.nan
    model = fit_baseline(changed, ml_config)
    training = changed.loc[changed["timestamp"].le(pd.Timestamp(ml_config.train_end)) & changed["eligible"]]
    column = list(FEATURE_COLUMNS).index("TP2_mean")
    assert model.preprocessor.statistics_[column] == training["TP2_mean"].median()


def test_saving_loading_and_scoring_batches_preserve_scores(ml_features, ml_config, tmp_path):
    model = fit_baseline(ml_features, ml_config)
    before = predict_anomaly(model, ml_features, batch_size=7)
    path = save_model(model, tmp_path)
    loaded = load_model(path)
    after = predict_anomaly(loaded, ml_features, batch_size=100)
    pd.testing.assert_frame_equal(before, after)
    metadata = json.loads(path.with_name("manifest.json").read_text())
    assert metadata["feature_columns"] == list(FEATURE_COLUMNS)
    assert len(metadata["artifact_sha256"]) == 64
    assert "probabilité" in metadata["score_definition"]


@pytest.mark.parametrize("options", [
    {"train_end": "2020-03-02"}, {"train_end": "NaT"}, {"train_end": "2020-02-22T00:00:00Z"},
    {"calibration_tail_fraction": 0}, {"n_estimators": 1},
])
def test_invalid_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        ModelConfig(**options)


def test_invalid_schema_and_insufficient_partitions_are_rejected(ml_features, ml_config):
    with pytest.raises(ValueError, match="manquantes"):
        fit_baseline(ml_features.drop(columns=[FEATURE_COLUMNS[0]]), ml_config)
    with pytest.raises(ValueError, match="Pas assez"):
        fit_baseline(ml_features.iloc[:70], ml_config)
    with pytest.raises(ValueError, match="chronologiques"):
        fit_baseline(ml_features.iloc[::-1], ml_config)
    model = fit_baseline(ml_features, ml_config)
    with pytest.raises(ValueError):
        predict_anomaly(model, ml_features, batch_size=0)


def test_training_cap_is_deterministic_and_only_uses_train(ml_features, ml_config):
    from dataclasses import replace
    config = replace(ml_config, max_train_windows=20)
    first, second = fit_baseline(ml_features, config), fit_baseline(ml_features, config)
    assert first.metadata["train_windows_fitted"] == 20
    assert first.model_id == second.model_id and first.threshold == second.threshold
