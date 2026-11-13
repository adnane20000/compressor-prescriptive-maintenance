"""Rapports externes incomplets : pas de précision/F1 inventés."""

import pandas as pd
import pytest

from metropt_maintenance.ml.evaluation import DOCUMENTED_PERIODS, DocumentedPeriod, evaluate_predictions


def test_official_periods_are_source_bound_and_not_expanded_to_labels():
    assert len(DOCUMENTED_PERIODS) == 4
    assert DOCUMENTED_PERIODS[1].start == "2020-05-29 23:30:00"
    assert DOCUMENTED_PERIODS[2].end == "2020-06-07 14:30:00"
    assert all("archive.ics.uci.edu" in period.source for period in DOCUMENTED_PERIODS)


def test_overlap_counts_unknown_outside_and_unobserved_period():
    dates = pd.to_datetime(["2020-04-18 00:01:00", "2020-04-18 00:02:00", "2020-04-19 00:01:00"])
    data = pd.DataFrame({"timestamp": dates, "window_start": dates - pd.Timedelta(minutes=1),
                         "split": "test", "status": "scored", "mode": "charge probable",
                         "is_anomaly": [True, False, True], "anomaly_score": [0.7, 0.4, 0.8]})
    metrics = evaluate_predictions(data)
    period = metrics["documented_period_comparisons"][0]
    assert period["scored_windows"] == 2 and period["alert_windows"] == 1
    assert period["delay_after_documented_start_seconds"] == 60
    assert metrics["documented_period_comparisons"][1]["has_alert"] is None
    assert metrics["test_alerts_outside_documented_periods"] == 1
    assert metrics["documented_periods_with_observations"] == 1
    assert metrics["test_alert_episodes"] == 2
    assert "F1" in metrics["unavailable_supervised_metrics"]
    assert "precision" not in metrics


def test_training_alerts_are_not_compared_to_failures(ml_features, ml_config):
    from metropt_maintenance.ml.model import fit_baseline, predict_anomaly
    predictions = predict_anomaly(fit_baseline(ml_features, ml_config), ml_features)
    period = DocumentedPeriod("synthetic-interval-for-test-only", "2020-02-01 00:00:00", "2020-02-01 00:59:00")
    metrics = evaluate_predictions(predictions, periods=[period])
    assert metrics["documented_period_comparisons"][0]["available_windows"] == 0
    with pytest.raises(ValueError):
        evaluate_predictions(predictions, periods=[DocumentedPeriod("bad", "2020-04-02", "2020-04-01")])
