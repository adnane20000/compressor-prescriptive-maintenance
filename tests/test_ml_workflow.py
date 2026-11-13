"""Exécution, HTML autonome, JSON agents et CLI sur données synthétiques."""

import json

import pandas as pd
import pytest

from metropt_maintenance.ml.__main__ import main
from metropt_maintenance.ml.model import fit_baseline, predict_anomaly
from metropt_maintenance.ml.report import score_figure, summarize_scores
from metropt_maintenance.ml.workflow import prediction_to_record, run_baseline


def test_agent_record_contains_observations_without_probabilities_or_causes(ml_features, ml_config):
    model = fit_baseline(ml_features, ml_config)
    predictions = predict_anomaly(model, ml_features)
    for index in [0, 20]:
        record = prediction_to_record(predictions.iloc[index].to_dict(), model_id=model.model_id)
        json.dumps(record, allow_nan=False)
        assert record["score_is_failure_probability"] is False
        assert record["cause"] is None and record["diagnosis_confirmed"] is False
        assert set(record["observed_signals"]) == {"Oil_temperature", "Motor_current", "TP2", "TP3"}
    assert prediction_to_record(predictions.iloc[0].to_dict(), model_id=model.model_id)["is_anomaly"] is None


def test_complete_workflow_writes_model_predictions_metrics_and_charts(ml_measurements, ml_config, tmp_path):
    source = tmp_path / "synthetic.csv"
    ml_measurements.to_csv(source, index=False)
    output = tmp_path / "reports"
    result = run_baseline(source, config=ml_config, chunksize=19, output=output, model_directory=tmp_path / "models")
    assert result.model_path.is_file() and result.report_path.is_file()
    assert result.metrics["raw_rows_read"] == len(ml_measurements)
    assert result.metrics["total_observed_windows"] == 180
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["protocol"]["random_state"] == 42
    assert len(metrics["provenance"]["csv_sha256"]) == 64
    assert (output / "predictions.csv").is_file()
    for line in (output / "agent_alerts.jsonl").read_text().splitlines():
        record = json.loads(line)
        assert record["is_anomaly"] is True and record["cause"] is None
        assert record["partition"] == "test"
    html = result.report_path.read_text()
    assert "probabilité" in html and "plotly" in html
    assert '<script src="https://cdn.plot.ly' not in html
    hourly = pd.read_csv(output / "score_plot_data.csv", parse_dates=["timestamp"])
    figure = score_figure(hourly, metrics["threshold"], metrics["protocol"])
    assert len(figure.data[0].x) < metrics["total_observed_windows"]


def test_cli_dispatch_and_errors(ml_measurements, tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace
    fake = SimpleNamespace(report_path=tmp_path / "report.html", model_path=tmp_path / "model.joblib",
                           metrics={"threshold": 0.5, "by_split": [], "documented_periods_with_alert": 0})
    calls = []
    def run(path, **kwargs):
        calls.append(kwargs)
        return fake
    monkeypatch.setattr("metropt_maintenance.ml.__main__.run_baseline", run)
    assert main(["synthetic.csv", "--trees", "10"]) == 0
    assert calls[0]["config"].n_estimators == 10
    assert "report.html" in capsys.readouterr().out
    with pytest.raises(SystemExit) as error:
        main(["absent.csv", "--tail-fraction", "0"])
    assert error.value.code == 2
