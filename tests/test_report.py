"""Agrégats comparés aux mesures et rapport autonome avec graphiques."""

import json

import pytest

from metropt_maintenance.ingestion.exploration import profile_csv
from metropt_maintenance.ingestion.report import aggregate_charts, build_figures, write_report


def test_aggregation_keeps_extrema_across_chunks(make_csv, tmp_path):
    path = make_csv([{"Oil_temperature": value} for value in [50, 200, 40, 60, 70, 80]])
    profile = profile_csv(path, chunksize=2)
    charts = aggregate_charts(path, profile, max_bins=1)
    assert len(charts) == 1
    assert charts["Oil_temperature_min"].iloc[0] == 40
    assert charts["Oil_temperature_max"].iloc[0] == 200
    assert charts["Oil_temperature_mean"].iloc[0] == pytest.approx(500 / 6)
    assert charts["Oil_temperature_count"].iloc[0] == 6
    figures = build_figures(profile, charts)
    assert len(figures) == 5
    assert "°C" in figures[0].layout.yaxis.title.text
    target = write_report(profile, charts, tmp_path / "report")
    content = target.read_text()
    assert "plotly" in content.lower() and "Aucune acquisition en temps réel" in content
    assert '<script src="https://cdn.plot.ly' not in content
    stored = json.loads(target.with_name("profile.json").read_text())
    assert stored["charts"]["bins"] == 1
    assert stored["failure_labels_created"] is False
    assert target.with_name("chart_aggregates.csv").is_file()


def test_empty_windows_remain_empty_and_prefix_chart_matches_profile(make_csv):
    path = make_csv([{}, {"timestamp": "2020-02-01 01:00:00"}, {"timestamp": "2020-02-01 02:00:00"}])
    profile = profile_csv(path, max_rows=2)
    charts = aggregate_charts(path, profile, max_bins=10)
    assert len(charts) <= 10
    assert charts["TP2_count"].sum() == 2
    assert charts["TP2_mean"].isna().sum() > 0
    with pytest.raises(ValueError):
        aggregate_charts(path, profile, max_bins=0)


def test_all_invalid_dates_still_produce_quality_report(make_csv, tmp_path):
    path = make_csv([{"timestamp": "bad"}])
    profile = profile_csv(path)
    charts = aggregate_charts(path, profile)
    assert charts.empty
    assert len(build_figures(profile, charts)) == 1
    assert write_report(profile, charts, tmp_path / "no-dates").is_file()
