"""Campagne de test SYNTHÉTIQUE, aucune API ni donnée industrielle requise."""

import copy

import pytest

from metropt_maintenance.evaluation.campaign import (CountingMock, SCENARIOS, audit_diagnostic,
    run_ablation, run_campaign, summarize)
from metropt_maintenance.evaluation.real_data import (evaluate_real_retrieval, evaluate_saved_predictions, reference_scores)


@pytest.fixture(scope="module")
def campaign(tmp_path_factory):
    return run_campaign(tmp_path_factory.mktemp("campaign_SYNTHETIC"), repeats=1)


def test_all_ten_scenarios_three_configurations_and_gaps_are_reported(campaign):
    assert len(campaign["rows"]) == 30
    assert {row["scenario"] for row in campaign["rows"]} == {case.identifier for case in SCENARIOS}
    assert all(row["data_kind"] == "synthetic_test" and row["calls"]["real_llm_api"] == 0 for row in campaign["rows"])
    assert {row["configuration"] for row in campaign["rows"]} == {"A", "B", "C"}
    assert not campaign["ml_performance_evaluated"]
    assert all(summary["precision"] is summary["recall"] is summary["f1"] is None for summary in campaign["summary"])
    assert all(summary["diagnostic_clinical_quality"] is None for summary in campaign["summary"])


def test_actual_graph_skips_llm_waits_and_persists_synthetic_reviews(campaign):
    rows = {row["scenario"]: row for row in campaign["rows"] if row["configuration"] == "C"}
    assert rows["normal"]["status"] == "passed" and rows["normal"]["calls"]["mock"] == 0
    assert rows["normal"]["calls"]["retrieve"] == 0
    assert rows["no_documents"]["status"] == "passed" and rows["no_documents"]["calls"]["mock"] == 0
    assert rows["api_failure"]["diagnostic_status"] == "erreur_api"
    assert rows["human_approval"]["paused_before_review"] and rows["human_approval"]["decision"] == "approved_documentary_proposal"
    assert rows["human_rejection"]["decision"] == "rejected"
    assert rows["resume"]["resumed"] and not rows["resume"]["recomputed_on_resume"]
    assert not any(row["approved_before_review"] or row["physical_action_triggered"] for row in rows.values())


def test_conflicts_are_an_explicit_failure_not_a_fabricated_success(campaign):
    for row in campaign["rows"]:
        if row["scenario"] == "contradiction" and row["configuration"] in ("B", "C"):
            assert len(row["references"]) == 2
            assert row["status"] == "failed" and not row["conflict_exposed"]
            assert row["audit"]["fidelity"] == 1
    assert all(row["status"] == "not_supported" for row in campaign["rows"] if row["configuration"] != "C" and row["scenario"] == "resume")


def test_baselines_do_not_receive_fake_ml_facts_or_invent_sources(tmp_path):
    provider = CountingMock()
    diagnostic = run_ablation({"TP3": 8.2}, "Question sans document", [], documents_directory=tmp_path, provider=provider, configuration="A")
    assert provider.calls == 1 and diagnostic["etat_generation"] == "information_insuffisante"
    assert all(fact["origin"] == "measurement" for fact in diagnostic["faits_observes"])
    assert audit_diagnostic(diagnostic, {"TP3": 8.2})["fidelity"] is None
    altered = copy.deepcopy(diagnostic)
    altered["faits_observes"][0]["value"] = 99.0
    assert not audit_diagnostic(altered, {"TP3": 8.2})["facts_preserved"]
    with pytest.raises(ValueError):
        run_ablation({}, "question", [], documents_directory=tmp_path, provider=provider, configuration="C")
    with pytest.raises(ValueError):
        run_campaign(tmp_path, repeats=0)


def test_recall_at_k_counts_all_annotated_pages_and_deduplicates():
    expected = [{"file": "a.pdf", "page": 1}, {"file": "a.pdf", "page": 2}]
    returned = [expected[0], expected[0], {"file": "b.pdf", "page": 1}]
    scores = reference_scores(expected, returned, k=3)
    assert scores["recall"] == 0.5 and scores["hit"] is True
    assert reference_scores([], returned, k=1)["recall"] is None
    with pytest.raises(ValueError): reference_scores(expected, returned, k=0)


def test_real_retrieval_protocol_can_be_verified_with_local_fixtures(rag_index):
    docs, index, encoder = rag_index
    cases = [{"question": "Motor_current", "should_abstain": False, "expected_references": [{"file": "description_test.pdf", "page": 1}]},
             {"question": "Quelle recette de gateau ?", "should_abstain": True}]
    result = evaluate_real_retrieval(cases, index_directory=index, encoder=encoder)
    assert result["scores"]["3"]["macro_annotated_page_recall"] == 1
    assert result["unanswerable_abstention_rate"] == 1
    with pytest.raises(ValueError): evaluate_real_retrieval([], index_directory=index, encoder=encoder)
    with pytest.raises(ValueError): evaluate_real_retrieval([{"question": "Motor_current", "should_abstain": False}], index_directory=index, encoder=encoder)


def test_streaming_predictions_do_not_label_outside_periods(tmp_path):
    from metropt_maintenance.ml.evaluation import DocumentedPeriod
    path = tmp_path / "predictions.csv"
    path.write_text("timestamp,window_start,split,status,is_anomaly,anomaly_score,threshold\n"
                    "2020-01-01 00:01:00,2020-01-01 00:00:00,test,scored,False,0.2,0.6\n"
                    "2020-01-01 00:02:00,2020-01-01 00:01:00,test,scored,True,0.8,0.6\n"
                    "2020-01-01 00:03:00,2020-01-01 00:02:00,test,insufficient_data,,,0.6\n")
    result = evaluate_saved_predictions(path, chunksize=1, periods=[DocumentedPeriod("test", "2020-01-01 00:01:00", "2020-01-01 00:01:00")])
    assert result["counts_by_split"]["test"]["scored_windows"] == 2
    assert result["documented_periods"][0]["alert_windows"] == 1
    assert result["outside_documented_periods"]["scored_windows"] == 1
    assert result["precision"] is result["recall"] is result["f1"] is None
    assert result["representative_timestamps"]["model_flagged"] == "2020-01-01T00:02:00"


def test_report_graphs_and_tables_use_measured_values_and_keep_na(campaign, tmp_path):
    from metropt_maintenance.evaluation.report import figures, write_reports
    paths = write_reports(campaign, None, tmp_path)
    assert all(path.is_file() for path in paths)
    markdown = (tmp_path / "rapport.md").read_text()
    html = (tmp_path / "rapport.html").read_text()
    assert "NA" in markdown and "contradictoire" in markdown
    assert "plotly" in html and "<table" in html
    plots = figures(campaign, None)
    assert sum(plots[0].data[0].y) == sum(row["status"] == "passed" for row in campaign["rows"])
    assert len(plots[2].data[0].y) == 10


def test_cli_produces_offline_report_and_reads_actual_junit(tmp_path):
    from metropt_maintenance.evaluation.__main__ import main, read_test_results
    junit = tmp_path / "junit.xml"
    junit.write_text('<testsuites><testsuite tests="4" failures="1" errors="0" skipped="1" time="1.5" timestamp="2026-11-16T10:00:00"><testcase/></testsuite></testsuites>')
    result = read_test_results(junit)
    assert result["passed"] == 2 and result["failures"] == 1 and result["skipped"] == 1
    assert read_test_results(None) is None
    assert main(["--repeats", "1", "--work-directory", str(tmp_path / "runs"), "--output", str(tmp_path / "report"), "--junit", str(junit)]) == 0
    assert (tmp_path / "report/metrics.json").is_file()


def test_missing_real_artifacts_remain_unmeasured(tmp_path):
    from metropt_maintenance.evaluation.real_data import run_real_evaluation
    result = run_real_evaluation(directory=tmp_path, predictions_path=tmp_path / "absent.csv", index_directory=tmp_path / "no_index")
    assert result["ml"] is result["retrieval"] is result["workflows"] is None
    assert "real_workflow_prerequisites_missing" in result["errors"]


def test_historical_pipeline_evaluator_with_trained_synthetic_model(ml_measurements, ml_features, ml_config, rag_index, tmp_path, monkeypatch):
    from metropt_maintenance.evaluation import real_data
    from metropt_maintenance.ml.model import fit_baseline, save_model
    from metropt_maintenance.rag import search_documents
    docs, index, encoder = rag_index
    csv = tmp_path / "measures_SYNTHETIC.csv"
    ml_measurements.to_csv(csv, index=False)
    model = save_model(fit_baseline(ml_features, ml_config), tmp_path / "model_SYNTHETIC")
    monkeypatch.setattr(real_data, "search_documents", lambda question, **kwargs: search_documents("Motor_current", index_directory=index, encoder=encoder))
    result = real_data.evaluate_real_workflows(directory=tmp_path / "workflow", csv_path=csv, model_path=model,
        index_directory=index, documents_directory=docs, timestamps={"synthetic_window": "2020-02-01T02:10:00"}, repeats=1)
    assert len(result["rows"]) == 3
    assert all(not row["review_submitted"] and not row["human_validated"] for row in result["rows"])
    row = next(row for row in result["rows"] if row["configuration"] == "C")
    assert row["checkpoint_reopened_consistently"] and row["calls"]["monitor"] == 1
    assert row["model_id"] and row["raw_samples"] > 30


def test_reporting_escapes_untrusted_text_and_can_render_real_metrics(campaign, tmp_path):
    from metropt_maintenance.evaluation.report import report_html, write_reports
    assert "<script>" not in report_html("# <script>unsafe</script>\n\n| Name | Value |\n| --- | --- |\n| <script> | 2 |")
    real = {"errors": [], "ml": {"counts_by_split": {"test": {"scored_windows": 2, "alert_windows": 1, "alert_rate": .5}},
        "documented_periods": [], "period_coverage": {"with_observations": 0, "with_at_least_one_alert": 0}, "limits": "Synthétique"},
        "retrieval": {"answerable_count": 0, "unanswerable_count": 1, "indexed_pages": 1, "indexed_chunks": 1,
            "scores": {"1": {"macro_annotated_page_recall": None, "hit_rate": None}}, "unanswerable_abstention_rate": 1,
            "annotation_limits": "Synthétique", "cases": []}, "workflows": {"selection": "Synthétique", "latency_scope": "locale", "rows": []}}
    paths = write_reports(campaign, real, tmp_path)
    assert len(paths) == 3
    assert "50,0 %" in paths[1].read_text()


def test_suspicion_fixture_has_a_pressure_trend_without_failure_label():
    from metropt_maintenance.evaluation.campaign import synthetic_request
    from metropt_maintenance.ingestion.schema import REQUIRED_COLUMNS
    leak = synthetic_request(next(case for case in SCENARIOS if case.identifier == "air_leak"))
    anomaly = synthetic_request(next(case for case in SCENARIOS if case.identifier == "anomaly"))
    uncertain = synthetic_request(next(case for case in SCENARIOS if case.identifier == "uncertain"))
    assert len(leak["samples"]) == 43
    assert leak["samples"][0]["TP3"] > leak["samples"][-1]["TP3"]
    assert anomaly["samples"][0]["TP3"] == anomaly["samples"][-1]["TP3"]
    assert set(leak["samples"][0]) == set(REQUIRED_COLUMNS)
    assert uncertain["samples"][-1]["Oil_temperature"] is None
