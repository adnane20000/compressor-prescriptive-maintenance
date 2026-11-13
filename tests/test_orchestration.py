"""Workflow et décisions sur données SYNTHÉTIQUES, sans API ni équipement."""

import json
import sqlite3
from dataclasses import replace
from functools import partial

import pytest

from metropt_maintenance.diagnostic import generate_diagnostic
from metropt_maintenance.diagnostic.providers import LLMApiError, MockProvider
from metropt_maintenance.maintenance.proposal import prepare_proposal
from metropt_maintenance.orchestration import open_workflow, WorkflowRequest
from metropt_maintenance.orchestration.graph import build_workflow
from metropt_maintenance.orchestration.monitoring import monitor_measurements
from metropt_maintenance.orchestration.nodes import WorkflowPersistenceError, WorkflowServices
from metropt_maintenance.rag import search_documents
from metropt_maintenance.rag.search import insufficient
from metropt_maintenance.storage.decisions import DecisionRecord, DecisionStore


@pytest.fixture
def workflow_request(ml_measurements):
    samples = ml_measurements.tail(37).copy()
    samples["timestamp"] = samples["timestamp"].map(lambda value: value.isoformat())
    return {"samples": samples.to_dict(orient="records"), "data_context": "synthetic_test", "question": "Motor_current"}


@pytest.fixture
def workflow_services(rag_index, tmp_path):
    docs, index, encoder = rag_index
    calls = {"monitor": 0, "retrieve": 0, "diagnose": 0, "llm": 0}
    event = {"available_at": "2020-02-01T03:00:00", "model_id": "SYNTHETIC-WORKFLOW-TEST",
             "anomaly_score": 0.8, "threshold": 0.65, "is_anomaly": True,
             "status": "scored", "data_context": "synthetic_test"}
    def monitor(request):
        calls["monitor"] += 1
        return {"event": dict(event), "measurements": {"Motor_current": 4.0}, "summary": {"scope": "synthetic stub"}}
    def retrieve(question):
        calls["retrieve"] += 1
        return search_documents(question, index_directory=index, encoder=encoder)
    class CountingMock(MockProvider):
        def complete(self, payload):
            calls["llm"] += 1
            return super().complete(payload)
    def diagnose(*args):
        calls["diagnose"] += 1
        return generate_diagnostic(*args, documents_directory=docs, provider=CountingMock())
    services = WorkflowServices(monitor, retrieve, diagnose, DecisionStore(tmp_path / "decisions.sqlite"))
    return services, calls, event


def submission(state, decision="approve"):
    return {"proposal_id": state["proposal"]["proposal_id"], "decision": decision,
            "reviewer": "Test SYNTHETIQUE", "comment": "Validation de test du dossier documentaire simulé.", "acknowledged": True}


def test_normal_window_closes_without_rag_llm_or_human_validation(workflow_request, workflow_services, tmp_path, caplog):
    services, calls, event = workflow_services
    event.update(anomaly_score=0.5, is_anomaly=False)
    caplog.set_level("INFO", logger="metropt.workflow")
    with open_workflow(directory=tmp_path / "checkpoints", services=services) as runner:
        state = runner.start(workflow_request)
        assert state["phase"] == "finalized" and not state["interrupts"]
        assert state["proposal"] is None and state["review"] is None
        assert state["decision"]["outcome"] == "closed_without_alert"
        assert not state["decision"]["human_validated"] and not state["decision"]["intervention_authorized"]
        assert calls == {"monitor": 1, "retrieve": 0, "diagnose": 0, "llm": 0}
        assert [entry["node"] for entry in state["journal"]] == ["monitor_node", "anomaly_router", "finalize_node"]
        assert services.decisions.get(state["run_id"]) == state["decision"]
    assert len(caplog.records) == 3


def test_anomaly_pauses_with_intermediates_without_a_decision(workflow_request, workflow_services, tmp_path, monkeypatch):
    import httpx
    services, calls, _ = workflow_services
    monkeypatch.setenv("LLM_API_KEY", "must-never-be-checkpointed")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    def no_network(*args, **kwargs): raise AssertionError("Aucun appel distant attendu")
    monkeypatch.setattr(httpx.Client, "send", no_network)
    with open_workflow(directory=tmp_path / "checkpoints", services=services) as runner:
        state = runner.start(workflow_request)
        assert state["phase"] == "awaiting_human" and state["pending_nodes"] == ["human_review_node"]
        assert len(state["interrupts"]) == 1 and state["review"] is None and state["decision"] is None
        assert state["ml_event"] and state["retrieval"]["passages"] and state["diagnostic"]["simulation"]
        assert state["proposal"]["approval_allowed"] and state["proposal"]["physical_procedure"] is None
        assert all(ref["page"] >= 1 for ref in state["proposal"]["documentary_references"])
        assert services.decisions.get(state["run_id"]) is None
        assert calls == {"monitor": 1, "retrieve": 1, "diagnose": 1, "llm": 1}
        json.dumps(state, allow_nan=False)
    for path in (tmp_path / "checkpoints").glob("*"):
        assert b"must-never-be-checkpointed" not in path.read_bytes()


@pytest.mark.parametrize("decision, outcome, validated", [
    ("approve", "approved_documentary_proposal", True),
    ("reject", "rejected", False),
    ("request_information", "information_requested", False),
])
def test_only_explicit_reviews_finalize_and_never_trigger_equipment(workflow_request, workflow_services, tmp_path, decision, outcome, validated):
    services, calls, _ = workflow_services
    with open_workflow(directory=tmp_path / "checkpoints", services=services) as runner:
        initial = runner.start(workflow_request)
        state = runner.review(initial["run_id"], submission(initial, decision))
        assert state["decision"]["outcome"] == outcome and state["decision"]["human_validated"] is validated
        assert state["decision"]["review"]["submission"]["reviewer"] == "Test SYNTHETIQUE"
        assert not state["decision"]["intervention_authorized"] and not state["decision"]["physical_action_triggered"]
        assert not state["decision"]["diagnosis_confirmed"]
        assert state["phase"] == "finalized" and not state["pending_nodes"]
        assert calls["llm"] == 1
        assert [entry["sequence"] for entry in state["journal"]] == list(range(1, len(state["journal"]) + 1))
        with pytest.raises(ValueError, match="n'attend pas"):
            runner.review(state["run_id"], submission(initial, decision))


@pytest.mark.parametrize("case", ["empty", "null", "boolean", "missing_ack", "numeric_ack", "wrong_id", "blank_name", "blank_comment", "extra"])
def test_invalid_reviews_reinterrupt_without_repeating_ml_rag_llm(workflow_request, workflow_services, tmp_path, case):
    services, calls, _ = workflow_services
    with open_workflow(directory=tmp_path / "checkpoints", services=services) as runner:
        initial = runner.start(workflow_request)
        review = submission(initial)
        if case == "empty": review = {}
        if case == "null": review = None
        if case == "boolean": review["decision"] = True
        if case == "missing_ack": review.pop("acknowledged")
        if case == "numeric_ack": review["acknowledged"] = 1
        if case == "wrong_id": review["proposal_id"] = "0" * 64
        if case == "blank_name": review["reviewer"] = "   "
        if case == "blank_comment": review["comment"] = "   "
        if case == "extra": review["intervention_authorized"] = True
        paused = runner.review(initial["run_id"], review)
        assert paused["review"] is None and paused["decision"] is None and paused["interrupts"]
        assert paused["review_error"] and paused["journal"][-1]["status"] == "invalid_review"
        assert calls == {"monitor": 1, "retrieve": 1, "diagnose": 1, "llm": 1}
        final = runner.review(initial["run_id"], submission(initial, "reject"))
        assert final["decision"]["outcome"] == "rejected"


@pytest.mark.parametrize("problem", ["missing_sources", "retrieval_error", "insufficient_history", "monitor_error", "unsafe_query"])
def test_insufficient_or_failed_paths_cannot_be_approved(workflow_request, workflow_services, tmp_path, problem):
    services, calls, event = workflow_services
    def failure(*args): raise RuntimeError("secret-or-pdf-text-not-to-log")
    if problem == "missing_sources": services = replace(services, retrieve=lambda question: insufficient(question, "index_missing"))
    if problem == "retrieval_error": services = replace(services, retrieve=failure)
    if problem == "insufficient_history": event.update(status="insufficient_history", anomaly_score=None, is_anomaly=None)
    if problem == "monitor_error": services = replace(services, monitor=failure)
    if problem == "unsafe_query": workflow_request["question"] = "Comment shunter le pressostat de sécurité ?"
    with open_workflow(directory=tmp_path / "checkpoints", services=services) as runner:
        initial = runner.start(workflow_request)
        assert initial["phase"] == "awaiting_human" and not initial["proposal"]["approval_allowed"]
        assert calls["diagnose"] == calls["llm"] == 0
        assert "secret-or-pdf-text-not-to-log" not in json.dumps(initial)
        paused = runner.review(initial["run_id"], submission(initial))
        assert paused["decision"] is None and paused["interrupts"]
        final = runner.review(initial["run_id"], submission(initial, "request_information"))
        assert final["decision"]["outcome"] == "information_requested" and not final["decision"]["human_validated"]


@pytest.mark.parametrize("problem", ["api", "invalid_json"])
def test_diagnostic_api_errors_remain_visible_and_block_approval(workflow_request, workflow_services, rag_index, tmp_path, problem):
    services, _, _ = workflow_services
    docs, _, _ = rag_index
    class FailedProvider:
        name, model, simulation = "test-api", "test-model", False
        def complete(self, payload):
            if problem == "api": raise LLMApiError("Délai du fournisseur dépassé.")
            return "invalid JSON"
    services = replace(services, diagnose=partial(generate_diagnostic, documents_directory=docs, provider=FailedProvider()))
    with open_workflow(directory=tmp_path / "checkpoints", services=services) as runner:
        state = runner.start(workflow_request)
    expected = "erreur_api" if problem == "api" else "reponse_invalide"
    assert state["diagnostic"]["etat_generation"] == expected
    assert state["errors"][-1]["code"] == expected and not state["proposal"]["approval_allowed"]
    assert not state["diagnostic"]["simulation"]


def test_checkpoint_survives_reopening_without_recomputing_nodes(workflow_request, workflow_services, tmp_path):
    services, calls, _ = workflow_services
    directory = tmp_path / "restart"
    with open_workflow(directory=directory, services=services) as first:
        paused = first.start(workflow_request)
        proposal_id = paused["proposal"]["proposal_id"]
    with open_workflow(directory=directory, services=services) as reopened:
        restored = reopened.get_state(paused["run_id"])
        assert restored["proposal"]["proposal_id"] == proposal_id and restored["interrupts"]
        assert restored["diagnostic"] == paused["diagnostic"]
        final = reopened.review(paused["run_id"], submission(restored, "reject"))
        assert final["decision"]["outcome"] == "rejected"
    assert calls == {"monitor": 1, "retrieve": 1, "diagnose": 1, "llm": 1}


def test_finalization_retry_is_idempotent_after_a_crash(workflow_request, workflow_services, tmp_path):
    services, calls, _ = workflow_services
    class WriteThenFail(DecisionStore):
        failed = False
        def save(self, record):
            saved = super().save(record)
            if not self.failed:
                self.failed = True
                raise sqlite3.OperationalError("synthetic crash after commit")
            return saved
    store = WriteThenFail(tmp_path / "retry-decisions.sqlite")
    services = replace(services, decisions=store)
    with open_workflow(directory=tmp_path / "checkpoints", services=services) as runner:
        paused = runner.start(workflow_request)
        with pytest.raises(WorkflowPersistenceError) as error:
            runner.review(paused["run_id"], submission(paused))
        assert error.value.run_id == paused["run_id"]
        assert runner.get_state(paused["run_id"])["pending_nodes"] == ["finalize_node"]
        saved = store.get(paused["run_id"])
        final = runner.retry_finalization(paused["run_id"])
        assert final["decision"] == saved and final["phase"] == "finalized"
        with sqlite3.connect(store.path) as connection:
            assert connection.execute("SELECT count(*) FROM workflow_decisions").fetchone()[0] == 1
        with pytest.raises(ValueError, match="Aucune finalisation"):
            runner.retry_finalization(paused["run_id"])
    assert calls["llm"] == calls["monitor"] == 1


def test_default_adapters_use_existing_ml_without_refitting(workflow_request, ml_features, ml_config, tmp_path, monkeypatch):
    from metropt_maintenance.ml.model import fit_baseline, predict_anomaly, save_model
    from metropt_maintenance.ml.workflow import prediction_to_record
    model = fit_baseline(ml_features, ml_config)
    path = save_model(model, tmp_path / "model")
    expected = prediction_to_record(predict_anomaly(model, ml_features).iloc[-1].to_dict(), model_id=model.model_id)
    def forbid_fit(*args, **kwargs): raise AssertionError("Aucun ajustement pendant le workflow")
    monkeypatch.setattr("sklearn.ensemble.IsolationForest.fit", forbid_fit)
    monitored = monitor_measurements(WorkflowRequest.model_validate(workflow_request), model_path=str(path))
    assert monitored["event"]["anomaly_score"] == pytest.approx(expected["anomaly_score"])
    assert monitored["event"]["is_anomaly"] == expected["is_anomaly"]
    assert monitored["event"]["available_at"] == expected["available_at"]
    assert monitored["event"]["data_context"] == "synthetic_test"
    assert monitored["summary"]["closed_windows"] == 6
    with open_workflow(directory=tmp_path / "real-adapters", model_path=path, index_directory=tmp_path / "no-index") as runner:
        state = runner.start(workflow_request)
        assert state["ml_event"]["anomaly_score"] == pytest.approx(expected["anomaly_score"])
        if expected["is_anomaly"]:
            assert state["phase"] == "awaiting_human" and state["retrieval"]["reason"] == "index_missing"
        else:
            assert state["decision"]["outcome"] == "closed_without_alert"


@pytest.mark.parametrize("case", ["sensor", "nonfinite", "unmarked_context", "date", "preapproval", "too_many", "out_of_order"])
def test_invalid_input_and_unordered_measurements_are_not_silently_accepted(workflow_request, case, ml_features, ml_config, tmp_path):
    from metropt_maintenance.ml.model import fit_baseline, save_model
    if case == "sensor": workflow_request["samples"][0]["invented_sensor"] = 12
    if case == "nonfinite": workflow_request["samples"][0]["TP2"] = float("nan")
    if case == "unmarked_context": workflow_request.pop("data_context")
    if case == "date": workflow_request["samples"][0]["timestamp"] = "not-a-date"
    if case == "preapproval": workflow_request["review"] = {"decision": "approve"}
    if case == "too_many": workflow_request["samples"] = workflow_request["samples"] * 200
    if case == "out_of_order":
        path = save_model(fit_baseline(ml_features, ml_config), tmp_path / "model")
        workflow_request["samples"] = list(reversed(workflow_request["samples"]))
        with pytest.raises(ValueError, match="chronologiques"):
            monitor_measurements(WorkflowRequest.model_validate(workflow_request), model_path=str(path))
    else:
        with pytest.raises(ValueError): WorkflowRequest.model_validate(workflow_request)


def test_graph_requires_checkpoint_and_has_all_requested_nodes(workflow_services, tmp_path):
    services, _, _ = workflow_services
    with pytest.raises(ValueError, match="checkpointer"):
        build_workflow(checkpointer=None, services=services)
    with open_workflow(directory=tmp_path / "checkpoints", services=services) as runner:
        drawing = runner.graph.get_graph().draw_mermaid()
        for node in ("monitor_node", "anomaly_router", "retrieval_node", "diagnosis_node", "planning_node", "human_review_node", "finalize_node"):
            assert node in drawing
        with pytest.raises(ValueError): runner.get_state("invalid-thread")
        with pytest.raises(KeyError): runner.get_state("0" * 32)


def test_proposal_identity_is_stable_and_changes_with_content():
    first = prepare_proposal("synthetic", None, blockers=["Données absentes"])
    second = prepare_proposal("synthetic", None, blockers=["Données absentes"])
    changed = prepare_proposal("synthetic", None, blockers=["Documents absents"])
    assert first.proposal_id == second.proposal_id and first.proposal_id != changed.proposal_id
    assert first.status == "proposition non validée" and not first.approval_allowed


def test_decision_store_is_idempotent_and_refuses_inconsistent_decisions(tmp_path):
    store = DecisionStore(tmp_path / "decisions.sqlite")
    record = DecisionRecord(run_id="synthetic", recorded_at_utc="2026-11-13T00:00:00+00:00", outcome="closed_without_alert",
                            human_validated=False, review=None, proposal_id=None, ml_event=None, simulation=None)
    saved = store.save(record)
    assert store.save(record.model_copy(update={"recorded_at_utc": "later"})) == saved
    with pytest.raises(ValueError, match="différente"):
        store.save(record.model_copy(update={"ml_event": {"model_id": "different"}}))
    with pytest.raises(ValueError): DecisionRecord.model_validate({**record.model_dump(), "human_validated": True})
    with pytest.raises(ValueError): DecisionRecord.model_validate({**record.model_dump(), "intervention_authorized": True})


def test_loads_a_bounded_csv_window_without_silent_truncation(ml_measurements, tmp_path):
    from metropt_maintenance.orchestration.monitoring import load_measurement_window
    path = tmp_path / "synthetic.csv"
    ml_measurements.to_csv(path, index=False)
    kwargs = dict(data_context="synthetic_test", question="Motor_current")
    request = load_measurement_window(path, "2020-02-01 02:54:00", "2020-02-01 03:00:00", **kwargs)
    assert len(request.samples) == 37 and request.samples[-1]["timestamp"] == "2020-02-01T03:00:00"
    with pytest.raises(ValueError, match="Trop de mesures"):
        load_measurement_window(path, "2020-02-01 02:54:00", "2020-02-01 03:00:00", max_samples=10, **kwargs)
    with pytest.raises(ValueError): load_measurement_window(path, "bad", "bad", **kwargs)
    with pytest.raises(ValueError): load_measurement_window(path, "2020-02-01", "2020-02-01", **kwargs)
    ml_measurements.iloc[::-1].to_csv(path, index=False)
    with pytest.raises(ValueError, match="chronologiques"):
        load_measurement_window(path, "2020-02-01 02:54:00", "2020-02-01 03:00:00", **kwargs)


def test_unexpected_diagnostic_error_is_sanitized(workflow_request, workflow_services, tmp_path):
    services, _, _ = workflow_services
    def failure(*args): raise RuntimeError("untrusted-secret-not-to-publish")
    with open_workflow(directory=tmp_path / "checkpoints", services=replace(services, diagnose=failure)) as runner:
        state = runner.start(workflow_request)
    assert state["diagnostic"] is None and not state["proposal"]["approval_allowed"]
    assert state["errors"][-1]["code"] == "diagnostic_unavailable_or_invalid"
    assert "untrusted-secret-not-to-publish" not in json.dumps(state)


def test_streamlit_requires_manual_review_and_reopens_saved_thread(workflow_services, ml_measurements, tmp_path, monkeypatch):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    services, calls, _ = workflow_services
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    path = tmp_path / "synthetic.csv"
    ml_measurements.to_csv(path, index=False)
    def configured_workflow(**kwargs):
        return open_workflow(directory=tmp_path / "ui-workflow", services=services)
    monkeypatch.setattr("metropt_maintenance.orchestration.open_workflow", configured_workflow)
    app_path = str(Path(__file__).resolve().parents[1] / "app.py")
    app = AppTest.from_file(app_path).run(timeout=15)
    def widget(widgets, label): return next(item for item in widgets if item.label == label)
    widget(app.text_input, "Chemin du CSV local").set_value(str(path)).run()
    widget(app.text_input, "Début du contexte du workflow").set_value("2020-02-01 02:54:00").run()
    widget(app.text_input, "Fin des mesures du workflow").set_value("2020-02-01 03:00:00").run()
    widget(app.text_input, "Question documentaire du workflow").set_value("Motor_current").run()
    widget(app.selectbox, "Origine déclarée des mesures du workflow").select("Synthétique (test)").run()
    widget(app.button, "Démarrer le workflow sur cette fenêtre").click().run(timeout=20)
    assert not app.exception
    run_id = app.session_state["workflow_active_id"]
    assert widget(app.button, "Envoyer la revue humaine").disabled
    assert services.decisions.get(run_id) is None
    # Nouveau navigateur simulé : l'état est retrouvé dans SQLite, pas dans session_state.
    reopened = AppTest.from_file(app_path).run(timeout=15)
    widget(reopened.text_input, "Identifiant du dossier à rouvrir").set_value(run_id).run()
    widget(reopened.button, "Rouvrir le dossier enregistré").click().run()
    assert not reopened.exception and reopened.session_state["workflow_active_id"] == run_id
    widget(reopened.selectbox, "Décision humaine sur ce dossier").select("Approuver la préparation documentaire").run()
    widget(reopened.text_input, "Identité du relecteur").set_value("Relecteur SYNTHETIQUE").run()
    widget(reopened.text_area, "Commentaire de la revue").set_value("Test simulé du parcours documentaire.").run()
    assert widget(reopened.button, "Envoyer la revue humaine").disabled
    widget(reopened.checkbox, "Je confirme cette revue documentaire ; aucune intervention physique n'est autorisée.").check().run()
    widget(reopened.button, "Envoyer la revue humaine").click().run()
    assert not reopened.exception
    assert services.decisions.get(run_id)["outcome"] == "approved_documentary_proposal"
    assert not services.decisions.get(run_id)["intervention_authorized"]
    assert calls["monitor"] == calls["llm"] == 1
