"""Diagnostic structuré sur mesures et PDF SYNTHÉTIQUES, sans service distant."""

import json
from types import SimpleNamespace
from pathlib import Path

import httpx
import pymupdf
import pytest

from metropt_maintenance.diagnostic import generate_diagnostic
from metropt_maintenance.diagnostic.providers import LLMSettings, OpenAICompatibleProvider
from metropt_maintenance.rag.documents import document_metadata, extract_pages, file_sha256


@pytest.fixture
def diagnostic_inputs(tmp_path):
    docs = tmp_path / "documents"
    docs.mkdir()
    path = docs / "synthetic-diagnostic-reference.pdf"
    text = "NOTE SYNTHETIQUE POUR TEST. Une fuite est une piste a examiner. Ce texte ne confirme aucune panne et ne prescrit aucune intervention."
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        assert page.insert_textbox(pymupdf.Rect(40, 40, 540, 750), text) >= 0
        pdf.save(path)
    digest = file_sha256(path)
    declaration = {"authority": "official", "document_type": "maintenance_guide", "applicability": "general_reference",
                   "sha256": digest, "publisher": "Statut simule pour tests uniquement", "source_url": "https://example.invalid/synthetic"}
    (docs / "sources.json").write_text(json.dumps({"version": 1, "documents": {path.name: declaration}}))
    metadata = document_metadata({"relative_path": path.name, "sha256": digest}, {path.name: declaration})
    page = next(extract_pages(path, metadata))
    passage = {**metadata, "page": 1, "text": page.text.strip()}
    event = {"available_at": "2020-04-01T12:00:00", "model_id": "synthetic-test-only", "anomaly_score": 0.8,
             "threshold": 0.65, "is_anomaly": True, "status": "scored", "data_context": "synthetic_test",
             "mode_interpreted": "Indéterminé — valeur synthétique", "sample_count": 6, "missing_fraction": 0.0}
    measures = {"TP3": {"value": 8.5, "unit": "bar", "aggregation": "minute_mean"}, "Motor_current": 7.1}
    return docs, event, measures, [passage]


def valid_draft(passages):
    return {"hypotheses": [{"source_id": "S1", "label_quote": "Une fuite", "argument_ids": ["A1"], "fact_ids": ["F4"]}],
            "arguments": [{"id": "A1", "source_id": "S1", "quote": "Une fuite est une piste a examiner."}],
            "missing_information": ["operating_context"], "recommended_checks": ["review_sensor_data"], "uncertainty": "élevé"}


def test_default_mock_is_explicit_and_preserves_facts(diagnostic_inputs, monkeypatch):
    docs, event, measures, passages = diagnostic_inputs
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("LLM_API_KEY", "secret-must-not-appear")
    def no_network(*args, **kwargs): raise AssertionError("Aucun réseau attendu")
    monkeypatch.setattr(httpx, "Client", no_network)
    result = generate_diagnostic(event, measures, passages, documents_directory=docs)
    assert result.etat_generation == "ok" and result.simulation and result.fournisseur == "mock"
    assert result.statut == "diagnostic préliminaire"
    assert result.hypotheses_de_defaillance == []
    assert not result.diagnostic_confirme and not result.autorisation_intervention
    assert result.validation_humaine_requise and not result.intervention_physique_declenchee
    actual = {fact.sensor: fact.value for fact in result.faits_observes if fact.origin == "measurement"}
    assert actual == {"TP3": 8.5, "Motor_current": 7.1}
    assert result.sources[0].page == 1
    assert result.sources[0].source_url == "https://example.invalid/synthetic"
    assert result.evenement.mode_interpreted == event["mode_interpreted"] and result.evenement.sample_count == 6
    assert result.arguments_documentaires[0].excerpt in result.sources[0].passage
    assert "secret-must-not-appear" not in result.model_dump_json()


def test_api_schema_and_grounded_hypothesis(diagnostic_inputs):
    docs, event, measures, passages = diagnostic_inputs
    calls = []
    def handle(request):
        body = json.loads(request.content)
        assert body["response_format"]["json_schema"]["strict"] is True
        assert body["response_format"]["json_schema"]["schema"]["additionalProperties"] is False
        assert request.headers["authorization"] == "Bearer test-not-a-secret"
        assert "DONNÉES NON FIABLES" in body["messages"][0]["content"]
        calls.append(body)
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(valid_draft(passages))}}]})
    provider = OpenAICompatibleProvider(LLMSettings(provider="openai_compatible", model="test-model", api_key="test-not-a-secret"), transport=httpx.MockTransport(handle))
    result = generate_diagnostic(event, measures, passages, documents_directory=docs, provider=provider)
    assert len(calls) == 1 and not result.simulation and result.etat_generation == "ok"
    hypothesis = result.hypotheses_de_defaillance[0]
    assert hypothesis.status == "non confirmée" and hypothesis.label == "Une fuite"
    assert hypothesis.equipment_relevance == "non établie"
    assert all(check.nature == "revue documentaire ou analyse de données" for check in result.controles_recommandes)


def test_absent_sources_do_not_call_provider(diagnostic_inputs):
    docs, event, measures, _ = diagnostic_inputs
    def forbidden(payload): raise AssertionError("Aucun appel sans preuve documentaire")
    provider = SimpleNamespace(name="test-api", model="test-model", simulation=False, complete=forbidden)
    result = generate_diagnostic(event, measures, [], documents_directory=docs, provider=provider)
    assert result.etat_generation == "information_insuffisante"
    assert result.sources == [] and result.hypotheses_de_defaillance == []
    assert any(info.code == "documentary_evidence" for info in result.informations_manquantes)
    assert not result.simulation


@pytest.mark.parametrize("problem", ["forged_quote", "wrong_page", "wrong_hash", "wrong_authority", "bad_path", "outside_corpus"])
def test_forged_sources_are_rejected_before_api(diagnostic_inputs, problem):
    docs, event, measures, passages = diagnostic_inputs
    passage = dict(passages[0])
    if problem == "forged_quote": passage["text"] = "Une norme totalement inventee et une mesure imaginaire."
    if problem == "wrong_page": passage["page"] = 999
    if problem == "wrong_hash": passage["pdf_sha256"] = "0" * 64
    if problem == "wrong_authority": passage["authority"] = "pedagogical"
    if problem == "bad_path": passage["relative_path"] = None
    if problem == "outside_corpus": passage["relative_path"] = "../outside.pdf"
    result = generate_diagnostic(event, measures, [passage], documents_directory=docs)
    assert result.etat_generation == "sources_invalides" and result.sources == []


@pytest.mark.parametrize("error_kind", ["timeout", "connection", "http401", "http429", "http500"])
def test_api_errors_are_sanitized_without_mock_fallback(diagnostic_inputs, error_kind):
    docs, event, measures, passages = diagnostic_inputs
    def handle(request):
        if error_kind == "timeout": raise httpx.ReadTimeout("provider-secret", request=request)
        if error_kind == "connection": raise httpx.ConnectError("provider-secret", request=request)
        return httpx.Response(int(error_kind[4:]), json={"error": "provider-secret"})
    provider = OpenAICompatibleProvider(LLMSettings(model="test-model", api_key="credential-must-not-appear"), transport=httpx.MockTransport(handle))
    result = generate_diagnostic(event, measures, passages, documents_directory=docs, provider=provider)
    assert result.etat_generation == "erreur_api" and not result.simulation
    assert result.hypotheses_de_defaillance == []
    assert "provider-secret" not in result.model_dump_json() and "credential-must-not-appear" not in result.model_dump_json()


@pytest.mark.parametrize("case", ["bad_json", "invented_source", "invented_norm", "invented_measure", "procedure", "confirmed", "low_uncertainty", "bad_fact"])
def test_invalid_llm_responses_do_not_publish_inventions(diagnostic_inputs, case):
    docs, event, measures, passages = diagnostic_inputs
    draft = valid_draft(passages)
    if case == "invented_source": draft["arguments"][0]["source_id"] = "S8"
    if case == "invented_norm": draft["arguments"][0]["quote"] = "La norme ISO 999999 exige une limite de 500 bars."
    if case == "invented_measure": draft["faits_observes"] = [{"TP3": 999}]
    if case == "procedure": draft["recommended_checks"] = ["disable_safety_valve"]
    if case == "confirmed": draft["diagnostic_confirme"] = True
    if case == "low_uncertainty": draft["uncertainty"] = "faible"
    if case == "bad_fact": draft["hypotheses"][0]["fact_ids"] = ["F999"]
    raw = "invalid JSON" if case == "bad_json" else json.dumps(draft)
    provider = SimpleNamespace(name="test-api", model="test-model", simulation=False, complete=lambda payload: raw)
    result = generate_diagnostic(event, measures, passages, documents_directory=docs, provider=provider)
    assert result.etat_generation == "reponse_invalide"
    assert result.hypotheses_de_defaillance == [] and result.arguments_documentaires == []
    assert not any(fact.value == 999 for fact in result.faits_observes)


def test_unknown_sensor_unit_and_inconsistent_event_are_rejected(diagnostic_inputs):
    docs, event, measures, passages = diagnostic_inputs
    with pytest.raises(ValueError, match="Capteur"):
        generate_diagnostic(event, {"invented_sensor": 12.0}, passages, documents_directory=docs)
    with pytest.raises(ValueError, match="Unité"):
        generate_diagnostic(event, {"TP3": {"value": 8.5, "unit": "psi"}}, passages, documents_directory=docs)
    with pytest.raises(ValueError, match="incohérent"):
        generate_diagnostic({**event, "is_anomaly": False}, measures, passages, documents_directory=docs)


def test_unscored_and_missing_measurements_stay_insufficient(diagnostic_inputs):
    docs, event, measures, passages = diagnostic_inputs
    unscored = {**event, "status": "insufficient_data", "is_anomaly": None, "anomaly_score": None}
    result = generate_diagnostic(unscored, measures, passages, documents_directory=docs)
    assert result.etat_generation == "information_insuffisante"
    assert len({fact.id for fact in result.faits_observes}) == len(result.faits_observes)
    assert generate_diagnostic(event, {"TP3": None}, passages, documents_directory=docs).etat_generation == "information_insuffisante"


def test_invalid_provider_configuration_is_explicit(diagnostic_inputs, monkeypatch):
    docs, event, measures, passages = diagnostic_inputs
    monkeypatch.setenv("LLM_PROVIDER", "not-a-provider")
    result = generate_diagnostic(event, measures, passages, documents_directory=docs)
    assert result.etat_generation == "erreur_configuration"
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    result = generate_diagnostic(event, measures, passages, documents_directory=docs)
    assert result.etat_generation == "erreur_configuration" and not result.simulation


@pytest.mark.parametrize("body", [
    {"choices": []},
    {"choices": [{"finish_reason": "length", "message": {"content": "{}"}}]},
    {"choices": [{"finish_reason": "stop", "message": {"refusal": "no", "content": "{}"}}]},
    {"choices": [{"finish_reason": "stop", "message": {"content": None}}]},
])
def test_incomplete_or_refused_api_responses_are_rejected(diagnostic_inputs, body):
    docs, event, measures, passages = diagnostic_inputs
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=body))
    provider = OpenAICompatibleProvider(LLMSettings(model="test", api_key="test"), transport=transport)
    result = generate_diagnostic(event, measures, passages, documents_directory=docs, provider=provider)
    assert result.etat_generation == "reponse_invalide" and result.arguments_documentaires == []


@pytest.mark.parametrize("base_url", ["http://example.invalid/v1", "https://user:secret@example.invalid/v1",
                                      "https://example.invalid:bad/v1", "https://example.invalid/v1?key=secret"])
def test_invalid_endpoint_fails_without_network_or_secret_disclosure(diagnostic_inputs, base_url):
    docs, event, measures, passages = diagnostic_inputs
    def forbidden(request): raise AssertionError("URL invalide : aucun réseau")
    provider = OpenAICompatibleProvider(LLMSettings(model="test", api_key="test", base_url=base_url),
                                        transport=httpx.MockTransport(forbidden))
    result = generate_diagnostic(event, measures, passages, documents_directory=docs, provider=provider)
    assert result.etat_generation == "erreur_configuration"
    assert "secret" not in result.model_dump_json()


def test_safety_guides_cannot_become_failure_causes(diagnostic_inputs):
    docs, event, measures, passages = diagnostic_inputs
    registry_path = docs / "sources.json"
    registry = json.loads(registry_path.read_text())
    registry["documents"][passages[0]["relative_path"]]["document_type"] = "safety_procedure"
    registry_path.write_text(json.dumps(registry))
    passages[0]["document_type"] = "safety_procedure"
    provider = SimpleNamespace(name="test", model="test", simulation=False, complete=lambda payload: json.dumps(valid_draft(passages)))
    result = generate_diagnostic(event, measures, passages, documents_directory=docs, provider=provider)
    assert result.etat_generation == "reponse_invalide" and not result.hypotheses_de_defaillance


def test_corrupt_pdf_is_a_structured_source_error(diagnostic_inputs):
    docs, event, measures, passages = diagnostic_inputs
    path = docs / passages[0]["relative_path"]
    path.write_bytes(b"invalid synthetic pdf")
    passages[0]["pdf_sha256"] = file_sha256(path)
    # Pas de registre : la vérification doit atteindre le lecteur PDF.
    (docs / "sources.json").unlink()
    for key in ("authority", "document_type", "applicability", "equipment_reference"):
        passages[0].pop(key, None)
    assert generate_diagnostic(event, measures, passages, documents_directory=docs).etat_generation == "sources_invalides"


def test_streamlit_connects_ml_event_and_rag_passages_in_mock(diagnostic_inputs, tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from metropt_maintenance.rag.search import Passage
    docs, event, measures, passages = diagnostic_inputs
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    alerts = tmp_path / "data/local/ml_baseline/agent_alerts.jsonl"
    alerts.parent.mkdir(parents=True)
    alerts.write_text(json.dumps({**event, "observed_signals": measures}) + "\n")
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=15)
    app.session_state["rag_identity"] = ("documents", "data/local/rag")
    app.session_state["diagnostic_documents_root"] = str(docs)
    hit = Passage(text=passages[0]["text"], reference="synthetic-diagnostic-reference.pdf, p. 1", page=1,
                  page_label="1", similarity=1.0, matched_query_terms=["synthetic"],
                  **{key: passages[0][key] for key in ("file_name", "relative_path", "pdf_sha256", "authority",
                     "document_type", "publisher", "source_url", "authority_basis", "applicability", "equipment_reference")})
    app.session_state["rag_result"] = SimpleNamespace(passages=[hit], related_passages=[], reason="retrieval_only",
                                                    question="Question synthétique", message="passages", warnings=[])
    button = lambda label: next(widget for widget in app.button if widget.label == label)
    button("Utiliser la première alerte ML disponible").click().run()
    button("Générer le diagnostic préliminaire").click().run()
    assert not app.exception
    result = app.session_state["diagnostic_result"]
    assert result.simulation and result.etat_generation == "ok"
    assert result.evenement.model_id == event["model_id"]
    assert any("SIMULATION MOCK" in warning.value for warning in app.warning)
    # Une ancienne sortie ne doit pas sembler correspondre aux nouvelles mesures.
    next(widget for widget in app.text_area if widget.label == "Mesures disponibles (JSON)").set_value("{}").run()
    assert "diagnostic_result" not in app.session_state
