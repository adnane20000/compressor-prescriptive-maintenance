"""Test d'intégration du point d'entrée et de son module d'interface."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_app_starts_without_api_key(monkeypatch) -> None:
    """Démarrer la vraie page et vérifier le mode démo et ses limites."""
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LLM_API_KEY"):
        monkeypatch.delenv(key, raising=False)

    entrypoint = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(entrypoint)).run(timeout=15)

    assert not app.exception, "L'application a levé une exception au démarrage."
    assert app.title[0].value == "Dashboard"
    assert len(app.radio[0].options) == 5
    assert any("aucune clé API nécessaire" in caption.value for caption in app.caption)
    assert any("diagnostic confirmé" in caption.value for caption in app.caption)
    assert any("Validation humaine" in caption.value for caption in app.caption)


def _widget(widgets, label):
    return next(widget for widget in widgets if widget.label == label)


def test_dashboard_explores_manual_csv_and_replays_without_api(make_csv, tmp_path, monkeypatch):
    """Exercer les boutons, le rapport et la fin du rejeu sur des données synthétiques."""
    path = make_csv()
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    entrypoint = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(entrypoint)).run(timeout=15)
    _widget(app.text_input, "Chemin du CSV local").set_value(str(path)).run()
    _widget(app.checkbox, "Limiter l'analyse à un préfixe du fichier").check().run()
    _widget(app.number_input, "Nombre maximal de lignes (mode limité)").set_value(3).run()
    _widget(app.button, "Analyser le CSV").click().run(timeout=15)
    assert not app.exception
    assert app.session_state["analysis"][0]["rows_analyzed"] == 3
    assert "PRÉFIXE LIMITÉ" in app.success[0].value
    assert len(app.get("plotly_chart")) == 4
    assert (tmp_path / "data/local/exploration/report.html").exists()
    _widget(app.button, "Initialiser le rejeu").click().run()
    _widget(app.button, "Avancer dans le rejeu").click().run()
    assert len(app.session_state["replay_rows"]) == 1
    _widget(app.selectbox, "Mesures par pas").select(100).run()
    _widget(app.button, "Avancer dans le rejeu").click().run()
    assert not app.exception
    assert len(app.session_state["replay_rows"]) == 6
    assert any("Fin du rejeu" in success.value for success in app.success)
    _widget(app.text_input, "Chemin du CSV local").set_value(str(tmp_path / "absent.csv")).run()
    _widget(app.button, "Analyser le CSV").click().run()
    assert not app.exception
    assert any("introuvable" in error.value for error in app.error)


def test_dashboard_reports_invalid_replay_dates(make_csv, tmp_path, monkeypatch):
    path = make_csv([{"timestamp": "bad"}])
    monkeypatch.chdir(tmp_path)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run()
    _widget(app.text_input, "Chemin du CSV local").set_value(str(path)).run()
    _widget(app.button, "Initialiser le rejeu").click().run()
    _widget(app.button, "Avancer dans le rejeu").click().run()
    assert not app.exception
    assert any("horodatage" in error.value for error in app.error)


def test_dashboard_runs_ml_and_shows_scores(ml_measurements, ml_config, tmp_path, monkeypatch):
    from metropt_maintenance.ml.workflow import run_baseline
    path = tmp_path / "synthetic_ml.csv"
    ml_measurements.to_csv(path, index=False)
    monkeypatch.chdir(tmp_path)
    def small_baseline(path, **kwargs):
        return run_baseline(path, config=ml_config, chunksize=100)
    monkeypatch.setattr("metropt_maintenance.ml.workflow.run_baseline", small_baseline)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=15)
    _widget(app.radio, "Navigation").set_value("Détection d'anomalies").run()
    _widget(app.text_input, "Chemin du CSV local").set_value(str(path)).run()
    _widget(app.button, "Entraîner et évaluer Isolation Forest").click().run(timeout=30)
    assert not app.exception
    assert app.session_state["ml_result"].metrics["total_observed_windows"] == 180
    assert len(app.get("plotly_chart")) == 1
    _widget(app.text_input, "Fin de l'entraînement ML (exclusive pour les mesures brutes)").set_value("invalid").run()
    _widget(app.button, "Entraîner et évaluer Isolation Forest").click().run()
    assert not app.exception and len(app.error) > 0


def test_dashboard_indexes_pdf_searches_and_abstains(rag_corpus, rag_encoder, tmp_path, monkeypatch):
    directory, _ = rag_corpus
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("metropt_maintenance.rag.index.default_encoder", lambda: rag_encoder)
    monkeypatch.setattr("metropt_maintenance.rag.search.default_encoder", lambda: rag_encoder)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=15)
    _widget(app.radio, "Navigation").set_value("Assistant RAG").run()
    _widget(app.text_input, "Dossier des PDF").set_value(str(directory)).run()
    _widget(app.button, "Indexer les PDF").click().run(timeout=20)
    assert not app.exception and app.session_state["rag_index_summary"]["pdf_count"] == 3
    _widget(app.text_input, "Question documentaire").set_value("AMDEC criticite effets").run()
    _widget(app.button, "Rechercher dans les documents").click().run()
    assert not app.exception
    assert "rag_result" in app.session_state, [error.value for error in app.error]
    assert app.session_state["rag_result"].passages[0].authority == "pedagogical"
    _widget(app.text_input, "Question documentaire").set_value("Quel couple de serrage pour la culasse ?").run()
    _widget(app.button, "Rechercher dans les documents").click().run()
    assert not app.exception
    assert app.session_state["rag_result"].message == "information insuffisante"


def test_all_five_pages_handle_missing_local_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/"app.py")).run(timeout=15)
    for page in ["Dashboard","Détection d'anomalies","Assistant RAG","Diagnostic agentique","Gestion des interventions"]:
        _widget(app.radio,"Navigation").set_value(page).run(timeout=15)
        assert not app.exception, page
        assert app.title[0].value==page
    assert any("Aucune décision enregistrée" in item.value for item in app.info)
    assert all(button.label!='Approuver' for button in app.button)


def test_synthetic_demo_is_explicit_and_replay_survives_navigation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/"app.py")).run(timeout=15)
    _widget(app.button,"Créer un exemple synthétique").click().run()
    assert not app.exception
    assert any("DONNÉES SYNTHÉTIQUES" in warning.value for warning in app.warning)
    _widget(app.text_input,"Début du rejeu (facultatif, AAAA-MM-JJ HH:MM:SS)").set_value("2020-01-01 00:03:00").run()
    _widget(app.button,"Initialiser le rejeu").click().run()
    _widget(app.button,"Avancer dans le rejeu").click().run()
    assert len(app.session_state['replay_rows'])==1
    _widget(app.radio,"Navigation").set_value("Assistant RAG").run()
    _widget(app.radio,"Navigation").set_value("Dashboard").run()
    assert not app.exception and len(app.session_state['replay_rows'])==1
    assert _widget(app.text_input,"Début du rejeu (facultatif, AAAA-MM-JJ HH:MM:SS)").value=="2020-01-01 00:03:00"


def test_corrupt_model_metrics_are_reported_without_crashing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root=tmp_path/'data/local/ml_baseline'; root.mkdir(parents=True)
    (root/'metrics.json').write_text('not JSON')
    app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/"app.py")).run(timeout=15)
    _widget(app.radio,"Navigation").set_value("Détection d'anomalies").run()
    assert not app.exception and any("Résultats ML illisibles" in item.value for item in app.error)
