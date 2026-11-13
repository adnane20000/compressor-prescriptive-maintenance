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
    assert app.title[0].value == "MetroPT — Maintenance prescriptive"
    assert "aucune clé API nécessaire" in app.info[0].value
    assert "diagnostic confirmé" in app.warning[0].value
    assert "validation humaine" in app.warning[0].value


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
    assert len(app.get("plotly_chart")) == 5
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
