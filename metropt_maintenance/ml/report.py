"""Rapport ML descriptif avec scores, charge d'alertes et périodes sourcées."""

import html
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from .evaluation import DOCUMENTED_PERIODS


def summarize_scores(predictions: pd.DataFrame) -> pd.DataFrame:
    """Limiter la visualisation à un maximum et un effectif par heure observée."""
    data = predictions.set_index("timestamp")
    summary = data.resample("h").agg(
        score_max=("anomaly_score", "max"),
        scored_windows=("anomaly_score", "count"),
        alert_windows=("is_anomaly", "sum"),
    )
    return summary.reset_index()


def score_figure(hourly: pd.DataFrame, threshold: float, config: dict, *, periods=DOCUMENTED_PERIODS) -> go.Figure:
    """Tracer les maxima horaires, en laissant visibles les périodes sans score."""
    figure = go.Figure(go.Scatter(
        x=hourly["timestamp"], y=hourly["score_max"], mode="lines", connectgaps=False,
        name="Maximum horaire du score d'atypie",
    ))
    figure.add_hline(y=threshold, line_dash="dash", line_color="red")
    for boundary in (config["train_end"], config["calibration_end"]):
        figure.add_vline(x=pd.Timestamp(boundary).to_pydatetime(), line_dash="dot")
    for period in periods:
        figure.add_vrect(x0=period.start, x1=(pd.Timestamp(period.end) + pd.Timedelta(minutes=1)).isoformat(),
                         fillcolor="orange", opacity=0.2, line_width=0)
    figure.update_layout(title="Scores historiques — maximum par heure, seuil rouge, périodes UCI orange",
                         xaxis_title="Date historique ; point placé au début de l'heure agrégée",
                         yaxis_title="Score d'atypie — aucune probabilité de panne", template="plotly_white")
    return figure


def write_ml_report(metrics: dict, hourly: pd.DataFrame, directory: str | Path) -> Path:
    """Sauver les métriques et un HTML autonome, sans prétendre à une vérité terrain complète."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    hourly.to_csv(directory / "score_plot_data.csv", index=False)
    parts = [
        "<!doctype html><html lang='fr'><meta charset='utf-8'><title>Baseline Isolation Forest</title><body>",
        "<style>body{font-family:system-ui;max-width:1250px;margin:2rem auto;padding:1rem}"
        "table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:.4rem}pre{white-space:pre-wrap}</style>",
        "<h1>MetroPT-3 — baseline d'anomalies Isolation Forest</h1>",
        "<p>Données historiques. Un comportement inhabituel n'est ni une panne confirmée, "
        "ni une identification de cause. Le score n'est pas une probabilité.</p>",
        "<h2>Protocole</h2><pre>" + html.escape(json.dumps(metrics["protocol"], ensure_ascii=False, indent=2)) + "</pre>",
        "<p>Fenêtres d'une minute closes ; cinq minutes passées pour l'historique. "
        "Après une minute absente, redémarrage de l'historique ; aucune interpolation. "
        "Une fenêtre exige au moins trois mesures et au plus 20 % de cellules absentes. "
        "Le dernier intervalle brut non confirmé à EOF est omis.</p>",
        "<h2>Résultats par période</h2>", pd.DataFrame(metrics["by_split"]).to_html(index=False, escape=True),
        "<h2>Modes et charge d'alertes</h2>", pd.DataFrame(metrics["by_mode"]).to_html(index=False, escape=True),
        "<p>Modes majoritaires par minute interprétés à partir des signaux ; "
        "des transitions peuvent exister dans une même minute. Les fractions de mode "
        "sont des variables du modèle global, pas des étiquettes de panne.</p>",
        "<h2>Comparaison aux rapports documentés</h2>",
        f"<p>{html.escape(metrics['limitations'])}</p>",
        pd.DataFrame(metrics["documented_period_comparisons"]).to_html(index=False, escape=True),
        "<p>La table contient les dates et liens sources, le nombre de fenêtres "
        "observées et scorées, puis les alertes à l'intérieur des périodes. "
        "Le délai de première alerte est calculé après le début documenté, "
        "sans prétention d'anticipation. Une période sans données n'est pas un échec mesuré.</p>",
    ]
    figure = score_figure(hourly, metrics["threshold"], metrics["protocol"])
    parts.append(figure.to_html(full_html=False, include_plotlyjs=True))
    daily_counts = hourly.set_index("timestamp")[["alert_windows", "scored_windows"]].resample("D").sum()
    daily = daily_counts["alert_windows"].where(daily_counts["scored_windows"].gt(0))
    burden = go.Figure(go.Bar(x=daily.index, y=daily.values, name="Minutes en alerte"))
    burden.update_layout(title="Charge quotidienne d'alertes — fenêtres de test et référence confondues",
                         yaxis_title="Fenêtres d'une minute en alerte", template="plotly_white")
    parts.append(burden.to_html(full_html=False, include_plotlyjs=False))
    parts.append("<h2>Métriques complètes et provenance</h2><pre>" + html.escape(json.dumps(metrics, ensure_ascii=False, indent=2)) + "</pre></body></html>")
    target = directory / "report.html"
    target.write_text("\n".join(parts), encoding="utf-8")
    return target
