"""Données d'affichage bornées et graphiques Plotly, sans métriques inventées."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .ingestion.replay import iter_replay
from .ingestion.schema import SIGNAL_NAMES

COLORS = {"teal": "#078B92", "blue": "#4263EB", "orange": "#E68A31", "red": "#D95757", "muted": "#94A3B8"}
STEPS = [("monitor_node", "Mesures & ML"), ("anomaly_router", "Routage"),
         ("retrieval_node", "Recherche RAG"), ("diagnosis_node", "Diagnostic"),
         ("planning_node", "Proposition"), ("human_review_node", "Revue humaine"), ("finalize_node", "Décision")]


def read_json(path) -> dict | None:
    """Une absence est un état vide ; un fichier invalide reste une erreur explicite."""
    path = Path(path)
    if not path.exists():
        return None
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"Objet JSON attendu : {path.name}")
    return result


def file_identity(path) -> tuple | None:
    path = Path(path)
    if not path.is_file():
        return None
    stat = path.stat()
    return str(path.resolve()), stat.st_size, stat.st_mtime_ns


def preview_measurements(path, *, limit=360) -> pd.DataFrame:
    """Lire seulement le préfixe nécessaire, avec les contrôles du rejeu existant."""
    if not 1 <= limit <= 1000:
        raise ValueError("L'aperçu doit contenir au plus 1000 mesures.")
    return pd.DataFrame([{"timestamp": event.timestamp, **event.values, "mode_interprete": event.mode}
                         for event in iter_replay(path, chunksize=1000, max_events=limit)])


def read_alerts(path, *, offset=0, limit=50, start=None, end=None) -> list[dict]:
    """Historique paginé en flux ; aucune lecture complète du JSONL en mémoire."""
    if offset < 0 or not 1 <= limit <= 200:
        raise ValueError("Pagination des alertes invalide.")
    rows, matched = [], 0
    lower, upper = pd.Timestamp(start) if start else None, pd.Timestamp(end) if end else None
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            date = pd.Timestamp(record["available_at"])
            if (lower is not None and date < lower) or (upper is not None and date > upper):
                continue
            if matched >= offset:
                rows.append(record)
            matched += 1
            if len(rows) >= limit:
                break
    return rows


def read_score_view(path, *, start=None, end=None, max_points=1600) -> pd.DataFrame:
    """Réduire les maxima horaires en groupes contigus, sans moyenner les alertes."""
    if max_points < 2:
        raise ValueError("Au moins deux points sont nécessaires.")
    identity = file_identity(path)
    columns = ["timestamp", "score_max", "scored_windows", "alert_windows"]
    def blocks():
        for block in pd.read_csv(path, usecols=columns, chunksize=20_000, parse_dates=["timestamp"]):
            if start is not None:
                block = block.loc[block.timestamp >= pd.Timestamp(start)]
            if end is not None:
                block = block.loc[block.timestamp <= pd.Timestamp(end)]
            yield block
    # Deux passes sur le fichier horaire : comptage puis agrégation bornée.
    count, latest = 0, None
    for block in blocks():
        count += len(block)
        if not block.empty:
            candidate = block.timestamp.max()
            latest = candidate if latest is None else max(latest,candidate)
    buckets, offset = {}, 0
    for block in blocks():
        for date, score, scored, alerts in block[columns].itertuples(index=False, name=None):
            index = offset * max_points // count if count > max_points else offset
            if index not in buckets:
                buckets[index] = {"timestamp": date, "score_max": score, "scored_windows": scored, "alert_windows": alerts}
            else:
                row = buckets[index]
                row["score_max"] = score if pd.isna(row["score_max"]) else row["score_max"] if pd.isna(score) else max(row["score_max"],score)
                row["scored_windows"] += scored
                row["alert_windows"] += alerts
            offset += 1
    if file_identity(path) != identity:
        raise ValueError("Les scores ont changé pendant la lecture : actualiser la page.")
    result = pd.DataFrame(list(buckets.values()), columns=columns)
    result.attrs["aggregation"] = "maxima horaires" if count <= max_points else f"maxima de groupes contigus ; {count} heures réduites à {len(result)} points"
    result.attrs["last_timestamp"] = latest
    return result


def style_figure(figure, *, height=310, y_title=None):
    figure.update_layout(template="plotly_white", height=height, margin=dict(l=12, r=12, t=16, b=12),
                         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                         font=dict(family="Inter, Arial, sans-serif", color="#59677C", size=12),
                         hovermode="x unified", legend=dict(orientation="h", y=1.13, x=0),
                         yaxis_title=y_title)
    figure.update_layout(separators=", ")
    figure.update_xaxes(showgrid=False, zeroline=False)
    figure.update_yaxes(gridcolor="#EDF1F6", zeroline=False)
    return figure


def sensor_figure(frame: pd.DataFrame, signals: list[str], *, unit: str):
    figure = go.Figure()
    for signal, color in zip(signals, (COLORS["teal"], COLORS["blue"], COLORS["orange"])):
        # Séparateurs visuels lors de trous >30 s ; les mesures d'origine ne changent pas.
        dates, values, previous = [], [], None
        for date, value in zip(pd.to_datetime(frame.timestamp), frame[signal]):
            if previous is not None and (date - previous).total_seconds() > 30:
                dates.append(date); values.append(None)
            dates.append(date); values.append(value)
            previous = date
        figure.add_trace(go.Scatter(x=dates, y=values, name=signal, mode="lines" if len(frame) > 1 else "markers",
                                   line=dict(color=color, width=2), connectgaps=False))
    style_figure(figure, y_title=unit)
    figure.update_xaxes(tickformat="%H:%M", hoverformat="%d/%m/%Y %H:%M:%S")
    return figure


def anomaly_figure(data: pd.DataFrame, threshold: float):
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=data.timestamp, y=data.score_max, mode="lines", name="Score maximal",
                               line=dict(color=COLORS["blue"], width=1.6)))
    alerts = data.loc[data.alert_windows > 0]
    figure.add_trace(go.Scatter(x=alerts.timestamp, y=alerts.score_max, mode="markers", name="Fenêtre contenant une alerte",
                               marker=dict(color=COLORS["orange"], size=5), customdata=alerts.alert_windows,
                               hovertemplate="%{x}<br>Score max. %{y:.4f}<br>%{customdata} alertes<extra></extra>"))
    figure.add_hline(y=threshold, line_dash="dash", line_color=COLORS["red"], annotation_text="Seuil de calibration")
    style_figure(figure, height=370, y_title="Score d'atypie · pas une probabilité")
    figure.update_xaxes(tickformat="%d/%m/%Y", hoverformat="%d/%m/%Y %H:%M")
    return figure


def mode_figure(counts: dict):
    figure = go.Figure(go.Bar(x=list(counts.values()), y=list(counts.keys()), orientation="h",
                             marker_color=[COLORS["teal"], COLORS["blue"], COLORS["orange"], COLORS["muted"]][:len(counts)]))
    figure.update_xaxes(title="Nombre de mesures")
    return style_figure(figure, height=230)


def workflow_figure(state: dict):
    latest = {entry["node"]: entry["status"] for entry in state["journal"]}
    labels, colors, descriptions = [], [], []
    for node, label in STEPS:
        status = latest.get(node)
        waiting = node == "human_review_node" and bool(state.get("interrupts"))
        color = COLORS["orange"] if waiting or status in ("error", "blocked", "invalid_review") else COLORS["teal"] if status else "#CBD5E1"
        labels.append(label); colors.append(color)
        descriptions.append("En attente de validation" if waiting else "Étape parcourue" if status else "Non parcourue")
    figure = go.Figure(go.Scatter(x=list(range(7)), y=[0]*7, mode="lines+markers+text", text=labels,
                                 textposition="bottom center", customdata=descriptions,
                                 hovertemplate="%{text}<br>%{customdata}<extra></extra>",
                                 line=dict(color="#E2E8F0", width=3), marker=dict(size=19, color=colors)))
    style_figure(figure, height=155)
    figure.update_xaxes(visible=False, range=[-.4,6.4]); figure.update_yaxes(visible=False, range=[-.8,.7])
    return figure


def create_synthetic_demo(directory="data/local/demo") -> Path:
    """Échantillon pédagogique explicite, sans étiquette de défaillance ni métrique de modèle."""
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    x = np.arange(721)
    loaded, running = x % 120 < 35, x % 120 < 85
    frame = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=len(x), freq="10s")})
    for name in SIGNAL_NAMES:
        frame[name] = 1.0
    frame["Oil_temperature"] = 55 + 5*np.sin(x/90)
    frame["TP2"] = np.where(loaded, 8+0.3*np.sin(x/20), .02)
    frame["TP3"] = 9+.45*np.sin(x/25)
    frame["Motor_current"] = np.where(loaded, 6.2, np.where(running,4.1,.04))
    frame["COMP"] = (~loaded).astype(float); frame["DV_eletric"] = loaded.astype(float)
    path = directory / "metropt_SYNTHETIQUE.csv"
    frame.to_csv(path, index=False)
    path.with_suffix(".context.json").write_text(json.dumps({"data_context":"synthetic_test", "description":"DONNÉES SYNTHÉTIQUES pédagogiques, aucune mesure industrielle réelle."}), encoding="utf-8")
    return path
