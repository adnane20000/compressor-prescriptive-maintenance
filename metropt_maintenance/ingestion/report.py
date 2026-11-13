"""Graphiques agrégés bornés et rapport HTML autonome exportable."""

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .csv_reader import iter_chunks
from .schema import PLOT_COLUMNS, SIGNALS, SOURCE_URL


def aggregate_charts(path: str | Path, profile: dict, *, max_bins: int = 1200) -> pd.DataFrame:
    """Deuxième passe : min/max/moyenne par fenêtre, O(chunksize + max_bins).

    Aucune simple décimation : les extrema des quatre signaux sont conservés
    dans chaque fenêtre. Les dates invalides sont exclues des graphiques.
    """
    if not 1 <= max_bins <= 10_000:
        raise ValueError("Le nombre de fenêtres doit être compris entre 1 et 10 000.")
    if profile["date_min"] is None:
        return pd.DataFrame()
    start = pd.Timestamp(profile["date_min"])
    end = pd.Timestamp(profile["date_max"])
    width_ns = max(1_000_000_000, (end.value - start.value + max_bins) // max_bins)
    bins = min(max_bins, (end.value - start.value) // width_ns + 1)
    arrays = {
        name: {"min": np.full(bins, np.inf), "max": np.full(bins, -np.inf),
               "sum": np.zeros(bins), "count": np.zeros(bins, dtype="int64")}
        for name in PLOT_COLUMNS
    }
    for frame in iter_chunks(
        path, chunksize=profile["chunksize"], columns=("timestamp", *PLOT_COLUMNS),
        max_rows=profile["max_rows_requested"],
    ):
        frame = frame.loc[frame["timestamp"].notna()].copy()
        if frame.empty:
            continue
        frame["window"] = ((frame["timestamp"].astype("int64") - start.value) // width_ns).clip(0, bins - 1)
        for name in PLOT_COLUMNS:
            grouped = frame.groupby("window")[name].agg(["min", "max", "sum", "count"])
            positions = grouped.index.to_numpy(dtype=int)
            valid = grouped["count"].to_numpy() > 0
            selected = positions[valid]
            arrays[name]["min"][selected] = np.minimum(arrays[name]["min"][selected], grouped["min"].to_numpy()[valid])
            arrays[name]["max"][selected] = np.maximum(arrays[name]["max"][selected], grouped["max"].to_numpy()[valid])
            arrays[name]["sum"][positions] += grouped["sum"].to_numpy()
            arrays[name]["count"][positions] += grouped["count"].to_numpy()
    output = {"timestamp": pd.to_datetime(start.value + np.arange(bins, dtype="int64") * width_ns)}
    for name, values in arrays.items():
        count = values["count"]
        output[f"{name}_min"] = np.where(count > 0, values["min"], np.nan)
        output[f"{name}_max"] = np.where(count > 0, values["max"], np.nan)
        output[f"{name}_mean"] = np.divide(values["sum"], count, out=np.full(bins, np.nan), where=count > 0)
        output[f"{name}_count"] = count
    result = pd.DataFrame(output)
    result.attrs["window_seconds"] = width_ns / 1e9
    result.attrs["max_bins"] = max_bins
    return result


def build_figures(profile: dict, chart_data: pd.DataFrame) -> list[go.Figure]:
    """Construire quatre courbes min/max/moyenne et les effectifs des modes."""
    figures = []
    units = {signal.name: signal.unit for signal in SIGNALS}
    if not chart_data.empty:
        for name in PLOT_COLUMNS:
            figure = go.Figure()
            for suffix, label in (("min", "Minimum"), ("max", "Maximum"), ("mean", "Moyenne")):
                figure.add_trace(go.Scatter(
                    x=chart_data["timestamp"], y=chart_data[f"{name}_{suffix}"],
                    mode="lines", name=label, connectgaps=False,
                ))
            figure.update_layout(
                title=f"{name} — agrégation par fenêtre", xaxis_title="Date historique (début de fenêtre)",
                yaxis_title=f"{name} ({units[name]})", template="plotly_white",
            )
            figures.append(figure)
    modes = profile["mode_counts"]
    figure = go.Figure(go.Bar(x=list(modes), y=list(modes.values())))
    figure.update_layout(title="Modes interprétés — nombre de mesures", yaxis_title="Lignes (pas des durées)", template="plotly_white")
    figures.append(figure)
    return figures


def write_report(profile: dict, chart_data: pd.DataFrame, directory: str | Path) -> Path:
    """Écrire HTML autonome, JSON vérifiable et agrégats CSV locaux."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    profile = dict(profile)
    profile["charts"] = {
        "aggregation": "min/max/moyenne par fenêtre temporelle ; aucune imputation",
        "bins": len(chart_data), "window_seconds": chart_data.attrs.get("window_seconds"),
    }
    (directory / "profile.json").write_text(
        json.dumps(profile, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    chart_data.to_csv(directory / "chart_aggregates.csv", index=False)
    quality = pd.DataFrame({
        "manquantes brutes": profile["missing_raw"],
        "non convertibles / non finies": profile["invalid_values"],
    }).fillna(0).astype(int)
    scope = "Fichier complet" if profile["scope"] == "entire_file" else "PRÉFIXE LIMITÉ — résultats non extrapolables au fichier complet"
    intervals = profile["time_intervals"]
    summary = {
        "Périmètre": scope, "Lignes analysées": profile["rows_analyzed"],
        "Début observé": profile["date_min"], "Fin observée": profile["date_max"],
        "Fuseau": profile["timezone"], "SHA-256 du fichier entier": profile["file_sha256"],
        "Doublons de mesures": profile["duplicate_measurement_rows"],
        "Horodatages répétés": profile["duplicate_timestamps"],
        "Retours chronologiques": intervals["backwards_intervals"],
        "Intervalle médian observé (s)": intervals["median_seconds"],
        "Intervalle maximal observé (s)": intervals["max_seconds"],
        "Écarts > seuil descriptif (pas une panne)": intervals["gaps_above_threshold"],
        "Seuil descriptif des écarts (s)": intervals["gap_threshold_seconds"],
    }
    sections = [
        "<!doctype html><html lang='fr'><meta charset='utf-8'><title>Exploration MetroPT-3</title>",
        "<style>body{font-family:system-ui;max-width:1250px;margin:2rem auto;padding:1rem}"
        "table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:.4rem}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><body>",
        "<h1>Exploration MetroPT-3 — données historiques</h1>",
        f"<p>Référence : <a href='{SOURCE_URL}'>UCI, dataset 791</a>. "
        "Davari et al., DOI 10.24432/C5VW3R, CC BY 4.0. "
        "Le schéma compatible ne prouve pas à lui seul la provenance d'un CSV fourni manuellement.</p>",
        "<p>Aucune acquisition en temps réel, aucun diagnostic confirmé, aucune étiquette de défaillance créée.</p>",
        "<h2>Périmètre et temps</h2>", pd.DataFrame(summary.items(), columns=["Indicateur", "Valeur"]).to_html(index=False, escape=True),
        "<p>La documentation UCI mentionne 1 Hz et 0,1 Hz selon les sections ; "
        "le PDF évoque aussi un nombre de lignes différent. Les résultats ci-dessus "
        "sont mesurés dans le CSV fourni, sans supposer une cadence fixe. Les écarts "
        "ne prouvent ni perte de données ni défaillance.</p>",
        "<h2>Schéma et unités documentées</h2>", pd.DataFrame(profile["schema"]).to_html(index=False, escape=True),
        "<p>timestamp est converti en datetime64[ns], sans fuseau documenté. "
        "Les signaux sont convertis en float64 ; les huit signaux numériques "
        "sont attendus dans {0, 1}. L'index exporté n'est pas un capteur. "
        "Les unités viennent de la documentation, pas d'une estimation des valeurs.</p>",
        "<h2>Qualité</h2>", quality.to_html(escape=True),
        f"<p>{html.escape(profile['duplicate_definition'])}</p>",
        "<p>Les horodatages répétés sont comptés après conversion des dates. "
        "Les valeurs manquantes utilisent les marqueurs NA de Pandas ; les chaînes "
        "non convertibles et valeurs infinies sont comptées séparément, sans imputation.</p>",
        "<h3>Valeurs hors {0, 1} des signaux numériques</h3>",
        pd.DataFrame(profile["non_binary_values"].items(), columns=["Signal", "Effectif"]).to_html(index=False, escape=True),
        "<h2>Statistiques</h2>", pd.DataFrame(profile["statistics"]).to_html(index=False, escape=True),
        "<p>std_sample : écart-type avec ddof=1 ; valeurs invalides exclues, doublons conservés.</p>",
        "<h2>Modes de fonctionnement interprétés</h2>", f"<p>{html.escape(profile['mode_rules'])}</p>",
        "<h2>Graphiques</h2>",
        f"<p>{len(chart_data)} fenêtres ; largeur : {chart_data.attrs.get('window_seconds')} s. "
        "Les minima et maxima conservent les extrema ; la moyenne peut lisser les cycles. "
        "Une fenêtre vide reste vide. Les graphiques ne remplacent pas les données brutes.</p>",
    ]
    for number, figure in enumerate(build_figures(profile, chart_data)):
        sections.append(figure.to_html(full_html=False, include_plotlyjs=True if number == 0 else False))
    sections.append("<h2>Profil complet vérifiable</h2><pre>" + html.escape(json.dumps(profile, ensure_ascii=False, indent=2, allow_nan=False)) + "</pre></body></html>")
    target = directory / "report.html"
    target.write_text("\n".join(sections), encoding="utf-8")
    return target
