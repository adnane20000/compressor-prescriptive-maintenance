"""Comparer après entraînement aux périodes externes ; hors période = non étiqueté."""

from dataclasses import asdict, dataclass

import pandas as pd

from ..ingestion.schema import SOURCE_URL


@dataclass(frozen=True)
class DocumentedPeriod:
    identifier: str
    start: str
    end: str
    reported_condition: str = "Air leak"
    source: str = SOURCE_URL
    source_section: str = "Failure Information ; également Data Description_Metro.pdf de l'archive UCI"


DOCUMENTED_PERIODS = (
    DocumentedPeriod("uci-period-1", "2020-04-18 00:00:00", "2020-04-18 23:59:00"),
    DocumentedPeriod("uci-period-2", "2020-05-29 23:30:00", "2020-05-30 06:00:00"),
    DocumentedPeriod("uci-period-3", "2020-06-05 10:00:00", "2020-06-07 14:30:00"),
    DocumentedPeriod("uci-period-4", "2020-07-15 14:30:00", "2020-07-15 19:00:00"),
)
EVALUATION_LIMITS = (
    "Comparaison descriptive avec quatre périodes externes documentées, jamais utilisées pour ajuster le modèle ou le seuil. "
    "Les dates hors période sont non étiquetées, pas des exemples normaux confirmés. "
    "Précision, F1, ROC-AUC et taux de faux positifs ne sont pas estimables ici. "
    "Une association temporelle ne démontre pas une cause, une anticipation ni une détection de fuite. "
    "Bornes documentées à la minute : la minute de fin est incluse."
)


def evaluate_predictions(predictions: pd.DataFrame, *, periods=DOCUMENTED_PERIODS) -> dict:
    """Ne comparer que les fenêtres du test, en indiquant aussi leur couverture."""
    test = predictions.loc[predictions["split"].eq("test")].copy()
    observed = test["status"].eq("scored")
    alerts = test["is_anomaly"].fillna(False).astype(bool)
    known = pd.Series(False, index=test.index)
    comparisons = []
    for period in periods:
        start = pd.Timestamp(period.start)
        end_exclusive = pd.Timestamp(period.end) + pd.Timedelta(minutes=1)
        if start.tzinfo is not None or end_exclusive.tzinfo is not None or start >= end_exclusive:
            raise ValueError("Période documentée invalide.")
        # Les fenêtres observées recouvrant la période sont associées au rapport.
        overlaps = test["window_start"].lt(end_exclusive) & test["timestamp"].gt(start)
        known |= overlaps
        available = overlaps & observed
        flagged = available & alerts
        first = test.loc[flagged, "timestamp"].min()
        comparisons.append({
            **asdict(period), "end_exclusive_for_minute_comparison": end_exclusive.isoformat(),
            "available_windows": int(overlaps.sum()), "scored_windows": int(available.sum()),
            "unscored_windows": int((overlaps & ~observed).sum()),
            "expected_minutes_in_documented_interval": int((end_exclusive - start).total_seconds() / 60),
            "alert_windows": int(flagged.sum()),
            "alert_rate_among_scored": float(flagged.sum() / available.sum()) if available.any() else None,
            "has_alert": bool(flagged.any()) if available.any() else None,
            "first_alert_available_at": first.isoformat() if pd.notna(first) else None,
            "delay_after_documented_start_seconds": (first - start).total_seconds() if pd.notna(first) else None,
        })
    by_split = []
    for split, group in predictions.groupby("split", sort=False):
        valid = group["status"].eq("scored")
        flagged = group["is_anomaly"].fillna(False).astype(bool)
        by_split.append({"split": split, "windows": len(group), "scored_windows": int(valid.sum()),
                         "alert_windows": int(flagged.sum()), "alert_rate": float(flagged.sum() / valid.sum()) if valid.any() else None,
                         "score_median": float(group.loc[valid, "anomaly_score"].median()) if valid.any() else None})
    by_mode = []
    for (split, mode), group in predictions.groupby(["split", "mode"], sort=False):
        valid = group["status"].eq("scored")
        flagged = group["is_anomaly"].fillna(False).astype(bool)
        by_mode.append({"split": split, "mode": mode, "windows": len(group),
                        "scored_windows": int(valid.sum()), "alert_windows": int(flagged.sum()),
                        "alert_rate": float(flagged.sum() / valid.sum()) if valid.any() else None})
    # Deux alertes séparées par une minute sans alerte ou sans observation
    # ouvrent deux épisodes distincts ; on ne relie jamais un trou de données.
    flagged_times = test.loc[alerts & observed, "timestamp"]
    episodes = int(flagged_times.diff().ne(pd.Timedelta(minutes=1)).sum())
    observable = [entry for entry in comparisons if entry["scored_windows"] > 0]
    return {
        "limitations": EVALUATION_LIMITS, "by_split": by_split, "by_mode": by_mode,
        "documented_period_comparisons": comparisons,
        "documented_periods_with_observations": len(observable),
        "documented_periods_with_alert": sum(entry["has_alert"] for entry in observable),
        "test_alert_episodes": episodes,
        "test_alerts_outside_documented_periods": int((alerts & observed & ~known).sum()),
        "test_scored_windows_outside_documented_periods": int((observed & ~known).sum()),
        "outside_period_status": "non étiqueté ; ces alertes ne sont pas des faux positifs confirmés",
        "unavailable_supervised_metrics": ["precision", "F1", "ROC-AUC", "false_positive_rate"],
    }
