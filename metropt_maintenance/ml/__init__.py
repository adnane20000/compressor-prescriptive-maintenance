"""Baseline d'anomalies ; aucun diagnostic de cause ni contrôle physique."""

from .model import load_model, predict_anomaly

__all__ = ["load_model", "predict_anomaly"]
