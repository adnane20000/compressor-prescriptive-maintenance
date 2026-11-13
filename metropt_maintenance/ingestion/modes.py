"""Interprétation descriptive et prudente des états électriques documentés."""

import numpy as np
import pandas as pd

MODE_LABELS = ("arrêt probable", "marche à vide probable", "charge probable", "indéterminé")
MODE_RULES = (
    "Règles descriptives, sans étiquette de défaillance : arrêt probable si "
    "0 ≤ Motor_current ≤ 1 A, COMP=1 et DV_eletric=0 ; marche à vide probable "
    "si Motor_current > 1 A, COMP=1 et DV_eletric=0 ; charge probable si "
    "Motor_current > 1 A, COMP=0 et DV_eletric=1 ; sinon indéterminé. "
    "Le seuil de 1 A est un choix pédagogique configurable, pas une limite "
    "constructeur validée. Les courants de démarrage ne sont pas isolés par "
    "ces règles instantanées. Comptages de lignes, pas durées de fonctionnement."
)


def infer_modes(frame: pd.DataFrame, *, stopped_current_max: float = 1.0) -> pd.Series:
    """Croiser courant et deux signaux ; garder les incohérences indéterminées."""
    if not np.isfinite(stopped_current_max) or stopped_current_max < 0:
        raise ValueError("Le seuil de courant doit être fini et positif ou nul.")
    current = frame["Motor_current"]
    comp, valve = frame["COMP"], frame["DV_eletric"]
    finite = current.notna() & np.isfinite(current)
    stopped = finite & current.between(0, stopped_current_max) & comp.eq(1) & valve.eq(0)
    running = finite & current.gt(stopped_current_max)
    return pd.Series(
        np.select(
            [stopped, running & comp.eq(1) & valve.eq(0), running & comp.eq(0) & valve.eq(1)],
            MODE_LABELS[:3], default=MODE_LABELS[3],
        ), index=frame.index, name="mode_interprete",
    )
