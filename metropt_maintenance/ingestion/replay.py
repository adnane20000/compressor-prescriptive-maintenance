"""Rejeu déterministe en flux ; aucune connexion à un compresseur réel."""

import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .csv_reader import iter_chunks
from .modes import infer_modes
from .schema import SIGNAL_NAMES


@dataclass(frozen=True)
class ReplayEvent:
    timestamp: pd.Timestamp
    values: dict[str, float | None]
    mode: str
    source_delta_seconds: float
    delay_seconds: float
    gap_capped: bool


def iter_replay(
    path: str | Path, *, chunksize: int = 10_000, speed: float = 1.0,
    start: str | None = None, end: str | None = None,
    max_events: int | None = None, max_delay_seconds: float | None = None,
) -> Iterator[ReplayEvent]:
    """Émettre les mesures dans l'ordre du CSV avec délai source / vitesse.

    Aucun sommeil ici : l'appelant choisit un pilote temporel ou un pas-à-pas.
    Les dates absentes/invalides et retours temporels arrêtent le rejeu, y compris
    entre deux blocs. L'ordre est vérifié sur la portion lue, sans tri silencieux.
    Les horodatages égaux restent dans leur ordre source, avec délai nul.
    """
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError("La vitesse doit être finie et strictement positive.")
    if max_events is not None and max_events < 1:
        raise ValueError("max_events doit être strictement positif.")
    if max_delay_seconds is not None and (not math.isfinite(max_delay_seconds) or max_delay_seconds < 0):
        raise ValueError("Le plafond de délai doit être fini et positif ou nul.")
    lower = pd.Timestamp(start) if start else None
    upper = pd.Timestamp(end) if end else None
    if any(value is not None and (pd.isna(value) or value.tzinfo is not None) for value in (lower, upper)):
        raise ValueError("Utiliser des dates valides sans fuseau, comme le CSV source.")
    if lower is not None and upper is not None and lower > upper:
        raise ValueError("La date de début doit précéder la date de fin.")
    previous_read = previous_sent = None
    emitted = 0
    for frame in iter_chunks(path, chunksize=chunksize):
        if frame.empty:
            continue
        dates = frame["timestamp"]
        if dates.isna().any():
            raise ValueError("Rejeu refusé : horodatage manquant ou invalide dans la portion lue.")
        if not dates.is_monotonic_increasing or (previous_read is not None and dates.iloc[0] < previous_read):
            raise ValueError("Rejeu refusé : CSV non chronologique ; aucun tri implicite.")
        previous_read = dates.iloc[-1]
        selected = frame
        if lower is not None:
            selected = selected.loc[selected["timestamp"] >= lower]
        if upper is not None:
            selected = selected.loc[selected["timestamp"] <= upper]
        modes = infer_modes(selected)
        for position, row in enumerate(selected.itertuples(index=False, name=None)):
            timestamp, *numbers = row
            delta = (timestamp - previous_sent).total_seconds() if previous_sent is not None else 0.0
            delay = delta / speed
            capped = max_delay_seconds is not None and delay > max_delay_seconds
            yield ReplayEvent(
                timestamp=timestamp,
                values={name: float(value) if pd.notna(value) else None for name, value in zip(SIGNAL_NAMES, numbers)},
                mode=str(modes.iloc[position]), source_delta_seconds=delta,
                delay_seconds=min(delay, max_delay_seconds) if capped else delay, gap_capped=capped,
            )
            previous_sent = timestamp
            emitted += 1
            if max_events is not None and emitted >= max_events:
                return
