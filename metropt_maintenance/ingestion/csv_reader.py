"""Lecture bornée en mémoire ; aucune correction ou imputation silencieuse."""

import csv
import warnings
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import DATE_FORMAT, REQUIRED_COLUMNS, SIGNAL_NAMES


def inspect_header(path: str | Path) -> list[str]:
    """Valider un CSV UCI ou téléchargé manuellement, avec ou sans index exporté."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV introuvable : {path}")
    with path.open(encoding="utf-8-sig", newline="") as source:
        header = next(csv.reader(source), [])
    if not header:
        raise ValueError("Le CSV est vide.")
    names = [name if name else f"Unnamed: {i}" for i, name in enumerate(header)]
    if len(names) != len(set(names)):
        raise ValueError("Le CSV contient des noms de colonnes dupliqués.")
    missing = set(REQUIRED_COLUMNS) - set(names)
    if missing:
        raise ValueError("Colonnes MetroPT-3 manquantes : " + ", ".join(sorted(missing)))
    return names


def iter_raw_chunks(
    path: str | Path,
    *,
    chunksize: int = 50_000,
    columns: Sequence[str] | None = None,
    max_rows: int | None = None,
) -> Iterator[pd.DataFrame]:
    """Lire des chaînes pour distinguer valeurs absentes et valeurs non convertibles."""
    header = inspect_header(path)
    if chunksize < 1 or (max_rows is not None and max_rows < 1):
        raise ValueError("chunksize et max_rows doivent être strictement positifs.")
    if columns is not None and (not columns or set(columns) - set(header)):
        raise ValueError("Sélection de colonnes vide ou inconnue.")
    try:
        with warnings.catch_warnings():
            # Éviter que Pandas interprète implicitement un champ excédentaire
            # comme index, ou tronque des valeurs sans bloquer le chargement.
            warnings.simplefilter("error", pd.errors.ParserWarning)
            with pd.read_csv(
                path, encoding="utf-8-sig", dtype="string", chunksize=chunksize,
                usecols=columns, nrows=max_rows, on_bad_lines="error", index_col=False,
            ) as reader:
                yield from reader
    except pd.errors.ParserWarning as error:
        raise ValueError("Structure CSV incohérente : nombre de champs incompatible avec l'en-tête.") from error


def normalize_chunk(raw: pd.DataFrame) -> pd.DataFrame:
    """Convertir les dates sans fuseau inventé et les signaux en float64.

    Les valeurs invalides et non finies deviennent NaN et sont comptées à part
    par le profilage. Aucune ligne ni aucun doublon n'est supprimé.
    """
    frame = pd.DataFrame(index=raw.index)
    if "timestamp" in raw:
        frame["timestamp"] = pd.to_datetime(
            raw["timestamp"], format=DATE_FORMAT, errors="coerce"
        ).astype("datetime64[ns]")
    for name in SIGNAL_NAMES:
        if name in raw:
            values = pd.to_numeric(raw[name], errors="coerce").astype("float64")
            frame[name] = values.where(np.isfinite(values))
    return frame


def iter_chunks(
    path: str | Path,
    *,
    chunksize: int = 50_000,
    columns: Sequence[str] = REQUIRED_COLUMNS,
    max_rows: int | None = None,
) -> Iterator[pd.DataFrame]:
    """Exposer une lecture typée ne chargeant que les colonnes demandées."""
    if set(columns) - set(REQUIRED_COLUMNS):
        raise ValueError("La lecture typée accepte uniquement les colonnes documentées.")
    for raw in iter_raw_chunks(path, chunksize=chunksize, columns=columns, max_rows=max_rows):
        yield normalize_chunk(raw)


def load_preview(path: str | Path, *, rows: int = 100) -> pd.DataFrame:
    """Charger un petit aperçu explicite, jamais le fichier complet par défaut."""
    if not 1 <= rows <= 10_000:
        raise ValueError("L'aperçu doit contenir entre 1 et 10 000 lignes.")
    return next(iter_chunks(path, chunksize=rows, max_rows=rows))
