"""Agrégation par minute et variables causales, sans ajustement statistique global."""

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from ..ingestion.csv_reader import iter_chunks
from ..ingestion.modes import MODE_LABELS, infer_modes
from ..ingestion.schema import PLOT_COLUMNS, REQUIRED_COLUMNS, SIGNALS, SIGNAL_NAMES

FEATURE_VERSION = "metropt-minute-causal-v1"
ANALOG = tuple(signal.name for signal in SIGNALS if signal.kind == "analogique")
DIGITAL = tuple(signal.name for signal in SIGNALS if signal.kind == "numérique")
BASE_FEATURES = (
    *(f"{name}_mean" for name in SIGNAL_NAMES),
    *(f"{name}_std" for name in ANALOG),
    *(f"mode_fraction_{i}" for i in range(4)), "TP3_minus_TP2",
)
TEMPORAL_FEATURES = tuple(
    f"{name}_{suffix}" for name in PLOT_COLUMNS
    for suffix in ("lag1", "delta1", "past5_mean", "past5_std")
)
FEATURE_COLUMNS = (*BASE_FEATURES, *TEMPORAL_FEATURES)


def aggregate_minutes(chunks: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Finaliser [t-1 min, t[ uniquement après réception d'une mesure à t ou après.

    Ne conserve que le dernier intervalle brut entre blocs. Le dernier intervalle
    à EOF est écarté, faute de confirmation de fermeture. Les minutes sans mesure
    sont omises, sans rééchantillonnage artificiel ni remplissage vers le futur.
    """
    carry = pd.DataFrame()
    outputs = []
    previous = first_timestamp = None
    raw_rows = omitted = 0
    for source in chunks:
        if source.empty:
            continue
        if set(REQUIRED_COLUMNS) - set(source):
            raise ValueError("Les 15 signaux et timestamp sont nécessaires au ML.")
        frame = source[list(REQUIRED_COLUMNS)].copy()
        dates = pd.to_datetime(frame["timestamp"], errors="coerce").astype("datetime64[ns]")
        if dates.isna().any() or not dates.is_monotonic_increasing or dates.duplicated().any():
            raise ValueError("ML : dates invalides, répétées ou non chronologiques.")
        if previous is not None and dates.iloc[0] <= previous:
            raise ValueError("ML : ordre strict non respecté entre blocs.")
        previous = dates.iloc[-1]
        if first_timestamp is None:
            first_timestamp = dates.iloc[0]
        frame["timestamp"] = dates
        for name in SIGNAL_NAMES:
            frame[name] = pd.to_numeric(frame[name], errors="coerce").astype("float64")
            frame[name] = frame[name].where(np.isfinite(frame[name]))
        for name in DIGITAL:
            frame[name] = frame[name].where(frame[name].isin([0, 1]))
        raw_rows += len(frame)
        frame = pd.concat([carry, frame], ignore_index=True)
        window = frame["timestamp"].dt.floor("min")
        last_window = window.iloc[-1]
        carry = frame.loc[window == last_window].copy()
        complete = frame.loc[window < last_window].copy()
        if complete.empty:
            continue
        complete["window_start"] = complete["timestamp"].dt.floor("min")
        complete["missing_cells"] = complete[list(SIGNAL_NAMES)].isna().sum(axis=1)
        modes = infer_modes(complete)
        for index, label in enumerate(MODE_LABELS):
            complete[f"mode_fraction_{index}"] = modes.eq(label).astype(float)
        grouped = complete.groupby("window_start", sort=True)
        mean = grouped[list(SIGNAL_NAMES)].mean().rename(columns=lambda name: f"{name}_mean")
        std = grouped[list(ANALOG)].std(ddof=0).rename(columns=lambda name: f"{name}_std")
        fractions = grouped[[f"mode_fraction_{i}" for i in range(4)]].mean()
        result = pd.concat([mean, std, fractions], axis=1)
        result["sample_count"] = grouped.size()
        result["missing_fraction"] = grouped["missing_cells"].sum() / (result["sample_count"] * 15)
        result["timestamp"] = result.index + pd.Timedelta(minutes=1)
        result["partial_start"] = (result.index == first_timestamp.floor("min")) & (first_timestamp != first_timestamp.floor("min"))
        outputs.append(result.reset_index())
    omitted = len(carry)
    if not outputs:
        raise ValueError("Aucune minute finalisée : fournir davantage de mesures chronologiques.")
    result = pd.concat(outputs, ignore_index=True)
    result.attrs.update(raw_rows_read=raw_rows, final_unclosed_raw_rows=omitted, feature_version=FEATURE_VERSION)
    return result


def engineer_features(minutes: pd.DataFrame) -> pd.DataFrame:
    """Créer retards et fenêtres des cinq minutes précédentes, jamais centrées."""
    frame = minutes.copy()
    dates = frame["timestamp"]
    if not dates.is_monotonic_increasing or dates.duplicated().any() or dates.isna().any():
        raise ValueError("Les minutes doivent être strictement chronologiques.")
    # Une minute absente ouvre un nouveau segment : pas de mémoire à travers un trou.
    segment = dates.diff().ne(pd.Timedelta(minutes=1)).cumsum()
    frame["history_ready"] = frame.groupby(segment).cumcount().ge(5)
    frame["TP3_minus_TP2"] = frame["TP3_mean"] - frame["TP2_mean"]
    for name in PLOT_COLUMNS:
        series = frame[f"{name}_mean"]
        lag = series.groupby(segment).shift(1)
        frame[f"{name}_lag1"] = lag
        frame[f"{name}_delta1"] = series - lag
        frame[f"{name}_past5_mean"] = lag.groupby(segment).transform(lambda values: values.rolling(5, min_periods=5).mean())
        frame[f"{name}_past5_std"] = lag.groupby(segment).transform(lambda values: values.rolling(5, min_periods=5).std(ddof=0))
    fraction_columns = [f"mode_fraction_{i}" for i in range(4)]
    frame["mode"] = [MODE_LABELS[i] for i in frame[fraction_columns].to_numpy().argmax(axis=1)]
    frame["mode_mixed"] = frame[fraction_columns].max(axis=1).lt(1)
    frame["eligible"] = frame["history_ready"] & frame["missing_fraction"].le(0.2) & frame["sample_count"].ge(3) & ~frame["partial_start"]
    frame.attrs.update(minutes.attrs)
    return frame


def build_features(path: str | Path, *, chunksize: int = 50_000) -> pd.DataFrame:
    """Lire le CSV réel par blocs et ne conserver que les fenêtres réduites."""
    return engineer_features(aggregate_minutes(iter_chunks(path, chunksize=chunksize)))
