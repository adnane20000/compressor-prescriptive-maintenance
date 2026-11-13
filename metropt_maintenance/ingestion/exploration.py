"""Profilage exhaustif par blocs avec détection exacte des doublons sur disque."""

import hashlib
import json
import math
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .csv_reader import inspect_header, iter_raw_chunks, normalize_chunk
from .modes import MODE_LABELS, MODE_RULES, infer_modes
from .schema import REQUIRED_COLUMNS, SIGNALS, SOURCE_URL, schema_records


def _median_from_counts(counts: Counter) -> float | None:
    """Calculer une médiane pondérée exacte à partir des intervalles comptés."""
    total = sum(counts.values())
    if not total:
        return None
    positions = ((total - 1) // 2, total // 2)
    found, cumulative = [], 0
    for value, count in sorted(counts.items()):
        for position in positions[len(found):]:
            if cumulative <= position < cumulative + count:
                found.append(value)
            else:
                break
        cumulative += count
        if len(found) == 2:
            return float(sum(found) / 2)
    return None


def profile_csv(
    path: str | Path, *, chunksize: int = 50_000, max_rows: int | None = None,
    temp_dir: str | Path | None = None, gap_threshold_seconds: float = 30.0,
) -> dict:
    """Analyser tout le fichier ou un préfixe explicitement limité.

    RAM : O(chunksize), plus un histogramme borné à 4096 intervalles différents.
    Les clés exactes des doublons (sans hachage sujet aux collisions) sont stockées
    dans une base temporaire SQLite, supprimée même si l'analyse échoue.
    Les statistiques numériques combinent les moments par blocs (Welford).
    """
    if not math.isfinite(gap_threshold_seconds) or gap_threshold_seconds <= 0:
        raise ValueError("Le seuil d'écart temporel doit être fini et strictement positif.")
    path = Path(path)
    header = inspect_header(path)
    missing = Counter({name: 0 for name in header})
    invalid = Counter({name: 0 for name in REQUIRED_COLUMNS})
    binary_invalid = Counter()
    modes = Counter({name: 0 for name in MODE_LABELS})
    moments = {
        signal.name: {"count": 0, "mean": 0.0, "m2": 0.0, "min": None, "max": None}
        for signal in SIGNALS
    }
    rows = duplicates = duplicate_times = 0
    first_date = last_date = previous = None
    interval_counts = Counter()
    interval_overflow = 0
    interval_total = interval_sum = zeros = backwards = large_gaps = 0
    interval_min = interval_max = None
    with tempfile.TemporaryDirectory(prefix="metropt-profile-", dir=temp_dir) as directory:
        with sqlite3.connect(Path(directory) / "duplicates.sqlite") as database:
            database.execute("PRAGMA cache_size=-8192")
            database.execute("PRAGMA journal_mode=OFF")
            database.execute("PRAGMA synchronous=OFF")
            database.execute("CREATE TABLE seen (kind INTEGER, value TEXT, PRIMARY KEY(kind, value)) WITHOUT ROWID")
            for raw in iter_raw_chunks(path, chunksize=chunksize, max_rows=max_rows):
                if raw.empty:
                    continue
                frame = normalize_chunk(raw)
                rows += len(raw)
                missing.update({name: int(raw[name].isna().sum()) for name in header})
                for name in REQUIRED_COLUMNS:
                    invalid[name] += int((raw[name].notna() & frame[name].isna()).sum())
                for signal in SIGNALS:
                    values = frame[signal.name].dropna().to_numpy()
                    if signal.kind == "numérique":
                        binary_invalid[signal.name] += int((~np.isin(values, [0, 1])).sum())
                    count = len(values)
                    if not count:
                        continue
                    state = moments[signal.name]
                    mean = float(values.mean())
                    m2 = float(np.square(values - mean).sum())
                    combined = state["count"] + count
                    delta = mean - state["mean"]
                    state["m2"] += m2 + delta * delta * state["count"] * count / combined
                    state["mean"] += delta * count / combined
                    state["count"] = combined
                    minimum, maximum = float(values.min()), float(values.max())
                    state["min"] = minimum if state["min"] is None else min(state["min"], minimum)
                    state["max"] = maximum if state["max"] is None else max(state["max"], maximum)
                modes.update({key: int(value) for key, value in infer_modes(frame).value_counts().items()})

                # Le JSON canonique de toutes les chaînes utiles rend la comparaison
                # exacte. L'index exporté et les colonnes étrangères sont exclus.
                records = raw[list(REQUIRED_COLUMNS)].astype(object)
                records = records.where(records.notna(), None)
                before = database.total_changes
                database.executemany(
                    "INSERT OR IGNORE INTO seen VALUES (0, ?)",
                    ((json.dumps(row, ensure_ascii=False, separators=(",", ":")),)
                     for row in records.itertuples(index=False, name=None)),
                )
                duplicates += len(raw) - (database.total_changes - before)

                times = frame["timestamp"]
                valid = times.dropna()
                before = database.total_changes
                database.executemany(
                    "INSERT OR IGNORE INTO seen VALUES (1, ?)",
                    ((str(int(value)),) for value in valid.astype("int64")),
                )
                duplicate_times += len(valid) - (database.total_changes - before)
                database.commit()
                if not valid.empty:
                    minimum, maximum = valid.min(), valid.max()
                    first_date = minimum if first_date is None else min(first_date, minimum)
                    last_date = maximum if last_date is None else max(last_date, maximum)
                differences = times.diff().dt.total_seconds()
                if previous is not None and pd.notna(previous) and pd.notna(times.iloc[0]):
                    differences.iloc[0] = (times.iloc[0] - previous).total_seconds()
                previous = times.iloc[-1]
                differences = differences.dropna()
                if not differences.empty:
                    interval_total += len(differences)
                    interval_sum += float(differences.sum())
                    zeros += int(differences.eq(0).sum())
                    backwards += int(differences.lt(0).sum())
                    large_gaps += int(differences.gt(gap_threshold_seconds).sum())
                    lo, hi = float(differences.min()), float(differences.max())
                    interval_min = lo if interval_min is None else min(interval_min, lo)
                    interval_max = hi if interval_max is None else max(interval_max, hi)
                    for value, count in differences.value_counts().items():
                        if value in interval_counts or len(interval_counts) < 4096:
                            interval_counts[float(value)] += int(count)
                        else:
                            interval_overflow += int(count)
    if not rows:
        raise ValueError("Le CSV contient un en-tête mais aucune mesure.")
    statistics = []
    for signal in SIGNALS:
        state = moments[signal.name]
        count = state["count"]
        statistics.append({
            "column": signal.name, "unit": signal.unit, "kind": signal.kind,
            "parsed_type": "float64", "valid_count": count,
            "min": state["min"], "max": state["max"],
            "mean": state["mean"] if count else None,
            "std_sample": math.sqrt(max(0, state["m2"]) / (count - 1)) if count > 1 else None,
        })
    # Un aperçu limité ne doit pas provoquer une lecture supplémentaire du
    # fichier entier uniquement pour calculer son empreinte.
    digest = None
    if max_rows is None:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    return {
        "source_reference": SOURCE_URL, "file": str(path.resolve()),
        "file_bytes": path.stat().st_size, "file_sha256": digest.hexdigest() if digest else None,
        "hash_scope": "entire_file" if digest else "not_computed_for_prefix",
        "scope": "entire_file" if max_rows is None else "prefix_limit",
        "max_rows_requested": max_rows, "rows_analyzed": rows, "chunksize": chunksize,
        "csv_columns": header, "signal_count": len(SIGNALS), "schema": schema_records(),
        "export_index_columns": [name for name in header if name in ("index", "Unnamed: 0")],
        "extra_columns_ignored": [name for name in header if name not in REQUIRED_COLUMNS and name not in ("index", "Unnamed: 0")],
        "missing_raw": dict(missing), "invalid_values": dict(invalid),
        "non_binary_values": {signal.name: binary_invalid[signal.name] for signal in SIGNALS if signal.kind == "numérique"},
        "duplicate_measurement_rows": duplicates, "duplicate_timestamps": duplicate_times,
        "duplicate_definition": "Répétitions des 16 champs textuels documentés après reconnaissance des marqueurs NA, timestamp inclus, index exporté exclu ; aucune suppression. Les écritures 1 et 1.0 restent distinctes.",
        "timestamp_type": "datetime64[ns]", "timezone": "non documenté ; dates conservées sans fuseau",
        "date_min": first_date.isoformat() if first_date is not None else None,
        "date_max": last_date.isoformat() if last_date is not None else None,
        "time_intervals": {
            "count": interval_total, "min_seconds": interval_min, "max_seconds": interval_max,
            "mean_seconds": interval_sum / interval_total if interval_total else None,
            "median_seconds": _median_from_counts(interval_counts) if not interval_overflow else None,
            "most_common_seconds": [{"seconds": key, "count": value} for key, value in interval_counts.most_common(20)],
            "histogram_overflow_count": interval_overflow, "zero_intervals": zeros,
            "backwards_intervals": backwards, "gap_threshold_seconds": gap_threshold_seconds,
            "gaps_above_threshold": large_gaps,
        },
        "statistics": statistics, "mode_counts": dict(modes), "mode_rules": MODE_RULES,
        "failure_labels_created": False,
    }
