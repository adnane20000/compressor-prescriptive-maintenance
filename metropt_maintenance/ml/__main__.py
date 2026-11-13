"""Commande d'entraînement, de calibration et d'évaluation de la baseline."""

import argparse
import json
from pathlib import Path

from .model import ModelConfig
from .workflow import run_baseline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Isolation Forest MetroPT-3 — anomalies, pas diagnostic")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/local/ml_baseline"))
    parser.add_argument("--models", type=Path, default=Path("models/isolation_forest"))
    parser.add_argument("--chunksize", type=int, default=50_000)
    parser.add_argument("--train-end", default="2020-02-22 00:00:00")
    parser.add_argument("--calibration-end", default="2020-03-01 00:00:00")
    parser.add_argument("--tail-fraction", type=float, default=0.01)
    parser.add_argument("--trees", type=int, default=200)
    args = parser.parse_args(argv)
    try:
        config = ModelConfig(train_end=args.train_end, calibration_end=args.calibration_end,
                             calibration_tail_fraction=args.tail_fraction, n_estimators=args.trees)
        run = run_baseline(args.csv, output=args.output, model_directory=args.models, config=config, chunksize=args.chunksize)
    except (OSError, ValueError) as error:
        parser.exit(2, f"Erreur ML : {error}\n")
    print(run.report_path)
    print(json.dumps({"model": str(run.model_path), "threshold": run.metrics["threshold"],
                      "by_split": run.metrics["by_split"],
                      "documented_periods_with_alert": run.metrics["documented_periods_with_alert"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
