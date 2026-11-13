"""Commandes : download, explore et replay (sans API LLM)."""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import requests

from .acquisition import download_dataset
from .exploration import profile_csv
from .replay import iter_replay
from .report import aggregate_charts, write_report


def main(argv: list[str] | None = None) -> int:
    """Fournir une CLI réutilisable depuis les tests et le terminal."""
    parser = argparse.ArgumentParser(description="MetroPT-3 : acquisition et exploration historiques")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Télécharger explicitement l'archive officielle UCI")
    download.add_argument("--directory", type=Path, default=Path("data/raw"))
    download.add_argument("--overwrite", action="store_true")
    explore = commands.add_parser("explore", help="Analyser le CSV par blocs et créer le rapport")
    explore.add_argument("csv", type=Path)
    explore.add_argument("--output", type=Path, default=Path("data/local/exploration"))
    explore.add_argument("--chunksize", type=int, default=50_000)
    explore.add_argument("--max-rows", type=int, help="Limiter explicitement l'analyse au préfixe du fichier")
    explore.add_argument("--max-bins", type=int, default=1200)
    replay = commands.add_parser("replay", help="Rejouer des mesures historiques en JSON")
    replay.add_argument("csv", type=Path)
    replay.add_argument("--limit", type=int, default=10)
    replay.add_argument("--chunksize", type=int, default=10_000)
    replay.add_argument("--speed", type=float, default=1.0)
    replay.add_argument("--start")
    replay.add_argument("--end")
    replay.add_argument("--max-delay", type=float)
    replay.add_argument("--wait", action="store_true", help="Attendre les délais calculés ; sinon émission immédiate")
    args = parser.parse_args(argv)
    try:
        if args.command == "download":
            print(download_dataset(args.directory, overwrite=args.overwrite))
        elif args.command == "explore":
            profile = profile_csv(args.csv, chunksize=args.chunksize, max_rows=args.max_rows)
            charts = aggregate_charts(args.csv, profile, max_bins=args.max_bins)
            print(write_report(profile, charts, args.output))
            print(json.dumps({key: profile[key] for key in ("scope", "rows_analyzed", "date_min", "date_max", "duplicate_measurement_rows", "duplicate_timestamps", "mode_counts")}, ensure_ascii=False))
        else:
            for event in iter_replay(
                args.csv, chunksize=args.chunksize, speed=args.speed, start=args.start, end=args.end,
                max_events=args.limit, max_delay_seconds=args.max_delay,
            ):
                if args.wait:
                    time.sleep(event.delay_seconds)
                print(json.dumps(asdict(event), default=str, ensure_ascii=False, allow_nan=False))
    except (OSError, ValueError, requests.RequestException) as error:
        parser.exit(2, f"Erreur : {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
