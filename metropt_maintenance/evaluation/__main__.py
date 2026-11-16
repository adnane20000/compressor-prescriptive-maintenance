"""CLI de campagne hors ligne ; l'intégration réelle se demande explicitement."""

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from .campaign import run_campaign
from .real_data import run_real_evaluation
from .report import write_reports


def read_test_results(path):
    """Lire le JUnit réellement produit par pytest, sans inventer de résultats."""
    if path is None:
        return None
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    tests = sum(int(suite.get("tests", 0)) for suite in suites)
    failures = sum(int(suite.get("failures", 0)) for suite in suites)
    errors = sum(int(suite.get("errors", 0)) for suite in suites)
    skipped = sum(int(suite.get("skipped", 0)) for suite in suites)
    return {"source": Path(path).name, "tests": tests, "passed": tests - failures - errors - skipped,
            "failures": failures, "errors": errors, "skipped": skipped,
            "time_seconds": sum(float(suite.get("time", 0)) for suite in suites),
            "suite_timestamps": [suite.get("timestamp") for suite in suites]}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Évaluation MetroPT : mock explicite, aucun appel API LLM.")
    parser.add_argument("--real", action="store_true", help="Ajouter les artefacts réels locaux (ML, E5, CSV), générations toujours mock.")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--work-directory", type=Path, default=Path("data/local/evaluation/runs"))
    parser.add_argument("--output", type=Path, default=Path("data/local/evaluation/report"))
    parser.add_argument("--junit", type=Path)
    args = parser.parse_args(argv)
    campaign = run_campaign(args.work_directory, repeats=args.repeats)
    # Garder les dossiers réels d'intégration hors du stockage d'usage courant.
    real = run_real_evaluation(directory=args.work_directory / campaign["executed_at_utc"].replace(":", "_") / "real", repeats=args.repeats) if args.real else None
    paths = write_reports(campaign, real, args.output, tests=read_test_results(args.junit))
    print(json.dumps({"reports": [str(path) for path in paths], "simulated_executions": len(campaign["rows"]),
                      "real_integration_requested": args.real, "real_llm_api_called": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
