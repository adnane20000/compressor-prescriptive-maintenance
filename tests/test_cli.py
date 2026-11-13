"""Commandes réellement exercées sur un CSV synthétique."""

import json

import pytest

from metropt_maintenance.ingestion.__main__ import main


def test_explore_command_creates_report(make_csv, tmp_path, capsys):
    output = tmp_path / "exploration"
    assert main(["explore", str(make_csv()), "--output", str(output), "--chunksize", "2", "--max-rows", "3"]) == 0
    assert (output / "report.html").is_file()
    assert '"rows_analyzed": 3' in capsys.readouterr().out


def test_replay_command_and_optional_scheduler(make_csv, capsys, monkeypatch):
    delays = []
    monkeypatch.setattr("metropt_maintenance.ingestion.__main__.time.sleep", delays.append)
    assert main(["replay", str(make_csv()), "--limit", "2", "--speed", "10", "--wait"]) == 0
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(events) == 2 and delays == [0, 1]
    assert events[1]["source_delta_seconds"] == 10


def test_cli_download_dispatch(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("metropt_maintenance.ingestion.__main__.download_dataset", lambda *args, **kwargs: tmp_path / "data.csv")
    assert main(["download", "--directory", str(tmp_path)]) == 0
    assert "data.csv" in capsys.readouterr().out


def test_cli_reports_missing_file(capsys):
    with pytest.raises(SystemExit) as error:
        main(["explore", "absent.csv"])
    assert error.value.code == 2
    assert "introuvable" in capsys.readouterr().err
