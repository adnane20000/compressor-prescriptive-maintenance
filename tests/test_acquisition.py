"""Archive officielle simulée en mémoire ; les tests n'utilisent pas le réseau."""

import io
import json
from zipfile import ZipFile

import pytest
import requests

from metropt_maintenance.ingestion.acquisition import download_dataset
from metropt_maintenance.ingestion.schema import ARCHIVE_URL, CSV_NAME


def mock_response(monkeypatch, content):
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def raise_for_status(self): pass
        def iter_content(self, chunk_size):
            assert chunk_size == 1024 * 1024
            for offset in range(0, len(content), 64):
                yield content[offset:offset + 64]
    def get(url, **options):
        assert url == ARCHIVE_URL and options["stream"] is True
        return Response()
    monkeypatch.setattr(requests, "get", get)


def test_download_streams_validates_and_extracts_only_expected_member(make_csv, tmp_path, monkeypatch):
    buffer = io.BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr(CSV_NAME, make_csv().read_bytes())
        archive.writestr("../../unexpected.txt", "must not be extracted")
    mock_response(monkeypatch, buffer.getvalue())
    target = download_dataset(tmp_path / "raw")
    assert target.name == CSV_NAME
    assert target.is_file()
    assert not (tmp_path / "unexpected.txt").exists()
    metadata = json.loads(target.with_suffix(".source.json").read_text())
    assert metadata["historical_data"] is True
    assert len(metadata["archive_sha256"]) == 64
    with pytest.raises(FileExistsError):
        download_dataset(tmp_path / "raw")


def test_bad_archive_preserves_existing_csv_and_cleans_temporary_files(tmp_path, monkeypatch):
    target = tmp_path / CSV_NAME
    target.write_text("previous version")
    mock_response(monkeypatch, b"invalid zip")
    with pytest.raises(ValueError, match="Archive UCI invalide"):
        download_dataset(tmp_path, overwrite=True)
    assert target.read_text() == "previous version"
    assert sorted(path.name for path in tmp_path.iterdir()) == [CSV_NAME]


def test_network_error_leaves_no_partial_csv(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise requests.ConnectionError("network unavailable")
    monkeypatch.setattr(requests, "get", fail)
    with pytest.raises(requests.ConnectionError):
        download_dataset(tmp_path / "raw")
    assert list((tmp_path / "raw").iterdir()) == []
