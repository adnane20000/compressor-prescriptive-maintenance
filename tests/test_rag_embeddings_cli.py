"""Contrat Sentence Transformers sans téléchargement et commandes de recherche."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from metropt_maintenance.rag.__main__ import main
from metropt_maintenance.rag.embeddings import MODEL_REVISION, SentenceTransformerEncoder


def test_embeddings_use_e5_prefixes_normalization_and_local_cache(tmp_path, monkeypatch):
    calls = []
    class Model:
        def __init__(self, name, **kwargs):
            assert kwargs["local_files_only"] and not kwargs["trust_remote_code"]
            assert kwargs["revision"] == MODEL_REVISION
            self.tokenizer = SimpleNamespace(encode=lambda text, **kwargs: text.split())
        def encode(self, texts, **kwargs):
            calls.append((texts, kwargs))
            return np.ones((len(texts), 384), dtype="float32")
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=Model))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(set_num_threads=lambda count: None))
    encoder = SentenceTransformerEncoder(cache_directory=tmp_path)
    assert encoder.encode_documents(["plain PDF data"]).shape == (1, 384)
    assert encoder.encode_query("question").shape == (384,)
    assert calls[0][0] == ["passage: plain PDF data"]
    assert calls[1][0] == ["query: question"]
    assert calls[0][1]["normalize_embeddings"] is True
    with pytest.raises(ValueError, match="512 tokens"):
        encoder.encode_query("word " * 520)


def test_model_unavailable_error_does_not_silently_download(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        assert kwargs["local_files_only"] is True
        raise OSError("no cached model")
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=fail))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(set_num_threads=lambda count: None))
    with pytest.raises(RuntimeError, match="prepare-model"):
        SentenceTransformerEncoder(cache_directory=tmp_path).prepare()


def test_cli_search_empty_index_and_validation(tmp_path, capsys):
    assert main(["search", "AMDEC", "--index", str(tmp_path / "missing")]) == 0
    assert "information insuffisante" in capsys.readouterr().out
    with pytest.raises(SystemExit) as error:
        main(["search", ""])
    assert error.value.code == 2


def test_cli_prepare_and_index_dispatch(tmp_path, monkeypatch, capsys):
    class Encoder:
        signature = {"model": "fake-unit-test"}
        def __init__(self, **kwargs):
            if "allow_download" in kwargs:
                assert kwargs["allow_download"]
        def prepare(self): return self
    monkeypatch.setattr("metropt_maintenance.rag.__main__.SentenceTransformerEncoder", Encoder)
    assert main(["prepare-model", "--cache", str(tmp_path)]) == 0
    monkeypatch.setattr("metropt_maintenance.rag.__main__.index_documents", lambda *args, **kwargs: {"chunk_count": 0})
    assert main(["index", "--documents", str(tmp_path)]) == 0
    assert "chunk_count" in capsys.readouterr().out


def test_cli_evaluate_writes_measured_results(tmp_path, monkeypatch):
    import json
    benchmark = tmp_path / "questions.json"
    benchmark.write_text(json.dumps({"scope": "synthetic unit test", "cases": []}))
    monkeypatch.setattr("metropt_maintenance.rag.__main__.evaluate_retrieval", lambda *args, **kwargs: {"question_count": 0})
    output = tmp_path / "metrics.json"
    assert main(["evaluate", str(benchmark), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["scope"] == "synthetic unit test"
