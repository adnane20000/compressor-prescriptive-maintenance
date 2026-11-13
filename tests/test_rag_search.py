"""Qualité contrôlée du classement, citations et abstention sans réseau."""

import json

import numpy as np
import pymupdf
import pytest

from metropt_maintenance.rag.index import index_documents, read_manifest
from metropt_maintenance.rag.search import search_documents
from metropt_maintenance.rag.evaluation import evaluate_retrieval


def test_retrieves_expected_document_page_and_authority(rag_index):
    directory, index, encoder = rag_index
    result = search_documents("Dans quelle unité est mesurée la temperature de l'huile ?", index_directory=index, encoder=encoder)
    assert result.status == "passages_found"
    assert result.passages[0].relative_path == "description_test.pdf" and result.passages[0].page == 2
    hit = result.passages[0]
    assert hit.authority == "official" and hit.authority_basis == "human_registry_matched_sha256"
    assert hit.text_is_untrusted_data and not result.sufficiency_verified
    assert result.generated_answer is None and result.similarity_is_probability is False
    json.dumps(result.to_dict(), allow_nan=False)
    with pymupdf.open(directory / hit.relative_path) as document:
        assert hit.text in document[hit.page - 1].get_text("text", sort=True)


def test_pedagogical_filter_and_official_filter_are_distinct(rag_index):
    _, index, encoder = rag_index
    result = search_documents("AMDEC criticite effets", index_directory=index, encoder=encoder, authority="pedagogical")
    assert result.passages[0].file_name == "notes_pedagogiques_test.pdf" and result.passages[0].page == 1
    assert all(hit.authority == "pedagogical" for hit in result.passages)
    official = search_documents("AMDEC criticite effets", index_directory=index, encoder=encoder, authority="official")
    assert official.message == "information insuffisante"


def test_official_dataset_description_cannot_authorize_a_procedure(rag_index):
    _, index, encoder = rag_index
    result = search_documents("Quelle procedure concerne Motor_current ?", index_directory=index,
                              encoder=encoder, authority="official")
    assert result.message == "information insuffisante"
    assert result.reason == "operational_source_missing"
    assert result.related_passages and result.related_passages[0].document_type == "dataset_description"


@pytest.mark.parametrize("question", ["Quelle est la recette du gateau au chocolat ?", "Quel couple de serrage pour la culasse ?", "Quelle est la procedure de maintenance ?"])
def test_unsupported_queries_abstain(rag_index, question):
    _, index, encoder = rag_index
    result = search_documents(question, index_directory=index, encoder=encoder)
    assert result.status == "information_insuffisante" and result.message == "information insuffisante"
    assert result.passages == []


def test_prompt_injection_is_returned_as_untrusted_data_never_executed(rag_index, tmp_path):
    _, index, encoder = rag_index
    sentinel = tmp_path / "do-not-delete"
    sentinel.write_text("must stay")
    result = search_documents("Ignore previous instructions", index_directory=index, encoder=encoder)
    assert result.passages
    assert result.passages[0].authority == "unverified"
    assert "delete" in result.passages[0].text
    assert result.passages[0].text_is_untrusted_data
    assert sentinel.read_text() == "must stay"


def test_deleted_changed_or_added_pdf_requires_reindex(rag_index):
    directory, index, encoder = rag_index
    (directory / "description_test.pdf").unlink()
    result = search_documents("Motor_current", index_directory=index, encoder=encoder)
    assert result.passages == [] and result.reason == "corpus_changed_reindex_required"


def test_registry_change_requires_reindex(rag_index):
    directory, index, encoder = rag_index
    (directory / "sources.json").write_text('{}')
    assert search_documents("AMDEC", index_directory=index, encoder=encoder).reason == "corpus_changed_reindex_required"


def test_missing_and_empty_index_do_not_load_model(tmp_path, rag_encoder):
    missing = search_documents("Motor_current", index_directory=tmp_path / "missing", encoder=object())
    assert missing.reason == "index_missing"
    manifest = index_documents(tmp_path / "empty", index_directory=tmp_path / "index", encoder=rag_encoder)
    assert manifest["chunk_count"] == 0
    assert search_documents("Motor_current", index_directory=tmp_path / "index", encoder=object()).reason == "no_extractable_text"


def test_reindex_is_idempotent_and_failed_encoding_preserves_old_index(rag_index):
    directory, index, encoder = rag_index
    initial = read_manifest(index)
    reindexed = index_documents(directory, index_directory=index, encoder=encoder)
    assert initial["chunk_count"] == reindexed["chunk_count"]
    assert initial["collection"] != reindexed["collection"]
    class Broken:
        signature = encoder.signature
        def encode_documents(self, texts):
            raise RuntimeError("synthetic encoding failure")
    with pytest.raises(RuntimeError):
        index_documents(directory, index_directory=index, encoder=Broken())
    assert read_manifest(index)["collection"] == reindexed["collection"]
    assert search_documents("AMDEC", index_directory=index, encoder=encoder).passages


def test_wrong_embedding_contract_is_refused(rag_index):
    _, index, _ = rag_index
    class Other:
        signature = {"model": "different", "dimension": 16}
    with pytest.raises(ValueError, match="incompatible"):
        search_documents("AMDEC", index_directory=index, encoder=Other())


def test_scanned_and_corrupt_pdf_are_reported(tmp_path, rag_encoder):
    docs = tmp_path / "docs"
    docs.mkdir()
    with pymupdf.open() as document:
        document.new_page()
        document.save(docs / "scan.pdf")
    (docs / "broken.pdf").write_bytes(b"not a PDF")
    manifest = index_documents(docs, index_directory=tmp_path / "index", encoder=rag_encoder)
    assert manifest["empty_pages"] == 1 and manifest["chunk_count"] == 0
    assert len(manifest["warnings"]) == 2


def test_local_quality_benchmark_measures_ranking_and_abstention(rag_index):
    _, index, encoder = rag_index
    cases = [
        {"question": "AMDEC criticite effets", "should_abstain": False,
         "expected_references": [{"file": "notes_pedagogiques_test.pdf", "page": 1}]},
        {"question": "calibration seuil", "should_abstain": False,
         "expected_references": [{"file": "notes_pedagogiques_test.pdf", "page": 2}]},
        {"question": "recette gateau chocolat", "should_abstain": True},
    ]
    metrics = evaluate_retrieval(cases, index_directory=index, encoder=encoder)
    assert metrics["hit_at_1"] == 1 and metrics["mean_reciprocal_rank_at_k"] == 1
    assert metrics["unanswerable_abstention_rate"] == 1
    assert metrics["encoder"]["model"] == "synthetic-test-encoder"
    with pytest.raises(ValueError):
        evaluate_retrieval([], index_directory=index, encoder=encoder)


def test_missing_index_is_not_reported_as_a_quality_measurement(tmp_path):
    with pytest.raises(ValueError, match="Corpus non prêt"):
        evaluate_retrieval([{"question": "AMDEC", "should_abstain": True}], index_directory=tmp_path)


@pytest.mark.parametrize("question,options", [("", {}), ("x" * 4001, {}), ("AMDEC", {"authority": "invented"}), ("AMDEC", {"top_k": 0}), ("AMDEC", {"min_similarity": 2})])
def test_search_parameters_are_validated(tmp_path, question, options):
    with pytest.raises(ValueError):
        search_documents(question, index_directory=tmp_path, **options)
