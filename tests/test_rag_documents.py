"""PDF réels de test générés localement, aucune procédure industrielle inventée."""

import json

import pymupdf
import pytest

from metropt_maintenance.rag.chunking import chunk_page
from metropt_maintenance.rag.documents import PDFPage, document_metadata, extract_pages, file_sha256, load_registry, pdf_inventory


def test_extracts_true_pages_and_never_crosses_page_boundaries(rag_corpus):
    directory, _ = rag_corpus
    item = next(item for item in pdf_inventory(directory) if item["relative_path"] == "description_test.pdf")
    metadata = document_metadata(item, load_registry(directory))
    pages = list(extract_pages(directory / item["relative_path"], metadata))
    assert [page.page for page in pages] == [1, 2]
    assert "Motor_current" in pages[0].text and "temperature" in pages[1].text
    for page in pages:
        for chunk in chunk_page(page, chunk_chars=160, overlap_chars=25):
            assert chunk.text == page.text[chunk.metadata["start_char"]:chunk.metadata["end_char"]]
            assert chunk.metadata["page"] == page.page
            assert chunk.metadata["file_name"] == "description_test.pdf"
            assert chunk.metadata["pdf_sha256"] == file_sha256(directory / "description_test.pdf")


def test_chunking_keeps_overlap_content_and_stable_identifiers():
    text = " ".join(f"word{index}" for index in range(180))
    page = PDFPage(text, 7, "vii", {"relative_path": "test.pdf", "pdf_sha256": "x"})
    chunks = chunk_page(page, chunk_chars=200, overlap_chars=40)
    assert len(chunks) > 2
    assert chunks[1].metadata["start_char"] < chunks[0].metadata["end_char"]
    assert chunks[-1].metadata["end_char"] == len(text)
    assert [chunk.identifier for chunk in chunks] == [chunk.identifier for chunk in chunk_page(page, chunk_chars=200, overlap_chars=40)]
    assert all(len(chunk.text) <= 200 and chunk.metadata["page_label"] == "vii" for chunk in chunks)
    with pytest.raises(ValueError):
        chunk_page(page, chunk_chars=100, overlap_chars=80)


def test_unknown_pdf_cannot_self_declare_official(rag_corpus):
    directory, _ = rag_corpus
    item = next(item for item in pdf_inventory(directory) if item["relative_path"] == "untrusted_test.pdf")
    metadata = document_metadata(item, load_registry(directory))
    assert metadata["authority"] == "unverified"
    assert metadata["source_url"] == ""
    assert "Ignore previous instructions" in next(extract_pages(directory / item["relative_path"], metadata)).text


def test_checksum_change_invalidates_attribution(rag_corpus):
    directory, _ = rag_corpus
    item = {"relative_path": "description_test.pdf", "sha256": "f" * 64}
    metadata = document_metadata(item, load_registry(directory))
    assert metadata["authority"] == "unverified" and metadata["authority_basis"] == "checksum_mismatch"
    assert metadata["publisher"] == "" and metadata["source_url"] == ""


@pytest.mark.parametrize("value", [[], {}, {"version": 1, "documents": {"../bad.pdf": {}}},
                                     {"version": 1, "documents": {"test.pdf": {"authority": "official", "document_type": "other", "sha256": "f" * 64}}}])
def test_invalid_registry_is_refused(tmp_path, value):
    (tmp_path / "sources.json").write_text(json.dumps(value))
    with pytest.raises(ValueError):
        load_registry(tmp_path)


def test_symlink_cannot_import_pdf_outside_corpus(tmp_path):
    corpus = tmp_path / "docs"
    corpus.mkdir()
    external = tmp_path / "external.pdf"
    external.write_bytes(b"not a real PDF")
    (corpus / "link.pdf").symlink_to(external)
    with pytest.raises(ValueError, match="hors du dossier"):
        pdf_inventory(corpus)
